import torch
from torch.utils.data import DataLoader
from transformers import set_seed, default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from tqdm import tqdm
import copy
import os
import argparse
import signal
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import wandb
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3' 

# 忽略 SIGHUP 信号，防止 SSH 断开导致训练中断
signal.signal(signal.SIGHUP, signal.SIG_IGN) 




def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(42)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    # 只在需要时启用 wandb
    log_with = "wandb" if args.use_wandb else None

    accelerator = Accelerator(
        mixed_precision = "bf16",
        gradient_accumulation_steps = args.gradient_accumulate_every,
        log_with=log_with,
        kwargs_handlers=[timeout]
    )

    print("\n===== DeepSpeed 信息 =====")
    print(f"DeepSpeed 是否启用: {accelerator.state.deepspeed_plugin is not None}")
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")
    device = accelerator.device

    # === 3. wandb 初始化（只主进程，可选） ===
    if args.use_wandb and accelerator.is_main_process:
        # 设置 wandb 模式：offline（离线模式，避免网络超时）或 online（在线模式）
        wandb_mode = os.environ.get("WANDB_MODE", "offline")  # 默认离线模式
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode=wandb_mode,
            settings=wandb.Settings(
                _disable_stats=False,
                _disable_meta=False,
                start_method="thread"
            )
        )

    # try:
    #     train_dataset = load_dataset(args.dataset)
    # except:
    #     train_dataset = load_from_disk(args.dataset)
    train_dataset = load_from_disk(args.dataset)
    
    # train_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    if isinstance(train_dataset, DatasetDict):
        # train_dataset = train_dataset["train"]
        train_dataset = train_dataset["test"]
    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    if "labels" not in train_dataset.column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample
        train_dataset = train_dataset.map(
            add_labels, desc="Adding labels", num_proc=args.num_proc)
    if "attention_mask" not in train_dataset.column_names:
        def add_attention_mask(sample):
            sample["attention_mask"] = torch.ones(
                len(sample["input_ids"]), dtype=torch.int8)
            return sample
        train_dataset = train_dataset.map(
            add_attention_mask, desc="Adding attention mask", num_proc=args.num_proc)

    if args.truncate:
        def truncate(sample):
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            sample["labels"] = sample["labels"][0:args.truncate]
            sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        train_dataset = train_dataset.map(
            truncate, desc="Truncating", num_proc=args.num_proc)


    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size
    )


    # === 6. 加载分词器和模型 ===
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rope_scaling = None
    if args.rope_scaling_type is not None:
        if args.rope_scaling_type == "yarn":
            rope_scaling = {"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
        elif args.rope_scaling_type == "dynamic":
            # from dynamic.modeling_rope_utils import _compute_dynamic_ntk_parameters
            # import dynamic.modeling_rope_utils as rope_utils
            # rope_utils.ROPE_INIT_FUNCTIONS["dynamic"] = _compute_dynamic_ntk_parameters
            # rope_scaling = {"type": "dynamic", "factor": 1.0, "segment_ratio": args.dynamic_segment_ratio}
            rope_scaling = {"type": "dynamic", "factor": 1.0}
        elif args.rope_scaling_type == "ntk_yarn":
            rope_scaling = {"type": "ntk_yarn", "factor": args.rope_scaling_factor}
        elif args.rope_scaling_type == "ntk":
            rope_scaling = {"type": "ntk", "factor": 1.0, "scaling_factor":args.rope_scaling_factor}
        elif args.rope_scaling_type == "my_new":
            rope_scaling = {"type": "my_new", "factor": args.rope_scaling_factor}
        elif args.rope_scaling_type == "mixed":
            rope_scaling = {
                "type": "mixed_radix", 
                "factor": args.rope_scaling_factor,
                "power_exponent": args.power_exponent
            }

        else:
            rope_scaling = {"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
    # from ntk_yarn.configuration_olmo import OlmoConfig
    # config = OlmoConfig.from_pretrained(
    #     args.model,
    #     rope_scaling=rope_scaling
    # )
    # # config = AutoConfig.from_pretrained(
    # #     args.model,
    # #     rope_scaling=rope_scaling
    # # )
    # print(config)  # 检查是否有异常配置

    # from transformers import LlamaForCausalLM
    if "llama" in args.model.lower():
        from ntk_yarn.modeling_llama import LlamaForCausalLM
        from ntk_yarn.configuration_llama import LlamaConfig
        config = LlamaConfig.from_pretrained(
            args.model,
            rope_scaling=rope_scaling
        )
        model = LlamaForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        print("using llama")
    elif "olmo" in args.model.lower():
        from ntk_yarn.modeling_olmo import OlmoForCausalLM
        from ntk_yarn.configuration_olmo import OlmoConfig
        config = OlmoConfig.from_pretrained(
            args.model,
            rope_scaling=rope_scaling
        )
        model = OlmoForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        print("using olmo")
    # from ntk_yarn.modeling_olmo import OlmoForCausalLM
    # model = OlmoForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # print("using olmo")

    torch.cuda.synchronize() 
    print(f"lm_head.weight:{model.lm_head.weight.shape}") 
    model = model.train()
    model = model.to(device) 



    # === 7. 优化器和lr scheduler ===

    optimizer = DummyOptim(model.parameters(), lr=args.learning_rate)
    lr_scheduler = DummyScheduler(
        optimizer, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
    model.gradient_checkpointing_enable()
    model.lm_head.weight.requires_grad = False 
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    accelerator.register_for_checkpointing(lr_scheduler)
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = (
            int(training_difference.replace("step_", ""))
        )

    if args.resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    loss_file = open(args.log_loss, "a" if args.resume_from_checkpoint else "w") if args.log_loss and accelerator.is_main_process else None

    if not args.save_only:
        # wandb 初始化已在前面完成（如果需要）
        pass
            

        model.train()
        for step, batch in enumerate(train_loader):
            # Forward pass and loss calculation
            loss = model(**batch).loss
            
            # Backward pass
            accelerator.backward(loss)

            # Gradient accumulation handling
            if (step + 1) % args.gradient_accumulate_every == 0 or (step + 1) == len(train_loader):
                # Gradient clipping if specified
                if isinstance(args.grad_norm, float):
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Logging and progress tracking
                loss_log = {"loss": loss.item()}
                accelerator.log(loss_log, step=completed_steps)
                
                #############################################
                # WandB监控代码（可选）
                if args.use_wandb and accelerator.is_main_process:
                    try:
                        if wandb.run:
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "step": completed_steps,
                                "epoch": completed_steps / len(train_loader)
                            })
                    except:
                        pass  # 忽略 wandb 错误，不影响训练
                #############################################
                
                if loss_file is not None:
                    loss_file.write(f"{loss_log['loss']},")
                    loss_file.flush()
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(loss_log)
                
                # Increment completed steps
                completed_steps += 1
                
                # Checkpoint saving
                if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                    if completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        
                        # 同时保存标准格式的模型（便于后续加载）
                        if accelerator.is_main_process:
                            accelerator.wait_for_everyone()
                            
                            # 检查是否使用 DeepSpeed ZeRO-3
                            if accelerator.distributed_type == "DEEPSPEED" and accelerator.deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3:
                                # ZeRO-3 使用 save_model
                                accelerator.save_model(model, output_dir)
                            else:
                                # ZeRO-2 或其他情况
                                unwrapped_model = accelerator.unwrap_model(model)
                                state_dict = accelerator.get_state_dict(unwrapped_model)
                                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                            
                            # 保存配置和 tokenizer
                            config.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            print(f"标准格式模型已保存到 {output_dir}")
            
            # Early stopping if max steps reached
            if completed_steps >= args.max_train_steps:
                break

        accelerator.print("Training Finished")
        # accelerator.end_training()

        if args.output_dir:
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                os.makedirs(args.output_dir, exist_ok=True)
            
            # 检查是否使用 DeepSpeed ZeRO-3
            if accelerator.distributed_type == "DEEPSPEED" and accelerator.deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3:
                print("Using zero3")
                # ZeRO-3 特殊处理
                accelerator.wait_for_everyone()
                
                # 方法1: 使用 save_model (推荐)
                accelerator.save_model(model, args.output_dir)
                
                # 或者方法2: 手动收集参数
                # unwrapped_model = accelerator.unwrap_model(model)
                # state_dict = accelerator.get_state_dict(unwrapped_model)
                # if accelerator.is_main_process:
                #     torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
                
            else:
                # ZeRO-2 或其他情况的处理
                print("Using zero2")
                if accelerator.is_main_process:
                    # 获取完整的 state_dict
                    state_dict = accelerator.get_state_dict(model, unwrap=True)
                    torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
            
            # 保存配置和 tokenizer (只在主进程)
            if accelerator.is_main_process:
                config.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"模型保存成功到 {args.output_dir}")
            
            accelerator.wait_for_everyone()

        # 清理资源
        accelerator.end_training()

    # if args.output_dir:
    #     accelerator.wait_for_everyone()
        
    #     if accelerator.is_main_process:
    #         os.makedirs(args.output_dir, exist_ok=True)
            
    #         # 修复1: 直接保存完整权重 (绕过 DeepSpeed 分片)
    #         torch.save(
    #             accelerator.get_state_dict(model, unwrap=False),  # 关键参数
    #             os.path.join(args.output_dir, "pytorch_model.bin")
    #         )
            
    #         # 修复2: 使用原始 config 对象保存
    #         config.save_pretrained(args.output_dir)
            
    #         # 保存 tokenizer
    #         tokenizer.save_pretrained(args.output_dir)
            
    #         print(f"模型保存成功到 {args.output_dir}")

    # # === 关键修复：清理分布式资源 ===
    # accelerator.end_training()





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--use-wandb", action="store_true", help="启用 wandb 日志记录（默认禁用）")
    args.add_argument("--wandb-project", type=str, help="wandb 项目名称（需要 --use-wandb）")
    args.add_argument("--wandb-run-name", type=str, help="wandb 运行名称（需要 --use-wandb）")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", action="store_true")
    args.add_argument("--lora", action="store_true")
    args.add_argument("--model", type=str)
    args.add_argument("--power-exponent", type=float, default=1.0)
    args.add_argument("--rope-scaling-factor", type=float)
    args.add_argument("--rope-scaling-type", type=str)
    args.add_argument("--rope-theta", type=float, default=10000.0)
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset", type=str)
    args.add_argument("--deepspeed", action="store_true")
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--max-position-embeddings", type=int)
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--original-max-position-embeddings", type=int)
    main(args.parse_args())