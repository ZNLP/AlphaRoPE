import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaConfig
# from new_rerope_patch import _compute_dynamic_ntk_parameters
# from rerope_yarn_patch import _compute_yarn_parameters
import transformers.modeling_rope_utils as rope_utils
# rope_utils.ROPE_INIT_FUNCTIONS["dynamic"] = _compute_dynamic_ntk_parameters
# rope_utils.ROPE_INIT_FUNCTIONS["yarn"] = _compute_yarn_parameters
from datasets import load_dataset
from tqdm import tqdm
import math
import pandas as pd # 导入 pandas
import os 
import gc
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='4'

# ========== 1. 解析命令行参数 ==========
def get_args():
    parser = argparse.ArgumentParser(description="滑动窗口困惑度评估")
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="模型路径或HuggingFace模型名"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/ppl_after_finetune",
        help="结果保存目录"
    )
    parser.add_argument(
        "--segment_ratio", type=int, default=4,
        help="结果保存目录"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=16384,
        help="最大评估上下文长度"
    )
    parser.add_argument(
        "--min_seq_len", type=int, default=256,
        help="最小评估上下文长度"
    )
    parser.add_argument(
        "--token_steps", type=int, default=512,
        help="每次窗口起点之间的采样间隔"
    )
    parser.add_argument(
        "--sliding_window_step", type=int, default=256,
        help="滑动窗口内部每步新预测的token数量"
    )
    parser.add_argument(
        "--num_docs", type=int, default=100,
        help="用于拼接的大文本的文档数"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="模型与推理设备"
    )
    parser.add_argument(
        "--aggressive_memory", action="store_true",
        help="每步是否主动清理CUDA显存"
    )
    parser.add_argument(
        "--hide_progress", action="store_true",
        help="隐藏进度条"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext",
        help="HuggingFace数据集库名"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-2-raw-v1",
        help="HuggingFace数据集配置"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="validation",
        help="数据集分割名"
    )
    return parser.parse_args()

# ========== 2. 滑动窗口困惑度计算 ==========
def compute_perplexity_with_sliding_window(
    model,
    tokenizer,
    tokenized_full_text: torch.Tensor,
    current_context_len: int,
    token_steps: int,
    sliding_window_step: int,
    device: str,
    add_start_token: bool = True,
    hide_progress: bool = False,
    aggressive_memory: bool = False,
):
    model.eval()
    nlls = []
    input_len_for_model = current_context_len

    if input_len_for_model < 2:
        return float('inf')

    labels = tokenized_full_text.unsqueeze(0).to(device)
    seq_len = labels.size(1)
    num_windows = (seq_len - 1) // sliding_window_step + 1
    if num_windows <= 0:
        return float('inf')

    pbar = tqdm(
        range(0, seq_len - 1, token_steps),
        disable=hide_progress,
        desc="滑动窗口 PPL 计算"
    )
    for begin_loc in pbar:
        end_loc_for_model_input = min(begin_loc + input_len_for_model, seq_len)
        input_ids_chunk = labels[:, begin_loc:end_loc_for_model_input]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]]).to(device)
            input_ids_chunk = torch.cat([bos_tokens_tensor, input_ids_chunk], dim=1)

        padding_needed = (input_len_for_model + (1 if add_start_token else 0)) - input_ids_chunk.size(1)
        if padding_needed > 0:
            padding_tensor = torch.full((1, padding_needed), tokenizer.pad_token_id, dtype=torch.long).to(device)
            input_ids_chunk = torch.cat([input_ids_chunk, padding_tensor], dim=1)

        final_input_length = current_context_len + (1 if add_start_token else 0)
        if input_ids_chunk.size(1) > final_input_length:
            input_ids_chunk = input_ids_chunk[:, :final_input_length]

        trg_len = end_loc_for_model_input - begin_loc
        trg_len = min(trg_len, input_ids_chunk.size(1) - 1)
        if trg_len <= 0:
            break

        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        if torch.isnan(neg_log_likelihood):
            print(f"警告: 困惑度计算出现 NaN，跳过此窗口。Context len: {current_context_len}, begin_loc: {begin_loc}")
            continue

        if aggressive_memory:
            del outputs, input_ids_chunk, target_ids
            gc.collect()
            torch.cuda.empty_cache()

        nlls.append(neg_log_likelihood)

        if len(nlls) > 0:
            current_ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=f"{current_ppl:.4f}")

    if len(nlls) == 0:
        return float('inf')

    final_ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return final_ppl

# ========== 3. PPL 曲线采样点生成 ==========
def generate_ppl_eval_points(min_len, max_len, tokens_step):
    points = []
    current = min_len
    while current <= max_len:
        points.append(current)
        current += tokens_step
    if max_len not in points and max_len >= min_len:
        points.append(max_len)
    return sorted(list(set(points)))

# ========== 4. 主流程 ==========
def main():
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device if args.device.isdigit() else ""

    print(f"正在加载数据集 '{args.dataset_name}/{args.dataset_config}' 的 {args.dataset_split} 分割...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)

    print(f"限定加载前 {args.num_docs} 个文档...")
    all_raw_text = "\n\n".join([
        item["text"] for item in dataset if item["text"].strip() != ""
    ][:args.num_docs])
    print(f"总长度为 {len(all_raw_text)} 字符。")

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("对数据集进行一次性 Tokenize ...")
    full_text_token_ids = tokenizer(
        all_raw_text,
        return_tensors='pt',
        truncation=False,
        padding=False
    )['input_ids'][0]

    print("加载模型 ...")
    # from dynamic.modeling_llama import LlamaForCausalLM
    # rope_scaling = {"type": "dynamic", "factor": 1.0, "segment_ratio": args.segment_ratio}

    # rope_scaling = {"type": "dynamic", "factor": 8.0}
    # config = AutoConfig.from_pretrained(
    #     args.model_name_or_path,
    #     rope_scaling=rope_scaling
    # )
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,config=config).to(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    model.eval()

    print("生成 PPL 曲线采样点 ...")
    PPL_CURVE_EVAL_POINTS = generate_ppl_eval_points(args.min_seq_len, args.max_seq_len, args.token_steps)
    print(f"PPL 曲线将评估的上下文长度点: {PPL_CURVE_EVAL_POINTS}")

    results_for_config = []
    for current_seq_len in tqdm(PPL_CURVE_EVAL_POINTS, desc=f"正在计算PPL 点"):
        perplexity = compute_perplexity_with_sliding_window(
            model=model,
            tokenizer=tokenizer,
            tokenized_full_text=full_text_token_ids,
            current_context_len=current_seq_len,
            token_steps=args.token_steps,
            sliding_window_step=args.sliding_window_step,
            device=args.device,
            add_start_token=tokenizer.bos_token is not None,
            hide_progress=args.hide_progress,
            aggressive_memory=args.aggressive_memory
        )
        results_for_config.append({'seq_len': current_seq_len, 'perplexity': perplexity})

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(
        args.output_dir,
        f"{os.path.basename(args.model_name_or_path).replace('/', '_')}_ppl_{args.max_seq_len}_yarn_curve.csv"
    )
    df = pd.DataFrame(results_for_config)
    df.to_csv(output_csv_path, index=False)
    print(f"PPL 曲线结果已保存到 {output_csv_path}")

if __name__ == "__main__":
    main()


