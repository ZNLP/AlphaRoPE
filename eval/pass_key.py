import os
import math
from transformers import GenerationConfig
import torch
import json
import argparse
import random
import re
import numpy as np
from numpy import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from transformers import LlamaForCausalLM
# from dynamic.modeling_llama import LlamaForCausalLM
# from new_rerope_patch1 import _compute_dynamic_ntk_parameters
# import transformers.modeling_rope_utils as rope_utils
# rope_utils.ROPE_INIT_FUNCTIONS["dynamic"] = _compute_dynamic_ntk_parameters

# from ntk_patch import _compute_dynamic_ntk_parameters
# import transformers.modeling_rope_utils as rope_utils
# rope_utils.ROPE_INIT_FUNCTIONS["dynamic"] = _compute_dynamic_ntk_parameters

# model_path = "/workspace/RWKV-block/test/v7_goose/.hf_build/v7-1B5-world/"

def get_gpu_memory():
    """Returns the current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def parse_config():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments including:
            - Standard evaluation parameters
            - HF model path and cache directory
            - Optional HF model arguments as JSON string
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--hf_model', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--min_tokens', type=int, default=2048, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=65536, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2048, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--max_depth', type=float, default=1.0, help='max depth ratio to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computation')
    parser.add_argument('--hf_model_args', type=str, default='{}',
                      help='Additional HuggingFace model arguments as JSON string')
    args = parser.parse_args()
    return args

# def generate_prompt_landmark(tokenizer, pass_key, context_length, depth, final_context_length_buffer=250):
#     needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
#     task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
#     garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
#     question = "What is the pass key? The pass key is"

#     # encode all fixed parts
#     tokens_task = tokenizer.encode(task_description)
#     tokens_needle = tokenizer.encode('\n' + needle + '\n')
#     tokens_question = tokenizer.encode('\n' + question)

#     # 可用的 context token 数
#     meta_token_count = len(tokens_task) + len(tokens_needle) + len(tokens_question)
#     max_context_len = context_length - final_context_length_buffer - meta_token_count

#     tokens_garbage = tokenizer.encode(garbage)
#     multiplier = math.ceil(max_context_len / len(tokens_garbage))
#     context = (garbage * multiplier)[:10000]
#     tokens_context = tokenizer.encode(context)[:max_context_len]

#     # 插入 needle
#     if depth <= 0:
#         parts = [tokens_task, tokens_needle, tokens_context, tokens_question]
#     elif depth >= 1:
#         parts = [tokens_task, tokens_context, tokens_needle, tokens_question]
#     else:
#         insert_pos = int(len(tokens_context) * depth)
#         # 向前找到最近句号结尾
#         period_token = tokenizer.encode('.')
#         while insert_pos > 0 and tokens_context[insert_pos-1] not in period_token:
#             insert_pos -= 1
#         parts = [tokens_task, tokens_context[:insert_pos], tokens_needle, tokens_context[insert_pos:], tokens_question]

#     # 拼接所有 token
#     tokens_new_context = []
#     for p in parts:
#         tokens_new_context.extend(p)

#     # 严格控制长度（此时 tokens_new_context 必定长度一致）
#     tokens_new_context = tokens_new_context[:context_length - final_context_length_buffer]
#     new_context = tokenizer.decode(tokens_new_context)
#     print("Total Tokens in Context: ", len(tokens_new_context))
#     return new_context
def generate_prompt_landmark(tokenizer, pass_key, context_length, depth, final_context_length_buffer=250):
    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. "
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    question = "What is the pass key? The pass key is"
    
    tokens_in_garbage = len(tokenizer.encode(garbage))
    multiplier = math.ceil((context_length - len(tokenizer.encode(task_description)) - 25) / tokens_in_garbage)
    context = garbage * multiplier
    
    tokens_task = tokenizer.encode(task_description)
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(context)
    tokens_question = tokenizer.encode(question)
    tokens_newline = tokenizer.encode("\n")
    
    # Reduce context length by buffer
    context_length = context_length - final_context_length_buffer - len(tokens_task) - len(tokens_question)
    
    # Truncate context if needed
    if len(tokens_context) + len(tokens_task) + len(tokens_needle) + len(tokens_question) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]
    
    if depth >= 1:
        tokens_new_context = tokens_task + tokens_context + tokens_newline + tokens_needle + tokens_newline + tokens_question

    elif depth == 0:
        tokens_new_context = tokens_task + tokens_needle + tokens_newline + tokens_context + tokens_newline + tokens_question

    else:
        insertion_point = int(len(tokens_context) * depth)
        tokens_new_context = tokens_context[:insertion_point]
        
        # Find sentence break
        period_tokens = tokenizer.encode('.')
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        
        tokens_new_context = tokens_task + tokens_new_context + tokens_newline + tokens_needle + tokens_newline + tokens_context[insertion_point:] + tokens_question
    
    print("Total Tokens in Context: ", len(tokens_new_context))
    new_context = tokenizer.decode(tokens_new_context)
    return new_context

def passkey_retrieval_test(model, tokenizer, device, context_length, depth, seed=666):
    # Generate random pass key
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)

    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)
    input_token_ids = tokenizer(prompt, return_tensors=None).input_ids
    input_ids = torch.tensor([input_token_ids], device=device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer).input_ids # Get token IDs for answer length
    max_new_tokens = len(answer_ids) + 20

    past_key_values = None
    processed_len = 0
    prefill_ids = input_ids[:, :-1]
    prefill_len = prefill_ids.shape[1]
    chunk_size = 2048

    # Process the prompt in chunks (for long context)
    with torch.no_grad():
        # Chunked prefill stage
        for i in range(0, prefill_len, chunk_size):
            chunk = prefill_ids[:, i : min(i + chunk_size, prefill_len)]
            chunk_len = chunk.shape[1]

            current_position_ids = torch.arange(processed_len, processed_len + chunk_len, dtype=torch.long, device=device).unsqueeze(0)

            outputs = model(
                input_ids=chunk,
                past_key_values=past_key_values,
                position_ids=current_position_ids,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            processed_len += chunk_len

        # Final forward pass for the last token
        last_token_input_ids = input_ids[:, -1:]
        last_token_pos_id = torch.tensor([[processed_len]], dtype=torch.long, device=device)
        final_prompt_mask = torch.ones(1, processed_len + 1, dtype=torch.long, device=device)

        outputs = model(
            input_ids=last_token_input_ids,
            past_key_values=past_key_values,
            position_ids=last_token_pos_id,
            use_cache=True,
        )
        logits = outputs.logits  # Logits for the *next* token
        past_key_values = outputs.past_key_values  # Final cache after full prompt
        processed_len += 1

        # Generation stage
        generated_ids_list = []

        # Get the first generated token ID
        next_token_logits = logits[:, -1, :]  # Shape [batch_size, vocab_size]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Shape [batch_size, 1]

        for step in range(max_new_tokens):
            generated_ids_list.append(next_token_id.item())

            # Prepare inputs for the next step model call
            step_position_ids = torch.tensor([[processed_len]], dtype=torch.long, device=device)

            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                position_ids=step_position_ids,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values  # Update cache
            processed_len += 1  # Sequence length grows

            # Get the ID for the *next* token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    # Decode and evaluate the model's output
    model_output = tokenizer.decode(generated_ids_list)

    matches = re.findall(r"[\D]*(\d+)", model_output)
    model_answer = matches[0] if matches else ""
    is_correct = (model_answer == answer)
    
    print(f"Model's output: {model_output}")
    print(f"Found answer: {model_answer}")
    print(f"Correct answer: {answer}")
    print(f"Is correct: {is_correct}\n")

    # Clean up
    del past_key_values, input_ids, logits, generated_ids_list, model_output, next_token_id
    torch.cuda.empty_cache()

    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    print("HF Model", args.hf_model)

    # Parse additional HF model arguments
    hf_model_args = json.loads(args.hf_model_args)
    
    # Load model and tokenizer
    # from ntk_yarn.modeling_llama import LlamaForCausalLM #很重要！
    # model = LlamaForCausalLM.from_pretrained(
    #     args.hf_model,
    #     trust_remote_code=True,
    #     ignore_mismatched_sizes=True,
    #     attn_implementation="flash_attention_2",
    #     **hf_model_args
    # ).bfloat16().to(device)
    # from ntk_yarn.modeling_mistral import MistralForCausalLM #很重要！
    # model = MistralForCausalLM.from_pretrained(
    #     args.hf_model,
    #     trust_remote_code=True,
    #     ignore_mismatched_sizes=True,
    #     attn_implementation="flash_attention_2",
    #     **hf_model_args
    # ).bfloat16().to(device)
    if 'llama' in args.hf_model.lower():
        from ntk_yarn.modeling_llama import LlamaForCausalLM
        print('using llama')
        model = LlamaForCausalLM.from_pretrained(
            args.hf_model,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            attn_implementation="flash_attention_2",
            **hf_model_args
        ).bfloat16().to(device)

    elif 'olmo' in args.hf_model.lower():
        from ntk_yarn.modeling_olmo import OlmoForCausalLM #很重要！
        print('using olmo')
        model = OlmoForCausalLM.from_pretrained(
            args.hf_model,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            attn_implementation="flash_attention_2",
            **hf_model_args
        ).bfloat16().to(device)


    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model.eval()

    # Calculate number of test points starting from min_tokens
    total_test_points = (args.max_tokens - args.min_tokens) // args.interval + 1
    all_accuracies = []
    
    for i in range(total_test_points):
        # Calculate context length starting from min_tokens
        current_tokens = args.min_tokens + (i * args.interval)
        
        # Calculate depth steps to max_depth
        depth_steps = np.linspace(0, args.max_depth, 10) # 10 steps from 0 to max_depth
        
        for depth in depth_steps:
            passed_tests = 0
            total_tokens = 0
            
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(
                    model, tokenizer, device, 
                    context_length=current_tokens,
                    depth=depth,
                    seed=k
                )
                passed_tests += is_correct
                total_tokens += len_tokens
                
            avg_tokens = total_tokens // args.num_tests
            accuracy = float(passed_tests) / args.num_tests
            print(f"accuracy on the token length {avg_tokens}, depth {depth:.2f}, is {accuracy:.2f}")
            
            result = {
                "Context Length": current_tokens,
                "Document Depth": round(depth * 100, -1),
                "Score": passed_tests
            }
            all_accuracies.append(result)

    total_tests = len(all_accuracies)
    total_passed = sum(result['Score'] for result in all_accuracies)
    total_score = (total_passed / (total_tests * args.num_tests)) * 100

    print("\nFinal Results Summary:")
    print(f"Total Tests Run: {total_tests * args.num_tests}")
    print(f"Total Tests Passed: {total_passed}")
    print(f"Overall Score: {total_score:.2f}%")

    # Print detailed breakdown
    df_summary = pd.DataFrame(all_accuracies)
    print("\nDetailed Results by Context Length and Depth:")
    print(df_summary.groupby(['Context Length', 'Document Depth'])['Score'].mean().to_string())

    # Create visualization
    df = pd.DataFrame(all_accuracies)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(
        df, values='Score', index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )
    
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=5
    )
    model_path_parts = args.hf_model.split('/')
    sanitized_model_name = '_'.join(model_path_parts[-2:] if len(model_path_parts) > 1 else model_path_parts[-1:])

    plt.title(f"128k {sanitized_model_name}")
    plt.xlabel('Token Limit')
    plt.ylabel('Depth Percent')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Extract last 2 path components and create sanitized filename


    
    plt.savefig(f"/data/zyli/work/Fourier-Position-Embedding/passkey_result/128k/{sanitized_model_name}.png")
    df_summary.to_csv(f"/data/zyli/work/Fourier-Position-Embedding/passkey_result/128k/{sanitized_model_name}.csv", index=False)

if __name__ == "__main__":
    args = parse_config()
    main(args)