# test_sharded.py
import os
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator
import pandas as pd
import sys, os

from utils import get_dataset
import argparse
import random
import torch, gc, subprocess, os


print("🧹 Attempting full GPU memory cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()


accelerator = Accelerator()  # handles device placement, sharding, dtype

# ----------------------------------------------------------------------
# Perplexity code (assistant-only masking)
# ----------------------------------------------------------------------
def find_response_part(chat_data, tokenizer, response_part):
    """
    Find the index in token IDs where the assistant's response starts.
    Strips special tokens from response_part and searches backwards.
    
    Args:
        chat_data: list of messages or conversation object
        tokenizer: HuggingFace tokenizer
        response_part: string, snippet of assistant response to locate
    
    Returns:
        index (int) in input_ids where assistant response starts (after response_part)
    """
    # Tokenize response part and strip special tokens
    response_ids = tokenizer(response_part)['input_ids']
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)
    response_ids_clean = [
        rid for rid, tok in zip(response_ids, response_tokens) if tok not in tokenizer.all_special_tokens
    ]

    # Tokenize full chat
    input_ids = tokenizer.apply_chat_template(chat_data, return_tensors="pt")[0].tolist()

    resp_len = len(response_ids_clean)
    i = len(input_ids) - resp_len

    # Search backwards
    while i >= 0:
        if input_ids[i:i+resp_len] == response_ids_clean:
            return i + resp_len
        i -= 1

    # Debug output if not found
    print("Could not find response part!")
    print("Response tokens (cleaned):", tokenizer.convert_ids_to_tokens(response_ids_clean))
    print("Last 50 tokens of chat:", tokenizer.convert_ids_to_tokens(input_ids[-50:]))
    raise ValueError("Response part not found in input!")




def compute_response_perplexity(model, tokenizer, chat_data, response_part, debug=False):
    """
    Compute perplexity only on assistant's response (ignore prompt).
    
    Args:
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        chat_data: conversation data for tokenizer.apply_chat_template
        debug (bool): If True, print per-token stats.
    """
    encodings = tokenizer.apply_chat_template(chat_data, return_tensors="pt")
    input_ids = encodings.to(model.device)

    # Clone input to create targets
    target_ids = input_ids.clone()

    # Find where assistant response starts
    start_idx = find_response_part(chat_data, tokenizer, response_part)
    print(f"Length of Input {input_ids.shape[1]}, START IDX: {start_idx}")
    print("Decoded assistant start region:", tokenizer.decode(input_ids[0, start_idx:]))
    
    # Mask everything before assistant response
    target_ids[:, :start_idx] = -100

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():#, accelerator.autocast():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    shift_logits = logits[:, :-1, :].contiguous().float() 
    shift_labels = target_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)

    nlls, per_token_info = [], []

    for i in range(shift_labels.size(1)):
        true_id = shift_labels[0, i].item()
        if true_id == -100:
            continue

        token = tokenizer.convert_ids_to_tokens([true_id])[0]
        log_prob = log_probs[0, i, true_id].item()
        nll = -log_prob
        ppl = torch.exp(torch.tensor(nll)).item()
        #print(ppl)
        nlls.append(nll)
        per_token_info.append({
            "position": i+1,
            "token": token,
            "log_prob": log_prob,
            "nll": nll,
            "ppl": ppl,
        })
        import math
        avg_nll = sum(nlls) / len(nlls)
        ppl = math.exp(avg_nll)

    if debug:
        print("Tokens:", tokens)
        print("Masked before index:", start_idx)
        print("Target Tokens:", [
            tok if tid != -100 else "<IGNORED>"
            for tok, tid in zip(tokens, target_ids[0].tolist())
        ])
        print("\nPer-token stats:")
        for info in per_token_info:
            print(f"  pos {info['position']:2d} | {info['token']:15s} "
                  f"| log_prob={info['log_prob']:.4f} | nll={info['nll']:.4f} | ppl={info['ppl']:.2f}")

        print(f"\nAverage NLL: {avg_nll:.4f}")
        print(f"Perplexity (assistant response only): {ppl:.2f}")
        print("-" * 60)

    return ppl

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute perplexity on chat datasets.")
    parser.add_argument("--adapter_path", type=str, required=True, help="PEFT adapter path")
    parser.add_argument("--base_model", type=str, help="Base HuggingFace model path")
    parser.add_argument("--train_path", type=str, help="Path to training dataset JSON")
    parser.add_argument("--test_path", type=str,  help="Path to test dataset JSON")
    parser.add_argument("--save_name", type=str, required=True, help="Directory to save results CSVs")
    parser.add_argument("--system_message", type=int, default=0, help="Whether to include system message")
    parser.add_argument("--debug", action="store_true", help="Print token-level stats")
    parser.add_argument("--dataset_name", required=True, help="Dataset Name")
    
    args = parser.parse_args()
    print(args.base_model)
    print("Adapter path: ", args.adapter_path)
    if 'qwen' in args.base_model.lower():
        model_name = 'qwen3'
        chat_template = 'qwen3-instruct'
        chat_template = 'qwen25'
        response_part='<|im_start|>assistant\n'
        
    elif 'llama' in args.base_model.lower():
        chat_template = 'llama-3.3'
        model_name = 'llama'
        response_part = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    elif 'gpt' in args.base_model.lower():
        chat_template = "gpt-oss"
        instruction_part = '<|end|><|start|>user<|message|>'
        response_part = '<|end|><|start|>assistant<|channel|>final<|message|>'        
    else:
        raise ValueError(f"Unknown model_name {args.model_name}")



    model, tokenizer = FastLanguageModel.from_pretrained(
       args.base_model,
        max_seq_length=5000,
        device_map='balanced',
        load_in_4bit=False,
        fix_tokenizer=True,
    )                    

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.adapter_path != 'baseline':
        print("LOADING: ", args.adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.load_adapter(args.adapter_path, adapter_name="adapter_1")
        model.set_adapter("adapter_1")         


    if 'gpt' in args.base_model.lower():
        # make sure adapter params have same data type
        print("PATCHING GPT")
        target_dtype = next(p for p in model.parameters()).dtype
        print(target_dtype)
        for name, param in model.named_parameters():
            if "lora_" in name and param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)

    FastLanguageModel.for_inference(model) 
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)


    if args.dataset_name == 'Albermale':
        ALL_AGENT_NAMES = [
            'ellenosborne', 'davidoberg', 'grahampaige', 'jonnoalcaro',
            'katrinacallsen', 'kateacuff', 'judyle'
        ]
    elif args.dataset_name == 'DCAppeals':
        ALL_AGENT_NAMES =  ['judgemcleese', 'judgeglickman', 'judgedeahl', 'judgeeasterly', 'judgeblackburn', 'judgethompson']
        
    elif args.dataset_name == 'Waipa':
        ALL_AGENT_NAMES = ["susanoregan", "jimmylchreest","clairestpierre", "andrewbrown",
                           "rogergordon","loubrown", "angeholt", "suemilner"]

    # Identify agent name from adapter path
    for item in ALL_AGENT_NAMES:
        if item.strip().lower() in args.adapter_path.lower():
            agent_name = item
            print(f"Identified agent name: {agent_name}")
            ALL_AGENT_NAMES = [agent_name]  # only evaluate this agent
            break
        else:
            agent_name = 'baseline'
    agent_name = args.adapter_path.lower()

    rows = []
    for dataset in ALL_AGENT_NAMES:
        _, test_data = get_dataset(args.train_path, args.test_path, dataset, sys_message=args.system_message)
        
        #test_data = test_dataw
        print(f"COMPUTING PERPLEXITY for agent: {agent_name} on dataset: {dataset}")
        for sample in range(3):
            print(f"Sample Number: {sample} ")
            ppls = []
            sampled_elements = random.sample(test_data, min(10,len(test_data)))
            
            for chat in sampled_elements:  # iterate over all conversations
                #ppl = compute_response_perplexity(model, tokenizer, chat, response_part, debug=args.debug)
                try:
                    ppl = compute_response_perplexity(model, tokenizer, chat, response_part, debug=args.debug)
                    ppls.append(ppl)
                except Exception as e:
                    print(f"Skipping conversation due to error: {e}")

            if len(ppls) > 0:
                mean_ppl = sum(ppls) / len(ppls)
            else:
                mean_ppl = float("nan")

            print("Mean Perplexity: ", mean_ppl)
            rows.append({
                'sample':sample,
                'dataset': dataset,
                'mean_perplexity': mean_ppl,
                'agent': agent_name
            })
        
    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.save_name), exist_ok=True)
    output_df.to_csv(args.save_name, index=False)
    print(f"Saved results to {args.save_name}")
