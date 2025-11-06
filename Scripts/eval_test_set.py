import sys, os

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from tqdm import tqdm
import pandas as pd
import argparse
import gc
from peft import PeftModel
import torch._dynamo
import random
from utils import get_dataset


def get_longest_assistant_conversations(test_data, tokenizer, top_k=10):
    """
    Returns the top_k conversations with the largest total assistant token count.
    """
    conv_with_tokens = []

    for conv in test_data:
        total_tokens = 0
        for msg in conv:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                total_tokens += tokenizer(msg['content'], return_tensors="pt")['input_ids'].shape[1]
        conv_with_tokens.append((total_tokens, conv))

    conv_with_tokens.sort(key=lambda x: x[0], reverse=True)
    top_conversations = [conv for _, conv in conv_with_tokens[:top_k]]
    return top_conversations



def generate_responses(chat_template, chat_data, tokenizer, model, max_new_tokens=256):
    """
    Generates responses for assistant messages given the conversation context.
    Returns a list of dicts: {"ground_truth": ..., "model_response": ...}
    """
    results = []
    i = 0
    for conversation in tqdm(chat_data, desc="Generating responses"):
        i += 1
        
        speaker = conversation[-1]['content'].split(':')[0]

        prompt = tokenizer.apply_chat_template(conversation[:-1], tokenize=False, add_generation_prompt=True, reasoning_effort='low', max_length=3000)

        # If using reasoning GPT use below as a hack to disable reasoning
        #prompt = tokenizer.apply_chat_template(conversation[:-1], tokenize=False, add_generation_prompt=False, reasoning_effort='low', max_length=3000)
        #prompt = prompt + f"""<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|> {speaker}: """

        inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)  
        print("DEVICE: ", model.device)

        eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("assistant")[0]]

        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_tokens,


        )

        gen_text = tokenizer.decode(generation[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Generated: {gen_text}")

        results.append({
            "conversation_id": i,
            "prompt": prompt,
            "model_response": gen_text})
        
        del inputs, generation
        torch.cuda.empty_cache()
        gc.collect()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses on chat datasets.")
    parser.add_argument("--adapter_path", type=str, required=True, help="PEFT adapter path")
    parser.add_argument("--base_model", type=str, help="Base HuggingFace model path")
    parser.add_argument("--train_path", type=str, help="Path to training dataset JSON")
    parser.add_argument("--test_path", type=str, help="Path to test dataset JSON")
    parser.add_argument("--save_name", type=str, required=True, help="Directory to save results CSVs")
    parser.add_argument("--system_message", type=int, default=0, help="Include system message")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Max tokens to generate per assistant message")
    parser.add_argument("--max_seq_length", type=int, default=3000)
    parser.add_argument("--dataset_name", required=True, help="Dataset Name")

    args = parser.parse_args()

    # Load model and tokenizer
    if 'qwen' in args.base_model.lower():
        model_name = 'qwen3'
        chat_template = 'qwen3-instruct'
    elif 'llama' in args.base_model.lower():
        model_name = 'llama'
        chat_template = 'llama-3.3'
    elif 'gpt' in args.base_model.lower():
        model_name = 'gpt'
        chat_template = "gpt-oss"
        
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=5000,
        torch_dtype=torch.float16,
        device_map='balanced',
        fix_tokenizer=True,
        load_in_4bit=False,
    )                    
    
    if args.adapter_path != 'baseline':
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.load_adapter(args.adapter_path, adapter_name="adapter_1")
        model.set_adapter("adapter_1")

        # make sure adapter params have same data type
        target_dtype = next(p for p in model.parameters() if p.dtype != torch.int8).dtype
        for name, param in model.named_parameters():
            if "lora_" in name and param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
         
        
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    FastLanguageModel.for_inference(model) 

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
        if item in args.adapter_path.lower():
            agent_name = item
            print(f"Identified agent name: {agent_name}")
            ALL_AGENT_NAMES = [agent_name]  # only evaluate this agent
            break
        else:
            agent_name = 'baseline'
            print(f"Identified agent name: {agent_name}")

    output_df = pd.DataFrame()
    for dataset in ALL_AGENT_NAMES:
        print(f"COMPUTING Completions for agent: {agent_name} on dataset: {dataset}")

        for sample in range(5):
            print(f"Sample Number: {sample} ")

            _, test_data = get_dataset(args.train_path, args.test_path, dataset, sys_message=args.system_message)

            if args.system_message == 0:
                insert_loc = 0
            else:
                insert_loc = 1

            print(test_data[0])
            print(f"GENERATING RESPONSES for agent: {agent_name} on dataset: {dataset}")
            sampled_elements = random.sample(test_data, min(10,len(test_data)) )

            output = pd.DataFrame(generate_responses(chat_template, sampled_elements, tokenizer, model, max_new_tokens=args.max_new_tokens))
            output['agent'] = agent_name
            output['dataset'] = dataset
            output['sample'] = sample
            output_df = pd.concat([output_df, output], ignore_index=True)
        

    # Concatenate all datasets and save
    os.makedirs(os.path.dirname(args.save_name), exist_ok=True)
    output_df.to_csv(args.save_name, index=False)
    print(f"Saved generated responses to {args.save_name}")
