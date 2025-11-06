#!/usr/bin/env python3
import os, sys
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # default is usually 16 or 64
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth import train_on_responses_only
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from opensloth.patching.ddp_patch import ddp_patch, patch_optimize_sft_trainer_batch_samples

import argparse
from typing import Any, Tuple
from utils import debug_chat_dataloader_for_training

from utils import get_dataset, WAIPA_TAGS, ALBERMALE_TAGS, DCAPPEALS_TAGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with OpenSloth + LoRA")

    # Data
    parser.add_argument("--train_path", type=str, required=True, help="Path to dataset saved with save_to_disk")

    parser.add_argument("--test_path", type=str, required=True, help="Path to dataset saved with save_to_disk")

    parser.add_argument("--agent_name", type=str, required=True,  help="Name of agent to train")

    # Model
    parser.add_argument("--model_name", type=str, help="Base model name or path")

    # Training setup
    parser.add_argument("--max_seq_length", type=int, default=3000, help="Max sequence length")
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Per device batch size (before grad accumulation)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup Ratio")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--add_tags", type=int, default=0, help="Add Actin Tags or not")

    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency")
    parser.add_argument("--dataset_name", required=True, help="Dataset Name")

    args = parser.parse_args()
    return args



def init_model(args: argparse.Namespace, ACTION_TAGS: list) -> Tuple[FastLanguageModel, Any]:
    compute_dtype = torch.bfloat16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fix_tokenizer=False,
        device_map='balanced',
        dtype = compute_dtype,
        #full_finetuning=True
    )

    if args.add_tags:
        print("Adding Action Tags to Tokenizer")
        tokenizer.add_tokens(ACTION_TAGS)
        # Also resizes lm_head as well!
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
   
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    args: argparse.Namespace,
) -> UnslothTrainer:

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim=args.optim,
        learning_rate=args.learning_rate,
        save_only_model=False,
        save_strategy="epoch",
        output_dir=args.output_dir,
        resume_from_checkpoint=args.output_dir,
        bf16 = True,
        #gradient_checkpointing=True,
        #ddp_find_unused_parameters = False,
        #gradient_checkpointing_kwargs={"use_reentrant": True},
        eval_strategy="no",
        dataset_num_proc=4,
        seed=3407
    )
    return UnslothTrainer(
        model=model,
        #tokenizer=tokenizer,  # type: ignore[arg-type]
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),


    )


def main():
    #ddp_patch()
    #patch_optimize_sft_trainer_batch_samples()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # Select Action Tags
    # -----------------------------
    if 'waipa' in args.dataset_name.lower():
        ACTION_TAGS = WAIPA_TAGS
    elif 'dcappeals' in args.dataset_name.lower():
        ACTION_TAGS = DCAPPEALS_TAGS
    elif 'albermale' in args.dataset_name.lower():
        ACTION_TAGS = ALBERMALE_TAGS
    else:
        raise ValueError(f"Unknown dataset_name {args.dataset_name}")

    # -----------------------------
    # Identify chat template
    # -----------------------------
    if 'qwen' in args.model_name.lower():
        chat_template = "qwen3-instruct"
        instruction_part = '<|im_start|>user\n'
        response_part = '<|im_start|>assistant\n'
    elif 'llama' in args.model_name.lower():
        chat_template = "llama-3.3"
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'gpt' in args.model_name.lower():
        chat_template = "gpt-oss"
        instruction_part = '<|end|><|start|>user<|message|>'
        response_part = '<|end|><|start|>assistant<|channel|>final<|message|>'
    else:
        raise ValueError(f"Unknown model_name {args.model_name}")

    # -----------------------------
    # Load model + tokenizer
    # -----------------------------
    model, tokenizer = init_model(args, ACTION_TAGS)
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # -----------------------------
    # Load dataset
    # -----------------------------
    print("Loading Dataset: ", args.train_path)
    train_data, _ = get_dataset(args.train_path, args.test_path, args.agent_name, sys_message=0)
    train_data = [tokenizer.apply_chat_template(x, tokenize=False) for x in train_data]
    train_data = Dataset.from_list([{"text": text} for text in train_data])

    # -----------------------------
    # Build trainer
    # -----------------------------
    trainer = build_trainer(model, tokenizer, train_data, args)
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)

    # Optional debug
    debug_chat_dataloader_for_training(trainer.train_dataset, tokenizer, n_example=1)

    # -----------------------------
    # Start training
    # -----------------------------
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
