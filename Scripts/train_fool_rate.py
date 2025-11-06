#!/usr/bin/env python
import os
import gc
import shutil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from huggingface_hub import HfApi, create_repo, login
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
import argparse

# ====================== Loader ======================
def load_transcript(transcript_path, transcript):
    return np.load(os.path.join(transcript_path, transcript), allow_pickle=True)

def load_turns(path):
    return np.load(path, allow_pickle=True)

# ====================== Example Builder ======================
def make_examples(
    files,
    speaker,
    neg_dataset=None,
    min_word_count=5,
    chunk_size=(2, 3),
    targets=None
):
    """
    Create sentence chunks as examples.
    """
    pos, neg = [], []
    for f in files:
        for t in load_turns(f):
            spk = t.get('speaker', '').lower()
            if not spk or not t.get('text', '').strip():
                continue

            sentences = [s.strip() for s in sent_tokenize(t['text']) if len(s.split()) >= min_word_count]
            if not sentences:
                continue

            i = 0
            while i < len(sentences):
                chunk_len = random.randint(chunk_size[0], chunk_size[1])
                chunk = sentences[i:i+chunk_len]
                if chunk:
                    chunk_text = " ".join(chunk)
                    lbl = int(spk == speaker)
                    entry = {'text': chunk_text, 'label': lbl}
                    if lbl:
                        pos.append(entry)
                    elif targets and spk in targets and spk != speaker:
                        neg.append(entry)
                i += chunk_len


    # Balance negatives
    max_neg = len(pos)
    if len(neg) > max_neg:
        neg = random.sample(neg, max_neg)

    if neg_dataset:
        for neg_text in random.sample(neg_dataset, 25):
            if len(neg_text.split()) >= min_word_count:
                neg.append({'text': neg_text.strip(), 'label': 0})


    data = pos + neg
    random.shuffle(data)
    print(f"{speaker}: {len(pos)} pos, {len(neg)} neg (sentence chunks including external negatives)")
    return data

# ====================== Tokenization ======================
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=1024)

# ====================== Metrics ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {"eval_accuracy": acc, "eval_precision": prec, "eval_recall": rec, "eval_f1": f1}

# ====================== Main ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--hf_token', help='HF auth token')
    parser.add_argument('--save_dir',  help='Path to directory to save models')
    parser.add_argument('--transcript_path', help='Path to transcript')
    parser.add_argument('--model_name', default='microsoft/deberta-v3-large', help='Model Name to Fine-tune')
    parser.add_argument('--repo_id', help='HF repo id to save trained models')

    args = parser.parse_args()

    # Hugging Face login
    login(token=args.hf_token)
    api = HfApi()
    MIN_WORD_COUNT = 10

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # Transcript sources and targets
    transcript_sources = [
        (args.transcript_path.format('Albermale'), [
            'ellenosborne', 'davidoberg', 'grahampaige', 'jonnoalcaro',
            'katrinacallsen', 'kateacuff', 'judyle'
        ]),
        (args.transcript_path.format('DCAppeals'), [
            'judgethompson', 'judgemcleese', 'judgeglickman', 'judgedeahl',
            'judgeeasterly', 'judgeblackburn', 'judgethompson'
        ]),
        (args.transcript_path.format('Waipa'), [
            "susanoregan", "jimmylchreest", "clairestpierre", "andrewbrown",
            "rogergordon","loubrown", "angeholt", "suemilner"
        ])
    ]

    results = {}

    for transcript_path, TARGETS in transcript_sources:

        all_files = [os.path.join(transcript_path, f) for f in os.listdir(transcript_path)]

        # Optional: load external negatives
        neg_examples = []
        if 'albermale' in transcript_path.lower():
            dataset_name = 'Albermale'
        elif 'waipa' in transcript_path.lower():
            dataset_name = 'Waipa'
            print("STARTING WAIPA")
        elif 'dcappeals' in transcript_path.lower():
            dataset_name = 'DCAppeals'

        for speaker in TARGETS:
            repo_id = f"{args.repo_id}/{speaker}_prediction"

            try:
                repo_info = api.repo_info(repo_id, repo_type="model")
                files = [f.rfilename for f in repo_info.siblings]
                if "model.safetensors" in files:
                    api.delete_repo(repo_id, repo_type="model")  # permanent deletion

                    print(f"Repo {repo_id} already has model.safetensors — skipping {speaker}.")
                    final_model_dir = os.path.join(args.save_dir, f"{speaker}_final")
                    if os.path.exists(final_model_dir):
                        shutil.rmtree(final_model_dir)
                else:
                    print(f"Repo {repo_id} exists but missing model.safetensors — running training.")
            except Exception:
                print(f"Repo {repo_id} does NOT exist — creating and running training.")
                create_repo(repo_id, private=True)


            print(f"\n=== Speaker: {speaker} ===")
            gc.collect()
            torch.cuda.empty_cache()

            # Build dataset
            all_examples = make_examples(all_files, speaker, neg_dataset=neg_examples, min_word_count=MIN_WORD_COUNT, chunk_size=(2,5), targets=TARGETS)
            train_ex, test_ex = train_test_split(
                all_examples, test_size=0.3, random_state=42,
                stratify=[ex['label'] for ex in all_examples]
            )

            train_ds = Dataset.from_list(train_ex).map(tokenize, batched=True)
            test_ds = Dataset.from_list(test_ex).map(tokenize, batched=True)
            train_ds = train_ds.rename_column("label", "labels")
            test_ds = test_ds.rename_column("label", "labels")
            train_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])
            test_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

            # Compute class weights
            y = np.array(train_ds["labels"])
            cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
            cw_tensor = torch.tensor(cw, dtype=torch.float)

            # Load model
            cfg = AutoConfig.from_pretrained(args.model_name, num_labels=2)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg)
            model = model.to("cuda")
            model.class_weights = cw_tensor.to(model.device)
            model.resize_token_embeddings(len(tokenizer))

            # Training args
            output_dir = os.path.join(args.save_dir, speaker)
            args = TrainingArguments(
                output_dir=output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=2,
                num_train_epochs=25,
                weight_decay=0.01,
                logging_steps=50,
                seed=42,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                warmup_steps=0,
                fp16=True
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train & evaluate
            trainer.train()
            results[speaker] = trainer.evaluate()

            # Save model
            final_model_dir = os.path.join(args.save_dir, f"{speaker}_final")
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            os.makedirs(final_model_dir, exist_ok=True)
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)

            print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")

            # Upload to Hugging Face
            try:
                api.upload_large_folder(
                    folder_path=final_model_dir,
                    repo_id=repo_id,
                    repo_type="model"
                )
            except Exception as e:
                print(f"Upload Failed for {speaker}: {e}")
