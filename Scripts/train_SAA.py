#!/usr/bin/env python
import os
import gc
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, resample
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from huggingface_hub import HfApi, create_repo, login
import argparse


def split_into_chunks(txt, chunk_size=CHUNK_SIZE):
    words = txt.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def load_turns(path):
    return np.load(path, allow_pickle=True)

def make_multiclass_examples(files, TARGETS, SPEAKER2ID):
    data = []
    for f in files:
        for t in load_turns(f):
            spk = t.get('speaker', '').lower()
            if spk not in TARGETS:
                continue
            txt = t.get('text', '').strip()
            if not txt or len(txt.split()) < MIN_WORD_COUNT:
                continue
            chunks = split_into_chunks(txt)
            for chunk in chunks:
                data.append({'text': chunk, 'label': SPEAKER2ID[spk]})
    random.shuffle(data)
    print(f"✅ Built {len(data)} examples for {len(TARGETS)} speakers.")
    return data

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=1024)

def balance_dataset(data):
    label_counts = Counter([ex['label'] for ex in data])
    max_count = max(label_counts.values())
    balanced = []
    for label in label_counts:
        class_samples = [ex for ex in data if ex['label'] == label]
        upsampled = resample(class_samples, replace=True, n_samples=max_count)
        balanced.extend(upsampled)
    random.shuffle(balanced)
    print(f"🔁 Balanced dataset to {max_count} per class ({len(balanced)} total)")
    return balanced


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if not hasattr(self, 'class_weights'):
            y = np.array(self.train_dataset['labels']).astype(int)
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(model.device)

        loss_fct = FocalLoss(alpha=self.class_weights, gamma=2.0)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_multiclass_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precs, recs, f1s, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1
    }
    for i, name in enumerate(TARGETS):
        metrics[f"{name}_precision"] = precs[i]
        metrics[f"{name}_recall"] = recs[i]
        metrics[f"{name}_f1"] = f1s[i]
        metrics[f"{name}_accuracy"] = per_class_acc[i]
    return metrics


class TrainSetEvaluatorCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_init_end(self, args, state, control, model=None, **kwargs):
        self.trainer = kwargs.get("trainer", None)

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.trainer is None:
            return control

        model.eval()
        dataloader = DataLoader(self.trainer.train_dataset, batch_size=args.per_device_eval_batch_size)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()

        from transformers import EvalPrediction
        train_metrics = compute_multiclass_metrics(EvalPrediction(predictions=all_logits, label_ids=all_labels))
        print("\n📊 Training set metrics:")
        for k, v in train_metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
        return control


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--hf_token', help='HF auth token')
    parser.add_argument('--save_dir',  help='Path to directory to save models')
    parser.add_argument('--transcript_path', help='Path to transcript')
    parser.add_argument('--model_name', default='microsoft/deberta-v3-large', help='Model Name to Fine-tune')
    parser.add_argument('--repo_id', help='HF repo id to save trained models')

    MIN_WORD_COUNT = 20
    CHUNK_SIZE = 200

    api = HfApi()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    login(token=args.hf_token)

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

    for transcript_path, TARGETS in transcript_sources:
        dataset_name = os.path.basename(os.path.dirname(transcript_path)).capitalize()
        repo_id = f"{args.repo_id}/{dataset_name}_speaker_attribution"

        print(f"\n==============================")
        print(f"🎤 DATASET: {dataset_name}")
        print(f"==============================")

        # --- Clean repo if it exists ---
        try:
            repo_info = api.repo_info(repo_id, repo_type="model")
            files = [f.rfilename for f in repo_info.siblings]
            if "model.safetensors" in files:
                print(f"🗑️ Deleting existing repo {repo_id} before reupload.")
                api.delete_repo(repo_id, repo_type="model")
        except Exception:
            print(f"Repo {repo_id} does not exist — proceeding to create it.")

        # --- Create new private repo ---
        create_repo(repo_id, private=True, exist_ok=True)

        SPEAKER2ID = {name: i for i, name in enumerate(TARGETS)}

        all_files = [os.path.join(transcript_path, f) for f in os.listdir(transcript_path)]
        all_ex = make_multiclass_examples(all_files, TARGETS, SPEAKER2ID)
        labels = [ex['label'] for ex in all_ex]

        if len(set(labels)) < 2:
            print(f"⚠️ Not enough classes in {dataset_name}, skipping.")
            continue

        train_ex, test_ex = train_test_split(
            all_ex, test_size=0.3, stratify=labels)
        balanced_train_ex = balance_dataset(train_ex)

        train_ds = Dataset.from_list(balanced_train_ex).map(tokenize, batched=True)
        test_ds = Dataset.from_list(test_ex).map(tokenize, batched=True)
        train_ds = train_ds.rename_column("label", "labels")
        test_ds = test_ds.rename_column("label", "labels")
        train_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])
        test_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

        cfg = AutoConfig.from_pretrained(args.model_name, num_labels=len(TARGETS))
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to("cuda")

        output_dir = os.path.join(args.save_dir, dataset_name)
        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=50,
            learning_rate=1e-6,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            warmup_steps=200,
            fp16=True,
            save_total_limit=2,
            report_to="none",
            logging_dir="./logs",
        )

        trainer = FocalLossTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_multiclass_metrics,
            callbacks=[
                TrainSetEvaluatorCallback(),
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        gc.collect()
        torch.cuda.empty_cache()

        trainer.train()
        results = trainer.evaluate()
        print(f"\n✅ Final Results for {dataset_name}: {results}")

        final_model_dir = os.path.join(args.save_dir, f"{dataset_name}_final")
        if os.path.exists(final_model_dir):
            import shutil
            shutil.rmtree(final_model_dir)
        os.makedirs(final_model_dir, exist_ok=True)
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        try:
            print(f"📤 Uploading model for {dataset_name} to {repo_id}")
            api.upload_large_folder(
                folder_path=final_model_dir,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ Uploaded model for {dataset_name}")
        except Exception as e:
            print(f"⚠️ Upload failed for {dataset_name}: {e}")

        del model, trainer, train_ds, test_ds
        gc.collect()
        torch.cuda.empty_cache()
