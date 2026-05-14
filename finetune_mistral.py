"""
Fine-tune Mistral-7B on fake news detection datasets
Comparison study with published EPRVFL model (Gurjwar et al., PRL 2025)

Hardware: NVIDIA H100 NVL (100GB VRAM)
Method: LoRA fine-tuning via PEFT
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ── Configuration ─────────────────────────────────────────────────────────────

DATASETS = {
    "politifact":  "/home/akumar/llm_fakenews/politifact_final.csv",
    "gossipcop":   "/home/akumar/llm_fakenews/GossipCop_final.csv",
    "welfake":     "/home/akumar/llm_fakenews/WELFake_final.csv",
    "fake_real": "/home/akumar/llm_fakenews/fake_and_real_news_dataset.csv",
    "buzzfeed": "/home/akumar/llm_fakenews/BuzzFeed.csv",
    "liar2": "/home/akumar/llm_fakenews/LIAR2_skewed_final.csv",
}

MODEL_NAME    = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR    = "/home/akumar/llm_fakenews/mistral_output"
RESULTS_DIR   = "/home/akumar/llm_fakenews/results"
MAX_LENGTH    = 256
RANDOM_STATE  = 42

# EPRVFL published results for direct comparison
# Source: Gurjwar et al., Pattern Recognition Letters, 2025
EPRVFL_RESULTS = {
    "politifact": {"accuracy": 0.9177, "precision": 0.9180,
                   "recall": 0.9177, "f1": 0.9181},
}

# ── Dataset class ─────────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions,
                                     average="weighted", zero_division=0),
        "recall":    recall_score(labels, predictions,
                                  average="weighted", zero_division=0),
        "f1":        f1_score(labels, predictions,
                              average="weighted", zero_division=0),
    }

# ── Load and prepare data ─────────────────────────────────────────────────────

def load_data(dataset_name: str):
    path = DATASETS[dataset_name]
    df = pd.read_csv(path, encoding="latin-1")

    # Drop missing values
    df = df.dropna(subset=["title", "verdict"])
    df["verdict"] = df["verdict"].astype(int)

    texts  = df["title"].astype(str).tolist()
    labels = df["verdict"].tolist()

    print(f"\nDataset: {dataset_name}")
    print(f"Total samples: {len(texts):,}")
    print(f"Real (1): {sum(labels):,}  Fake (0): {len(labels)-sum(labels):,}")

    # 70/15/15 split — same as EPRVFL paper
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.30,
        random_state=RANDOM_STATE, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50,
        random_state=RANDOM_STATE, stratify=y_tmp)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ── Build model with LoRA ─────────────────────────────────────────────────────

def build_model(hf_token: str):
    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=hf_token, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                    # LoRA rank
        lora_alpha=32,           # scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # attention layers only
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# ── Train ─────────────────────────────────────────────────────────────────────

def train(dataset_name: str, hf_token: str, epochs: int = 5):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = \
        load_data(dataset_name)

    # Build model
    model, tokenizer = build_model(hf_token)

    # Tokenise
    print("\nTokenising datasets...")
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset   = FakeNewsDataset(X_val,   y_val,   tokenizer, MAX_LENGTH)
    test_dataset  = FakeNewsDataset(X_test,  y_test,  tokenizer, MAX_LENGTH)

    # Training arguments
    run_name = f"mistral_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{run_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        dataloader_num_workers=4,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print(f"\nStarting fine-tuning: {epochs} epochs on {dataset_name}")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time/60:.1f} minutes")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    t1 = time.time()
    predictions = trainer.predict(test_dataset)
    inf_time = time.time() - t1

    y_pred = np.argmax(predictions.predictions, axis=-1)

    results = {
        "model":          "Mistral-7B (LoRA)",
        "dataset":        dataset_name,
        "epochs":         epochs,
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "accuracy":       accuracy_score(y_test, y_pred),
        "precision":      precision_score(y_test, y_pred,
                                          average="weighted", zero_division=0),
        "recall":         recall_score(y_test, y_pred,
                                       average="weighted", zero_division=0),
        "f1":             f1_score(y_test, y_pred,
                                   average="weighted", zero_division=0),
        "training_time_minutes": round(train_time / 60, 2),
        "inference_time_seconds": round(inf_time, 4),
        "timestamp":      datetime.now().isoformat(),
    }

    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Train time: {results['training_time_minutes']:.1f} min")
    print(f"  Inf time:  {results['inference_time_seconds']:.4f} sec")

    # Compare with EPRVFL
    if dataset_name in EPRVFL_RESULTS:
        eprvfl = EPRVFL_RESULTS[dataset_name]
        print("\n" + "="*60)
        print("COMPARISON: Mistral-7B LoRA vs EPRVFL (PRL 2025)")
        print("="*60)
        print(f"  {'Metric':<12} {'Mistral-7B':>12} {'EPRVFL':>12} {'Diff':>10}")
        print(f"  {'-'*48}")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            diff = results[metric] - eprvfl[metric]
            sign = "+" if diff >= 0 else ""
            print(f"  {metric:<12} {results[metric]:>12.4f} "
                  f"{eprvfl[metric]:>12.4f} {sign}{diff:>9.4f}")
        print(f"\n  EPRVFL inference: ~0.001s (closed-form)")
        print(f"  Mistral-7B inf:   {results['inference_time_seconds']:.4f}s")
        print(f"  Speed ratio:      {results['inference_time_seconds']/0.001:.0f}x slower")

    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Fake (0)", "Real (1)"]))

    # Save results
    results_file = f"{RESULTS_DIR}/mistral_{dataset_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B for fake news detection")
    parser.add_argument("--dataset", type=str, default="politifact",
                        choices=list(DATASETS.keys()),
                        help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="HuggingFace access token")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run on all four datasets sequentially")
    args = parser.parse_args()

    if args.all_datasets:
        all_results = {}
        for dataset_name in DATASETS.keys():
            print(f"\n{'='*60}")
            print(f"Running on: {dataset_name}")
            print(f"{'='*60}")
            results = train(dataset_name, args.hf_token, args.epochs)
            all_results[dataset_name] = results

        # Save combined results
        combined_file = f"{RESULTS_DIR}/all_datasets_results.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_file}")
    else:
        train(args.dataset, args.hf_token, args.epochs)

if __name__ == "__main__":
    main()
