"""
Execution Script for RQ2:
Does the user's prompt help identify the LLM?
Compares three input regimes:
1. Input Only (User prompt alone)
2. Output Only (Model response alone)
3. Input + Output (Synergistic concatenation with [SEP])
Evaluates on both Controlled Benchmark (Dataset A) and Wild Arena (Dataset B).
"""

import os
import json
import time
import torch

from src.dataset import get_dataloaders
from src.models.cnn import TextCNN
from src.models.baseline import BaselineSuite, load_texts_and_labels
from src.train import train_model
from src.evaluate import evaluate_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def evaluate_regime(train_path: str, val_path: str, test_path: str, mode: str, dataset_name: str, device: torch.device):
    print(f"\n>>> Running Regime [{mode.upper()}] on [{dataset_name}] <<<")
    
    # 1. TF-IDF + Linear Classifier
    train_texts, train_labels = load_texts_and_labels(train_path, mode=mode)
    test_texts, test_labels = load_texts_and_labels(test_path, mode=mode)
    
    baseline = BaselineSuite(max_features=12000)
    baseline.fit(train_texts, train_labels)
    base_eval = baseline.evaluate(test_texts, test_labels)
    lr_acc = base_eval["logistic_regression"]["accuracy"]
    lr_f1 = base_eval["logistic_regression"]["macro_f1"]
    
    # 2. 1D-CNN
    train_loader, val_loader, test_loader, vocab, label2idx = get_dataloaders(
        train_path, val_path, test_path,
        mode=mode,
        max_len=384,
        batch_size=64
    )
    idx2label = {i: l for l, i in label2idx.items()}
    
    cnn = TextCNN(
        vocab_size=len(vocab),
        embed_dim=96,
        num_classes=len(label2idx),
        filter_sizes=[2, 3, 4],
        num_filters=48,
        dropout_p=0.4
    )
    
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"cnn_rq2_{dataset_name}_{mode}.pt")
    trained_cnn, history = train_model(
        cnn, train_loader, val_loader, idx2label, device,
        num_epochs=6,
        learning_rate=1.5e-3,
        patience=2,
        model_save_path=ckpt_path
    )
    
    cnn_eval = evaluate_model(trained_cnn, test_loader, device, idx2label)
    
    print(f"[{dataset_name} | {mode.upper()}] LR Acc: {lr_acc:.4f} (F1: {lr_f1:.4f}) | "
          f"CNN Acc: {cnn_eval['accuracy']:.4f} (F1: {cnn_eval['macro_f1']:.4f})")
          
    return {
        "lr_accuracy": lr_acc,
        "lr_macro_f1": lr_f1,
        "cnn_accuracy": cnn_eval["accuracy"],
        "cnn_macro_f1": cnn_eval["macro_f1"],
        "cnn_metrics": cnn_eval
    }


def run_rq2_experiments():
    print("=" * 65)
    print("CMPSC 448 MIDTERM PROJECT - RESEARCH QUESTION 2 (RQ2)")
    print("Target: Assessing Prompt Informativeness & Selection Bias")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    
    # Dataset A (Controlled Benchmark)
    bench_train = os.path.join("data", "processed", "benchmark", "train.jsonl")
    bench_val = os.path.join("data", "processed", "benchmark", "val.jsonl")
    bench_test = os.path.join("data", "processed", "benchmark", "test.jsonl")
    
    results = {"benchmark": {}, "arena": {}}
    
    for mode in ["input", "output", "both"]:
        results["benchmark"][mode] = evaluate_regime(
            bench_train, bench_val, bench_test, mode, "benchmark", device
        )
        
    # Dataset B (In-the-wild LMSYS Arena)
    arena_train = os.path.join("data", "processed", "arena", "train.jsonl")
    arena_val = os.path.join("data", "processed", "arena", "val.jsonl")
    arena_test = os.path.join("data", "processed", "arena", "test.jsonl")
    
    if os.path.exists(arena_train):
        for mode in ["input", "output", "both"]:
            results["arena"][mode] = evaluate_regime(
                arena_train, arena_val, arena_test, mode, "arena", device
            )
            
    save_path = os.path.join(RESULTS_DIR, "rq2_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] RQ2 results saved to {save_path}")


if __name__ == "__main__":
    run_rq2_experiments()
