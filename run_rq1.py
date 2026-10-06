"""
Execution Script for RQ1:
Can we identify which LLM generated a response?
Compares:
1. Baseline: TF-IDF + Logistic Regression & Linear SVM
2. Convolutional Neural Network (Kim 1D-CNN)
3. Recurrent Neural Network (BiLSTM with Gated Recurrence)
"""

import os
import time
import json
import torch
import numpy as np

from src.dataset import get_dataloaders
from src.models.baseline import BaselineSuite, load_texts_and_labels
from src.models.cnn import TextCNN
from src.models.rnn import TextBiLSTM
from src.train import train_model
from src.evaluate import evaluate_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)  # model weights land here


def run_rq1_experiments():
    print("=" * 65)
    print("CMPSC 448 MIDTERM PROJECT - RESEARCH QUESTION 1 (RQ1)")
    print("Target: Identifying LLM Authorship from Output Responses Alone")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    
    train_path = os.path.join("data", "processed", "benchmark", "train.jsonl")
    val_path = os.path.join("data", "processed", "benchmark", "val.jsonl")
    test_path = os.path.join("data", "processed", "benchmark", "test.jsonl")
    
    # -------------------------------------------------------------
    # 1. TRADITIONAL BASELINES (TF-IDF + LR / SVM)
    # -------------------------------------------------------------
    print("\n[PHASE 1.1] Training Traditional NLP Baselines...")
    train_texts, train_labels = load_texts_and_labels(train_path, mode="output")
    test_texts, test_labels = load_texts_and_labels(test_path, mode="output")
    
    baseline_suite = BaselineSuite(max_features=15000, ngram_range=(1, 2))
    b_start = time.time()
    baseline_suite.fit(train_texts, train_labels)
    b_fit_time = time.time() - b_start
    baseline_results = baseline_suite.evaluate(test_texts, test_labels)
    
    print(f"  -> TF-IDF + Logistic Regression Test Acc: {baseline_results['logistic_regression']['accuracy']:.4f} "
          f"| Macro-F1: {baseline_results['logistic_regression']['macro_f1']:.4f}")
    print(f"  -> TF-IDF + Linear SVM Test Acc:          {baseline_results['linear_svm']['accuracy']:.4f} "
          f"| Macro-F1: {baseline_results['linear_svm']['macro_f1']:.4f}")
          
    # -------------------------------------------------------------
    # 2. DATA LOADERS FOR DEEP MODELS
    # -------------------------------------------------------------
    print("\n[PHASE 1.2] Building DataLoaders for Neural Networks...")
    train_loader, val_loader, test_loader, vocab, label2idx = get_dataloaders(
        train_path, val_path, test_path,
        mode="output",
        max_len=512,
        batch_size=32
    )
    idx2label = {i: l for l, i in label2idx.items()}
    vocab_size = len(vocab)
    num_classes = len(label2idx)
    
    # -------------------------------------------------------------
    # 3. 1D CONVOLUTIONAL NEURAL NETWORK (CNN)
    # -------------------------------------------------------------
    print("\n[PHASE 1.3] Training 1D-CNN (Kim Multi-Filter CNN)...")
    cnn_model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=128,
        num_classes=num_classes,
        filter_sizes=[2, 3, 4, 5],
        num_filters=64,
        dropout_p=0.4
    )
    
    cnn_ckpt = os.path.join(CHECKPOINTS_DIR, "best_cnn_rq1.pt")
    trained_cnn, cnn_history = train_model(
        cnn_model, train_loader, val_loader, idx2label, device,
        num_epochs=12,
        learning_rate=1e-3,
        patience=4,
        model_save_path=cnn_ckpt
    )
    
    # Inference Latency Benchmark
    t0 = time.time()
    cnn_test_metrics = evaluate_model(trained_cnn, test_loader, device, idx2label)
    cnn_infer_time_ms = ((time.time() - t0) / len(test_loader.dataset)) * 1000.0
    
    print(f"\nCNN Test Results:")
    print(f"  Test Accuracy: {cnn_test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:      {cnn_test_metrics['macro_f1']:.4f}")
    print(f"  Inference Latency: {cnn_infer_time_ms:.2f} ms/sample")
    
    # -------------------------------------------------------------
    # 4. RECURRENT NEURAL NETWORK (BiLSTM)
    # -------------------------------------------------------------
    print("\n[PHASE 1.4] Training Bidirectional LSTM (BiLSTM)...")
    rnn_model = TextBiLSTM(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        bidirectional=True,
        dropout_p=0.4
    )
    
    rnn_ckpt = os.path.join(CHECKPOINTS_DIR, "best_rnn_rq1.pt")
    trained_rnn, rnn_history = train_model(
        rnn_model, train_loader, val_loader, idx2label, device,
        num_epochs=12,
        learning_rate=8e-4,
        patience=4,
        model_save_path=rnn_ckpt
    )
    
    # Inference Latency Benchmark
    t0 = time.time()
    rnn_test_metrics = evaluate_model(trained_rnn, test_loader, device, idx2label)
    rnn_infer_time_ms = ((time.time() - t0) / len(test_loader.dataset)) * 1000.0
    
    print(f"\nBiLSTM Test Results:")
    print(f"  Test Accuracy: {rnn_test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:      {rnn_test_metrics['macro_f1']:.4f}")
    print(f"  Inference Latency: {rnn_infer_time_ms:.2f} ms/sample")
    
    # -------------------------------------------------------------
    # 5. AGGREGATE RESULTS & SAVE
    # -------------------------------------------------------------
    final_results = {
        "classes": [idx2label[i] for i in range(num_classes)],
        "baseline_lr": baseline_results["logistic_regression"],
        "baseline_svm": baseline_results["linear_svm"],
        "cnn": {
            "metrics": cnn_test_metrics,
            "history": cnn_history,
            "infer_time_ms": cnn_infer_time_ms,
            "avg_epoch_time_sec": float(np.mean(cnn_history["epoch_times_sec"]))
        },
        "rnn": {
            "metrics": rnn_test_metrics,
            "history": rnn_history,
            "infer_time_ms": rnn_infer_time_ms,
            "avg_epoch_time_sec": float(np.mean(rnn_history["epoch_times_sec"]))
        }
    }
    
    save_path = os.path.join(RESULTS_DIR, "rq1_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[SUCCESS] RQ1 full experimental results saved to {save_path}")


if __name__ == "__main__":
    run_rq1_experiments()
