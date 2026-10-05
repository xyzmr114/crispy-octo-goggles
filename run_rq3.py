"""
Execution Script for RQ3 (Extra Credit +5 Points):
Do LLM fingerprints generalize across tasks or domains?
Investigates out-of-domain transfer:
- Domain A: Technical & STEM (Coding, Math, Algorithms, Technical Explanations)
- Domain B: Creative Writing & Humanities (Narratives, Roleplay, Ethics, Dialogue)
Measures the Domain Generalization Drop (Delta_OOD) and per-family fingerprint robustness.
"""

import os
import json
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.models.baseline import BaselineSuite
from src.dataset import Vocabulary, LLMAuthorshipDataset, collate_fn
from torch.utils.data import DataLoader
from src.models.cnn import TextCNN
from src.train import train_model
from src.evaluate import evaluate_model

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def load_domain_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_rq3_experiments():
    print("=" * 65)
    print("CMPSC 448 MIDTERM PROJECT - RESEARCH QUESTION 3 (RQ3 - EXTRA CREDIT)")
    print("Target: Cross-Domain Generalization of LLM Fingerprints")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tech_path = os.path.join("data", "processed", "domains", "technical_stem.jsonl")
    creative_path = os.path.join("data", "processed", "domains", "creative_humanities.jsonl")
    
    tech_records = load_domain_records(tech_path)
    creative_records = load_domain_records(creative_path)
    
    print(f"Loaded Technical Domain: {len(tech_records)} samples")
    print(f"Loaded Creative Domain:  {len(creative_records)} samples")
    
    from sklearn.model_selection import train_test_split
    
    tech_train, tech_test = train_test_split(
        tech_records, test_size=0.20, random_state=42,
        stratify=[r["family"] for r in tech_records]
    )
    creat_train, creat_test = train_test_split(
        creative_records, test_size=0.20, random_state=42,
        stratify=[r["family"] for r in creative_records]
    )
    
    families = sorted(list(set(r["family"] for r in tech_records)))
    
    # ---------------------------------------------------------
    # Baseline TF-IDF + Logistic Regression Cross-Domain Study
    # ---------------------------------------------------------
    print("\n--- 1. Traditional NLP Cross-Domain Transfer ---")
    
    # Fit on Technical
    tech_texts_train = [r["llm_output"] for r in tech_train]
    tech_labels_train = [r["family"] for r in tech_train]
    
    suite_tech = BaselineSuite(max_features=10000)
    suite_tech.fit(tech_texts_train, tech_labels_train)
    
    # Eval on Technical Test (In-Domain)
    tech_eval_in = suite_tech.evaluate([r["llm_output"] for r in tech_test], [r["family"] for r in tech_test])
    tech_in_acc = tech_eval_in["logistic_regression"]["accuracy"]
    tech_in_f1 = tech_eval_in["logistic_regression"]["macro_f1"]
    
    # Eval on Creative All (Out-of-Domain)
    tech_eval_ood = suite_tech.evaluate([r["llm_output"] for r in creative_records], [r["family"] for r in creative_records])
    tech_ood_acc = tech_eval_ood["logistic_regression"]["accuracy"]
    tech_ood_f1 = tech_eval_ood["logistic_regression"]["macro_f1"]
    
    print(f"Trained on Technical:")
    print(f"  In-Domain (Technical Test)  Acc: {tech_in_acc:.4f} | F1: {tech_in_f1:.4f}")
    print(f"  Out-of-Domain (Creative)    Acc: {tech_ood_acc:.4f} | F1: {tech_ood_f1:.4f}")
    print(f"  Generalization Drop (Delta): {(tech_in_acc - tech_ood_acc) * 100:.2f}%")
    
    # Fit on Creative
    creat_texts_train = [r["llm_output"] for r in creat_train]
    creat_labels_train = [r["family"] for r in creat_train]
    
    suite_creat = BaselineSuite(max_features=10000)
    suite_creat.fit(creat_texts_train, creat_labels_train)
    
    # Eval on Creative Test (In-Domain)
    creat_eval_in = suite_creat.evaluate([r["llm_output"] for r in creat_test], [r["family"] for r in creat_test])
    creat_in_acc = creat_eval_in["logistic_regression"]["accuracy"]
    creat_in_f1 = creat_eval_in["logistic_regression"]["macro_f1"]
    
    # Eval on Technical All (Out-of-Domain)
    creat_eval_ood = suite_creat.evaluate([r["llm_output"] for r in tech_records], [r["family"] for r in tech_records])
    creat_ood_acc = creat_eval_ood["logistic_regression"]["accuracy"]
    creat_ood_f1 = creat_eval_ood["logistic_regression"]["macro_f1"]
    
    print(f"\nTrained on Creative:")
    print(f"  In-Domain (Creative Test)   Acc: {creat_in_acc:.4f} | F1: {creat_in_f1:.4f}")
    print(f"  Out-of-Domain (Technical)   Acc: {creat_ood_acc:.4f} | F1: {creat_ood_f1:.4f}")
    print(f"  Generalization Drop (Delta): {(creat_in_acc - creat_ood_acc) * 100:.2f}%")
    
    # Per-Model Robustness Analysis
    report_ood = tech_eval_ood["logistic_regression"]["report"]
    per_model_f1 = {fam: report_ood[fam]["f1-score"] for fam in families if fam in report_ood}
    print("\nPer-Model Robustness on Unseen Domain (Trained on Tech -> Tested on Creative):")
    for fam, f1 in sorted(per_model_f1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fam:15s}: F1 = {f1:.4f}")
        
    # ---------------------------------------------------------
    # 2. General QA -> Coding & Creative Transfer (Lecture Syllabus Pattern)
    # ---------------------------------------------------------
    print("\n--- 2. General QA -> Coding & Creative Transfer ---")
    all_path = os.path.join("data", "processed", "benchmark", "train.jsonl")
    with open(all_path, "r", encoding="utf-8") as f:
        all_train = [json.loads(l) for l in f]
    test_path = os.path.join("data", "processed", "benchmark", "test.jsonl")
    with open(test_path, "r", encoding="utf-8") as f:
        all_test = [json.loads(l) for l in f]
        
    bench_data = all_train + all_test
    qa_data = [r for r in bench_data if r["domain"] == "General_QA"]
    qa_train, qa_test = train_test_split(qa_data, test_size=0.2, random_state=42, stratify=[r["family"] for r in qa_data])
    
    suite_qa = BaselineSuite(max_features=12000)
    suite_qa.fit([r["llm_output"] for r in qa_train], [r["family"] for r in qa_train])
    
    ev_qa_in = suite_qa.evaluate([r["llm_output"] for r in qa_test], [r["family"] for r in qa_test])
    ev_qa_to_code = suite_qa.evaluate([r["llm_output"] for r in tech_records], [r["family"] for r in tech_records])
    ev_qa_to_creat = suite_qa.evaluate([r["llm_output"] for r in creative_records], [r["family"] for r in creative_records])
    
    qa_in_acc = ev_qa_in["logistic_regression"]["accuracy"]
    qa_in_f1 = ev_qa_in["logistic_regression"]["macro_f1"]
    qa_code_acc = ev_qa_to_code["logistic_regression"]["accuracy"]
    qa_code_f1 = ev_qa_to_code["logistic_regression"]["macro_f1"]
    qa_creat_acc = ev_qa_to_creat["logistic_regression"]["accuracy"]
    qa_creat_f1 = ev_qa_to_creat["logistic_regression"]["macro_f1"]
    
    print(f"General QA In-Domain Acc: {qa_in_acc:.4f} | F1: {qa_in_f1:.4f}")
    print(f"OOD Transfer -> Coding:   Acc: {qa_code_acc:.4f} | F1: {qa_code_f1:.4f}")
    print(f"OOD Transfer -> Creative: Acc: {qa_creat_acc:.4f} | F1: {qa_creat_f1:.4f}")
    
    results = {
        "technical_to_creative": {
            "in_domain_accuracy": tech_in_acc,
            "in_domain_macro_f1": tech_in_f1,
            "ood_accuracy": tech_ood_acc,
            "ood_macro_f1": tech_ood_f1,
            "generalization_gap": tech_in_acc - tech_ood_acc,
            "per_model_f1_ood": per_model_f1
        },
        "creative_to_technical": {
            "in_domain_accuracy": creat_in_acc,
            "in_domain_macro_f1": creat_in_f1,
            "ood_accuracy": creat_ood_acc,
            "ood_macro_f1": creat_ood_f1,
            "generalization_gap": creat_in_acc - creat_ood_acc
        },
        "qa_to_coding_and_creative": {
            "qa_in_accuracy": qa_in_acc,
            "qa_in_f1": qa_in_f1,
            "qa_to_coding_accuracy": qa_code_acc,
            "qa_to_coding_f1": qa_code_f1,
            "qa_to_creative_accuracy": qa_creat_acc,
            "qa_to_creative_f1": qa_creat_f1
        }
    }
    
    save_path = os.path.join(RESULTS_DIR, "rq3_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] RQ3 results saved to {save_path}")


if __name__ == "__main__":
    run_rq3_experiments()
