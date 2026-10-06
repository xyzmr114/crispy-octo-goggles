"""
Execution Script for RQ4 (Extra Credit +5 Points):
What characteristics distinguish different LLMs?
1. Quantitative Stylometrics & AI Detection Signatures:
   - Burstiness (GPTZero sentence-length variance)
   - Type-Token Ratio (TTR Lexical Diversity)
   - Structural Markdown Markers (Headers, Bullets, Fences, Math)
2. Controlled Ablation Experiments:
   - Raw Text Baseline
   - Ablation 1: Structural Stripping (Remove Markdown, bullets, code blocks)
   - Ablation 2: Length Normalization (Uniform word truncation to 150 words)
   - Ablation 3: Punctuation Stripping & Case Folding
"""

import os
import re
import json
import numpy as np
from collections import Counter
from typing import Dict, List

from src.models.baseline import BaselineSuite


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """Computes linguistic and structural fingerprint metrics."""
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)
    if total_words == 0:
        return {
            "word_count": 0, "char_count": 0, "ttr": 0.0, "burstiness": 0.0,
            "header_density": 0.0, "bullet_density": 0.0, "code_block_count": 0,
            "bold_density": 0.0, "semicolon_count": 0
        }
        
    char_count = len(text)
    unique_words = len(set(words))
    ttr = unique_words / total_words  # Type-Token Ratio
    
    # Burstiness: std deviation of sentence lengths (GPTZero feature)
    sentences = re.split(r"[.!?]+", text)
    sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    burstiness = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    
    # Structural Markdown indicators
    header_count = len(re.findall(r"(?m)^#{1,6}\s", text))
    bullet_count = len(re.findall(r"(?m)^\s*[-*+]\s", text))
    code_block_count = len(re.findall(r"```", text)) // 2
    bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
    semicolon_count = text.count(";")
    
    return {
        "word_count": float(total_words),
        "char_count": float(char_count),
        "ttr": float(ttr),
        "burstiness": burstiness,
        "header_density": float(header_count / total_words * 1000),
        "bullet_density": float(bullet_count / total_words * 1000),
        "code_block_count": float(code_block_count),
        "bold_density": float(bold_count / total_words * 1000),
        "semicolon_count": float(semicolon_count)
    }


def strip_structure(text: str) -> str:
    """Ablation 1: Strips all Markdown headers, bullet points, and code fences."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " [CODE_REMOVED] ", text)
    # Remove headers
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    # Remove bullet markers
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    # Remove bold/italic syntax
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Normalize spaces
    return " ".join(text.split())


def truncate_length(text: str, max_words: int = 150) -> str:
    """Ablation 2: Enforces uniform length limit to eliminate verbosity shortcut."""
    words = text.split()
    return " ".join(words[:max_words])


def strip_punctuation(text: str) -> str:
    """Ablation 3: Strips all punctuation and lowercases entirely."""
    return re.sub(r"[^\w\s]", "", text.lower())


def run_rq4_experiments():
    print("=" * 65)
    print("CMPSC 448 MIDTERM PROJECT - RESEARCH QUESTION 4 (RQ4 - EXTRA CREDIT)")
    print("Target: Stylometric Profiling & Structural Ablation Analysis")
    print("=" * 65)
    
    train_path = os.path.join("data", "processed", "benchmark", "train.jsonl")
    test_path = os.path.join("data", "processed", "benchmark", "test.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]
        
    families = sorted(list(set(r["family"] for r in train_records)))
    
    # -------------------------------------------------------------
    # 1. QUANTITATIVE STYLOMETRIC PROFILING
    # -------------------------------------------------------------
    print("\n--- 1. Computing Stylometric Fingerprints by Family ---")
    family_metrics = {fam: [] for fam in families}
    
    all_records = train_records + test_records
    for r in all_records:
        feat = extract_stylometric_features(r["llm_output"])
        family_metrics[r["family"]].append(feat)
        
    stylometric_summary = {}
    print(f"{'Family':12s} | {'Words':>6s} | {'TTR':>6s} | {'Burst':>6s} | {'Hdr/k':>6s} | {'Blt/k':>6s} | {'Bld/k':>6s}")
    print("-" * 62)
    for fam in families:
        feats = family_metrics[fam]
        mean_words = np.mean([f["word_count"] for f in feats])
        mean_ttr = np.mean([f["ttr"] for f in feats])
        mean_burst = np.mean([f["burstiness"] for f in feats])
        mean_hdr = np.mean([f["header_density"] for f in feats])
        mean_blt = np.mean([f["bullet_density"] for f in feats])
        mean_bld = np.mean([f["bold_density"] for f in feats])
        
        stylometric_summary[fam] = {
            "mean_word_count": float(mean_words),
            "mean_ttr": float(mean_ttr),
            "mean_burstiness": float(mean_burst),
            "mean_header_density": float(mean_hdr),
            "mean_bullet_density": float(mean_blt),
            "mean_bold_density": float(mean_bld)
        }
        print(f"{fam:12s} | {mean_words:6.1f} | {mean_ttr:6.3f} | {mean_burst:6.2f} | {mean_hdr:6.2f} | {mean_blt:6.2f} | {mean_bld:6.2f}")
        
    # -------------------------------------------------------------
    # 2. ABLATION EXPERIMENTS
    # -------------------------------------------------------------
    print("\n--- 2. Controlled Feature Ablation Experiments ---")
    
    # Raw Baseline
    raw_train_texts = [r["llm_output"] for r in train_records]
    raw_train_labels = [r["family"] for r in train_records]
    raw_test_texts = [r["llm_output"] for r in test_records]
    raw_test_labels = [r["family"] for r in test_records]
    
    model_raw = BaselineSuite(max_features=12000)
    model_raw.fit(raw_train_texts, raw_train_labels)
    eval_raw = model_raw.evaluate(raw_test_texts, raw_test_labels)
    raw_acc = eval_raw["logistic_regression"]["accuracy"]
    raw_f1 = eval_raw["logistic_regression"]["macro_f1"]
    print(f"Raw Text Baseline:            Accuracy = {raw_acc:.4f} | Macro-F1 = {raw_f1:.4f}")
    
    # Ablation 1: Structural Stripping
    s_train_texts = [strip_structure(t) for t in raw_train_texts]
    s_test_texts = [strip_structure(t) for t in raw_test_texts]
    model_struct = BaselineSuite(max_features=12000)
    model_struct.fit(s_train_texts, raw_train_labels)
    eval_struct = model_struct.evaluate(s_test_texts, raw_test_labels)
    s_acc = eval_struct["logistic_regression"]["accuracy"]
    s_f1 = eval_struct["logistic_regression"]["macro_f1"]
    print(f"Ablation 1 (Structure Strip): Accuracy = {s_acc:.4f} | Drop = {(raw_acc - s_acc)*100:+.2f}%")
    
    # Ablation 2: Length Truncation (150 words)
    l_train_texts = [truncate_length(t, 150) for t in raw_train_texts]
    l_test_texts = [truncate_length(t, 150) for t in raw_test_texts]
    model_len = BaselineSuite(max_features=12000)
    model_len.fit(l_train_texts, raw_train_labels)
    eval_len = model_len.evaluate(l_test_texts, raw_test_labels)
    l_acc = eval_len["logistic_regression"]["accuracy"]
    l_f1 = eval_len["logistic_regression"]["macro_f1"]
    print(f"Ablation 2 (Length 150w):     Accuracy = {l_acc:.4f} | Drop = {(raw_acc - l_acc)*100:+.2f}%")
    
    # Ablation 3: Punctuation Stripping
    p_train_texts = [strip_punctuation(t) for t in raw_train_texts]
    p_test_texts = [strip_punctuation(t) for t in raw_test_texts]
    model_punct = BaselineSuite(max_features=12000)
    model_punct.fit(p_train_texts, raw_train_labels)
    eval_punct = model_punct.evaluate(p_test_texts, raw_test_labels)
    p_acc = eval_punct["logistic_regression"]["accuracy"]
    p_f1 = eval_punct["logistic_regression"]["macro_f1"]
    print(f"Ablation 3 (Punctuation Off): Accuracy = {p_acc:.4f} | Drop = {(raw_acc - p_acc)*100:+.2f}%")
    
    results = {
        "stylometric_summary": stylometric_summary,
        "ablation_results": {
            "raw_text": {"accuracy": raw_acc, "macro_f1": raw_f1},
            "structure_stripped": {"accuracy": s_acc, "macro_f1": s_f1, "drop": raw_acc - s_acc},
            "length_truncated_150w": {"accuracy": l_acc, "macro_f1": l_f1, "drop": raw_acc - l_acc},
            "punctuation_stripped": {"accuracy": p_acc, "macro_f1": p_f1, "drop": raw_acc - p_acc}
        }
    }
    
    save_path = os.path.join("results", "rq4_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] RQ4 results saved to {save_path}")


if __name__ == "__main__":
    run_rq4_experiments()
