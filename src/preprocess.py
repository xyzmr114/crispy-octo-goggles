"""
Dataset Preprocessing and Stratification Pipeline
Curates and balances multi-family LLM datasets:
1. AlpacaEval 6-model benchmark suite (DeepSeek-67B, Qwen2-72B, GPT-4-Turbo, Claude-2.1, Gemini-Pro, Llama-3-70B)
2. LMSYS Chatbot Arena pairwise human chat logs (GPT-4, Claude, PaLM-2, ChatGLM-6B, Vicuna/Llama)
"""

import os
import json
import re
import random
from collections import Counter
import polars as pl

# Set random seed for perfect scientific reproducibility
SEED = 42
random.seed(SEED)

PROCESSED_DIR = os.path.join("data", "processed")
BENCHMARK_DIR = os.path.join(PROCESSED_DIR, "benchmark")
ARENA_DIR = os.path.join(PROCESSED_DIR, "arena")
DOMAINS_DIR = os.path.join(PROCESSED_DIR, "domains")

os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(ARENA_DIR, exist_ok=True)
os.makedirs(DOMAINS_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic whitespace normalization preserving formatting characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")
    return text.strip()


def infer_domain(instruction: str, dataset_tag: str) -> str:
    """Categorizes prompt into one of four core academic domains for RQ3."""
    ins_lower = instruction.lower()
    
    # Coding & Technical Reasoning
    code_keywords = ["python", "code", "function", "program", "sql", "java", "c++", 
                     "algorithm", "script", "bug", "html", "css", "docker", "git", "api"]
    math_keywords = ["calculate", "solve", "math", "equation", "integral", "derivative", 
                     "probability", "proof", "matrix", "geometry", "theorem"]
    if any(k in ins_lower for k in code_keywords) or any(k in ins_lower for k in math_keywords):
        return "Technical_STEM"
    
    # Creative Writing & Roleplay
    creative_keywords = ["write a story", "poem", "essay", "dialogue", "character", 
                         "roleplay", "pretend", "narrative", "scene", "script", "creative"]
    if any(k in ins_lower for k in creative_keywords):
        return "Creative_Writing"
    
    # Humanities, Ethics, & Reasoning
    humanities_keywords = ["history", "philosophy", "ethics", "argue", "compare and contrast", 
                           "literature", "politics", "society", "moral", "analysis", "opinion"]
    if any(k in ins_lower for k in humanities_keywords):
        return "Humanities_Reasoning"
        
    # Default / General QA
    return "General_QA"


def process_alpaca_eval():
    """Processes the 6-model AlpacaEval benchmark dataset."""
    raw_dir = os.path.join("data", "raw", "alpaca_eval")
    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    
    print("\n--- Processing AlpacaEval 6-Model Benchmark ---")
    data_by_family = {}
    
    for filename in sorted(files):
        # Format: Family_model-id.json
        parts = filename.split("_", 1)
        family = parts[0]
        model_id = parts[1].replace(".json", "")
        filepath = os.path.join(raw_dir, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        print(f"Loading {family} ({model_id}): {len(records)} entries")
        data_by_family[family] = []
        
        for idx, item in enumerate(records):
            instruction = clean_text(item.get("instruction", ""))
            output = clean_text(item.get("output", ""))
            dataset_source = item.get("dataset", "unknown")
            domain = infer_domain(instruction, dataset_source)
            
            data_by_family[family].append({
                "sample_id": f"{family}_{idx}",
                "prompt_id": idx,
                "family": family,
                "model_name": model_id,
                "domain": domain,
                "dataset_source": dataset_source,
                "llm_input": instruction,
                "llm_output": output,
                "input_len": len(instruction.split()),
                "output_len": len(output.split())
            })
            
    # Combine and create paired train/val/test splits (80 / 10 / 10)
    # We split by prompt_id so the SAME prompt never leaks across train and test!
    num_prompts = len(records)
    prompt_ids = list(range(num_prompts))
    random.shuffle(prompt_ids)
    
    train_cutoff = int(0.80 * num_prompts)
    val_cutoff = int(0.90 * num_prompts)
    
    train_prompts = set(prompt_ids[:train_cutoff])
    val_prompts = set(prompt_ids[train_cutoff:val_cutoff])
    test_prompts = set(prompt_ids[val_cutoff:])
    
    train_set, val_set, test_set = [], [], []
    
    for family, samples in data_by_family.items():
        for s in samples:
            pid = s["prompt_id"]
            if pid in train_prompts:
                train_set.append(s)
            elif pid in val_prompts:
                val_set.append(s)
            else:
                test_set.append(s)
                
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_jsonl(train_set, os.path.join(BENCHMARK_DIR, "train.jsonl"))
    save_jsonl(val_set, os.path.join(BENCHMARK_DIR, "val.jsonl"))
    save_jsonl(test_set, os.path.join(BENCHMARK_DIR, "test.jsonl"))
    
    print(f"\nAlpacaEval Splits Saved:")
    print(f"  Train: {len(train_set)} samples ({Counter(s['family'] for s in train_set)})")
    print(f"  Val:   {len(val_set)} samples ({Counter(s['family'] for s in val_set)})")
    print(f"  Test:  {len(test_set)} samples ({Counter(s['family'] for s in test_set)})")
    
    # Domain partitions for RQ3
    all_samples = train_set + val_set + test_set
    domain_counts = Counter(s["domain"] for s in all_samples)
    print("\nDomain Distribution for RQ3:")
    for d, c in domain_counts.most_common():
        print(f"  {d:25s}: {c} samples")
        
    # Save technical vs creative partitions
    tech_samples = [s for s in all_samples if s["domain"] == "Technical_STEM"]
    creative_samples = [s for s in all_samples if s["domain"] in ("Creative_Writing", "Humanities_Reasoning")]
    
    save_jsonl(tech_samples, os.path.join(DOMAINS_DIR, "technical_stem.jsonl"))
    save_jsonl(creative_samples, os.path.join(DOMAINS_DIR, "creative_humanities.jsonl"))
    print(f"  -> Saved Technical partition: {len(tech_samples)} samples")
    print(f"  -> Saved Creative partition:  {len(creative_samples)} samples")


def process_lmsys_arena():
    """Processes the LMSYS Chatbot Arena pairwise human chat logs."""
    raw_path = os.path.join("data", "raw", "lmsys_chatbot_arena.parquet")
    if not os.path.exists(raw_path):
        print(f"\n[Warning] LMSYS file not found at {raw_path}")
        return
        
    print("\n--- Processing LMSYS Chatbot Arena In-The-Wild Dataset ---")
    df = pl.read_parquet(raw_path)
    df_en = df.filter(pl.col("language") == "English")
    
    # We focus on the premier 4 families with substantial representations
    FAMILY_MAPPING = {
        "gpt-4": "OpenAI",
        "gpt-3.5-turbo": "OpenAI",
        "claude-v1": "Anthropic",
        "claude-instant-v1": "Anthropic",
        "palm-2": "Google",
        "chatglm-6b": "Chinese_ChatGLM"
    }
    
    extracted_records = []
    
    for row in df_en.iter_rows(named=True):
        m_a = row["model_a"]
        m_b = row["model_b"]
        
        # Check model A
        if m_a in FAMILY_MAPPING:
            conv = row["conversation_a"]
            if conv and len(conv) >= 2:
                user_msg = clean_text(conv[0].get("content", ""))
                assistant_msg = clean_text(conv[1].get("content", ""))
                if len(user_msg) > 5 and len(assistant_msg) > 10:
                    fam = FAMILY_MAPPING[m_a]
                    extracted_records.append({
                        "family": fam,
                        "model_name": m_a,
                        "llm_input": user_msg,
                        "llm_output": assistant_msg,
                        "domain": infer_domain(user_msg, "lmsys")
                    })
                    
        # Check model B
        if m_b in FAMILY_MAPPING:
            conv = row["conversation_b"]
            if conv and len(conv) >= 2:
                user_msg = clean_text(conv[0].get("content", ""))
                assistant_msg = clean_text(conv[1].get("content", ""))
                if len(user_msg) > 5 and len(assistant_msg) > 10:
                    fam = FAMILY_MAPPING[m_b]
                    extracted_records.append({
                        "family": fam,
                        "model_name": m_b,
                        "llm_input": user_msg,
                        "llm_output": assistant_msg,
                        "domain": infer_domain(user_msg, "lmsys")
                    })
                    
    # Balance across families to ~1,200 per family
    by_fam = {}
    for r in extracted_records:
        by_fam.setdefault(r["family"], []).append(r)
        
    TARGET_PER_FAM = 1200
    balanced_records = []
    for fam, items in by_fam.items():
        random.shuffle(items)
        selected = items[:TARGET_PER_FAM]
        balanced_records.extend(selected)
        print(f"LMSYS Family [{fam}]: Available={len(items)}, Selected={len(selected)}")
        
    random.shuffle(balanced_records)
    
    # Train / Val / Test (80/10/10)
    n = len(balanced_records)
    t_cut = int(0.80 * n)
    v_cut = int(0.90 * n)
    
    t_set = balanced_records[:t_cut]
    v_set = balanced_records[t_cut:v_cut]
    te_set = balanced_records[v_cut:]
    
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_jsonl(t_set, os.path.join(ARENA_DIR, "train.jsonl"))
    save_jsonl(v_set, os.path.join(ARENA_DIR, "val.jsonl"))
    save_jsonl(te_set, os.path.join(ARENA_DIR, "test.jsonl"))
    
    print(f"\nLMSYS Arena Balanced Splits Saved:")
    print(f"  Train: {len(t_set)} samples")
    print(f"  Val:   {len(v_set)} samples")
    print(f"  Test:  {len(te_set)} samples")


if __name__ == "__main__":
    process_alpaca_eval()
    process_lmsys_arena()
    print("\n[SUCCESS] Preprocessing completed successfully.")
