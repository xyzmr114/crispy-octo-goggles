"""
PyTorch Dataset and Vocabulary Pipeline
Supports three input regimes:
- 'output': Output response only (RQ1, core classification)
- 'input': Prompt/instruction only (RQ2, prompt leakage test)
- 'both': Prompt + SEP + Response (RQ2, prompt synergy test)
"""

import os
import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SEP_IDX = 4


def tokenize(text: str) -> List[str]:
    """Splits text into words, punctuation, and markdown structural tokens."""
    if not text:
        return []
    # Lowercase and split on words and non-whitespace punctuation
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    def __init__(self, min_freq: int = 2, max_size: int = 25000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # Initialize with special tokens
        for idx, tok in enumerate(SPECIAL_TOKENS):
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
            
        # Sort by frequency descending
        sorted_tokens = [tok for tok, cnt in counter.most_common() if cnt >= self.min_freq]
        
        for tok in sorted_tokens:
            if len(self.token2idx) >= self.max_size:
                break
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok
                
        print(f"Vocabulary built: {len(self.token2idx)} unique tokens (min_freq={self.min_freq}, max_size={self.max_size})")

    def encode(self, text: str, max_length: int = 512, add_sos_eos: bool = False) -> List[int]:
        tokens = tokenize(text)
        indices = [self.token2idx.get(tok, UNK_IDX) for tok in tokens]
        
        if add_sos_eos:
            indices = [SOS_IDX] + indices[:max_length - 2] + [EOS_IDX]
        else:
            indices = indices[:max_length]
            
        return indices

    def decode(self, indices: List[int]) -> str:
        tokens = [self.idx2token.get(idx, UNK_TOKEN) for idx in indices if idx != PAD_IDX]
        return " ".join(tokens)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"token2idx": self.token2idx, "min_freq": self.min_freq, "max_size": self.max_size}, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"], max_size=data["max_size"])
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        return vocab

    def __len__(self):
        return len(self.token2idx)


class LLMAuthorshipDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Vocabulary, mode: str = "output", max_len: int = 512, label2idx: Optional[Dict[str, int]] = None):
        self.vocab = vocab
        self.mode = mode  # 'output', 'input', or 'both'
        self.max_len = max_len
        self.records = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    
        # Construct or use label2idx
        if label2idx is None:
            unique_labels = sorted(list(set(r["family"] for r in self.records)))
            self.label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx
            
        self.idx2label = {i: lbl for lbl, i in self.label2idx.items()}
        print(f"Loaded {len(self.records)} records from {jsonl_path} | Classes: {self.label2idx}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        
        if self.mode == "output":
            text = item["llm_output"]
            token_ids = self.vocab.encode(text, max_length=self.max_len)
        elif self.mode == "input":
            text = item["llm_input"]
            token_ids = self.vocab.encode(text, max_length=self.max_len)
        elif self.mode == "both":
            in_ids = self.vocab.encode(item["llm_input"], max_length=self.max_len // 2)
            out_ids = self.vocab.encode(item["llm_output"], max_length=self.max_len // 2)
            token_ids = in_ids + [SEP_IDX] + out_ids
            token_ids = token_ids[:self.max_len]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        label = self.label2idx[item["family"]]
        return torch.tensor(token_ids, dtype=torch.long), label


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences dynamically to the maximum sequence length in the batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) if len(s) > 0 else 1 for s in sequences], dtype=torch.long)
    
    # Pad sequences with PAD_IDX
    padded_seqs = torch.nn.utils.rnn.pad_sequence(
        [s if len(s) > 0 else torch.tensor([UNK_IDX], dtype=torch.long) for s in sequences],
        batch_first=True,
        padding_value=PAD_IDX
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels


def get_dataloaders(train_path: str, val_path: str, test_path: str, 
                    mode: str = "output", max_len: int = 512, batch_size: int = 32):
    """Convenience factory creating train, val, and test DataLoaders."""
    # Build vocab from training set
    with open(train_path, "r", encoding="utf-8") as f:
        train_texts = []
        for line in f:
            if line.strip():
                item = json.loads(line)
                if mode == "output":
                    train_texts.append(item["llm_output"])
                elif mode == "input":
                    train_texts.append(item["llm_input"])
                else:
                    train_texts.append(item["llm_input"] + " " + item["llm_output"])
                    
    vocab = Vocabulary(min_freq=2, max_size=25000)
    vocab.build_vocab(train_texts)
    
    # Train dataset sets label mapping
    train_ds = LLMAuthorshipDataset(train_path, vocab, mode=mode, max_len=max_len)
    val_ds = LLMAuthorshipDataset(val_path, vocab, mode=mode, max_len=max_len, label2idx=train_ds.label2idx)
    test_ds = LLMAuthorshipDataset(test_path, vocab, mode=mode, max_len=max_len, label2idx=train_ds.label2idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, vocab, train_ds.label2idx
