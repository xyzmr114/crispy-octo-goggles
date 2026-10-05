"""
Evaluation Engine:
Computes Macro/Micro Accuracy, Precision, Recall, Macro-F1, and Confusion Matrices.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_model(model: torch.nn.Module, data_loader, device: torch.device, idx2label: Dict[int, str]) -> Dict:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, lengths, y in data_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
            
    total_samples = len(all_targets)
    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    
    target_names = [idx2label[i] for i in range(len(idx2label))]
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": [int(p) for p in all_preds],
        "targets": [int(t) for t in all_targets]
    }
