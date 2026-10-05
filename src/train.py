"""
Modular PyTorch Training Engine:
- Optimized for fast CPU/GPU convergence
- AdamW + Cosine Annealing Learning Rate Schedule
- Gradient norm clipping to prevent gradient explosion
- Early stopping tracking validation loss and macro-F1
- Execution time profiling per epoch (demonstrating CNN vs RNN speed)
"""

import os
import time
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.evaluate import evaluate_model


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    idx2label: Dict[int, str],
    device: torch.device,
    num_epochs: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    model_save_path: str = "best_model.pt"
) -> Tuple[nn.Module, Dict]:
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_times_sec": []
    }
    
    best_val_f1 = -1.0
    patience_counter = 0
    
    print(f"\n--- Starting Training for {model.__class__.__name__} ({num_epochs} Epochs) ---")
    total_start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            
            # Clip gradient norm to 2.0
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            
        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation pass
        val_metrics = evaluate_model(model, val_loader, device, idx2label)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["epoch_times_sec"].append(epoch_time)
        
        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.2f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        # Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            # print(f"  -> Best model saved (Macro-F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f} seconds. Best Val Macro-F1: {best_val_f1:.4f}")
    
    # Load best weights
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        
    return model, history
