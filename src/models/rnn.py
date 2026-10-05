"""
Bidirectional Recurrent Neural Network (BiLSTM Sequence Classifier)
Directly implements architecture referenced in CMPSC 448 Slides 52-82:
- Explicit LSTM gating mechanism: i_t (input), f_t (forget), o_t (output), c_t (cell state)
- Bidirectional recurrence enables perceiving both preceding (left) and succeeding (right) context
- Sequence pooling over valid tokens to avoid zero-padding corruption
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 6,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_p: float = 0.4,
        padding_idx: int = 0
    ):
        super(TextBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.padding_idx = padding_idx
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Recurrent Neural Network Layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )
        
        # Self-Attention Pooling over time (Slides 83-93)
        lstm_out_dim = hidden_dim * self.num_directions
        self.attn_fc = nn.Linear(lstm_out_dim, 1)
        
        # Regularization & Classification Head
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch, Seq_Len)
        embedded = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  # (Batch, Seq_Len, hidden_dim * directions)
        
        # Create attention mask for padding tokens
        mask = (x != self.padding_idx).float()  # (Batch, Seq_Len)
        
        # Compute scalar attention score for each token
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (Batch, Seq_Len)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (Batch, Seq_Len, 1)
        
        # Weighted sequence representation
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # (Batch, hidden_dim * directions)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
