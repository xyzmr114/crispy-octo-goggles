"""
1D Convolutional Neural Network (Kim CNN for Text Authorship Classification)
Directly implements architecture referenced in CMPSC 448 Slides 36 & 82:
- Embedding -> Parallel multi-scale Conv1D filter bank -> ReLU -> Global Max-Pooling -> FC
- Captures invariant local n-gram stylistic phrases (2-gram, 3-gram, 4-gram, 5-gram)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 6,
        filter_sizes: List[int] = [2, 3, 4, 5],
        num_filters: int = 64,
        dropout_p: float = 0.4,
        padding_idx: int = 0
    ):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # Word Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Parallel 1D Convolution layers
        # Input to Conv1D: (Batch, Channels=embed_dim, Length=seq_len)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0
            ) for k in filter_sizes
        ])
        
        # Regularization & Classification Head
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch, Seq_Len)
        embedded = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        embedded = embedded.transpose(1, 2)  # (Batch, Embed_Dim, Seq_Len)
        
        pooled_outputs = []
        for conv in self.convs:
            # Check sequence length is at least kernel size
            if embedded.shape[2] < conv.kernel_size[0]:
                pad_size = conv.kernel_size[0] - embedded.shape[2]
                curr_embedded = F.pad(embedded, (0, pad_size))
            else:
                curr_embedded = embedded
                
            conv_out = F.relu(conv(curr_embedded))  # (Batch, Num_Filters, L_out)
            # Global 1D Max Pooling across time dimension (Slide 41 & 82)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)  # (Batch, Num_Filters)
            pooled_outputs.append(pooled)
            
        # Concatenate multi-scale representations
        features = torch.cat(pooled_outputs, dim=1)  # (Batch, len(filter_sizes) * num_filters)
        features = self.dropout(features)
        logits = self.fc(features)  # (Batch, Num_Classes)
        return logits
