"""GraFuSE project module.
"""

import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """Graph queries SMILES/functional-group context."""
    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, graph, context):
        q = graph.unsqueeze(1)
        out, weights = self.attn(q, context, context,
                                 need_weights=True)
        out = self.norm(q + out)
        out = self.norm(out + self.ffn(out))
        return out.squeeze(1), weights
