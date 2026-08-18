"""GraFuSE project module.
"""

import torch
import torch.nn as nn

class FunctionalGroupEncoder(nn.Module):
    def __init__(self, num_groups, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_groups, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, group_ids):
        x = self.embedding(group_ids)
        return self.mlp(x)
