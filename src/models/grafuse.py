"""GraFuSE project module.

Refactored from the supplied Test_Val(FG+SMILES+GRAPH).ipynb.
"""

import torch
import torch.nn as nn
from .graph_transformer import GraphTransformer
from .smiles_transformer import SMILESTransformer
from .functional_group_encoder import FunctionalGroupEncoder
from .cross_attention import CrossAttentionFusion

class GraFuSE(nn.Module):
    """Graph + SMILES + Functional Group cross-attention model."""
    def __init__(self, graph_in_channels, vocab_size, num_groups,
                 hidden_dim=128, num_classes=1, max_length=100,
                 model_option=2):
        super().__init__()
        self.model_option = model_option
        self.graph_encoder = GraphTransformer(graph_in_channels, hidden_dim,
                                              out_dim=hidden_dim)
        self.smiles_encoder = SMILESTransformer(vocab_size, hidden_dim,
                                                 max_length=max_length)
        self.fg_encoder = FunctionalGroupEncoder(num_groups, hidden_dim)

        self.cross_attention = CrossAttentionFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, graph, smiles_ids, fg_ids=None):
        g = self.graph_encoder(graph)
        if self.model_option == 0:
            s = self.smiles_encoder(smiles_ids)
            fused = (g + s) / 2
            weights = None
        elif self.model_option == 1:
            if fg_ids is None:
                raise ValueError("fg_ids required for model_option=1")
            f = self.fg_encoder(fg_ids).mean(1)
            fused = (g + f) / 2
            weights = None
        else:
            s = self.smiles_encoder(smiles_ids)
            if fg_ids is None:
                raise ValueError("fg_ids required for model_option=2")
            f = self.fg_encoder(fg_ids).mean(1)
            context = torch.stack([s, f], dim=1)
            fused, weights = self.cross_attention(g, context)
        return self.classifier(fused).squeeze(-1), weights
