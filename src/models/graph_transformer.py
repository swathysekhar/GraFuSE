"""GraFuSE project module.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, heads=4, out_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_dim // heads,
                                     heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads,
                                     heads=heads, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return self.proj(global_mean_pool(x, batch))
