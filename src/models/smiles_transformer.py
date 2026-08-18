"""GraFuSE project module.
"""

import torch
import torch.nn as nn

class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, max_length=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, input_ids):
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        x = self.embedding(input_ids) + self.position(pos)
        pad_mask = input_ids.eq(0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)
