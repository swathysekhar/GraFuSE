"""GraFuSE project module.
"""

import torch
from .metrics import classification_metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    ys, ps = [], []
    for batch in loader:
        optimizer.zero_grad()
        graph = batch["graph"].to(device)
        smiles_ids = batch["smiles_ids"].to(device)
        fg_ids = batch.get("fg_ids")
        if fg_ids is not None:
            fg_ids = fg_ids.to(device)
        y = batch["label"].float().to(device)
        logits, _ = model(graph, smiles_ids, fg_ids)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        ys.extend(y.detach().cpu().numpy())
        ps.extend(torch.sigmoid(logits).detach().cpu().numpy())
    return total_loss / max(1, len(loader)), classification_metrics(ys, ps)
