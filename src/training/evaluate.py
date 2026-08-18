"""GraFuSE project module.

"""

import torch
from .metrics import classification_metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    for batch in loader:
        graph = batch["graph"].to(device)
        smiles_ids = batch["smiles_ids"].to(device)
        fg_ids = batch.get("fg_ids")
        if fg_ids is not None:
            fg_ids = fg_ids.to(device)
        y = batch["label"].float().to(device)
        logits, _ = model(graph, smiles_ids, fg_ids)
        total_loss += criterion(logits, y).item()
        ys.extend(y.cpu().numpy())
        ps.extend(torch.sigmoid(logits).cpu().numpy())
    return total_loss / max(1, len(loader)), classification_metrics(ys, ps)
