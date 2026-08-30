"""Architecture-agnostic full-graph transductive trainer, with optional
multi-task lineage supervision.

Generalises src/gnn/train.py (which is hardcoded to MultiTaskGCN) so the
benchmark can train every architecture in src/gnn/architectures.py through one
code path. src/gnn/train.py is left in place and unchanged -- it is what
produced the committed results/gnn_metrics.json, and rewriting it would
invalidate that provenance.

Transductive setting, unchanged from train.py: all nodes and edges are visible
in the forward pass; only the LOSS is masked to train rows. Labels never enter
the features or the graph.

Multi-task loss
---------------
    L = BCE(amr_logits, y_amr) + lambda_lineage * CE(lineage_logits, y_lineage)

Isolates with an unknown lineage carry IGNORE_INDEX and are dropped by the
cross-entropy term only -- they still contribute their AMR loss and still sit
in the graph, so the sample alignment never changes.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from ..gnn.architectures import ARCHITECTURES, GRAPH_FREE

IGNORE_INDEX = -100


def compute_pos_weight(y_train):
    """Per-drug BCE pos_weight from train-split balance (same as train.py)."""
    n_pos = y_train.sum(dim=0)
    n_neg = y_train.shape[0] - n_pos
    return torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1), torch.ones_like(n_pos))


class MultiTaskWrapper(nn.Module):
    """Adds a lineage head on top of any architecture's shared trunk.

    The AMR head lives inside the architecture (its `head`); the lineage head
    is attached here and reads the same trunk embedding, which is what makes
    this hard-parameter-sharing multi-task learning rather than two models.
    """

    def __init__(self, backbone, trunk_dim, n_lineage_classes):
        super().__init__()
        self.backbone = backbone
        self.lineage_head = nn.Linear(trunk_dim, n_lineage_classes)

    def forward(self, x, edge_index, edge_weight):
        # One trunk pass feeds BOTH heads. Calling the backbone's own forward
        # here instead would run the trunk twice, and with dropout active the
        # two heads would then see different stochastic representations --
        # which is not hard parameter sharing.
        h = self.backbone.embed(x, edge_index, edge_weight)
        return self.backbone.head(h), self.lineage_head(h)


def _trunk_dim(model, x, edge_index, edge_weight):
    with torch.no_grad():
        return model.embed(x, edge_index, edge_weight).shape[1]


def train(
    X,
    y_amr,
    edge_index,
    edge_weight,
    masks,
    architecture="gcn_skip",
    hidden_dim=64,
    lr=0.01,
    weight_decay=5e-4,
    dropout=0.3,
    max_epochs=300,
    patience=20,
    seed=42,
    y_lineage=None,
    n_lineage_classes=None,
    lambda_lineage=0.0,
    selection_drug_idx=None,
    verbose=False,
):
    """Train one model. Returns (model, probabilities, info).

    probabilities is the sigmoid of the AMR logits for ALL nodes, (N, D).

    Early stopping is on val macro-F1 over the AMR task -- the lineage task is
    auxiliary and never selects the checkpoint.

    selection_drug_idx restricts that early-stopping metric to a subset of
    drugs. This matters: with drugs that have single-digit positives in val,
    an all-drug macro-F1 is dominated by whether one or two examples happened
    to cross the threshold, so it stops training on noise. Passing the core-6
    indices selects on the drugs that actually carry support. It changes only
    WHICH CHECKPOINT is kept -- every reported metric is still computed over
    all drugs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph_free = architecture in GRAPH_FREE
    ei = edge_index if not graph_free else edge_index
    ew = edge_weight if not graph_free else None

    backbone = ARCHITECTURES[architecture](
        in_dim=X.shape[1], hidden_dim=hidden_dim,
        n_out=y_amr.shape[1], dropout=dropout,
    )

    multitask = (
        y_lineage is not None and lambda_lineage > 0.0 and n_lineage_classes
        and hasattr(backbone, "embed")
    )
    if multitask:
        backbone.eval()
        tdim = _trunk_dim(backbone, X, ei, ew)
        model = MultiTaskWrapper(backbone, tdim, n_lineage_classes)
    else:
        model = backbone

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask, val_mask = masks["train"], masks["val"]
    pos_weight = compute_pos_weight(y_amr[train_mask])
    amr_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lin_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_val, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        if multitask:
            amr_logits, lin_logits = model(X, ei, ew)
            loss = amr_criterion(amr_logits[train_mask], y_amr[train_mask])
            loss = loss + lambda_lineage * lin_criterion(
                lin_logits[train_mask], y_lineage[train_mask]
            )
        else:
            amr_logits = model(X, ei, ew)
            loss = amr_criterion(amr_logits[train_mask], y_amr[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X, ei, ew)
            logits = out[0] if multitask else out
            pred = (torch.sigmoid(logits[val_mask]) >= 0.5).int().numpy()
            truth = y_amr[val_mask].int().numpy()
            if selection_drug_idx is not None:
                pred = pred[:, selection_drug_idx]
                truth = truth[:, selection_drug_idx]
            val_f1 = f1_score(truth, pred, average="macro", zero_division=0)

        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                        "val_macro_f1": float(val_f1)})
        if verbose and epoch % 25 == 0:
            print("    epoch %3d loss=%.4f val_macro_f1=%.4f" % (epoch, loss.item(), val_f1))

        if val_f1 > best_val:
            best_val, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(X, ei, ew)
        logits = out[0] if multitask else out
        proba = torch.sigmoid(logits).numpy()
        lineage_pred = out[1].argmax(dim=1).numpy() if multitask else None

    info = {
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_val_macro_f1_selection_metric": float(best_val),
        "n_epochs_run": len(history),
        "multitask_lineage": bool(multitask),
        "lambda_lineage": lambda_lineage if multitask else 0.0,
        "history_tail": history[-5:],
    }
    gated = backbone if not multitask else model.backbone
    if hasattr(gated, "gate_stats"):
        info["gate_stats"] = gated.gate_stats()
    if lineage_pred is not None:
        info["lineage_pred"] = lineage_pred
    return model, proba, info
