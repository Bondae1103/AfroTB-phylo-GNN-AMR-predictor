"""Training loop for MultiTaskGCN: full-graph transductive training, loss
masked to the train split only, early stopping on val macro-F1.

Transductive setting, stated explicitly: every node and edge (including
those touching val/test isolates) is visible to the model during the forward
pass -- this is standard transductive graph learning, not leakage, since
edges/features never encode labels. Only the loss is restricted to train
rows. See README Sec. 12's cross-split near-duplicate diagnostic (98 val/test
isolates with a near-identical train neighbor) for the one real caveat this
implies for interpreting test performance.
"""

import torch
from sklearn.metrics import f1_score

from .model import MultiTaskGCN


def compute_pos_weight(y_train):
    """Per-drug BCE pos_weight from train-split class balance (mirrors
    train_random_forest.py's class_weight="balanced" for the same reason:
    several drugs are heavily imbalanced, e.g. ETH/LZD/CAP).
    """
    n_pos = y_train.sum(dim=0)
    n_neg = y_train.shape[0] - n_pos
    pos_weight = torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1), torch.ones_like(n_pos))
    return pos_weight


def train_model(
    data,
    hidden_dim=64,
    lr=0.01,
    weight_decay=5e-4,
    dropout=0.3,
    max_epochs=300,
    patience=20,
    seed=42,
    verbose=False,
    use_edge_weight=True,
    skip_connection=False,
):
    """use_edge_weight=False ablates the Jaccard-derived edge_weight: GCNConv
    then treats every edge as weight 1 (uniform), isolating whether the
    *weighting* itself matters, independent of the graph's structure
    (which edges exist at all) -- structure is untouched either way.

    skip_connection=True concatenates each isolate's raw feature vector onto
    the head's input -- see model.py's docstring for the motivation.
    """
    torch.manual_seed(seed)

    X, y = data["X"], data["y"]
    edge_index = data["edge_index"]
    edge_weight = data["edge_weight"] if use_edge_weight else None
    train_mask, val_mask = data["masks"]["train"], data["masks"]["val"]
    drug_columns = data["drug_columns"]

    model = MultiTaskGCN(
        in_dim=X.shape[1],
        hidden_dim=hidden_dim,
        n_drugs=len(drug_columns),
        dropout=dropout,
        skip_connection=skip_connection,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = compute_pos_weight(y[train_mask])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, edge_index, edge_weight)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X, edge_index, edge_weight)
            val_pred = (torch.sigmoid(val_logits[val_mask]) >= 0.5).int().numpy()
            val_true = y[val_mask].int().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_macro_f1": float(val_f1)})
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1}
