"""Training loop for TorchMLP: mini-batch BCEWithLogitsLoss, early stopping
on val macro-F1 only -- mirrors src/gnn/train.py's protocol so the two are
directly comparable. Supports both unweighted and per-drug-weighted BCE via
a configurable, train-only pos_weight vector with an optional cap (see the
class-imbalance audit: raw pos_weight reaches 4813.5 for LZD, which would
destabilize a shared-trunk network if used uncapped).

train_model() only ever receives X_train/y_train/X_val/y_val -- test data
never reaches this function, so it structurally cannot influence the loss,
early stopping, or (via compute_pos_weight) the class weights.
"""

import random

import numpy as np
import torch
from sklearn.metrics import f1_score

from .torch_mlp_model import TorchMLP


def compute_pos_weight(y_train, cap=None):
    """Per-drug BCE pos_weight = train_negatives / train_positives, computed
    strictly from y_train (never val/test). Returns (raw, effective) tensors;
    effective is raw clipped to `cap` when given, else identical to raw.
    A drug with zero train positives would fall back to weight 1.0 (pos_weight
    is undefined otherwise) -- doesn't occur on the actual AfroTB train split
    (minimum support is LZD's 2), but guards the general case.
    """
    n_pos = y_train.sum(dim=0)
    n_neg = y_train.shape[0] - n_pos
    raw = torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1), torch.ones_like(n_pos))
    effective = torch.clamp(raw, max=cap) if cap is not None else raw.clone()
    return raw, effective


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dims=(128, 64),
    dropout=0.3,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=256,
    max_epochs=300,
    patience=20,
    seed=42,
    use_class_weight=True,
    pos_weight_cap=20.0,
    verbose=False,
    model_factory=None,
    selection_drug_idx=None,
):
    """use_class_weight=False trains with plain unweighted BCEWithLogitsLoss.
    use_class_weight=True (default) uses per-drug pos_weight, clipped to
    pos_weight_cap (raw and effective values are both returned regardless of
    which mode is active, so the run's output always records what the raw
    imbalance was and what was actually used).
    """
    set_seed(seed)

    n_drugs = y_train.shape[1]
    if model_factory is not None:
        model = model_factory(X_train.shape[1], n_drugs)
    else:
        model = TorchMLP(in_dim=X_train.shape[1], hidden_dims=hidden_dims, n_drugs=n_drugs, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raw_pos_weight, effective_pos_weight = compute_pos_weight(y_train, cap=pos_weight_cap)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight if use_class_weight else None)

    generator = torch.Generator().manual_seed(seed)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred = (torch.sigmoid(val_logits) >= 0.5).int().numpy()
            val_true = y_val.int().numpy()
            if selection_drug_idx is not None:
                # same checkpoint criterion as src/experiments/trainer.py: drugs with real support only
                val_true, val_pred = val_true[:, selection_drug_idx], val_pred[:, selection_drug_idx]
            val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": float(val_f1)})
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")

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
    return model, {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "raw_pos_weight": raw_pos_weight.tolist(),
        "effective_pos_weight": effective_pos_weight.tolist(),
    }
