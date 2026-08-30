"""Skip-connection ablation: train MultiTaskGCN with and without concatenating
raw features onto the head's input, everything else identical (including
use_edge_weight=True, the setting the edge-weight ablation validated).
Writes results/gnn_skip_connection_ablation.json.
"""

import json
import time
from pathlib import Path

import torch

from ..baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from .data import load_gnn_data
from .train import train_model
from .run_train import DROPOUT, HIDDEN_DIM, LR, MAX_EPOCHS, PATIENCE, SEED, WEIGHT_DECAY

RESULTS_PATH = Path("results/gnn_skip_connection_ablation.json")


def run_variant(data, skip_connection):
    t0 = time.time()
    model, train_info = train_model(
        data,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        verbose=False,
        use_edge_weight=True,
        skip_connection=skip_connection,
    )
    fit_seconds = time.time() - t0

    with torch.no_grad():
        logits = model(data["X"], data["edge_index"], data["edge_weight"])
        proba = torch.sigmoid(logits).numpy()

    y_np = data["y"].numpy()
    masks_np = data["masks_np"]
    drug_columns = data["drug_columns"]
    y_val, y_test = y_np[masks_np["val"]], y_np[masks_np["test"]]
    val_proba, test_proba = proba[masks_np["val"]], proba[masks_np["test"]]
    tuned_thresholds = tune_per_drug_thresholds(y_val, val_proba, drug_columns)

    return {
        "skip_connection": skip_connection,
        "fit_seconds": round(fit_seconds, 2),
        "best_epoch": train_info["best_epoch"],
        "best_val_macro_f1_during_training": train_info["best_val_macro_f1"],
        "default_threshold_0.5": {
            "val": evaluate_multilabel(y_val, val_proba, drug_columns, threshold=0.5),
            "test": evaluate_multilabel(y_test, test_proba, drug_columns, threshold=0.5),
        },
        "tuned_per_drug_threshold": {
            "thresholds": tuned_thresholds,
            "val": evaluate_multilabel(y_val, val_proba, drug_columns, threshold=tuned_thresholds),
            "test": evaluate_multilabel(y_test, test_proba, drug_columns, threshold=tuned_thresholds),
        },
    }


def main():
    data = load_gnn_data()

    print("Training WITHOUT skip connection (baseline GCN)...")
    no_skip = run_variant(data, skip_connection=False)
    print("Training WITH skip connection (raw features concatenated at head)...")
    with_skip = run_variant(data, skip_connection=True)

    results = {
        "seed": SEED,
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "use_edge_weight": True,
        },
        "without_skip_connection": no_skip,
        "with_skip_connection": with_skip,
    }

    for label, r in (("WITHOUT skip connection", no_skip), ("WITH skip connection", with_skip)):
        d, tu = r["default_threshold_0.5"], r["tuned_per_drug_threshold"]
        print(
            f"{label:24s} best_epoch={r['best_epoch']:3d}  "
            f"[0.5] test macro-F1={d['test']['macro_f1']:.3f} macro-ROC-AUC={d['test']['macro_roc_auc']:.4f}  "
            f"[tuned] test macro-F1={tu['test']['macro_f1']:.3f}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
