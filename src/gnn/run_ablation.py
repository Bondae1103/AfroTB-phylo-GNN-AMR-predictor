"""Edge-weight ablation: train MultiTaskGCN with and without the Jaccard-
derived edge_weight from pyg_graph.pt, everything else identical (same
graph structure/edges, same architecture, hyperparameters, and seed), to
isolate whether the *weighting* itself matters, independent of the graph's
connectivity. Writes results/gnn_edge_weight_ablation.json.
"""

import json
import time
from pathlib import Path

import torch

from ..baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from .data import load_gnn_data
from .train import train_model
from .run_train import DROPOUT, HIDDEN_DIM, LR, MAX_EPOCHS, PATIENCE, SEED, WEIGHT_DECAY

RESULTS_PATH = Path("results/gnn_edge_weight_ablation.json")


def run_variant(data, use_edge_weight):
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
        use_edge_weight=use_edge_weight,
    )
    fit_seconds = time.time() - t0

    edge_weight = data["edge_weight"] if use_edge_weight else None
    with torch.no_grad():
        logits = model(data["X"], data["edge_index"], edge_weight)
        proba = torch.sigmoid(logits).numpy()

    y_np = data["y"].numpy()
    masks_np = data["masks_np"]
    drug_columns = data["drug_columns"]
    y_val, y_test = y_np[masks_np["val"]], y_np[masks_np["test"]]
    val_proba, test_proba = proba[masks_np["val"]], proba[masks_np["test"]]
    tuned_thresholds = tune_per_drug_thresholds(y_val, val_proba, drug_columns)

    return {
        "use_edge_weight": use_edge_weight,
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

    print("Training WITH edge_weight (Jaccard-derived)...")
    weighted = run_variant(data, use_edge_weight=True)
    print("Training WITHOUT edge_weight (uniform, weight=1 for every edge)...")
    unweighted = run_variant(data, use_edge_weight=False)

    results = {
        "seed": SEED,
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        },
        "with_edge_weight": weighted,
        "without_edge_weight": unweighted,
    }

    for label, r in (("WITH edge_weight", weighted), ("WITHOUT edge_weight", unweighted)):
        d, tu = r["default_threshold_0.5"], r["tuned_per_drug_threshold"]
        print(
            f"{label:22s} best_epoch={r['best_epoch']:3d}  "
            f"[0.5] test macro-F1={d['test']['macro_f1']:.3f} macro-ROC-AUC={d['test']['macro_roc_auc']:.4f}  "
            f"[tuned] test macro-F1={tu['test']['macro_f1']:.3f}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
