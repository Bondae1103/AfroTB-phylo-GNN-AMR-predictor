"""Train MultiTaskGCN on the fixed splits and write results/gnn_metrics.json.

Same evaluation protocol as src/baselines/run_all.py, reusing its metrics
code directly for a fair, apples-to-apples comparison against
results/baseline_metrics.json: default-0.5 and val-tuned-per-drug-threshold
results, both val and test, per-drug detail included. Thresholds are tuned
on val only and applied unchanged to test (src/baselines/metrics.py's
tune_per_drug_thresholds) -- same non-leakage discipline as Phase 1.

AMR-only for this first pass (9 drug columns) -- no Lineage head yet (see
README Sec. 9/12: labels.csv's Lineage column isn't cleaned).

Defaults as of README Sec. 13's ablations: use_edge_weight=True (beats
uniform edges on every metric, results/gnn_edge_weight_ablation.json) and
skip_connection=True (beats plain GCN on every metric except a single-example
LZD noise case, results/gnn_skip_connection_ablation.json).
"""

import json
import time
from pathlib import Path

import torch

from ..baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds
from ..baselines.results_path import resolve as resolve_results_path
from .data import load_gnn_data
from .train import train_model

RESULTS_PATH = Path("results/gnn_metrics.json")
SEED = 42

HIDDEN_DIM = 64
LR = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3
MAX_EPOCHS = 300
PATIENCE = 20
USE_EDGE_WEIGHT = True
SKIP_CONNECTION = True


def main():
    data = load_gnn_data()
    # See src/baselines/results_path.py: a run redirected away from
    # data/processed writes beside its own input, never into results/.
    results_path = resolve_results_path(
        data.get("processed_dir", "data/processed"), RESULTS_PATH)
    drug_columns = data["drug_columns"]
    masks_np = data["masks_np"]

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
        verbose=True,
        use_edge_weight=USE_EDGE_WEIGHT,
        skip_connection=SKIP_CONNECTION,
    )
    fit_seconds = time.time() - t0
    print(f"Trained to epoch {train_info['best_epoch']} (best val macro-F1={train_info['best_val_macro_f1']:.4f}) in {fit_seconds:.1f}s")

    eval_edge_weight = data["edge_weight"] if USE_EDGE_WEIGHT else None
    with torch.no_grad():
        logits = model(data["X"], data["edge_index"], eval_edge_weight)
        proba = torch.sigmoid(logits).numpy()

    y_np = data["y"].numpy()
    val_mask, test_mask = masks_np["val"], masks_np["test"]
    y_val, y_test = y_np[val_mask], y_np[test_mask]
    val_proba, test_proba = proba[val_mask], proba[test_mask]

    tuned_thresholds = tune_per_drug_thresholds(y_val, val_proba, drug_columns)

    results = {
        "seed": SEED,
        "model": "multi_task_gcn",
        "n_train": int(masks_np["train"].sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": int(data["X"].shape[1]),
        "n_edges": int(data["edge_index"].shape[1]),
        "drug_columns": drug_columns,
        "threshold_for_f1": 0.5,
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "use_edge_weight": USE_EDGE_WEIGHT,
            "skip_connection": SKIP_CONNECTION,
        },
        "fit_seconds": round(fit_seconds, 2),
        "best_epoch": train_info["best_epoch"],
        "best_val_macro_f1_during_training": train_info["best_val_macro_f1"],
        "training_history_tail": train_info["history"][-10:],
        "default_threshold_0.5": {
            "val": evaluate_multilabel(y_val, val_proba, drug_columns, threshold=0.5),
            "test": evaluate_multilabel(y_test, test_proba, drug_columns, threshold=0.5),
        },
        "tuned_per_drug_threshold": {
            "thresholds": tuned_thresholds,
            "tuned_on": "val (F1-maximizing per drug via precision_recall_curve)",
            "val": evaluate_multilabel(y_val, val_proba, drug_columns, threshold=tuned_thresholds),
            "test": evaluate_multilabel(y_test, test_proba, drug_columns, threshold=tuned_thresholds),
        },
    }

    d, tu = results["default_threshold_0.5"], results["tuned_per_drug_threshold"]
    print(
        f"[0.5] test macro-F1={d['test']['macro_f1']:.3f}  "
        f"[tuned] test macro-F1={tu['test']['macro_f1']:.3f}"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
