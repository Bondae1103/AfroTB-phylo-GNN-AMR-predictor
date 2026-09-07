"""Train TorchMLP (new PyTorch multilabel baseline) on the fixed splits and
write results/torch_mlp_metrics.json. Does not touch results/baseline_metrics.json,
results/gnn_metrics.json, or any other existing result file.

Reuses src/baselines/data_loader.py (identical loader/fixed split as
RF/XGBoost/sklearn-MLP) and src/baselines/metrics.py (identical val-tuned-
per-drug-threshold protocol) so results are directly comparable across all
baselines and the GNN.

Per the class-imbalance/preprocessing audit: CAP/ETH/LZD have raw train
pos_weight of ~457.5/2406.3/4813.5 -- using these directly would destabilize
BCE training on a shared trunk, so this first experiment runs per-drug
weighted BCE with pos_weight capped at POS_WEIGHT_CAP (default 20), per the
audit's recommendation. Both raw and effective (capped) weights are recorded
in the output regardless of which was actually used (USE_CLASS_WEIGHT).

The 157 features are kept as raw float32 0/1 -- no scaling/normalization is
applied (see audit: standardizing an already-uniform binary indicator matrix
has no clear benefit and just adds a fit-on-train surface for no reason).
"""

import json
import os
import time
from pathlib import Path

import torch

from .data_loader import load_aligned_data, split_masks
from .metrics import evaluate_multilabel, tune_per_drug_thresholds
from .results_path import resolve as resolve_results_path
from .train_torch_mlp import train_model

RESULTS_PATH = Path("results/torch_mlp_metrics.json")
SEED = 42

HIDDEN_DIMS = (128, 64)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 20
USE_CLASS_WEIGHT = True
POS_WEIGHT_CAP = 20.0


def main():
    data = load_aligned_data()
    # See src/baselines/results_path.py: a run redirected away from
    # data/processed writes beside its own input, never into results/.
    results_path = resolve_results_path(data["processed_dir"], RESULTS_PATH)
    drug_columns = data["drug_columns"]
    masks = split_masks(data)

    X = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"])
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    t0 = time.time()
    model, train_info = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        use_class_weight=USE_CLASS_WEIGHT,
        pos_weight_cap=POS_WEIGHT_CAP,
        verbose=True,
    )
    fit_seconds = time.time() - t0
    print(
        f"Trained to epoch {train_info['best_epoch']} "
        f"(best val macro-F1={train_info['best_val_macro_f1']:.4f}) in {fit_seconds:.1f}s"
    )

    # Test tensors are used here for the first time, purely for a frozen-
    # weight forward pass to score already-selected (val-only-tuned)
    # thresholds -- never for loss, early stopping, or class-weight/threshold
    # selection, all of which happened above using only X_train/y_train/X_val/y_val.
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(X_val)).numpy()
        test_proba = torch.sigmoid(model(X_test)).numpy()

    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    tuned_thresholds = tune_per_drug_thresholds(y_val_np, val_proba, drug_columns)

    raw_pos_weight = dict(zip(drug_columns, train_info["raw_pos_weight"]))
    effective_pos_weight = dict(zip(drug_columns, train_info["effective_pos_weight"]))

    train_class_support = {}
    val_class_support = {}
    test_class_support = {}
    for i, drug in enumerate(drug_columns):
        tr_pos = int(y_train[:, i].sum().item())
        va_pos = int(y_val[:, i].sum().item())
        te_pos = int(y_test[:, i].sum().item())
        train_class_support[drug] = {"n_positive": tr_pos, "n_negative": int(y_train.shape[0] - tr_pos)}
        val_class_support[drug] = {"n_positive": va_pos, "n_negative": int(y_val.shape[0] - va_pos)}
        test_class_support[drug] = {"n_positive": te_pos, "n_negative": int(y_test.shape[0] - te_pos)}

    zero_test_positive_drugs = [d for d, s in test_class_support.items() if s["n_positive"] == 0]

    results = {
        "seed": SEED,
        "model": "torch_multi_task_mlp",
        "note": (
            "New PyTorch multilabel MLP baseline (157 -> 128 -> ReLU -> Dropout(0.3) "
            "-> 64 -> ReLU -> Dropout(0.3) -> 9 raw logits), BCEWithLogitsLoss. "
            "Same fixed split, same data, same val-tuned-threshold eval protocol as "
            "results/baseline_metrics.json and results/gnn_metrics.json; does not "
            "modify or overwrite either. Class weights computed from y_train only; "
            "features kept as raw binary 0/1, no scaling applied."
        ),
        "dataset_provenance": {
            "processed_dir": str(data["processed_dir"]),
            "afrotb_processed_dir_env": os.environ.get("AFROTB_PROCESSED_DIR"),
        },
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "n_features": int(X.shape[1]),
        "drug_columns": drug_columns,
        "threshold_for_f1": 0.5,
        "architecture": {
            "input_dim": int(X.shape[1]),
            "hidden_dims": list(HIDDEN_DIMS),
            "output_dim": len(drug_columns),
            "activation": "relu",
            "dropout": DROPOUT,
            "final_layer": "raw logits (no sigmoid inside the model)",
        },
        "hyperparameters": {
            "optimizer": "adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
        "loss": {
            "type": "BCEWithLogitsLoss",
            "use_class_weight": USE_CLASS_WEIGHT,
            "pos_weight_cap": POS_WEIGHT_CAP,
            "raw_pos_weight": raw_pos_weight,
            "effective_pos_weight": effective_pos_weight,
            "note": "raw = train_negatives / train_positives from y_train only; effective = raw clipped to pos_weight_cap; effective is what BCEWithLogitsLoss actually used (only when use_class_weight=True).",
        },
        "train_class_support": train_class_support,
        "val_class_support": val_class_support,
        "test_class_support": test_class_support,
        "rare_drug_caveat": (
            "CAP/ETH/LZD have extremely low positive support (train n=21/4/2 "
            "respectively). Any F1 change for these drugs, up or down, reflects "
            "a handful of examples and should not be read as a reliable model "
            "improvement. ETH has zero positive samples in the test split, so its "
            "test F1 is trivially defined (no positives to recover) and its "
            "ROC-AUC/PR-AUC are undefined -- see per_drug 'auc_note' below, "
            "consistent with how results/gnn_metrics.json and "
            "results/baseline_metrics.json already report ETH."
        ),
        "zero_test_positive_drugs": zero_test_positive_drugs,
        "fit_seconds": round(fit_seconds, 2),
        "best_epoch": train_info["best_epoch"],
        "best_val_macro_f1_during_training": train_info["best_val_macro_f1"],
        "training_history_tail": train_info["history"][-10:],
        "default_threshold_0.5": {
            "val": evaluate_multilabel(y_val_np, val_proba, drug_columns, threshold=0.5),
            "test": evaluate_multilabel(y_test_np, test_proba, drug_columns, threshold=0.5),
        },
        "tuned_per_drug_threshold": {
            "thresholds": tuned_thresholds,
            "tuned_on": "val (F1-maximizing per drug via precision_recall_curve); never uses test",
            "val": evaluate_multilabel(y_val_np, val_proba, drug_columns, threshold=tuned_thresholds),
            "test": evaluate_multilabel(y_test_np, test_proba, drug_columns, threshold=tuned_thresholds),
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
