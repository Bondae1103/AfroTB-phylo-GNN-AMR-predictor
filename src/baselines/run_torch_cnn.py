"""Train TorchCNN (1D-CNN deep-learning baseline) on the fixed splits and
write results/torch_cnn_metrics.json. Does not touch any other result file.

Identical protocol to run_torch_mlp.py so the CNN, MLP and GNN are directly
comparable: same loader and fixed split, same capped per-drug pos_weight
(train only), same early stopping on val macro-F1, same val-tuned per-drug
thresholds. Multi-seed: results are per seed plus mean/std.

    .venv/Scripts/python.exe -m src.baselines.run_torch_cnn
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data_loader import load_aligned_data, split_masks
from .metrics import evaluate_multilabel, tune_per_drug_thresholds
from .results_path import resolve as resolve_results_path
from .torch_cnn_model import TorchCNN
from .train_torch_mlp import train_model

RESULTS_PATH = Path("results/torch_cnn_metrics.json")
SEEDS = (42, 123, 456, 789, 2026)

CHANNELS = (32, 64)
KERNEL_SIZE = 5
HIDDEN_DIM = 128
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 20
POS_WEIGHT_CAP = 10.0  # matches the torch MLP cap10 configuration


def _factory(in_dim, n_drugs):
    return TorchCNN(in_dim, channels=CHANNELS, kernel_size=KERNEL_SIZE,
                    hidden_dim=HIDDEN_DIM, n_drugs=n_drugs, dropout=DROPOUT)


def main():
    data = load_aligned_data()
    results_path = resolve_results_path(data["processed_dir"], RESULTS_PATH)
    drugs = data["drug_columns"]
    masks = split_masks(data)

    X, y = torch.from_numpy(data["X"]), torch.from_numpy(data["y"])
    X_tr, y_tr = X[masks["train"]], y[masks["train"]]
    X_va, y_va = X[masks["val"]], y[masks["val"]]
    X_te, y_te = X[masks["test"]], y[masks["test"]]

    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        model, info = train_model(
            X_tr, y_tr, X_va, y_va, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed, use_class_weight=True,
            pos_weight_cap=POS_WEIGHT_CAP, model_factory=_factory,
        )
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(X_va)).numpy()
            test_p = torch.sigmoid(model(X_te)).numpy()
        thr = tune_per_drug_thresholds(y_va.numpy(), val_p, drugs)
        run = {
            "seed": seed,
            "best_epoch": info["best_epoch"],
            "best_val_macro_f1": info["best_val_macro_f1"],
            "fit_seconds": round(time.time() - t0, 2),
            "default_threshold_0.5": {"test": evaluate_multilabel(y_te.numpy(), test_p, drugs, threshold=0.5)},
            "tuned_per_drug_threshold": {
                "thresholds": thr,
                "test": evaluate_multilabel(y_te.numpy(), test_p, drugs, threshold=thr),
            },
        }
        per_seed.append(run)
        print(f"seed {seed}: [0.5] test macro-F1={run['default_threshold_0.5']['test']['macro_f1']:.3f}  "
              f"[tuned] {run['tuned_per_drug_threshold']['test']['macro_f1']:.3f}")

    def agg(key):
        v = np.array([r[key]["test"]["macro_f1"] for r in per_seed])
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    results = {
        "model": "torch_multi_task_cnn1d",
        "note": ("1D-CNN baseline: Conv1d blocks over the 157-column binary mutation vector, shared head, "
                 "9 logits. Same split, class-weighting (cap 10, train-only), early stopping and val-tuned "
                 "thresholds as results/torch_mlp_cap10_multiseed.json. Catalogue columns are not in genome "
                 "order, so conv locality is not biologically meaningful on this input."),
        "dataset_provenance": {"processed_dir": str(data["processed_dir"])},
        "n_train": int(masks["train"].sum()), "n_val": int(masks["val"].sum()), "n_test": int(masks["test"].sum()),
        "n_features": int(X.shape[1]),
        "drug_columns": drugs,
        "architecture": {"channels": list(CHANNELS), "kernel_size": KERNEL_SIZE,
                         "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT},
        "hyperparameters": {"lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "pos_weight_cap": POS_WEIGHT_CAP},
        "seeds": list(SEEDS),
        "aggregate": {"test_f1_0.5": agg("default_threshold_0.5"), "test_f1_tuned": agg("tuned_per_drug_threshold")},
        "per_seed": per_seed,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
