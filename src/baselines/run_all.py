"""Train and evaluate all Phase 1 tabular baselines on the fixed splits.

Uses data/processed/{sample_ids,features,y_amr,splits}.csv exactly as
produced by Person 1's pipeline -- no relabeling, no re-splitting. Trains
each model on `train` only; `val` is scored for monitoring (and used
internally by the MLP's early stopping) but no hyperparameter search is
performed here; `test` is scored once, after training, and never used to
select a model or hyperparameters. Writes results/baseline_metrics.json.
"""

import json
import time
from pathlib import Path

from . import train_mlp, train_random_forest, train_xgboost
from .data_loader import load_aligned_data, split_masks
from .results_path import resolve as resolve_results_path
from .metrics import evaluate_multilabel, tune_per_drug_thresholds
from .multilabel_utils import get_proba_matrix

RESULTS_PATH = Path("results/baseline_metrics.json")
SEED = 42

MODEL_BUILDERS = {
    "random_forest": train_random_forest.build_model,
    "xgboost": train_xgboost.build_model,
    "multi_task_mlp": train_mlp.build_model,
}


def run_model(build_fn, X_train, y_train, X_val, y_val, X_test, y_test, drug_columns):
    t0 = time.time()
    model = build_fn(random_state=SEED)
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    val_proba = get_proba_matrix(model, X_val)
    test_proba = get_proba_matrix(model, X_test)

    # Thresholds are selected on val only (F1-maximizing per drug) and then
    # applied unchanged to test -- test probabilities are never used to pick
    # a threshold, so the tuned-threshold test score stays a single-shot,
    # unbiased evaluation rather than something implicitly fit to test.
    tuned_thresholds = tune_per_drug_thresholds(y_val, val_proba, drug_columns)

    return {
        "fit_seconds": round(fit_seconds, 2),
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


def main():
    data = load_aligned_data()
    # Write beside the input when the run was redirected away from
    # data/processed, so a synthetic or subset run can never overwrite the
    # committed Afro-TB metrics (src/baselines/results_path.py).
    results_path = resolve_results_path(data["processed_dir"], RESULTS_PATH)
    masks = split_masks(data)
    X, y = data["X"], data["y"]
    drug_columns = data["drug_columns"]

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]

    results = {
        "seed": SEED,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "n_features": int(X.shape[1]),
        "drug_columns": drug_columns,
        "threshold_for_f1": 0.5,
        "models": {},
    }

    for name, build_fn in MODEL_BUILDERS.items():
        print(f"Training {name} ...")
        model_result = run_model(build_fn, X_train, y_train, X_val, y_val, X_test, y_test, drug_columns)
        results["models"][name] = model_result
        d, tu = model_result["default_threshold_0.5"], model_result["tuned_per_drug_threshold"]
        print(
            f"  fit={model_result['fit_seconds']:.1f}s  "
            f"[0.5] test macro-F1={d['test']['macro_f1']:.3f}  "
            f"[tuned] test macro-F1={tu['test']['macro_f1']:.3f} "
            f"(val macro-F1={tu['val']['macro_f1']:.3f})"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
