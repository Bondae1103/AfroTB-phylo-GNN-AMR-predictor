"""Oracle upper bound for the synthetic fixture. FIXTURE-ONLY.

In the replica, resistance is drawn per isolate from its clade's probability
(src/synthetic/afrotb_replica.py). Under a protocol where the model cannot see
a drug's own catalogue mutations, the best any predictor can do is recover the
clade and emit that clade's base rate -- the residual is coin flips the
generator made and no feature records.

This computes that bound by cheating with the true clade labels. It exists so a
genome-wide score can be read against the ceiling of its own problem instead of
against 1.0. It is meaningless for the real Afro-TB data, where no such ground
truth exists, and must never be reported as a model result.
"""

import csv
from pathlib import Path

import numpy as np

from ..baselines.metrics import evaluate_multilabel, tune_per_drug_thresholds


def read_true_clade(processed_dir):
    path = Path(processed_dir) / "true_clade.csv"
    if not path.exists():
        raise FileNotFoundError(
            "%s not found -- regenerate the fixture with the current "
            "src/synthetic/afrotb_replica.py" % path
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["clade_id"]) for r in rows], dtype=np.int64)


def clade_rate_oracle(processed_dir, y, masks, drug_columns):
    """Predict each isolate's clade base rate, estimated on TRAIN rows only.

    Returns the same metric dict shape the benchmark uses, so the bound can be
    printed in the same table as the models.
    """
    clade = read_true_clade(processed_dir)
    train = masks["train"]
    n_clades = int(clade.max()) + 1

    proba = np.zeros_like(y, dtype=float)
    global_rate = y[train].mean(axis=0)
    for c in range(n_clades):
        in_c = clade == c
        tr = in_c & train
        # a clade with no training members falls back to the global rate;
        # estimating from its own val/test rows would be leakage
        rate = y[tr].mean(axis=0) if tr.sum() > 0 else global_rate
        proba[in_c] = rate

    thr = tune_per_drug_thresholds(y[masks["val"]], proba[masks["val"]], drug_columns)
    ev = evaluate_multilabel(y[masks["test"]], proba[masks["test"]],
                             drug_columns, threshold=thr)
    core6 = [ev["per_drug"][d]["f1"] for d in
             ("RIF", "INH", "EMB", "PZA", "STM", "LEV") if d in ev["per_drug"]]
    return {
        "what": "oracle: true clade -> that clade's train-split resistance rate",
        "test_macro_f1_core6": float(np.mean(core6)),
        "test_macro_f1_all9": ev["macro_f1"],
        "test_macro_pr_auc": ev["macro_pr_auc"],
        "test_per_drug_f1": {d: ev["per_drug"][d]["f1"] for d in drug_columns},
        "caveat": "FIXTURE-ONLY upper bound; uses ground truth unavailable in real data",
    }
