"""Multi-label evaluation metrics for AMR baseline models.

F1 metrics are computed on thresholded predictions -- either a single global
threshold or a per-drug threshold dict (see tune_per_drug_thresholds below).
ROC-AUC and PR-AUC are threshold-independent, computed on predicted
probabilities. A drug column that has only one class present in a given split
(e.g. all-zero) has an undefined ROC-AUC/PR-AUC for that split; such drugs are
reported as null for that split and excluded from the macro average, rather
than silently assigned a default score.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _threshold_array(threshold, drug_columns):
    if isinstance(threshold, dict):
        return np.array([threshold[d] for d in drug_columns], dtype=float)
    if np.isscalar(threshold):
        return np.full(len(drug_columns), float(threshold))
    return np.asarray(threshold, dtype=float)


def tune_per_drug_thresholds(y_val, val_prob, drug_columns, default=0.5):
    """Pick the F1-maximizing decision threshold per drug, using val only.

    Must never be called with test-set data: the whole point is that the
    threshold is selected before test is touched, so test evaluation stays
    an unbiased, single-shot check rather than something implicitly tuned.
    A drug with a single class in val (no positives, usually) cannot have a
    threshold tuned meaningfully and keeps `default`.
    """
    y_val = y_val.astype(int)
    thresholds = {}
    for i, drug in enumerate(drug_columns):
        yt, yprob = y_val[:, i], val_prob[:, i]
        if len(np.unique(yt)) < 2:
            thresholds[drug] = default
            continue
        precision, recall, cand_thresholds = precision_recall_curve(yt, yprob)
        f1s = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )[:-1]  # last precision/recall point has no corresponding threshold
        best_idx = int(np.argmax(f1s))
        thresholds[drug] = float(np.clip(cand_thresholds[best_idx], 0.01, 0.99))
    return thresholds


def evaluate_multilabel(y_true, y_prob, drug_columns, threshold=0.5):
    """Compute macro/micro F1, per-drug F1, and macro ROC-AUC/PR-AUC.

    y_true: (N, D) binary array. y_prob: (N, D) predicted-probability array.
    threshold: a single float applied to every drug, or a dict {drug: thresh}
    (e.g. from tune_per_drug_thresholds) for per-drug decision thresholds.
    """
    y_true = y_true.astype(int)
    thresh_arr = _threshold_array(threshold, drug_columns)
    y_pred = (y_prob >= thresh_arr[None, :]).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    per_drug = {}
    roc_aucs = []
    pr_aucs = []
    for i, drug in enumerate(drug_columns):
        yt, yp, yprob = y_true[:, i], y_pred[:, i], y_prob[:, i]
        entry = {
            "threshold": float(thresh_arr[i]),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "n_positive": int(yt.sum()),
            "n_total": int(len(yt)),
        }
        if len(np.unique(yt)) < 2:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
            entry["auc_note"] = "single-class in this split; AUC undefined, excluded from macro average"
        else:
            roc = roc_auc_score(yt, yprob)
            pr = average_precision_score(yt, yprob)
            entry["roc_auc"] = float(roc)
            entry["pr_auc"] = float(pr)
            roc_aucs.append(roc)
            pr_aucs.append(pr)
        per_drug[drug] = entry

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_roc_auc": float(np.mean(roc_aucs)) if roc_aucs else None,
        "macro_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
        "n_drugs_with_defined_auc": len(roc_aucs),
        "n_drugs_total": len(drug_columns),
        "per_drug": per_drug,
    }
