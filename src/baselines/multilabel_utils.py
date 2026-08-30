"""Shared helper to extract a dense (N, D) positive-class probability matrix.

RandomForestClassifier (fit natively on a 2D binary y) and
sklearn.multioutput.MultiOutputClassifier (used to wrap XGBoost) both expose
predict_proba() as a *list* of per-label (N, n_classes_for_that_label) arrays
plus a matching list-of-arrays `classes_`. MLPClassifier instead returns a
single dense (N, D) array directly for multilabel-indicator targets. This
module normalizes both cases to one (N, D) array of P(label=1).
"""

import numpy as np


def _positive_class_proba(proba_list, classes_list):
    n_samples = proba_list[0].shape[0]
    n_labels = len(proba_list)
    out = np.zeros((n_samples, n_labels), dtype=np.float64)
    for i, (proba, classes) in enumerate(zip(proba_list, classes_list)):
        classes = list(classes)
        if 1 in classes:
            out[:, i] = proba[:, classes.index(1)]
        else:
            out[:, i] = 0.0
    return out


def get_proba_matrix(model, X):
    """Return (N, D) array of P(label=1) for a fitted multi-label classifier."""
    proba = model.predict_proba(X)
    if isinstance(proba, list):
        return _positive_class_proba(proba, model.classes_)
    return proba
