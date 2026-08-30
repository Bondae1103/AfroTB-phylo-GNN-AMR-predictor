"""Random Forest multi-label baseline for AMR prediction.

RandomForestClassifier natively supports a 2D binary y (multilabel-indicator
format) -- it fits one tree ensemble per drug internally without any wrapper.
"""

from sklearn.ensemble import RandomForestClassifier


def build_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
