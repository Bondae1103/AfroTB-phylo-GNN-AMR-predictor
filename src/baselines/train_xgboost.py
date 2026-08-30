"""XGBoost multi-label baseline for AMR prediction.

XGBoost's core estimator is single-output, so MultiOutputClassifier fits one
independent XGBClassifier per drug (no parameter sharing across drugs -- in
contrast to the shared-trunk multi-task MLP baseline). This is the standard
approach to multi-label classification with gradient-boosted trees.
"""

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


def build_model(random_state=42):
    base = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )
    return MultiOutputClassifier(base, n_jobs=1)
