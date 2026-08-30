"""Hard-parameter-sharing multi-task MLP baseline for AMR prediction.

sklearn's MLPClassifier natively supports multilabel-indicator targets: a
shared stack of hidden layers feeds one independent sigmoid output unit per
drug. Architecturally this already is a hard-parameter-sharing multi-task
network (shared trunk, per-task heads), so no torch dependency is needed for
this baseline.
"""

from sklearn.neural_network import MLPClassifier


def build_model(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=random_state,
    )
