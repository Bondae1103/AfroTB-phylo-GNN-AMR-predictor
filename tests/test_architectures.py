"""Architectures and the shared trainer."""

import numpy as np
import pytest
import torch

from src.baselines.data_loader import load_aligned_data, split_masks
from src.gnn.architectures import ARCHITECTURES, GatedHybridGNN
from src.experiments.trainer import IGNORE_INDEX, train


@pytest.fixture(scope="module")
def tensors(replica):
    tab = load_aligned_data(replica["dir"])
    masks = split_masks(tab)
    graph = torch.load(replica["dir"] / "pyg_graph.pt", weights_only=False)
    return {
        "X": torch.tensor(tab["X"], dtype=torch.float32),
        "y": torch.tensor(tab["y"], dtype=torch.float32),
        "edge_index": graph.edge_index.long(),
        "edge_weight": graph.edge_weight.float(),
        "masks": {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()},
        "drug_columns": tab["drug_columns"],
    }


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_trains_and_returns_valid_probabilities(arch, tensors):
    _, proba, info = train(
        tensors["X"], tensors["y"], tensors["edge_index"], tensors["edge_weight"],
        tensors["masks"], architecture=arch, max_epochs=12, patience=12, seed=0,
    )
    assert proba.shape == tuple(tensors["y"].shape)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert info["architecture"] == arch


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_exposes_embed_for_multitask(arch, tensors):
    model = ARCHITECTURES[arch](in_dim=tensors["X"].shape[1], hidden_dim=16,
                                n_out=9, dropout=0.0)
    model.eval()
    h = model.embed(tensors["X"], tensors["edge_index"], tensors["edge_weight"])
    assert h.shape[0] == tensors["X"].shape[0]
    assert model.head(h).shape == (tensors["X"].shape[0], 9)


def test_training_is_deterministic_given_a_seed(tensors):
    out = []
    for _ in range(2):
        _, proba, _ = train(
            tensors["X"], tensors["y"], tensors["edge_index"], tensors["edge_weight"],
            tensors["masks"], architecture="gcn_skip", max_epochs=8, patience=8, seed=3,
        )
        out.append(proba)
    assert np.allclose(out[0], out[1])


def test_gated_hybrid_reports_its_gate(tensors):
    _, _, info = train(
        tensors["X"], tensors["y"], tensors["edge_index"], tensors["edge_weight"],
        tensors["masks"], architecture="gated_hybrid", max_epochs=10, patience=10, seed=0,
    )
    g = info["gate_stats"]
    assert g is not None
    assert 0.0 <= g["gamma_mean"] <= 1.0
    assert 0.0 <= g["fraction_above_0.9"] <= 1.0


def test_multitask_lineage_ignores_missing_labels(tensors, replica):
    """Isolates with IGNORE_INDEX must not break the loss or change row count."""
    from src.protocols.lineage import build_y_lineage

    y_lin, classes, rep = build_y_lineage(replica["dir"])
    assert rep["n_missing"] >= 1
    _, proba, info = train(
        tensors["X"], tensors["y"], tensors["edge_index"], tensors["edge_weight"],
        tensors["masks"], architecture="gcn_skip", max_epochs=10, patience=10, seed=0,
        y_lineage=torch.tensor(y_lin, dtype=torch.long),
        n_lineage_classes=len(classes), lambda_lineage=0.5,
    )
    assert info["multitask_lineage"] is True
    assert proba.shape[0] == tensors["X"].shape[0]
    assert np.isfinite(proba).all()


def test_selection_metric_subset_changes_only_checkpoint_not_output_shape(tensors):
    _, proba, info = train(
        tensors["X"], tensors["y"], tensors["edge_index"], tensors["edge_weight"],
        tensors["masks"], architecture="gcn_skip", max_epochs=10, patience=10, seed=0,
        selection_drug_idx=[0, 1, 2, 3, 4, 5],
    )
    assert proba.shape[1] == 9, "reported metrics must still cover every drug"


def test_mlp_only_ignores_the_graph(tensors):
    """The no-graph control must be invariant to the edge set."""
    kw = dict(architecture="mlp_only", max_epochs=8, patience=8, seed=1)
    _, p1, _ = train(tensors["X"], tensors["y"], tensors["edge_index"],
                     tensors["edge_weight"], tensors["masks"], **kw)
    shuffled = tensors["edge_index"][:, torch.randperm(tensors["edge_index"].shape[1])]
    _, p2, _ = train(tensors["X"], tensors["y"], shuffled,
                     tensors["edge_weight"], tensors["masks"], **kw)
    assert np.allclose(p1, p2)
