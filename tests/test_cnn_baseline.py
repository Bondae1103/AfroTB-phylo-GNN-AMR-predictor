import numpy as np
import torch

from src.baselines.torch_cnn_model import TorchCNN
from src.experiments.benchmark import run_cell


def test_cnn_output_shape_for_catalogue_and_genomewide_widths():
    for width in (157, 1500):
        out = TorchCNN(width, n_drugs=9)(torch.zeros(4, width))
        assert out.shape == (4, 9)


def test_cnn1d_run_cell_returns_probabilities_for_every_row():
    rng = np.random.default_rng(0)
    n, f = 120, 64
    X = (rng.random((n, f)) < 0.3).astype(np.float32)
    drugs = ["RIF", "INH", "EMB", "PZA", "STM", "LEV", "CAP", "ETH", "LZD"]
    y = (rng.random((n, 9)) < 0.3).astype(np.float32)
    split = np.array(["train"] * 80 + ["val"] * 20 + ["test"] * 20)
    masks = {s: split == s for s in ("train", "val", "test")}
    proba, info = run_cell("cnn1d", X, y, masks, None, None, drugs, seed=0,
                           max_epochs=3, patience=3)
    assert proba.shape == (n, 9)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert info["architecture"] == "cnn1d"
