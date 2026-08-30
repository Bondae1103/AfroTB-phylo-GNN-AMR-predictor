"""Load node features, the AMR target, splits, and the phylogenetic graph,
strictly aligned to sample_ids.csv's canonical order.

Reuses src/baselines/data_loader.py for features.csv/y_amr.csv/splits.csv
(same validation, same DRUG_COLUMNS -- AMR-only for this first GNN pass, no
Lineage head yet) and data/processed/pyg_graph.pt (built by
src/phylogeny/run_build_graph.py) for edge_index/edge_weight. Fails loudly if
the graph's stored sample_ids don't match the canonical order exactly.
"""

from pathlib import Path

import torch

from ..baselines.data_loader import load_aligned_data, processed_dir, split_masks

GRAPH_PATH = Path("data/processed/pyg_graph.pt")


def load_gnn_data(dirpath=None):
    """Returns a dict: X (N,157) float32, y (N,9) float32, masks (train/val/test
    bool tensors), edge_index (2,E) long, edge_weight (E,) float32,
    drug_columns, sample_ids.

    dirpath (or the AFROTB_PROCESSED_DIR env var) redirects every artifact,
    graph included, to another processed-data directory -- see
    src/baselines/data_loader.py:processed_dir.
    """
    base = processed_dir(dirpath)
    tabular = load_aligned_data(base)
    graph = torch.load(base / "pyg_graph.pt", weights_only=False)

    graph_sample_ids = list(graph.sample_ids)
    if graph_sample_ids != tabular["sample_ids"]:
        raise ValueError(
            "pyg_graph.pt sample_ids do not match sample_ids.csv canonical order -- "
            "re-run src/phylogeny/run_build_graph.py before training."
        )

    masks_np = split_masks(tabular)

    return {
        "X": torch.tensor(tabular["X"], dtype=torch.float32),
        "y": torch.tensor(tabular["y"], dtype=torch.float32),
        "masks": {name: torch.tensor(mask, dtype=torch.bool) for name, mask in masks_np.items()},
        "masks_np": masks_np,
        "edge_index": graph.edge_index.long(),
        "edge_weight": graph.edge_weight.float(),
        "drug_columns": tabular["drug_columns"],
        "sample_ids": tabular["sample_ids"],
        "processed_dir": base,
    }
