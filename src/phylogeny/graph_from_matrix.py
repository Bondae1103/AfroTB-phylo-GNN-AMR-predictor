"""Build pyg_graph.pt from an already-materialised SNP matrix.

run_build_graph.py owns the full VCF path (parse 13,753 VCFs -> site index ->
sparse matrix -> distance -> k-NN). This module covers the case where the
sparse matrix already exists on disk and only the distance + k-NN stages need
running -- which is what the synthetic fixture needs, and is also useful for
re-running graph construction with a different k or distance metric without
re-parsing 20.8M VCF records.

It deliberately calls the SAME primitives as the VCF path
(compute_pairwise_jaccard_distance, build_knn_edges) and the same defaults
(k=20, union mode, w = 1 - jaccard), so a graph built here is directly
comparable to one built by run_build_graph.py.
"""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .distance import compute_pairwise_jaccard_distance
from .knn_graph import build_knn_edges
from .run_build_graph import (
    JACCARD_EDGE_WEIGHT_FN,
    K_NEIGHBORS,
    KNN_MODE,
    NEAR_DUPLICATE_THRESHOLD,
    degree_stats,
    n_connected_components,
    save_pyg_graph,
)


def cross_split_near_duplicates(D, sample_ids, split_by_id, threshold=NEAR_DUPLICATE_THRESHOLD):
    """val/test isolates having a train neighbour at distance <= threshold.

    Same diagnostic as run_build_graph.py's, but taking the split mapping as an
    argument instead of reading splits.csv, so it works for any processed dir.
    """
    split = np.array([split_by_id[s] for s in sample_ids])
    train_idx = np.where(split == "train")[0]
    flagged = []
    if train_idx.size:
        for i, sid in enumerate(sample_ids):
            if split[i] == "train":
                continue
            d = D[i, train_idx]
            md = float(d.min())
            if md <= threshold:
                flagged.append({
                    "sample_id": sid,
                    "split": str(split[i]),
                    "nearest_train_id": sample_ids[int(train_idx[int(np.argmin(d))])],
                    "distance": md,
                })
    return {"threshold": threshold, "n_flagged": len(flagged), "examples": flagged[:50]}


def build(processed_dir, k=None, mode=None, keep_distance=False):
    """Read <processed_dir>/snp_matrix.npz + sample_ids.csv, write pyg_graph.pt.

    Returns (report dict, D) where D is the dense Jaccard distance matrix if
    keep_distance else None. Callers that need D for a phylogeny-aware split
    should pass keep_distance=True rather than recomputing it.
    """
    k = K_NEIGHBORS if k is None else k
    mode = KNN_MODE if mode is None else mode
    processed_dir = Path(processed_dir)

    with open(processed_dir / "sample_ids.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        sample_ids = [row[0] for row in reader]

    X = sp.load_npz(processed_dir / "snp_matrix.npz")
    if X.shape[0] != len(sample_ids):
        raise ValueError(
            "snp_matrix.npz has %d rows but sample_ids.csv has %d IDs"
            % (X.shape[0], len(sample_ids))
        )

    D = compute_pairwise_jaccard_distance(X)
    edge_index, edge_weight = build_knn_edges(
        D, k=k, mode=mode, weight_fn=JACCARD_EDGE_WEIGHT_FN
    )
    save_pyg_graph(edge_index, edge_weight, sample_ids, processed_dir / "pyg_graph.pt")

    report = {
        "source": "graph_from_matrix.build",
        "n_isolates": len(sample_ids),
        "n_sites": int(X.shape[1]),
        "knn_k": k,
        "knn_mode": mode,
        "n_edges": int(edge_index.shape[1]),
        "degree_stats": degree_stats(edge_index, len(sample_ids)),
        "n_connected_components": n_connected_components(edge_index, len(sample_ids)),
    }

    splits_path = processed_dir / "splits.csv"
    if splits_path.exists():
        with open(splits_path, newline="", encoding="utf-8") as f:
            split_by_id = {r["Name"]: r["split"] for r in csv.DictReader(f)}
        report["cross_split_near_duplicate_diagnostic"] = cross_split_near_duplicates(
            D, sample_ids, split_by_id
        )

    with open(processed_dir / "graph_construction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report, (D if keep_distance else None)
