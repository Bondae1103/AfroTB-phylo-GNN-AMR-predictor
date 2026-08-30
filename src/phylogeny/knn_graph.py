"""Build a sparse k-NN (or mutual-k-NN) graph from a pairwise distance matrix.

Edge weight is w = 1 / (1 + distance): monotonically decreasing in distance,
bounded in (0, 1], never divides by zero. This module never reads any label
file (y_amr.csv, labels.csv, splits.csv) -- edges are purely a function of
genomic SNP distance passed in by the caller.
"""

import numpy as np


def _knn_indices(D, k):
    """Per-row indices of the k smallest distances, excluding the diagonal."""
    n = D.shape[0]
    D_masked = D.copy()
    np.fill_diagonal(D_masked, np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(D_masked, k, axis=1)[:, :k]
    return idx


def build_knn_edges(D, k=10, mode="mutual", weight_fn=None):
    """Return (edge_index, edge_weight) as numpy arrays: edge_index (2, E) int64,
    edge_weight (E,) float32.

    mode="mutual": keep an edge i->j only if j is among i's k nearest AND i is
    among j's k nearest (fewer, cleaner edges). mode="union": keep an edge if
    either direction holds (symmetrized union). Either way the returned edge
    set is undirected (both (i,j) and (j,i) present).

    weight_fn: distance -> weight, applied elementwise. Defaults to
    1/(1+distance) (monotonic, bounded in (0,1], works for any nonnegative
    distance). For a distance already bounded in [0,1] (e.g. Jaccard), pass
    weight_fn=lambda d: 1.0 - d to use full [0,1] dynamic range instead of
    the compressed [0.5,1] that 1/(1+d) would give in that case.
    """
    if mode not in ("mutual", "union"):
        raise ValueError(f"mode must be 'mutual' or 'union', got {mode!r}")
    if weight_fn is None:
        weight_fn = lambda d: 1.0 / (1.0 + d)

    n = D.shape[0]
    knn_idx = _knn_indices(D, k)  # (n, k)

    directed = set()
    for i in range(n):
        for j in knn_idx[i]:
            directed.add((i, int(j)))

    if mode == "mutual":
        kept = {(i, j) for (i, j) in directed if (j, i) in directed}
    else:
        kept = set(directed) | {(j, i) for (i, j) in directed}

    if not kept:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    src = np.array([i for i, _ in kept], dtype=np.int64)
    dst = np.array([j for _, j in kept], dtype=np.int64)
    dist = D[src, dst].astype(np.float32)
    weight = weight_fn(dist).astype(np.float32)

    edge_index = np.stack([src, dst], axis=0)
    return edge_index, weight
