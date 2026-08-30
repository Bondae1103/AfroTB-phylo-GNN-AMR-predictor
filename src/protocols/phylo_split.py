"""Phylogeny-aware train/val/test split.

Why the existing split is not enough
------------------------------------
splits.csv is a per-sample stratified split. README Sec. 12 already records the
consequence: 98 val/test isolates have a train isolate at Jaccard distance
<= 0.01 (>= 99% shared calls). In a transductive full-graph GNN with a union
k-NN graph, those near-identical train isolates are also direct graph
neighbours of the test nodes. A test score earned that way partly measures
memorisation of a near-duplicate, not generalisation to a new isolate.

What this module does
---------------------
Builds a SECOND split in which whole phylogenetic clusters, not individual
isolates, are assigned to train/val/test. Every member of a cluster lands in
the same split, so no test isolate has a near-identical train counterpart.

This does NOT modify splits.csv. Project rules keep that file fixed, and the
comparison is the point: the same models are reported under both splits, and
the gap between them is itself a result (it quantifies how much of the
headline number was phylogenetic leakage).

Clustering: single-linkage connected components of the graph "isolates closer
than `threshold` Jaccard distance". Single linkage is the right choice here --
it is exactly the relation "no isolate in cluster A is near-identical to any
isolate in cluster B", which is the property the split needs to guarantee.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cluster_by_distance(D, threshold=0.05):
    """Single-linkage clusters of isolates within `threshold` Jaccard distance.

    D: dense (N, N) distance matrix. Returns an (N,) array of cluster labels.
    """
    n = D.shape[0]
    close = D <= threshold
    np.fill_diagonal(close, False)
    src, dst = np.nonzero(close)
    adj = coo_matrix(
        (np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n)
    )
    n_clusters, labels = connected_components(adj, directed=False)
    return labels, n_clusters


def grouped_split(labels, y_strat=None, frac=(0.70, 0.15, 0.15), seed=42):
    """Assign whole clusters to train/val/test, targeting the given fractions.

    Clusters are shuffled and then greedily placed into whichever split is
    furthest below its target share, largest cluster first. Greedy largest-first
    matters: a few very large clusters otherwise blow past a split's quota and
    leave the fractions badly skewed.

    y_strat, if given, is only used to report the resulting class balance --
    it never influences the assignment, since stratifying by label while
    grouping by cluster would reintroduce label-driven placement.
    """
    rng = np.random.default_rng(seed)
    uniq, sizes = np.unique(labels, return_counts=True)
    order = np.argsort(-sizes)
    # break size ties randomly so the split is not an artefact of label order
    tie = rng.permutation(uniq.size)
    order = order[np.argsort(tie[order], kind="stable")]
    order = order[np.argsort(-sizes[order], kind="stable")]

    n_total = labels.size
    targets = np.array(frac, dtype=float) * n_total
    current = np.zeros(3, dtype=float)
    names = np.array(["train", "val", "test"])
    assign = {}
    for o in order:
        deficit = targets - current
        k = int(np.argmax(deficit))
        assign[uniq[o]] = k
        current[k] += sizes[o]

    split = np.array([names[assign[c]] for c in labels], dtype=object)
    report = {
        "n_isolates": int(n_total),
        "n_clusters": int(uniq.size),
        "largest_cluster_size": int(sizes.max()),
        "n_singleton_clusters": int((sizes == 1).sum()),
        "counts": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "fractions": {s: round(float((split == s).mean()), 4) for s in ("train", "val", "test")},
    }
    if y_strat is not None:
        report["positive_rate_by_split"] = {
            s: [round(float(y_strat[split == s, j].mean()), 5) for j in range(y_strat.shape[1])]
            for s in ("train", "val", "test")
        }
    return split, report


def near_duplicate_count(D, split, threshold=0.01):
    """How many val/test isolates still have a train isolate within threshold.

    For a correctly grouped split with cluster threshold >= this threshold, the
    answer must be 0. Reported rather than asserted so it can be shown next to
    the original split's count.
    """
    train_idx = np.where(split == "train")[0]
    if train_idx.size == 0:
        return 0
    other = np.where(split != "train")[0]
    if other.size == 0:
        return 0
    return int((D[np.ix_(other, train_idx)].min(axis=1) <= threshold).sum())


def build_phylo_split(D, y=None, cluster_threshold=0.05, frac=(0.70, 0.15, 0.15), seed=42):
    """Convenience wrapper: cluster, split by cluster, verify no leakage remains."""
    labels, n_clusters = cluster_by_distance(D, threshold=cluster_threshold)
    split, report = grouped_split(labels, y_strat=y, frac=frac, seed=seed)
    report["cluster_threshold"] = cluster_threshold
    report["n_connected_clusters"] = int(n_clusters)
    report["residual_near_duplicates_at_0.01"] = near_duplicate_count(D, split, 0.01)
    report["residual_near_duplicates_at_threshold"] = near_duplicate_count(
        D, split, cluster_threshold
    )
    return split, labels, report
