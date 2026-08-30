"""Build the core SNP matrix, pairwise distance, and k-NN graph for the GNN.

Strictly reuses data/processed/sample_ids.csv for canonical isolate order and
data/raw/Afro_TB/AFRO_TB_VCF/ for genomic data (via src/phylogeny/vcf_paths.py,
which asserts the prior 1:1 mapping audit still holds). Never reads
features.csv, y_amr.csv, labels.csv's Drug/Lineage columns, or the split
column of splits.csv for anything that affects the graph itself -- edges are
purely a function of genomic SNP distance. splits.csv is read only AFTER the
graph is built, solely to compute an informational cross-split near-duplicate
diagnostic; it is never modified.

Accepts a sample_id_subset so the exact same code path runs at every scale
(tiny / ~100 / full 13,753) per CLAUDE.md's staged-rollout rule -- there is no
separate "test version" of this pipeline.

Writes:
  data/processed/snp_matrix.npz
  data/processed/snp_matrix_sites.csv
  data/processed/pyg_graph.pt
  data/processed/graph_construction_report.json
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .distance import compute_pairwise_jaccard_distance
from .knn_graph import build_knn_edges
from .snp_matrix import build_site_index, build_sparse_matrix
from .vcf_paths import assert_mapping_is_exact_1to1

SAMPLE_IDS_CSV = Path("data/processed/sample_ids.csv")
SPLITS_CSV = Path("data/processed/splits.csv")

SNP_MATRIX_NPZ = Path("data/processed/snp_matrix.npz")
SNP_MATRIX_SITES_CSV = Path("data/processed/snp_matrix_sites.csv")
PYG_GRAPH_PT = Path("data/processed/pyg_graph.pt")
GRAPH_REPORT_JSON = Path("data/processed/graph_construction_report.json")

# Jaccard distance is the validated default (see distance.py module docstring):
# raw Hamming distance produced degree-thousands hub artifacts from low-call-count
# isolates at full (13,753-isolate) scale. Union mode is the validated default too:
# strict mutual-kNN left 98.4% of nodes isolated at full scale (most isolates have
# no *reciprocal* top-k match even though they have a clear best candidate).
K_NEIGHBORS = 20
KNN_MODE = "union"
JACCARD_EDGE_WEIGHT_FN = lambda d: 1.0 - d
NEAR_DUPLICATE_THRESHOLD = 0.01  # Jaccard distance <=0.01 => >=99% shared calls


def read_canonical_ids():
    with open(SAMPLE_IDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


def read_split_by_id():
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["Name"]: row["split"] for row in reader}


def cross_split_near_duplicate_diagnostic(D, sample_ids, threshold=NEAR_DUPLICATE_THRESHOLD):
    """Informational only: val/test isolates with a train neighbor at distance <= threshold.

    Reads splits.csv for this diagnostic only, after D is already computed
    from genomic distance alone -- this cannot leak into edge_index/edge_weight.
    """
    split_by_id = read_split_by_id()
    split = np.array([split_by_id[sid] for sid in sample_ids])
    train_mask = split == "train"
    train_idx = np.where(train_mask)[0]

    flagged = []
    for i, sid in enumerate(sample_ids):
        if split[i] == "train" or train_idx.size == 0:
            continue
        d_to_train = D[i, train_idx]
        min_d = float(d_to_train.min())
        if min_d <= threshold:
            nearest_train_id = sample_ids[int(train_idx[np.argmin(d_to_train)])]
            flagged.append(
                {
                    "sample_id": sid,
                    "split": split[i],
                    "nearest_train_id": nearest_train_id,
                    "distance": min_d,
                }
            )
    return {
        "threshold": threshold,
        "n_flagged": len(flagged),
        "examples": flagged[:50],
    }


def degree_stats(edge_index, n):
    degree = np.zeros(n, dtype=np.int64)
    if edge_index.shape[1] > 0:
        np.add.at(degree, edge_index[0], 1)
    return {
        "min": int(degree.min()) if n else 0,
        "median": float(np.median(degree)) if n else 0,
        "mean": float(degree.mean()) if n else 0,
        "max": int(degree.max()) if n else 0,
        "n_isolated_nodes": int((degree == 0).sum()),
    }


def n_connected_components(edge_index, n):
    adj = [[] for _ in range(n)]
    for a, b in edge_index.T:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return n_components


def save_pyg_graph(edge_index, edge_weight, sample_ids, path):
    import torch

    graph_obj = {
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
        "edge_weight": torch.as_tensor(edge_weight, dtype=torch.float32),
        "num_nodes": len(sample_ids),
        "sample_ids": sample_ids,
    }
    try:
        from torch_geometric.data import Data

        data = Data(
            edge_index=graph_obj["edge_index"],
            edge_weight=graph_obj["edge_weight"],
            num_nodes=graph_obj["num_nodes"],
        )
        data.sample_ids = sample_ids
        torch.save(data, path)
    except ImportError:
        torch.save(graph_obj, path)


def run(sample_id_subset=None, k=None, mode=None):
    k = K_NEIGHBORS if k is None else k
    mode = KNN_MODE if mode is None else mode
    assert_mapping_is_exact_1to1()

    sample_ids = read_canonical_ids()
    if sample_id_subset is not None:
        subset_set = set(sample_id_subset)
        sample_ids = [sid for sid in sample_ids if sid in subset_set]
        if len(sample_ids) != len(sample_id_subset):
            raise ValueError("sample_id_subset contains IDs not in sample_ids.csv")

    timings = {}

    t0 = time.time()
    site_index = build_site_index(sample_ids)
    timings["pass1_site_index_seconds"] = time.time() - t0

    t0 = time.time()
    X = build_sparse_matrix(sample_ids, site_index["core_sites"])
    timings["pass2_sparse_matrix_seconds"] = time.time() - t0

    t0 = time.time()
    D = compute_pairwise_jaccard_distance(X)
    timings["pairwise_distance_seconds"] = time.time() - t0

    t0 = time.time()
    edge_index, edge_weight = build_knn_edges(D, k=k, mode=mode, weight_fn=JACCARD_EDGE_WEIGHT_FN)
    timings["knn_graph_seconds"] = time.time() - t0

    SNP_MATRIX_NPZ.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(SNP_MATRIX_NPZ, X)

    with open(SNP_MATRIX_SITES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pos", "alt", "global_call_count", "missing_rate", "kept_in_core"])
        core_set = set(site_index["core_sites"])
        for site in site_index["all_sites"]:
            pos, alt = site
            writer.writerow(
                [
                    pos,
                    alt,
                    site_index["call_count"][site],
                    round(site_index["site_missing_rate"][site], 6),
                    site in core_set,
                ]
            )

    save_pyg_graph(edge_index, edge_weight, sample_ids, PYG_GRAPH_PT)

    near_dup = cross_split_near_duplicate_diagnostic(D, sample_ids)

    report = {
        "n_isolates": len(sample_ids),
        "is_subset_run": sample_id_subset is not None,
        "n_all_sites": len(site_index["all_sites"]),
        "n_core_sites": len(site_index["core_sites"]),
        "max_site_missing_rate": site_index["max_site_missing_rate"],
        "knn_k": k,
        "knn_mode": mode,
        "n_edges": int(edge_index.shape[1]),
        "degree_stats": degree_stats(edge_index, len(sample_ids)),
        "n_connected_components": n_connected_components(edge_index, len(sample_ids)),
        "cross_split_near_duplicate_diagnostic": near_dup,
        "timings_seconds": timings,
        "outputs": {
            "snp_matrix": SNP_MATRIX_NPZ.as_posix(),
            "snp_matrix_sites": SNP_MATRIX_SITES_CSV.as_posix(),
            "pyg_graph": PYG_GRAPH_PT.as_posix(),
        },
    }
    with open(GRAPH_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "cross_split_near_duplicate_diagnostic"}, indent=2))
    print(f"cross_split_near_duplicate_diagnostic.n_flagged: {near_dup['n_flagged']}")
    print(f"\nWrote {GRAPH_REPORT_JSON}")
    return report


def main():
    run()


if __name__ == "__main__":
    main()
