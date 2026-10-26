"""Build a true phylogenetic network combining the clonal tree backbone (MST)
with localized transmission cluster edges, derived from VCF core SNPs.
"""

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
IN_GRAPH_PT = PROCESSED_DIR / "pyg_graph.pt"
OUT_GRAPH_PT = PROCESSED_DIR / "phylo_tree_graph.pt"


def main():
    print("Loading base k-NN graph...")
    g = torch.load(IN_GRAPH_PT, weights_only=False)
    sample_ids = list(g.sample_ids)
    n = len(sample_ids)

    src = g.edge_index[0].numpy()
    dst = g.edge_index[1].numpy()
    weights = g.edge_weight.numpy()
    # Distance is 1 - weight
    dists = np.clip(1.0 - weights, 1e-6, 1.0)

    print(f"Loaded {len(src)} directed edges across {n} isolates.")

    print("Computing clonal evolutionary tree backbone via Minimum Spanning Tree (MST)...")
    t0 = time.time()
    adj = coo_matrix((dists, (src, dst)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(adj).tocoo()
    print(f"Extracted clonal MST with {mst.nnz} tree edges in {time.time() - t0:.2f}s.")

    # Symmetrize MST edges
    mst_edges = set()
    mst_dists = {}
    for u, v, d in zip(mst.row, mst.col, mst.data):
        mst_edges.add((u, v))
        mst_edges.add((v, u))
        mst_dists[(u, v)] = d
        mst_dists[(v, u)] = d

    # Overlay close transmission edges (Jaccard distance <= 0.05 => >= 95% shared SNP calls)
    transmission_edges = set()
    trans_dists = {}
    for u, v, w, d in zip(src, dst, weights, dists):
        if d <= 0.05:
            transmission_edges.add((u, v))
            trans_dists[(u, v)] = d

    # Union of MST and transmission edges
    all_edges = mst_edges | transmission_edges
    print(f"Phylogenetic network: {len(mst_edges)} tree edges + {len(transmission_edges)} transmission edges -> {len(all_edges)} total directed edges.")

    edge_src = []
    edge_dst = []
    edge_w = []
    for u, v in all_edges:
        edge_src.append(u)
        edge_dst.append(v)
        d = mst_dists.get((u, v), trans_dists.get((u, v), 0.05))
        w = max(0.01, 1.0 - float(d))
        edge_w.append(w)

    phylo_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    phylo_edge_weight = torch.tensor(edge_w, dtype=torch.float32)

    phylo_graph = Data(
        edge_index=phylo_edge_index,
        edge_weight=phylo_edge_weight,
        sample_ids=sample_ids,
    )

    torch.save(phylo_graph, OUT_GRAPH_PT)
    print(f"Saved phylogenetic network to {OUT_GRAPH_PT}")


if __name__ == "__main__":
    main()
