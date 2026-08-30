"""Pairwise distance from a sparse binary matrix: raw Hamming (SNP) count and
normalized Jaccard, both via the same block-matmul trick (no O(N^2 * M) loop):
    S[i,j] = (X @ X.T)[i,j]                      # shared confident-call count
    Hamming[i,j] = row_sum[i] + row_sum[j] - 2*S[i,j]
    Jaccard[i,j] = 1 - S[i,j] / (row_sum[i] + row_sum[j] - S[i,j])
Computed in row blocks so the full (N, N) result is the only thing ever held
dense in memory (float32), never (N, N, M) or similar.

Jaccard is the default for graph construction (see run_build_graph.py):
raw Hamming distance was found, empirically, to make isolates with an
unusually LOW total call count spuriously "close" to nearly everyone (their
distance to any other isolate is bounded above by their own tiny row_sum),
producing degree-thousands hub nodes with no real phylogenetic meaning.
Jaccard normalizes by each isolate's own call count and removes that
artifact -- an isolate with few calls is no longer automatically "cheap" to
be close to.
"""

import numpy as np


def compute_pairwise_distance(X, block_size=1000):
    """X: scipy.sparse binary matrix (N, M). Returns a dense (N, N) float32 array.

    D[i, j] is the number of core SNP sites at which isolates i and j differ
    (raw Hamming count -- kept for reference/comparison; NOT used by default
    for graph construction, see module docstring).
    """
    n = X.shape[0]
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    D = np.zeros((n, n), dtype=np.float32)

    # dtype MUST be widened before the dot product, not after: scipy sparse
    # matmul accumulates in the input dtype, so leaving X as uint8 silently
    # overflows/wraps past 255 shared calls (a real bug caught by
    # distance.py's own correctness spot-check during development -- see
    # git history / session log).
    X_csr = X.tocsr().astype(np.float32)
    X_T = X_csr.T.tocsr()
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = X_csr[start:end]
        S_block = block.dot(X_T).toarray().astype(np.float32)
        D[start:end, :] = row_sum[start:end, None] + row_sum[None, :] - 2 * S_block

    np.fill_diagonal(D, 0.0)
    return D


def compute_pairwise_jaccard_distance(X, block_size=1000):
    """X: scipy.sparse binary matrix (N, M). Returns a dense (N, N) float32 array.

    D[i,j] = 1 - |calls_i ∩ calls_j| / |calls_i ∪ calls_j|, in [0, 1].
    Pairs where BOTH isolates have zero core-site calls (union size 0) are an
    undefined edge case for Jaccard; by convention they're treated as
    identical (distance 0) rather than left as NaN -- this only ever affects
    isolates with row_sum == 0 against each other (a handful in this
    dataset), never a zero-call isolate against a normal one (which is
    correctly distance 1, i.e. maximally dissimilar).
    """
    n = X.shape[0]
    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    D = np.zeros((n, n), dtype=np.float32)

    # See compute_pairwise_distance above: dtype must be widened before the
    # dot product to avoid silent uint8 overflow in the matmul accumulator.
    X_csr = X.tocsr().astype(np.float64)
    X_T = X_csr.T.tocsr()
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = X_csr[start:end]
        S_block = block.dot(X_T).toarray().astype(np.float64)
        union = row_sum[start:end, None] + row_sum[None, :] - S_block
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union > 0, S_block / np.where(union > 0, union, 1.0), 1.0)
        D[start:end, :] = (1.0 - similarity).astype(np.float32)

    np.fill_diagonal(D, 0.0)
    return D
