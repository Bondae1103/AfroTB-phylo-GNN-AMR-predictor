"""Feature protocols, phylogeny-aware splitting, and lineage cleaning."""

import json

import numpy as np
import pytest
import scipy.sparse as sp

from src.audit.label_tautology import load_mutation_drug_map
from src.baselines.data_loader import load_aligned_data, split_masks
from src.protocols import features as F
from src.protocols.lineage import IGNORE_INDEX, build_y_lineage, normalise_label
from src.protocols.phylo_split import build_phylo_split, near_duplicate_count


# ---------------------------------------------------------------- features --
def test_leave_drug_out_removes_exactly_that_drugs_columns(replica):
    tab = load_aligned_data(replica["dir"])
    mapping = load_mutation_drug_map(replica["dir"])
    for drug in ("RIF", "INH", "PZA"):
        X, names = F.catalogue_features(tab["X"], tab["feature_names"], mapping, drug=drug)
        expected = sum(1 for n in tab["feature_names"] if mapping[n] == drug)
        assert X.shape[1] == tab["X"].shape[1] - expected
        assert all(mapping[n] != drug for n in names), "a masked column survived"


def test_leave_drug_out_actually_destroys_the_or_signal(replica):
    """After masking, no remaining column may equal the drug's label exactly.

    This is the property that makes LDO worth running: if some other column
    still encoded the answer, LDO would not have removed the tautology.
    """
    tab = load_aligned_data(replica["dir"])
    mapping = load_mutation_drug_map(replica["dir"])
    d_i = tab["drug_columns"].index("RIF")
    y = tab["y"][:, d_i]
    X, _ = F.catalogue_features(tab["X"], tab["feature_names"], mapping, drug="RIF")
    or_of_rest = X.max(axis=1)
    assert not np.array_equal(or_of_rest, y)


def test_genomewide_excludes_catalogue_sites(replica):
    tab = load_aligned_data(replica["dir"])
    masks = split_masks(tab)
    excluded = F.load_catalogue_site_indices(replica["dir"])
    assert excluded is not None and len(excluded) == 157

    X, names, info = F.genomewide_features(replica["dir"], masks["train"], max_sites=300)
    assert info["catalogue_sites_excluded"] is True
    assert "WARNING" not in info
    chosen = {int(n.split("_")[1]) for n in names}
    assert chosen.isdisjoint(set(int(e) for e in excluded))
    assert X.shape[0] == tab["X"].shape[0]


def test_site_selection_uses_train_rows_only(replica):
    """Changing val/test rows must not change which sites are selected."""
    tab = load_aligned_data(replica["dir"])
    masks = split_masks(tab)
    snp = sp.load_npz(replica["dir"] / "snp_matrix.npz").tolil()
    excl = F.load_catalogue_site_indices(replica["dir"])

    sites_a, _ = F.select_genomewide_sites(snp.tocsr(), masks["train"], excl, max_sites=200)
    # scramble every non-train row
    rng = np.random.default_rng(0)
    dense = np.asarray(snp.todense())
    non_train = ~masks["train"]
    dense[non_train] = rng.integers(0, 2, size=dense[non_train].shape)
    sites_b, _ = F.select_genomewide_sites(
        sp.csr_matrix(dense), masks["train"], excl, max_sites=200)
    assert np.array_equal(sites_a, sites_b)


# ------------------------------------------------------------------ splits --
def test_phylo_split_removes_near_duplicate_leakage(replica):
    tab = load_aligned_data(replica["dir"])
    D = np.load(replica["dir"] / "_test_D.npy") if (replica["dir"] / "_test_D.npy").exists() else None
    if D is None:
        from src.phylogeny.distance import compute_pairwise_jaccard_distance
        D = compute_pairwise_jaccard_distance(sp.load_npz(replica["dir"] / "snp_matrix.npz"))
        np.save(replica["dir"] / "_test_D.npy", D)

    original = split_masks(tab)
    orig_split = np.where(original["train"], "train",
                          np.where(original["val"], "val", "test"))
    baseline_leak = near_duplicate_count(D, orig_split, 0.01)

    split, labels, rep = build_phylo_split(D, y=tab["y"], cluster_threshold=0.05)
    assert rep["residual_near_duplicates_at_0.01"] == 0
    assert baseline_leak >= rep["residual_near_duplicates_at_0.01"]
    # every cluster lands wholly in one split
    for c in np.unique(labels):
        assert len(set(split[labels == c])) == 1


def test_phylo_split_hits_target_fractions(replica):
    from src.phylogeny.distance import compute_pairwise_jaccard_distance
    D = compute_pairwise_jaccard_distance(sp.load_npz(replica["dir"] / "snp_matrix.npz"))
    split, _, rep = build_phylo_split(D, cluster_threshold=0.05)
    for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(rep["fractions"][name] - target) < 0.08, rep["fractions"]


# ----------------------------------------------------------------- lineage --
def test_normalise_merges_spelling_variants_without_renaming_taxa():
    assert normalise_label("BOV-AFRI") == normalise_label("BOV_AFRI") == "BOV_AFRI"
    assert normalise_label("-") is None
    assert normalise_label("  Lineage 4 ") == "LINEAGE_4"
    # no taxonomic substitution happens
    assert "AFRICANUM" not in normalise_label("BOV_AFRI")


def test_lineage_missing_becomes_ignore_index_and_rows_are_kept(replica):
    tab = load_aligned_data(replica["dir"])
    y, classes, rep = build_y_lineage(replica["dir"])
    assert len(y) == len(tab["sample_ids"]), "row count must never change"
    assert rep["n_missing"] == int((y == IGNORE_INDEX).sum())
    assert rep["n_missing"] >= 1
    assert all(0 <= v < len(classes) for v in y[y != IGNORE_INDEX])
    assert "BOV_AFRI" in rep["raw_labels_merged_by_normalisation"]
