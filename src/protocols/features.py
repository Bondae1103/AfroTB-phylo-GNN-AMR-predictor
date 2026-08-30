"""Feature protocols: what a model is allowed to see when predicting resistance.

src/audit/label_tautology.py establishes that y_amr is an exact Boolean OR over
features.csv columns. Under the default protocol a model therefore has the
answer in its input, and every reported score measures OR-gate reconstruction.
This module defines the alternatives that do not.

    CATALOGUE       all 157 curated mutation columns. The status quo.
                    Tautological -- kept only as the reference point that
                    every other protocol is compared against.

    CATALOGUE_LDO   leave-drug-out: when predicting drug d, drug d's own
                    mutation columns are removed. Drug d must then be inferred
                    from OTHER drugs' mutations (co-resistance, e.g. the MDR
                    RIF+INH pattern) and, for a graph model, from neighbouring
                    isolates. Requires one binary model per drug.

    GENOMEWIDE      genome-wide SNP sites from snp_matrix.npz, with the
                    catalogue sites EXCLUDED and the remainder selected on the
                    TRAIN SPLIT ONLY. Multi-label, one model for all 9 drugs,
                    and no drug's answer is present in its own input. This is
                    the protocol under which "does phylogeny help?" is a real
                    question rather than a rhetorical one.

Leakage rules enforced here
---------------------------
* Site selection statistics (call rate, minor-allele frequency) are computed
  on train rows only. Val/test rows never influence which columns exist.
* Catalogue-site exclusion is applied BEFORE selection, so a resistance site
  cannot be picked up by frequency filtering.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CATALOGUE = "catalogue"
CATALOGUE_LDO = "catalogue_ldo"
GENOMEWIDE = "genomewide"
ALL_PROTOCOLS = (CATALOGUE, CATALOGUE_LDO, GENOMEWIDE)


def catalogue_features(X, feature_names, mutation_to_drug, drug=None):
    """Catalogue protocol. If drug is given, apply leave-drug-out masking.

    Returns (X_out, kept_feature_names). Columns are DROPPED, not zeroed, so a
    model cannot recover the removed signal from a constant column and so the
    input dimensionality honestly reflects what the model may use.
    """
    if drug is None:
        return X, list(feature_names)
    keep = [j for j, name in enumerate(feature_names)
            if mutation_to_drug.get(name) != drug]
    return X[:, keep], [feature_names[j] for j in keep]


def load_catalogue_site_indices(processed_dir):
    """snp_matrix.npz column indices occupied by catalogue mutations, or None.

    Returning None means the mapping is unavailable. Callers must then treat a
    genome-wide run as NOT catalogue-free and say so, rather than quietly
    assuming the resistance sites happened to be absent.
    """
    path = Path(processed_dir) / "catalogue_sites.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f)["site_indices"], dtype=np.int64)


def select_genomewide_sites(
    snp,
    train_mask,
    exclude_sites=None,
    max_sites=4000,
    min_minor_freq=0.01,
    strategy="random",
    seed=0,
):
    """Choose informative genome-wide SNP columns using TRAIN ROWS ONLY.

    snp: scipy sparse (N, n_sites) binary. Returns (site_indices, info).

    Both strategies first drop the excluded (catalogue) sites and then keep
    only sites whose TRAIN-split minor-allele frequency is at least
    min_minor_freq. They differ in how they cut down to max_sites:

    strategy="random" (default)
        Uniform random sample of the eligible sites.

    strategy="variance"
        The max_sites most variable eligible sites, ranked by p*(1-p).

    Why random is the default, despite "most variable" sounding better:
    variance of a binary site is maximised at frequency 0.5, so ranking by it
    systematically selects the sites that split the population most evenly --
    which are the DEEPEST splits, i.e. lineage-level markers. Markers of a
    small sublineage sit near frequency 1/n_clades and score near the bottom,
    so they get discarded wholesale. Drug resistance in M. tuberculosis is
    largely clonal at the sublineage level, so variance ranking throws away
    precisely the resolution the task needs. This was not a theoretical
    worry: under variance ranking every model scored ~0.31 core-6 F1 on the
    synthetic fixture, far below what its own generative structure allows.
    Uniform sampling keeps sites in proportion to how many exist at each
    depth, so fine-grained structure survives.
    """
    if strategy not in ("random", "variance"):
        raise ValueError("strategy must be 'random' or 'variance', got %r" % strategy)

    snp = snp.tocsr()
    n_sites = snp.shape[1]
    train_rows = np.where(train_mask)[0]
    if train_rows.size == 0:
        raise ValueError("train_mask selects no rows")

    freq = np.asarray(snp[train_rows].sum(axis=0)).ravel().astype(np.float64)
    freq /= train_rows.size

    eligible = np.ones(n_sites, dtype=bool)
    n_excluded = 0
    if exclude_sites is not None and len(exclude_sites):
        eligible[np.asarray(exclude_sites, dtype=np.int64)] = False
        n_excluded = int(len(exclude_sites))

    minor = np.minimum(freq, 1.0 - freq)
    eligible &= minor >= min_minor_freq

    eligible_idx = np.where(eligible)[0]
    k = min(max_sites, eligible_idx.size)
    if strategy == "variance":
        variance = freq[eligible_idx] * (1.0 - freq[eligible_idx])
        chosen = np.sort(eligible_idx[np.argsort(-variance)[:k]])
    else:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(eligible_idx, size=k, replace=False))

    info = {
        "n_sites_total": int(n_sites),
        "n_sites_excluded_as_catalogue": n_excluded,
        "n_sites_passing_maf": int(eligible_idx.size),
        "n_sites_selected": int(chosen.size),
        "min_minor_freq": min_minor_freq,
        "max_sites": max_sites,
        "strategy": strategy,
        "selection_seed": seed,
        "selection_fitted_on": "train split only",
    }
    return chosen, info


def genomewide_features(processed_dir, train_mask, max_sites=4000,
                        min_minor_freq=0.01, strategy="random", seed=0):
    """Build the genome-wide feature matrix. Returns (X, feature_names, info)."""
    processed_dir = Path(processed_dir)
    snp = sp.load_npz(processed_dir / "snp_matrix.npz")
    exclude = load_catalogue_site_indices(processed_dir)

    sites, info = select_genomewide_sites(
        snp, train_mask, exclude_sites=exclude,
        max_sites=max_sites, min_minor_freq=min_minor_freq,
        strategy=strategy, seed=seed,
    )
    X = np.asarray(snp[:, sites].todense(), dtype=np.float32)
    info["catalogue_sites_excluded"] = exclude is not None
    if exclude is None:
        info["WARNING"] = (
            "catalogue_sites.json absent -- resistance-catalogue sites could NOT "
            "be excluded from the genome-wide features. Results under this "
            "protocol are not catalogue-free and must not be described as such."
        )
    names = ["site_%d" % s for s in sites]
    return X, names, info
