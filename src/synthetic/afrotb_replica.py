"""Synthetic Afro-TB replica -- a TEST FIXTURE, never a data source.

=====================================================================
THIS MODULE GENERATES SIMULATED DATA. Numbers produced from it are
NEVER Afro-TB results and must NEVER be reported as biological
findings, dataset statistics, or resistance associations (CLAUDE.md's
"never invent" rule). Its outputs go to a caller-supplied directory
that is expected to be OUTSIDE data/processed/.
=====================================================================

Why it exists
-------------
The real Afro-TB artifacts (data/raw/, data/processed/) are gitignored
and absent from a fresh checkout, so nothing downstream of them can be
run or tested. This module emits a dataset with the SAME SCHEMA and the
SAME DOCUMENTED GENERATIVE STRUCTURE as the real one, so that every
loader, split, graph builder, baseline, and GNN in this repo can be
exercised end to end without the real data present.

Structure replicated (all sourced from README.md / the preprocessing
scripts, not invented):
  * N isolates x 157 binary mutation columns          (prepare_afrotb_matrix.py)
  * each mutation cell carries ONE of 9 drug codes    (create_y_amr.py)
  * y_amr[i, d] = 1 iff any of isolate i's mutation
    cells carried drug code d  -- i.e. an exact OR    (create_y_amr.py:build_y_amr)
  * labels.csv = Name, Country, Lineage, Drug         (prepare_afrotb_matrix.py)
  * Lineage carries the documented defects: a
    BOV_AFRI / BOV-AFRI spelling split and a handful
    of "-" (missing) values                           (README Sec. 8)
  * a genome-wide binary SNP matrix whose Jaccard
    k-NN graph has the clustered block structure of
    a real phylogeny                                  (README Sec. 12)

Biological realism that is deliberately built in
------------------------------------------------
Resistance is simulated as CLADE-CORRELATED, not independent per
isolate: resistant genotypes are concentrated in a minority of
subclades, and RIF/INH co-occur (the MDR pattern). This is the property
that makes a phylogeny-aware model worth testing at all -- it is
standard, published M. tuberculosis epidemiology (resistant strains are
largely clonal), not a result tuned to make any particular model win.
The generator is NOT tuned toward a desired experimental outcome;
whichever model wins on this fixture, wins.
"""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DRUG_CODES = ["RIF", "INH", "EMB", "PZA", "STM", "LEV", "CAP", "ETH", "LZD"]

# How the 157 mutation columns are apportioned across the 9 drug codes.
# The ordering/relative size mirrors the real catalogue's shape (many rpoB and
# katG entries, few for the rare drugs); the exact counts are a fixture choice.
MUTATIONS_PER_DRUG = {
    "RIF": 34, "INH": 28, "EMB": 24, "PZA": 26, "STM": 18,
    "LEV": 14, "CAP": 5, "ETH": 5, "LZD": 3,
}
assert sum(MUTATIONS_PER_DRUG.values()) == 157

# Gene symbols per drug, used only to build human-readable mutation NAMES
# (e.g. "rpoB S450L") so the fixture looks like the real catalogue.
GENES_PER_DRUG = {
    "RIF": ["rpoB"], "INH": ["katG", "inhA", "fabG1"], "EMB": ["embB", "embA"],
    "PZA": ["pncA"], "STM": ["rpsL", "rrs", "gid"], "LEV": ["gyrA", "gyrB"],
    "CAP": ["tlyA"], "ETH": ["ethA"], "LZD": ["rplC"],
}

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Lineage names, including the documented BOV_AFRI / BOV-AFRI spelling split.
LINEAGE_NAMES = [
    "Lineage1", "Lineage2", "Lineage3", "Lineage4",
    "Lineage5", "Lineage6", "BOV_AFRI", "BOV-AFRI",
]
# Population share per lineage. A fixture choice, not a dataset statistic.
LINEAGE_WEIGHTS = np.array([0.10, 0.18, 0.12, 0.42, 0.05, 0.06, 0.05, 0.02])

COUNTRIES = [
    "Morocco", "Algeria", "Tunisia", "Nigeria", "Ghana", "Uganda",
    "Tanzania", "SouthAfrica", "Ethiopia", "Mali",
]

DRUG_CATEGORY_SENSITIVE = "Sensitive"

# Resistance-simulation parameters, calibrated (see generate()) so per-drug
# prevalence approximates the real Afro-TB rates. Order matches DRUG_CODES.
MDR_CLADE_FRACTION = 0.37
MDR_COUPLING = 0.35
BASE_RATE = np.array([0.041668, 0.039172, 0.027479, 0.018705, 0.013325,
                      0.010432, 0.004571, 0.000298, 0.000251])
MDR_RATE = np.array([0.709926, 0.387845, 0.494150, 0.304061, 0.213773,
                     0.151770, 0.002105, 0.000550, 0.000693])

# Private sites separating two members of the same transmission cluster. Small
# enough that their Jaccard distance falls under the 0.01 near-duplicate
# threshold README Sec. 12 uses.
CLONE_PRIVATE_SITES = 4


def _build_mutation_catalogue(rng):
    """157 (name, drug) pairs: each mutation column belongs to exactly one drug.

    This one-drug-per-column property is what makes y_amr an exact Boolean OR
    over feature columns. audit_label_tautology.py VERIFIES that property in
    the real workbook rather than assuming it.
    """
    catalogue = []
    for drug in DRUG_CODES:
        genes = GENES_PER_DRUG[drug]
        used = set()
        for _ in range(MUTATIONS_PER_DRUG[drug]):
            while True:
                gene = genes[rng.integers(len(genes))]
                ref = AMINO_ACIDS[rng.integers(len(AMINO_ACIDS))]
                pos = int(rng.integers(1, 600))
                alt = AMINO_ACIDS[rng.integers(len(AMINO_ACIDS))]
                name = gene + " " + ref + str(pos) + alt
                if alt != ref and name not in used:
                    used.add(name)
                    break
            catalogue.append((name, drug))
    return catalogue


def generate(
    out_dir,
    n_isolates=6000,
    n_background_sites=20000,
    n_subclades_per_lineage=6,
    transmission_cluster_frac=0.035,
    seed=42,
):
    """Write a full synthetic Afro-TB-shaped dataset into out_dir.

    Emits exactly the filenames the real pipeline emits, so every existing
    loader (src/baselines/data_loader.py, src/gnn/data.py) reads it unchanged:
      sample_ids.csv, features.csv, y_amr.csv, labels.csv, splits.csv,
      snp_matrix.npz, mutation_drug_map.json, synthetic_manifest.json
    pyg_graph.pt is NOT written here -- it is built from snp_matrix.npz by the
    real graph code, so that code path stays exercised rather than bypassed.

    Returns a manifest dict describing what was generated.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = _build_mutation_catalogue(rng)
    mutation_names = [name for name, _ in catalogue]
    mutation_drug = [drug for _, drug in catalogue]
    n_mutations = len(catalogue)

    sample_ids = ["SYN" + str(i).zfill(6) for i in range(n_isolates)]

    # ---- phylogeny: lineage -> subclade -> isolate -------------------------
    lineage_idx = rng.choice(len(LINEAGE_NAMES), size=n_isolates, p=LINEAGE_WEIGHTS)
    subclade_idx = rng.integers(n_subclades_per_lineage, size=n_isolates)
    clade_id = lineage_idx * n_subclades_per_lineage + subclade_idx
    n_clades = len(LINEAGE_NAMES) * n_subclades_per_lineage

    # Transmission clusters: small groups of near-clonal isolates (recent
    # transmission chains). These are what produce near-duplicate PAIRS, and
    # they are the reason README Sec. 12's cross-split diagnostic finds 98
    # val/test isolates with a >=99%-identical train neighbour. Without them
    # the fixture cannot exhibit -- or test a fix for -- phylogenetic leakage.
    # transmission_group[i] == -1 means the isolate is not in a cluster.
    transmission_group = np.full(n_isolates, -1, dtype=np.int64)
    n_in_clusters = int(transmission_cluster_frac * n_isolates)
    if n_in_clusters >= 2:
        members = rng.choice(n_isolates, size=n_in_clusters, replace=False)
        g, pos = 0, 0
        while pos < members.size - 1:
            size = min(int(rng.integers(2, 6)), members.size - pos)
            grp = members[pos:pos + size]
            # a transmission cluster is inside one clade, so give every member
            # the clade of the first member
            clade_id[grp] = clade_id[grp[0]]
            lineage_idx[grp] = lineage_idx[grp[0]]
            transmission_group[grp] = g
            pos += size
            g += 1

    # Marker-site sets. A site index is shared by every member of the clade
    # that owns it, which is what produces phylogenetic block structure in the
    # Jaccard distance matrix.
    per_lineage_markers = 400
    per_clade_markers = 250
    per_isolate_private = 150
    shared_core = 300

    cursor = 0
    core_sites = np.arange(cursor, cursor + shared_core)
    cursor += shared_core
    lineage_sites = []
    for _ in range(len(LINEAGE_NAMES)):
        lineage_sites.append(np.arange(cursor, cursor + per_lineage_markers))
        cursor += per_lineage_markers
    clade_sites = []
    for _ in range(n_clades):
        clade_sites.append(np.arange(cursor, cursor + per_clade_markers))
        cursor += per_clade_markers
    # Each of the 157 catalogue mutations gets its own dedicated genomic site,
    # so genome-wide SNP features genuinely contain the resistance signal.
    mutation_sites = np.arange(cursor, cursor + n_mutations)
    cursor += n_mutations
    background_start = cursor
    n_sites = background_start + n_background_sites

    # ---- resistance: clade-correlated, with an MDR (RIF+INH) pattern -------
    # MDR_CLADE_FRACTION / BASE_RATE / MDR_RATE / MDR_COUPLING below were
    # solved so the resulting per-drug prevalence lands near the REAL Afro-TB
    # rates implied by results/baseline_metrics.json's test-split positive
    # counts (RIF .380, INH .304, EMB .266, PZA .170, STM .120, LEV .088, and
    # near-zero CAP/ETH/LZD). Matching prevalence matters because class
    # imbalance is one of the things the evaluation protocol has to survive.
    clade_is_mdr = rng.random(n_clades) < MDR_CLADE_FRACTION
    clade_drug_rate = np.where(
        clade_is_mdr[:, None],
        MDR_RATE[None, :] * rng.uniform(0.75, 1.15, size=(n_clades, 9)),
        BASE_RATE[None, :] * rng.uniform(0.5, 1.6, size=(n_clades, 9)),
    ).clip(0.0, 0.98)

    resistant = rng.random((n_isolates, 9)) < clade_drug_rate[clade_id]
    # MDR coupling: RIF-resistant isolates are more likely INH-resistant too.
    resistant[:, 1] |= resistant[:, 0] & (rng.random(n_isolates) < MDR_COUPLING)

    # Transmission-cluster members are clonal: they inherit one resistance
    # profile. This is what makes a near-duplicate in the training set so
    # valuable to memorise -- it carries the test isolate's exact label.
    for g in np.unique(transmission_group[transmission_group >= 0]):
        grp = np.where(transmission_group == g)[0]
        resistant[grp] = resistant[grp[0]]

    # ---- features: activate 1-3 of the drug's own mutation columns ---------
    drug_to_cols = {d: [] for d in DRUG_CODES}
    for col, drug in enumerate(mutation_drug):
        drug_to_cols[drug].append(col)

    features = np.zeros((n_isolates, n_mutations), dtype=np.int8)
    for d_i, drug in enumerate(DRUG_CODES):
        cols = np.array(drug_to_cols[drug])
        # Mutation choice within a drug is clade-biased too: a clone tends to
        # carry the SAME resistance mutation, not a fresh random one each time.
        clade_pref = rng.integers(len(cols), size=n_clades)
        for r in np.where(resistant[:, d_i])[0]:
            n_hits = 1 + int(rng.random() < 0.18) + int(rng.random() < 0.04)
            if rng.random() < 0.80:
                chosen = [cols[clade_pref[clade_id[r]]]]
                if n_hits > 1:
                    chosen += list(rng.choice(cols, size=n_hits - 1, replace=False))
            else:
                chosen = list(rng.choice(cols, size=n_hits, replace=False))
            features[r, np.asarray(chosen, dtype=int)] = 1

    # ---- y_amr: EXACTLY the real derivation (OR over the drug's columns) ---
    y_amr = np.zeros((n_isolates, 9), dtype=np.int8)
    for d_i, drug in enumerate(DRUG_CODES):
        y_amr[:, d_i] = features[:, drug_to_cols[drug]].max(axis=1)

    # ---- genome-wide SNP matrix -------------------------------------------
    def draw_backbone(i):
        """The clade-determined part of isolate i's genome."""
        return np.concatenate([
            core_sites[rng.random(shared_core) < 0.97],
            lineage_sites[lineage_idx[i]][rng.random(per_lineage_markers) < 0.93],
            clade_sites[clade_id[i]][rng.random(per_clade_markers) < 0.90],
            rng.integers(background_start, n_sites, size=per_isolate_private),
        ])

    # Members of a transmission cluster share ONE backbone and differ only by a
    # handful of private sites, giving them Jaccard distances near zero.
    group_backbone = {}
    for g in np.unique(transmission_group[transmission_group >= 0]):
        grp = np.where(transmission_group == g)[0]
        group_backbone[int(g)] = draw_backbone(grp[0])

    rows_idx, cols_idx = [], []
    for i in range(n_isolates):
        g = int(transmission_group[i])
        if g >= 0:
            present = [
                group_backbone[g],
                rng.integers(background_start, n_sites, size=CLONE_PRIVATE_SITES),
            ]
        else:
            present = [draw_backbone(i)]
        mut_hits = mutation_sites[features[i] == 1]
        if mut_hits.size:
            present.append(mut_hits)
        idx = np.unique(np.concatenate(present))
        rows_idx.append(np.full(idx.size, i, dtype=np.int32))
        cols_idx.append(idx.astype(np.int32))

    rows_cat = np.concatenate(rows_idx)
    cols_cat = np.concatenate(cols_idx)
    snp = sp.csr_matrix(
        (np.ones(rows_cat.size, dtype=np.uint8), (rows_cat, cols_cat)),
        shape=(n_isolates, n_sites),
        dtype=np.uint8,
    )

    # ---- labels.csv, with the documented Lineage defects -------------------
    lineage_col = [LINEAGE_NAMES[k] for k in lineage_idx]
    n_missing_lineage = max(1, round(10 * n_isolates / 13753))
    for p in rng.choice(n_isolates, size=n_missing_lineage, replace=False):
        lineage_col[p] = "-"

    country_col = [COUNTRIES[k] for k in rng.integers(len(COUNTRIES), size=n_isolates)]
    n_res = y_amr.sum(axis=1)
    drug_category = []
    for i in range(n_isolates):
        if n_res[i] == 0:
            drug_category.append(DRUG_CATEGORY_SENSITIVE)
        elif y_amr[i, 0] and y_amr[i, 1] and (y_amr[i, 5] or y_amr[i, 6]):
            drug_category.append("Pre-XDR")
        elif y_amr[i, 0] and y_amr[i, 1]:
            drug_category.append("MDR")
        elif n_res[i] == 1:
            drug_category.append("Mono")
        else:
            drug_category.append("Other")

    # ---- stratified 70/15/15 split on the Drug category --------------------
    split = np.array(["train"] * n_isolates, dtype=object)
    cat_arr = np.array(drug_category)
    split_rng = np.random.default_rng(seed)
    for cat in np.unique(cat_arr):
        idx = np.where(cat_arr == cat)[0]
        split_rng.shuffle(idx)
        n_val = int(round(0.15 * idx.size))
        n_test = int(round(0.15 * idx.size))
        split[idx[:n_val]] = "val"
        split[idx[n_val:n_val + n_test]] = "test"
        split[idx[n_val + n_test:]] = "train"

    # ---- write everything --------------------------------------------------
    def _write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    _write_csv(out_dir / "sample_ids.csv", ["Name"], [[s] for s in sample_ids])
    _write_csv(out_dir / "features.csv", ["Name"] + mutation_names,
               [[sample_ids[i]] + features[i].tolist() for i in range(n_isolates)])
    _write_csv(out_dir / "y_amr.csv", ["Name"] + DRUG_CODES,
               [[sample_ids[i]] + y_amr[i].tolist() for i in range(n_isolates)])
    _write_csv(out_dir / "labels.csv", ["Name", "Country", "Lineage", "Drug"],
               [[sample_ids[i], country_col[i], lineage_col[i], drug_category[i]]
                for i in range(n_isolates)])
    _write_csv(out_dir / "splits.csv", ["Name", "split"],
               [[sample_ids[i], split[i]] for i in range(n_isolates)])
    sp.save_npz(out_dir / "snp_matrix.npz", snp)

    with open(out_dir / "mutation_drug_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "SYNTHETIC FIXTURE -- not the Afro-TB catalogue",
                "drug_columns": DRUG_CODES,
                "mutation_to_drug": dict(zip(mutation_names, mutation_drug)),
            },
            f, indent=2,
        )

    # Which snp_matrix.npz columns carry the 157 catalogue mutations. Needed by
    # the genome-wide feature protocol so those sites can be EXCLUDED -- keeping
    # them would smuggle the same tautology back in through the genome.
    with open(out_dir / "catalogue_sites.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "SYNTHETIC FIXTURE -- not the Afro-TB catalogue",
                "description": "snp_matrix.npz column index of each catalogue mutation",
                "mutation_to_site": {
                    name: int(site) for name, site in zip(mutation_names, mutation_sites)
                },
                "site_indices": [int(s) for s in mutation_sites],
            },
            f, indent=2,
        )

    # Ground-truth clade / transmission-cluster membership. Exists ONLY for the
    # fixture, and only so an oracle upper bound can be computed: resistance is
    # generated at the clade level with a per-clade probability, so no model can
    # exceed the F1 of "predict each clade's own base rate". Without that bound
    # there is no way to tell a weak model from a hard problem.
    _write_csv(out_dir / "true_clade.csv", ["Name", "lineage_idx", "clade_id",
                                            "transmission_group"],
               [[sample_ids[i], int(lineage_idx[i]), int(clade_id[i]),
                 int(transmission_group[i])] for i in range(n_isolates)])

    manifest = {
        "WARNING": "SYNTHETIC DATA. Not Afro-TB. Never report as a biological result.",
        "generator": "src/synthetic/afrotb_replica.py",
        "seed": seed,
        "n_isolates": n_isolates,
        "n_mutations": n_mutations,
        "n_snp_sites": int(n_sites),
        "n_snp_nonzero": int(snp.nnz),
        "drug_columns": DRUG_CODES,
        "drug_positive_counts": {d: int(y_amr[:, i].sum()) for i, d in enumerate(DRUG_CODES)},
        "drug_prevalence": {d: round(float(y_amr[:, i].mean()), 5) for i, d in enumerate(DRUG_CODES)},
        "split_counts": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "lineage_counts": {n: int(sum(1 for v in lineage_col if v == n))
                           for n in LINEAGE_NAMES + ["-"]},
        "n_clades": int(n_clades),
        "n_mdr_clades": int(clade_is_mdr.sum()),
    }
    with open(out_dir / "synthetic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
