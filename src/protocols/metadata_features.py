"""Phylogenetic and phylogeographic metadata feature loaders.

Extracts Lineage (8 clades) and Country (26 African nations) from labels.csv,
strictly aligned to sample_ids.csv canonical order.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .lineage import normalise_label

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_metadata_features(processed_dir=PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    with open(processed_dir / "sample_ids.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        canonical_ids = [row[0] for row in r]

    with open(processed_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if [r["Name"] for r in rows] != canonical_ids:
        raise ValueError("labels.csv row order does not match sample_ids.csv")

    # Clean Lineage
    norm_lineages = [normalise_label(r["Lineage"]) for r in rows]
    unique_lineages = sorted({v for v in norm_lineages if v is not None})
    lineage_to_idx = {lin: i + 1 for i, lin in enumerate(unique_lineages)}  # 0 reserved for unknown
    lineage_indices = np.array([lineage_to_idx.get(v, 0) for v in norm_lineages], dtype=np.int64)

    # Clean Country
    raw_countries = [r["Country"].strip() if r["Country"] else None for r in rows]
    unique_countries = sorted({c for c in raw_countries if c is not None and c not in ("-", "")})
    country_to_idx = {c: i + 1 for i, c in enumerate(unique_countries)}  # 0 reserved for unknown
    country_indices = np.array([country_to_idx.get(c, 0) for c in raw_countries], dtype=np.int64)

    info = {
        "n_isolates": len(canonical_ids),
        "n_lineage_classes": len(unique_lineages),
        "lineage_classes": unique_lineages,
        "n_countries": len(unique_countries),
        "countries": unique_countries,
        "n_missing_lineage": int((lineage_indices == 0).sum()),
        "n_missing_country": int((country_indices == 0).sum()),
    }

    return {
        "lineage_indices": torch.tensor(lineage_indices, dtype=torch.long),
        "country_indices": torch.tensor(country_indices, dtype=torch.long),
        "info": info,
    }
