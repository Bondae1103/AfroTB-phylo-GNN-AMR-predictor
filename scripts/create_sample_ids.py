"""Generate data/processed/sample_ids.csv, the canonical sample ID list.

Reads the already-generated data/processed/features.csv (produced by
prepare_afrotb_matrix.py) and writes a single-column CSV of Name values,
preserving features.csv's exact row order. This file is the canonical
row-order reference for every other processed artifact (y_amr.csv,
splits.csv, and anything Person 2/3 add later).

Does not read or modify the raw dataset, features.csv, or labels.csv.
"""

import csv
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_CSV = PROCESSED_DIR / "features.csv"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
OUT_CSV = PROCESSED_DIR / "sample_ids.csv"

EXPECTED_N_ISOLATES = 13753


def read_ids(path):
    """Read the Name column (first column) from a processed CSV, in row order."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [row[0] for row in reader]


def main():
    feature_ids = read_ids(FEATURES_CSV)
    label_ids = read_ids(LABELS_CSV)

    if len(feature_ids) != EXPECTED_N_ISOLATES:
        raise AssertionError(
            f"Expected {EXPECTED_N_ISOLATES} IDs in {FEATURES_CSV}, "
            f"found {len(feature_ids)}."
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise AssertionError(f"Duplicate Name values found in {FEATURES_CSV}.")
    if feature_ids != label_ids:
        raise AssertionError(
            f"{FEATURES_CSV} and {LABELS_CSV} do not have identical Name "
            f"order; sample_ids.csv must be derivable unambiguously from "
            f"either file."
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name"])
        for name in feature_ids:
            writer.writerow([name])

    print(f"Wrote {len(feature_ids)} canonical sample IDs to {OUT_CSV}")
    print("Validation: row count OK, all IDs unique, order matches labels.csv.")


if __name__ == "__main__":
    main()
