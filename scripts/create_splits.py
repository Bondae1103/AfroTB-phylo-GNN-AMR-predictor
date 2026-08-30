"""Generate data/processed/splits.csv and splits_metadata.json.

Fixed 70/15/15 train/validation/test split, stratified on labels.csv's
Drug column, random seed 42. Within each Drug class, IDs are shuffled with
random.Random(42) (classes processed in sorted order for reproducibility)
and cut at rounded 70/15/15 boundaries, with train size = n - round(0.15n)
- round(0.15n) so counts always sum exactly to the class size.

The single Drug="Other" sample cannot be split three ways and is
deterministically assigned to train (documented below and in the output
metadata) -- train is the only split where a class represented by one
example is usable at all.

Original Drug labels in labels.csv are used unmodified; no classes are
merged. Does not modify features.csv, labels.csv, or sample_ids.csv.
Requires data/processed/sample_ids.csv and labels.csv to already exist.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SAMPLE_IDS_CSV = PROCESSED_DIR / "sample_ids.csv"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
OUT_CSV = PROCESSED_DIR / "splits.csv"
OUT_METADATA = PROCESSED_DIR / "splits_metadata.json"

EXPECTED_N_ISOLATES = 13753
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def read_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


def read_drug_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    return {row[0]: row[3] for row in rows}, [row[0] for row in rows]


def build_splits(canonical_ids, drug_by_id):
    """Stratified 70/15/15 split by Drug label; singleton classes go to train."""
    groups = {}
    for name in canonical_ids:
        groups.setdefault(drug_by_id[name], []).append(name)

    split_of = {}
    singleton_info = None

    for drug_label in sorted(groups.keys()):
        ids = groups[drug_label]
        n = len(ids)

        if n == 1:
            split_of[ids[0]] = "train"
            singleton_info = {
                "drug_label": drug_label,
                "sample_id": ids[0],
                "assigned_split": "train",
            }
            continue

        rng = random.Random(SEED)
        shuffled = ids[:]
        rng.shuffle(shuffled)

        n_val = round(n * VAL_FRAC)
        n_test = round(n * TEST_FRAC)
        n_train = n - n_val - n_test

        for name in shuffled[:n_train]:
            split_of[name] = "train"
        for name in shuffled[n_train:n_train + n_val]:
            split_of[name] = "val"
        for name in shuffled[n_train + n_val:]:
            split_of[name] = "test"

    return split_of, singleton_info


def main():
    canonical_ids = read_ids(SAMPLE_IDS_CSV)
    drug_by_id, labels_ids = read_drug_labels(LABELS_CSV)

    if len(canonical_ids) != EXPECTED_N_ISOLATES:
        raise AssertionError(f"sample_ids.csv does not have {EXPECTED_N_ISOLATES} IDs.")
    if canonical_ids != labels_ids:
        raise AssertionError("sample_ids.csv order does not match labels.csv order.")

    split_of, singleton_info = build_splits(canonical_ids, drug_by_id)

    if set(split_of.keys()) != set(canonical_ids):
        raise AssertionError("Split assignment does not cover exactly the canonical ID set.")
    if any(v not in ("train", "val", "test") for v in split_of.values()):
        raise AssertionError("Invalid split value produced.")
    if len(split_of) != EXPECTED_N_ISOLATES:
        raise AssertionError("Split assignment count mismatch.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "split"])
        for name in canonical_ids:
            writer.writerow([name, split_of[name]])

    split_counts = Counter(split_of.values())
    drug_by_split = {}
    for name in canonical_ids:
        s = split_of[name]
        d = drug_by_id[name]
        drug_by_split.setdefault(s, Counter())[d] += 1

    metadata = {
        "seed": SEED,
        "proportions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "stratify_column": "Drug",
        "stratify_note": "Drug labels used exactly as in labels.csv, unmodified. No classes merged.",
        "singleton_handling": {
            "description": (
                "Any Drug class with exactly 1 member cannot be stratified across "
                "3 splits and is deterministically assigned to train, since train "
                "is the only split where a class represented by a single example "
                "is usable (val/test metrics on a 1-example class would be "
                "meaningless, and a class absent from train could never be "
                "learned)."
            ),
            "singleton": singleton_info,
        },
        "split_method": (
            "Per-Drug-class deterministic shuffle (random.Random(42), classes "
            "processed in sorted order) then cut at rounded 70/15/15 boundaries; "
            "train size = n - round(0.15n) - round(0.15n) so counts sum exactly "
            "to class size."
        ),
        "split_counts": dict(split_counts),
        "drug_distribution_by_split": {s: dict(c) for s, c in drug_by_split.items()},
        "source_files": {
            "sample_ids": SAMPLE_IDS_CSV.as_posix(),
            "labels": LABELS_CSV.as_posix(),
        },
        "canonical_order_preserved": True,
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(canonical_ids)} split assignments to {OUT_CSV}")
    print(f"Wrote {OUT_METADATA}")
    print("split_counts:", dict(split_counts))
    print("singleton_info:", singleton_info)


if __name__ == "__main__":
    main()
