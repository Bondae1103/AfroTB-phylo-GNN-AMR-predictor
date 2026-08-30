"""The tautology audit must confirm an exact OR, and must NOT confirm one when
the relationship is broken. The negative case is the important one: an audit
that always reports "tautology" would be worthless.
"""

import csv

import numpy as np

from src.audit.label_tautology import (
    audit,
    drug_to_column_indices,
    load_mutation_drug_map,
)


def test_detects_exact_tautology(replica):
    report = audit(replica["dir"])
    assert report["is_exact_tautology"] is True
    assert report["overall"]["exact_cell_match_rate"] == 1.0
    assert report["overall"]["total_mismatched_cells"] == 0
    assert report["n_features_unmapped_to_any_drug"] == 0
    # every drug with positives is reproduced perfectly by a zero-parameter rule
    for drug, stats in report["per_drug"].items():
        if stats["n_positive_true"] > 0:
            assert stats["or_rule_f1"] == 1.0, drug


def test_does_not_report_tautology_when_labels_are_perturbed(replica, tmp_path):
    """Flip a handful of label cells; the audit must notice."""
    src = replica["dir"]
    dst = tmp_path / "perturbed"
    dst.mkdir()
    for name in ("features.csv", "mutation_drug_map.json"):
        (dst / name).write_bytes((src / name).read_bytes())

    with open(src / "y_amr.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    flipped = 0
    for r in body:
        if flipped >= 7:
            break
        if r[1] == "0":
            r[1] = "1"
            flipped += 1
    assert flipped == 7
    with open(dst / "y_amr.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + body)

    report = audit(dst)
    assert report["is_exact_tautology"] is False
    assert report["overall"]["total_mismatched_cells"] == 7
    assert report["per_drug"]["RIF"]["or_rule_f1"] < 1.0


def test_column_grouping_is_a_partition(replica):
    mapping = load_mutation_drug_map(replica["dir"])
    import pandas as pd
    names = [c for c in pd.read_csv(replica["dir"] / "features.csv", nrows=0).columns
             if c != "Name"]
    by_drug, unmapped = drug_to_column_indices(names, mapping)
    assert unmapped == []
    all_cols = sorted(c for cols in by_drug.values() for c in cols)
    assert all_cols == list(range(len(names))), "columns must partition exactly once"
