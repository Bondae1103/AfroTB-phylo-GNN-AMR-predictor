"""Generate data/processed/y_amr.csv and y_amr_metadata.json.

Re-parses the raw data/raw/Afro_TB/0-StartHERE_Afro-TB.xlsx workbook
READ-ONLY, preserving the drug code contained in each of the 157 mutation
cells (features.csv discards this — it only keeps a flat 0/1 presence
flag). For each isolate, y_amr[drug] = 1 if any of its 157 mutation cells
carried that drug's code, else 0.

D=9 by design: only the 9 drug codes that actually occur as values in this
dataset's mutation cells are used (RIF, INH, EMB, PZA, STM, LEV, CAP, ETH,
LZD). AMI, KAN, MXF appear in the separate WHO-resistance-associated-
mutations.xlsx global catalog but never as a value in this dataset, so they
are deliberately NOT added as all-zero columns.

Also runs a non-destructive QC comparison between y_amr and labels.csv's
aggregate Drug column, reported in y_amr_metadata.json without correcting
either file.

Does not modify the raw workbook, features.csv, or labels.csv. Requires
data/processed/sample_ids.csv and labels.csv to already exist.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import openpyxl

RAW_XLSX = Path("data/raw/Afro_TB/0-StartHERE_Afro-TB.xlsx")
SHEET_NAME = "AfroTB"

MUTATION_HEADER_ROW = 4
BASE_HEADER_ROW = 5
DATA_START_ROW = 6
MUTATION_COL_START = 5   # 1-based, inclusive
MUTATION_COL_END = 161   # 1-based, inclusive

PROCESSED_DIR = Path("data/processed")
SAMPLE_IDS_CSV = PROCESSED_DIR / "sample_ids.csv"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
OUT_CSV = PROCESSED_DIR / "y_amr.csv"
OUT_METADATA = PROCESSED_DIR / "y_amr_metadata.json"

EXPECTED_N_ISOLATES = 13753
DRUG_CODES = ["RIF", "INH", "EMB", "PZA", "STM", "LEV", "CAP", "ETH", "LZD"]
PLACEHOLDER_CODES = {"_", "-", None}

RESISTANT_DRUG_LABELS = {"MDR", "Mono", "Pre-XDR", "Other", "Other*"}


def load_raw_mutation_rows():
    """Load per-isolate mutation cell rows from the raw workbook, unmodified."""
    wb = openpyxl.load_workbook(RAW_XLSX, read_only=True, data_only=True)
    ws = wb[SHEET_NAME]

    mutation_header_row = None
    data_rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if i == MUTATION_HEADER_ROW:
            mutation_header_row = row
        elif i >= DATA_START_ROW:
            if row[0] is None:
                continue
            data_rows.append(row)
    wb.close()

    mutation_names = list(mutation_header_row[MUTATION_COL_START - 1:MUTATION_COL_END])
    return mutation_names, data_rows


def build_y_amr(data_rows):
    """Aggregate each isolate's mutation-cell drug codes into a 9-drug binary row."""
    y_amr_by_name = {}
    seen_names = set()

    for row in data_rows:
        name = row[0]
        if name in seen_names:
            raise AssertionError(f"Duplicate Name in raw workbook: {name!r}")
        seen_names.add(name)

        mutation_values = row[MUTATION_COL_START - 1:MUTATION_COL_END]
        drugs_present = set()
        for value in mutation_values:
            if value in PLACEHOLDER_CODES:
                continue
            if value not in DRUG_CODES:
                raise ValueError(
                    f"Unexpected/unlisted drug code {value!r} for isolate {name!r}."
                )
            drugs_present.add(value)

        y_amr_by_name[name] = [1 if d in drugs_present else 0 for d in DRUG_CODES]

    return y_amr_by_name, seen_names


def read_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    return {row[0]: row[3] for row in rows}, [row[0] for row in rows]


def run_qc(canonical_ids, y_amr_by_name, drug_by_id):
    """Non-destructive comparison between y_amr and the aggregate Drug label."""
    inconsistent = {
        "sensitive_but_yamr_positive": [],
        "mono_but_yamr_count_ne_1": [],
        "resistant_category_but_yamr_allzero": [],
    }
    sum_by_drug_category = defaultdict(list)

    for name in canonical_ids:
        drug_label = drug_by_id[name]
        yamr_sum = sum(y_amr_by_name[name])
        sum_by_drug_category[drug_label].append(yamr_sum)

        if drug_label == "Sensitive" and yamr_sum > 0:
            inconsistent["sensitive_but_yamr_positive"].append(name)
        if drug_label == "Mono" and yamr_sum != 1:
            inconsistent["mono_but_yamr_count_ne_1"].append(name)
        if drug_label in RESISTANT_DRUG_LABELS and yamr_sum == 0:
            inconsistent["resistant_category_but_yamr_allzero"].append(name)

    summary = {
        "sensitive_but_yamr_positive_count": len(inconsistent["sensitive_but_yamr_positive"]),
        "mono_but_yamr_count_ne_1_count": len(inconsistent["mono_but_yamr_count_ne_1"]),
        "resistant_category_but_yamr_allzero_count": len(inconsistent["resistant_category_but_yamr_allzero"]),
        "yamr_positive_drug_count_stats_by_drug_label": {
            label: {
                "n": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": round(sum(vals) / len(vals), 3),
            }
            for label, vals in sum_by_drug_category.items()
        },
    }
    return summary, inconsistent


def main():
    mutation_names, data_rows = load_raw_mutation_rows()
    if len(data_rows) != EXPECTED_N_ISOLATES:
        raise AssertionError(
            f"Expected {EXPECTED_N_ISOLATES} isolates in raw workbook, found {len(data_rows)}."
        )
    if len(mutation_names) != 157:
        raise AssertionError(f"Expected 157 mutation columns, found {len(mutation_names)}.")

    y_amr_by_name, seen_names = build_y_amr(data_rows)

    canonical_ids = read_ids(SAMPLE_IDS_CSV)
    drug_by_id, labels_ids = read_labels(LABELS_CSV)

    if len(canonical_ids) != EXPECTED_N_ISOLATES:
        raise AssertionError(f"sample_ids.csv does not have {EXPECTED_N_ISOLATES} IDs.")
    if canonical_ids != labels_ids:
        raise AssertionError("sample_ids.csv order does not match labels.csv order.")
    if set(canonical_ids) != set(y_amr_by_name.keys()):
        raise AssertionError("Raw workbook IDs do not match canonical sample_ids.csv IDs.")

    validation = {
        "row_count_ok": len(canonical_ids) == EXPECTED_N_ISOLATES,
        "id_set_matches_canonical": set(canonical_ids) == set(y_amr_by_name.keys()),
        "no_duplicate_ids_in_raw": len(seen_names) == EXPECTED_N_ISOLATES,
        "all_values_binary": all(
            v in (0, 1) for name in canonical_ids for v in y_amr_by_name[name]
        ),
        "only_allowed_drug_codes_used": True,  # enforced during construction; ValueError otherwise
        "allowed_drug_codes": DRUG_CODES,
    }
    if not validation["all_values_binary"]:
        raise AssertionError("Non-binary value found in constructed y_amr matrix.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name"] + DRUG_CODES)
        for name in canonical_ids:
            writer.writerow([name] + y_amr_by_name[name])

    qc_summary, qc_inconsistent = run_qc(canonical_ids, y_amr_by_name, drug_by_id)

    metadata = {
        "source_file": RAW_XLSX.as_posix(),
        "sheet_name": SHEET_NAME,
        "derivation": (
            "Re-parsed the raw Afro-TB mutation workbook (read-only), preserving the "
            "drug code contained in each of the 157 mutation cells (rather than "
            "collapsing to a flat presence/absence 0/1 as prepare_afrotb_matrix.py "
            "does for features.csv). For each isolate, y_amr[drug] = 1 if any of its "
            "157 mutation cells carried that drug code, else 0."
        ),
        "D_decision": (
            "D=9. Only the 9 drug codes that actually occur as values in this "
            "dataset's 157 mutation columns are used: RIF, INH, EMB, PZA, STM, LEV, "
            "CAP, ETH, LZD. AMI, KAN, MXF (present in the separate WHO-resistance-"
            "associated-mutations.xlsx global catalog) were deliberately NOT added "
            "as all-zero columns, since this dataset provides no ground truth for them."
        ),
        "drug_columns": DRUG_CODES,
        "n_isolates": len(canonical_ids),
        "canonical_order_source": SAMPLE_IDS_CSV.as_posix(),
        "features_labels_untouched": True,
        "validation": validation,
        "qc_vs_aggregate_drug_label": {
            "method": (
                "Non-destructive comparison only; no data modified or corrected. "
                "Sensitive is expected to have all-zero y_amr; Mono is expected to "
                "have exactly one positive drug; MDR/Mono/Pre-XDR/Other/Other* are "
                "expected to have at least one positive drug."
            ),
            "summary": qc_summary,
            "inconsistent_ids": qc_inconsistent,
        },
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(canonical_ids)} isolates x {len(DRUG_CODES)} drugs to {OUT_CSV}")
    print(f"Wrote {OUT_METADATA}")
    print("Validation:", json.dumps(validation, indent=2))
    print("QC summary:", json.dumps(qc_summary, indent=2))


if __name__ == "__main__":
    main()
