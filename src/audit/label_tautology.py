"""Is y_amr a deterministic function of features.csv?

Why this exists
---------------
scripts/prepare_afrotb_matrix.py and scripts/create_y_amr.py read the SAME 157
mutation cells of the SAME workbook rows:

    features.csv[i, m] = 1   iff cell m of isolate i holds a drug code
    y_amr.csv[i, d]    = 1   iff ANY cell of isolate i holds drug code d

If each mutation column only ever carries one drug code, the second line is a
Boolean OR over a fixed subset of the first line's columns -- i.e. y = f(X)
exactly, with no noise term and nothing to learn. Every supervised metric
computed on X -> y would then be measuring how well a model reproduces an OR
gate, not how well it predicts drug resistance.

That is a claim about the data, so this module MEASURES it rather than
asserting it:

  1. the one-drug-per-column property is checked against the raw workbook
     (mutation_drug_map_from_workbook) and reported, never assumed;
  2. the OR rule is reconstructed and compared to y_amr cell by cell;
  3. the rule's per-drug F1 is reported in the SAME units as the model
     metrics in results/, so it can be read directly against them.

The OR rule has zero learned parameters and never sees a training split. If
its F1 sits at or above the trained baselines', the modelling task as
currently specified is vacuous, and that is the finding.

Nothing here modifies any data file.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DRUG_CODES = ["RIF", "INH", "EMB", "PZA", "STM", "LEV", "CAP", "ETH", "LZD"]
PLACEHOLDER_CODES = {"_", "-", None, ""}

# Raw-workbook geometry, identical to scripts/create_y_amr.py.
SHEET_NAME = "AfroTB"
MUTATION_HEADER_ROW = 4
DATA_START_ROW = 6
MUTATION_COL_START = 5
MUTATION_COL_END = 161


def mutation_drug_map_from_workbook(xlsx_path):
    """Derive {mutation_name: drug_code} from the raw workbook.

    Returns (mapping, diagnostics). diagnostics records, per column, every
    distinct non-placeholder value seen -- so a column carrying more than one
    drug code shows up as a violation instead of being silently collapsed.
    Only columns with exactly one distinct code enter the mapping.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME]

    header = None
    values_per_col = defaultdict(set)
    n_rows = 0
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if i == MUTATION_HEADER_ROW:
            header = list(row[MUTATION_COL_START - 1:MUTATION_COL_END])
        elif i >= DATA_START_ROW:
            if row[0] is None:
                continue
            n_rows += 1
            cells = row[MUTATION_COL_START - 1:MUTATION_COL_END]
            for c, value in enumerate(cells):
                if value in PLACEHOLDER_CODES:
                    continue
                values_per_col[c].add(str(value).strip())
    wb.close()

    mapping = {}
    multi_code_columns = {}
    empty_columns = []
    for c, name in enumerate(header):
        codes = sorted(values_per_col.get(c, set()))
        if len(codes) == 1:
            mapping[name] = codes[0]
        elif len(codes) == 0:
            empty_columns.append(name)
        else:
            multi_code_columns[name] = codes

    diagnostics = {
        "n_rows_scanned": n_rows,
        "n_columns": len(header),
        "n_columns_single_drug_code": len(mapping),
        "n_columns_never_positive": len(empty_columns),
        "n_columns_with_multiple_drug_codes": len(multi_code_columns),
        "columns_with_multiple_drug_codes": multi_code_columns,
        "columns_never_positive": empty_columns,
        "one_drug_per_column_holds": len(multi_code_columns) == 0,
    }
    return mapping, diagnostics


def write_mutation_drug_map(mapping, diagnostics, out_path):
    """Persist the mutation -> drug mapping.

    This artifact did not previously exist anywhere in the repo: create_y_amr.py
    computed drug codes per cell and discarded the column-level mapping. It is
    required by any protocol that needs to know which feature columns belong to
    which drug -- notably leave-drug-out evaluation.
    """
    out_path = Path(out_path)
    payload = {
        "source": "raw Afro-TB workbook, one distinct drug code per mutation column",
        "drug_columns": DRUG_CODES,
        "mutation_to_drug": mapping,
        "diagnostics": diagnostics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_mutation_drug_map(processed_dir):
    """Read mutation_drug_map.json from a processed-data directory."""
    path = Path(processed_dir) / "mutation_drug_map.json"
    if not path.exists():
        raise FileNotFoundError(
            "%s not found. Generate it with src.audit.run_label_tautology "
            "(needs the raw workbook), or use a processed dir that ships one."
            % path
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["mutation_to_drug"]


def _read_matrix(path, id_col="Name"):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    ids = [r[0] for r in rows]
    cols = header[1:]
    M = np.array([[int(v) for v in r[1:]] for r in rows], dtype=np.int8)
    return ids, cols, M


def drug_to_column_indices(feature_names, mutation_to_drug):
    """{drug: [feature column indices belonging to that drug]}."""
    out = {d: [] for d in DRUG_CODES}
    unmapped = []
    for j, name in enumerate(feature_names):
        drug = mutation_to_drug.get(name)
        if drug is None:
            unmapped.append(name)
        else:
            out[drug].append(j)
    return out, unmapped


def _binary_f1(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0 if (fp or fn) else float("nan"), tp, fp, fn
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec), tp, fp, fn


def audit(processed_dir, mutation_to_drug=None):
    """Reconstruct y_amr as an OR over each drug's own feature columns.

    Returns a report dict. The headline field is
    reconstruction["overall"]["exact_cell_match_rate"]: the fraction of the
    N x 9 label matrix a zero-parameter rule reproduces.
    """
    processed_dir = Path(processed_dir)
    if mutation_to_drug is None:
        mutation_to_drug = load_mutation_drug_map(processed_dir)

    f_ids, feature_names, X = _read_matrix(processed_dir / "features.csv")
    y_ids, drug_cols, Y = _read_matrix(processed_dir / "y_amr.csv")
    if f_ids != y_ids:
        raise ValueError("features.csv and y_amr.csv row order differ")
    if drug_cols != DRUG_CODES:
        raise ValueError("y_amr.csv columns %r != %r" % (drug_cols, DRUG_CODES))

    cols_by_drug, unmapped = drug_to_column_indices(feature_names, mutation_to_drug)

    per_drug = {}
    Y_hat = np.zeros_like(Y)
    for d_i, drug in enumerate(DRUG_CODES):
        cols = cols_by_drug[drug]
        if cols:
            Y_hat[:, d_i] = X[:, cols].max(axis=1)
        y_true, y_pred = Y[:, d_i], Y_hat[:, d_i]
        f1, tp, fp, fn = _binary_f1(y_true, y_pred)
        mism = int(np.sum(y_true != y_pred))
        per_drug[drug] = {
            "n_feature_columns": len(cols),
            "n_positive_true": int(y_true.sum()),
            "n_positive_or_rule": int(y_pred.sum()),
            "n_mismatches": mism,
            "exact_match_rate": round(1.0 - mism / len(y_true), 6),
            "or_rule_f1": None if f1 != f1 else round(f1, 6),
            "or_rule_says_resistant_label_says_not": fp,
            "label_says_resistant_or_rule_says_not": fn,
        }

    total_cells = Y.size
    total_mismatch = int(np.sum(Y != Y_hat))
    n_rows_exact = int(np.sum((Y == Y_hat).all(axis=1)))
    defined = [per_drug[d]["or_rule_f1"] for d in DRUG_CODES
               if per_drug[d]["or_rule_f1"] is not None]
    core6 = [per_drug[d]["or_rule_f1"] for d in DRUG_CODES[:6]
             if per_drug[d]["or_rule_f1"] is not None]

    is_tautology = total_mismatch == 0
    return {
        "processed_dir": processed_dir.as_posix(),
        "question": "Can y_amr be reproduced exactly by OR-ing each drug's own feature columns?",
        "n_isolates": int(Y.shape[0]),
        "n_features": len(feature_names),
        "n_features_unmapped_to_any_drug": len(unmapped),
        "features_unmapped_to_any_drug": unmapped[:20],
        "per_drug": per_drug,
        "overall": {
            "total_label_cells": int(total_cells),
            "total_mismatched_cells": total_mismatch,
            "exact_cell_match_rate": round(1.0 - total_mismatch / total_cells, 8),
            "n_isolates_reproduced_exactly": n_rows_exact,
            "isolate_exact_match_rate": round(n_rows_exact / Y.shape[0], 8),
            "or_rule_macro_f1_all9": round(float(np.mean(defined)), 6) if defined else None,
            "or_rule_macro_f1_core6": round(float(np.mean(core6)), 6) if core6 else None,
        },
        "verdict": (
            "TAUTOLOGY CONFIRMED: y_amr is an exact Boolean OR of features.csv "
            "columns. A model trained on X -> y is reproducing a lookup rule, "
            "not predicting resistance. Supervised metrics on this pair do not "
            "measure biological prediction."
            if is_tautology else
            "NOT an exact tautology: %d of %d label cells are not reproduced by "
            "the OR rule. Inspect per_drug before drawing conclusions."
            % (total_mismatch, total_cells)
        ),
        "is_exact_tautology": bool(is_tautology),
    }
