"""Entry point: is y_amr a deterministic function of features.csv?

Run this FIRST, before interpreting any model metric in results/. It takes
seconds, trains nothing, and determines whether the supervised task those
metrics describe is a real prediction problem or a lookup.

Usage
-----
    # real data (derives the mutation -> drug map from the raw workbook and
    # writes data/processed/mutation_drug_map.json, which nothing produced before)
    .venv/Scripts/python.exe -m src.audit.run_label_tautology

    # any other processed dir that already ships a mutation_drug_map.json
    .venv/Scripts/python.exe -m src.audit.run_label_tautology --dir data/synthetic/full

Writes <dir>/label_tautology_report.json (and, for the real data,
results/label_tautology_report.json as well).
"""

import argparse
import json
from pathlib import Path

from .label_tautology import (
    audit,
    mutation_drug_map_from_workbook,
    write_mutation_drug_map,
)

REAL_PROCESSED = Path("data/processed")
RAW_XLSX = Path("data/raw/Afro_TB/0-StartHERE_Afro-TB.xlsx")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REAL_PROCESSED))
    ap.add_argument("--xlsx", default=str(RAW_XLSX))
    args = ap.parse_args()

    processed = Path(args.dir)
    xlsx = Path(args.xlsx)
    map_path = processed / "mutation_drug_map.json"

    if not map_path.exists():
        if not xlsx.exists():
            raise SystemExit(
                "Neither %s nor the raw workbook %s is present.\n"
                "The mutation -> drug mapping must come from one of them."
                % (map_path, xlsx)
            )
        print("Deriving mutation -> drug map from %s ..." % xlsx)
        mapping, diagnostics = mutation_drug_map_from_workbook(xlsx)
        write_mutation_drug_map(mapping, diagnostics, map_path)
        print("Wrote %s" % map_path)
        print(json.dumps(
            {k: v for k, v in diagnostics.items()
             if k not in ("columns_with_multiple_drug_codes", "columns_never_positive")},
            indent=2))
        if not diagnostics["one_drug_per_column_holds"]:
            print("\nNOTE: %d column(s) carry more than one drug code, so y_amr is "
                  "NOT a clean per-column OR. Those columns are excluded from the "
                  "mapping and the audit below will show them as mismatches."
                  % diagnostics["n_columns_with_multiple_drug_codes"])

    report = audit(processed)

    print("\n" + "=" * 72)
    print(report["verdict"])
    print("=" * 72)
    o = report["overall"]
    print("exact label-cell match rate : %.6f  (%d of %d cells)" % (
        o["exact_cell_match_rate"],
        o["total_label_cells"] - o["total_mismatched_cells"], o["total_label_cells"]))
    print("isolates reproduced exactly : %d / %d" % (
        o["n_isolates_reproduced_exactly"], report["n_isolates"]))
    print("OR-rule macro-F1 (core 6)   : %s" % o["or_rule_macro_f1_core6"])
    print("OR-rule macro-F1 (all 9)    : %s" % o["or_rule_macro_f1_all9"])
    print("\nThe OR rule has ZERO learned parameters and never saw a training split.")
    print("Compare the above against the trained models in results/.\n")

    print("%-6s %6s %9s %9s %10s" % ("DRUG", "#COLS", "#POS", "#MISMATCH", "OR-RULE F1"))
    for drug, s in report["per_drug"].items():
        print("%-6s %6d %9d %9d %10s" % (
            drug, s["n_feature_columns"], s["n_positive_true"],
            s["n_mismatches"], s["or_rule_f1"]))

    out = processed / "label_tautology_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nWrote %s" % out)

    if processed.resolve() == REAL_PROCESSED.resolve():
        mirror = Path("results/label_tautology_report.json")
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with open(mirror, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Wrote %s" % mirror)


if __name__ == "__main__":
    main()
