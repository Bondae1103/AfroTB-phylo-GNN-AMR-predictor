"""Read-only structural integrity audit of every VCF in AFRO_TB_VCF/.

Step 2 of the Person 1 VCF handoff audit (Step 1 was filename/canonical-ID
identity mapping, done separately in validate_vcf_mapping.py). This script
checks only STRUCTURE -- gzip validity, VCF header/column shape, per-record
field counts and basic type sanity, sample-column identity vs. filename,
CHROM-value consistency, variant-count statistics, and genotype-field
column-count parseability. It does NOT interpret biological meaning of any
variant, does NOT compute SNP distances, and does NOT modify any VCF or any
existing processed file.

Anomalies are reported, never silently fixed. A malformed record in one
file does not stop the audit of the rest of the dataset.

Writes data/processed/vcf_structural_qc_report.json. Rerunning this script
against an unchanged VCF directory reproduces the same report.
"""

import gzip
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VCF_DIR = Path("data/raw/Afro_TB/AFRO_TB_VCF")
OUT_REPORT = Path("data/processed/vcf_structural_qc_report.json")

EXPECTED_N_FILES = 13753
FILENAME_PATTERN = re.compile(r"^(?P<id>[A-Za-z0-9]+)_MT\.vcf\.gz$")
REQUIRED_FIXED_COLUMNS = [
    "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
]

# Objective outlier rule (Tukey fences), stated explicitly rather than
# implied: a file's variant count is "unusual" if it falls outside
# [Q1 - 1.5*IQR, Q3 + 1.5*IQR] across the whole dataset. Files are still
# audited and reported in full regardless of this flag -- nothing is
# discarded.
IQR_FENCE_MULTIPLIER = 1.5


def canonical_id_from_filename(path):
    match = FILENAME_PATTERN.match(path.name)
    return match.group("id") if match else None


def audit_one_file(path):
    """Run all structural checks on a single VCF. Never raises for a bad VCF.

    Returns a dict describing the file's status and any anomalies found.
    """
    result = {
        "path": path.as_posix(),
        "filename_id": canonical_id_from_filename(path),
        "readable": False,
        "valid_gzip": False,
        "non_empty_compressed": path.stat().st_size > 0 if path.exists() else False,
        "non_empty_decompressed": False,
        "has_header_lines": False,
        "has_chrom_header": False,
        "header_columns": None,
        "sample_columns": [],
        "sample_id_match": None,   # True/False/None (None = no sample column to compare)
        "n_variants": 0,
        "n_malformed_records": 0,
        "malformed_record_examples": [],   # up to a few (line_no, reason)
        "n_malformed_genotypes": 0,
        "malformed_genotype_examples": [],
        "chrom_values_seen": Counter(),
        "passed_all_checks": False,
        "failure_reasons": [],
    }

    if result["filename_id"] is None:
        result["failure_reasons"].append("filename_does_not_match_convention")

    try:
        with gzip.open(path, "rb") as fb:
            fb.read(1)
        result["valid_gzip"] = True
    except OSError:
        result["failure_reasons"].append("invalid_gzip")
        return result  # nothing further can be read reliably

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            header_columns = None
            expected_field_count = None
            format_col_index = None
            line_no = 0
            any_content = False

            for line in f:
                line_no += 1
                any_content = True
                line = line.rstrip("\n")

                if line.startswith("##"):
                    result["has_header_lines"] = True
                    continue

                if line.startswith("#CHROM"):
                    result["has_chrom_header"] = True
                    header_columns = line.split("\t")
                    result["header_columns"] = header_columns
                    expected_field_count = len(header_columns)
                    if len(header_columns) > 9:
                        result["sample_columns"] = header_columns[9:]
                    if len(header_columns) >= 9 and header_columns[8] == "FORMAT":
                        format_col_index = 8
                    if header_columns[:8] != REQUIRED_FIXED_COLUMNS:
                        result["failure_reasons"].append("unexpected_fixed_header_columns")
                    continue

                # variant record
                if header_columns is None:
                    result["n_malformed_records"] += 1
                    if len(result["malformed_record_examples"]) < 5:
                        result["malformed_record_examples"].append(
                            {"line": line_no, "reason": "record_before_chrom_header"}
                        )
                    continue

                fields = line.split("\t")
                record_ok = True

                if len(fields) != expected_field_count:
                    record_ok = False
                    if len(result["malformed_record_examples"]) < 5:
                        result["malformed_record_examples"].append(
                            {
                                "line": line_no,
                                "reason": f"field_count_{len(fields)}_expected_{expected_field_count}",
                            }
                        )
                else:
                    chrom, pos, _id, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
                    result["chrom_values_seen"][chrom] += 1

                    try:
                        int(pos)
                    except ValueError:
                        record_ok = False
                        if len(result["malformed_record_examples"]) < 5:
                            result["malformed_record_examples"].append(
                                {"line": line_no, "reason": f"non_integer_POS_{pos!r}"}
                            )

                    if ref == "":
                        record_ok = False
                        if len(result["malformed_record_examples"]) < 5:
                            result["malformed_record_examples"].append(
                                {"line": line_no, "reason": "empty_REF"}
                            )
                    if alt == "":
                        record_ok = False
                        if len(result["malformed_record_examples"]) < 5:
                            result["malformed_record_examples"].append(
                                {"line": line_no, "reason": "empty_ALT"}
                            )

                    if format_col_index is not None and len(fields) > format_col_index:
                        format_subfields = fields[format_col_index].split(":")
                        for sample_field in fields[format_col_index + 1:]:
                            if len(sample_field.split(":")) != len(format_subfields):
                                result["n_malformed_genotypes"] += 1
                                if len(result["malformed_genotype_examples"]) < 5:
                                    result["malformed_genotype_examples"].append(
                                        {
                                            "line": line_no,
                                            "reason": "genotype_field_count_mismatch",
                                            "format": fields[format_col_index],
                                            "sample_field": sample_field,
                                        }
                                    )

                result["n_variants"] += 1
                if not record_ok:
                    result["n_malformed_records"] += 1

            result["non_empty_decompressed"] = any_content
            result["readable"] = True
    except UnicodeDecodeError:
        result["failure_reasons"].append("non_text_content")
        return result
    except OSError:
        result["failure_reasons"].append("read_error")
        return result

    if not result["has_chrom_header"]:
        result["failure_reasons"].append("missing_chrom_header")
    if not result["non_empty_decompressed"]:
        result["failure_reasons"].append("empty_after_decompression")
    if result["n_malformed_records"] > 0:
        result["failure_reasons"].append("has_malformed_records")
    if result["n_malformed_genotypes"] > 0:
        result["failure_reasons"].append("has_malformed_genotypes")

    if result["filename_id"] is not None:
        if len(result["sample_columns"]) == 0:
            result["sample_id_match"] = None
        elif len(result["sample_columns"]) == 1:
            result["sample_id_match"] = (result["sample_columns"][0] == result["filename_id"])
            if not result["sample_id_match"]:
                result["failure_reasons"].append("sample_id_header_mismatch")
        else:
            result["sample_id_match"] = False
            result["failure_reasons"].append("unexpected_multiple_sample_columns")

    result["passed_all_checks"] = (
        result["valid_gzip"]
        and result["readable"]
        and result["non_empty_compressed"]
        and result["non_empty_decompressed"]
        and result["has_header_lines"]
        and result["has_chrom_header"]
        and result["n_malformed_records"] == 0
        and result["n_malformed_genotypes"] == 0
        and result["filename_id"] is not None
        and result["sample_id_match"] in (True, None)
        and len(result["failure_reasons"]) == 0
    )

    return result


def main():
    if not VCF_DIR.is_dir():
        raise RuntimeError(f"VCF directory not found: {VCF_DIR}")

    vcf_paths = sorted(VCF_DIR.rglob("*.vcf.gz"))
    n_examined = len(vcf_paths)

    per_file_results = []
    for path in vcf_paths:
        per_file_results.append(audit_one_file(path))

    n_passed = sum(1 for r in per_file_results if r["passed_all_checks"])
    n_failed = n_examined - n_passed

    failure_counts = Counter()
    for r in per_file_results:
        for reason in r["failure_reasons"]:
            failure_counts[reason] += 1

    sample_mismatches = [
        {
            "path": r["path"],
            "filename_id": r["filename_id"],
            "header_sample_columns": r["sample_columns"],
        }
        for r in per_file_results
        if r["sample_id_match"] is False
    ]

    invalid_gzip_files = [r["path"] for r in per_file_results if not r["valid_gzip"]]
    empty_files = [
        r["path"] for r in per_file_results
        if r["non_empty_compressed"] and not r["non_empty_decompressed"]
    ]
    zero_size_files = [r["path"] for r in per_file_results if not r["non_empty_compressed"]]
    missing_header_files = [r["path"] for r in per_file_results if not r["has_chrom_header"]]

    malformed_record_files = [
        {"path": r["path"], "n_malformed_records": r["n_malformed_records"],
         "examples": r["malformed_record_examples"]}
        for r in per_file_results if r["n_malformed_records"] > 0
    ]
    malformed_genotype_files = [
        {"path": r["path"], "n_malformed_genotypes": r["n_malformed_genotypes"],
         "examples": r["malformed_genotype_examples"]}
        for r in per_file_results if r["n_malformed_genotypes"] > 0
    ]

    # chromosome/reference consistency, aggregated across the whole dataset
    global_chrom_counts = Counter()
    files_with_multiple_chroms = []
    for r in per_file_results:
        for chrom, count in r["chrom_values_seen"].items():
            global_chrom_counts[chrom] += count
        if len(r["chrom_values_seen"]) > 1:
            files_with_multiple_chroms.append(
                {"path": r["path"], "chroms": dict(r["chrom_values_seen"])}
            )

    # variant-count statistics
    counts = [r["n_variants"] for r in per_file_results]
    zero_variant_files = [r["path"] for r in per_file_results if r["n_variants"] == 0]

    variant_stats = {}
    if counts:
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        q1 = sorted_counts[n // 4]
        q3 = sorted_counts[(3 * n) // 4]
        iqr = q3 - q1
        low_fence = q1 - IQR_FENCE_MULTIPLIER * iqr
        high_fence = q3 + IQR_FENCE_MULTIPLIER * iqr

        unusually_low = [
            {"path": r["path"], "n_variants": r["n_variants"]}
            for r in per_file_results if r["n_variants"] < low_fence
        ]
        unusually_high = [
            {"path": r["path"], "n_variants": r["n_variants"]}
            for r in per_file_results if r["n_variants"] > high_fence
        ]

        variant_stats = {
            "min": min(counts),
            "max": max(counts),
            "mean": round(statistics.mean(counts), 3),
            "median": statistics.median(counts),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "outlier_rule": (
                f"Tukey fence: unusual if outside "
                f"[Q1 - {IQR_FENCE_MULTIPLIER}*IQR, Q3 + {IQR_FENCE_MULTIPLIER}*IQR] "
                f"= [{low_fence:.1f}, {high_fence:.1f}]. Files are reported, not discarded."
            ),
            "n_zero_variant_files": len(zero_variant_files),
            "n_unusually_low": len(unusually_low),
            "n_unusually_high": len(unusually_high),
            "unusually_low_examples": unusually_low[:20],
            "unusually_high_examples": unusually_high[:20],
        }

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "vcf_directory": VCF_DIR.as_posix(),
        "expected_n_files": EXPECTED_N_FILES,
        "n_files_examined": n_examined,
        "n_passed_all_structural_checks": n_passed,
        "n_failed": n_failed,
        "failure_category_counts": dict(failure_counts),
        "sample_identity": {
            "n_mismatches": len(sample_mismatches),
            "mismatches": sample_mismatches,
        },
        "invalid_gzip_files": invalid_gzip_files,
        "zero_size_files": zero_size_files,
        "empty_after_decompression_files": empty_files,
        "missing_chrom_header_files": missing_header_files,
        "malformed_record_files": malformed_record_files,
        "malformed_genotype_files": malformed_genotype_files,
        "chromosome_reference_observations": {
            "distinct_chrom_values": dict(global_chrom_counts),
            "n_distinct_chrom_values": len(global_chrom_counts),
            "single_consistent_convention": len(global_chrom_counts) == 1,
            "files_with_multiple_chrom_values_in_one_file": files_with_multiple_chroms,
        },
        "variant_count_statistics": variant_stats,
        "overall_structural_qc_result": (
            "PASS" if n_failed == 0 and n_examined == EXPECTED_N_FILES else "ANOMALIES_FOUND"
        ),
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Examined: {n_examined} (expected {EXPECTED_N_FILES})")
    print(f"Passed all structural checks: {n_passed}")
    print(f"Failed: {n_failed}")
    print("Failure category counts:", json.dumps(dict(failure_counts), indent=2))
    print("Variant count stats:", json.dumps(
        {k: v for k, v in variant_stats.items() if k not in
         ("unusually_low_examples", "unusually_high_examples")}, indent=2))
    print("Chromosome conventions:", dict(global_chrom_counts))
    print("Overall result:", report["overall_structural_qc_result"])
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
