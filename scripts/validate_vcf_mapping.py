"""Validate the mapping between canonical sample IDs and raw VCF files.

Identity/file-mapping audit only: does NOT parse variant records, compute
SNP distances, or touch any processed artifact. Confirms whether Person 2
can safely use data/processed/sample_ids.csv's row order to look up each
isolate's VCF path in data/raw/Afro_TB/AFRO_TB_VCF/.

Confirmed filename convention (inspected manually before writing this
script): every VCF file is named "<ID>_MT.vcf.gz" and sits directly in
AFRO_TB_VCF/ with no subdirectories -- e.g. "ERR036186_MT.vcf.gz" maps to
canonical ID "ERR036186". The two accession prefixes observed are ERR and
SRR. This script still enumerates recursively and validates the pattern
against every file rather than assuming it holds.

Writes data/processed/vcf_mapping_report.json documenting the audit
result. Does not modify sample_ids.csv or any other processed file.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

SAMPLE_IDS_CSV = Path("data/processed/sample_ids.csv")
VCF_DIR = Path("data/raw/Afro_TB/AFRO_TB_VCF")
OUT_REPORT = Path("data/processed/vcf_mapping_report.json")

EXPECTED_N_CANONICAL = 13753

FILENAME_PATTERN = re.compile(r"^(?P<id>[A-Za-z0-9]+)_MT\.vcf\.gz$")


def read_canonical_ids():
    with open(SAMPLE_IDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


def enumerate_vcf_files():
    """Recursively enumerate VCF files; each is a candidate for ID extraction."""
    return sorted(VCF_DIR.rglob("*.vcf.gz"))


def extract_id(vcf_path):
    """Extract the canonical ID from a VCF filename using the confirmed convention.

    Returns (id_or_None, matched_bool). A filename that does not match the
    confirmed "<ID>_MT.vcf.gz" convention is reported, not guessed at.
    """
    match = FILENAME_PATTERN.match(vcf_path.name)
    if match is None:
        return None, False
    return match.group("id"), True


def main():
    canonical_ids = read_canonical_ids()
    canonical_id_set = set(canonical_ids)
    canonical_duplicates = [
        name for name, count in Counter(canonical_ids).items() if count > 1
    ]

    vcf_paths = enumerate_vcf_files()

    vcf_id_by_path = {}
    unmatched_filenames = []
    for path in vcf_paths:
        extracted_id, matched = extract_id(path)
        if not matched:
            unmatched_filenames.append(path.as_posix())
            continue
        vcf_id_by_path[path.as_posix()] = extracted_id

    vcf_ids = list(vcf_id_by_path.values())
    vcf_id_counts = Counter(vcf_ids)
    vcf_duplicate_ids = [name for name, count in vcf_id_counts.items() if count > 1]
    unique_vcf_ids = set(vcf_ids)

    missing_vcf = sorted(canonical_id_set - unique_vcf_ids)
    extra_vcf = sorted(unique_vcf_ids - canonical_id_set)

    # deterministic canonical-order -> VCF-path mapping (only meaningful if 1:1)
    path_by_id = {}
    for path_str, sample_id in vcf_id_by_path.items():
        path_by_id.setdefault(sample_id, []).append(path_str)

    is_exact_1to1 = (
        len(unmatched_filenames) == 0
        and len(canonical_duplicates) == 0
        and len(vcf_duplicate_ids) == 0
        and len(missing_vcf) == 0
        and len(extra_vcf) == 0
        and len(vcf_paths) == EXPECTED_N_CANONICAL
        and len(canonical_ids) == EXPECTED_N_CANONICAL
    )

    ordered_vcf_paths = None
    if is_exact_1to1:
        ordered_vcf_paths = [path_by_id[name][0] for name in canonical_ids]

    report = {
        "filename_convention": {
            "pattern": r"^[A-Za-z0-9]+_MT\.vcf\.gz$",
            "example": "ERR036186_MT.vcf.gz -> canonical ID 'ERR036186'",
            "observed_prefixes": sorted({sid[:3] for sid in vcf_ids if len(sid) >= 3}),
            "vcf_directory": VCF_DIR.as_posix(),
            "subdirectories_present": any(p.parent != VCF_DIR for p in vcf_paths),
        },
        "counts": {
            "expected_canonical_ids": EXPECTED_N_CANONICAL,
            "total_canonical_ids": len(canonical_ids),
            "total_vcf_files": len(vcf_paths),
            "unique_vcf_derived_ids": len(unique_vcf_ids),
            "unmatched_filenames": len(unmatched_filenames),
        },
        "discrepancies": {
            "canonical_ids_missing_a_vcf": missing_vcf,
            "canonical_ids_missing_a_vcf_count": len(missing_vcf),
            "vcf_ids_not_in_canonical": extra_vcf,
            "vcf_ids_not_in_canonical_count": len(extra_vcf),
            "duplicate_vcf_derived_ids": vcf_duplicate_ids,
            "duplicate_canonical_ids": canonical_duplicates,
            "unmatched_filenames_list": unmatched_filenames,
        },
        "exact_1to1_mapping": is_exact_1to1,
        "canonical_order_safely_mappable_to_vcf_path": is_exact_1to1,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["counts"], indent=2))
    print(json.dumps({"exact_1to1_mapping": is_exact_1to1}, indent=2))
    if not is_exact_1to1:
        print()
        print("MAPPING IS NOT EXACT -- see", OUT_REPORT, "for full discrepancy detail.")
        print(json.dumps(report["discrepancies"], indent=2)[:2000])
    print()
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
