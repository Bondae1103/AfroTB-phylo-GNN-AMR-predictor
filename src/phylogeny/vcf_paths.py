"""Resolve canonical sample IDs to raw VCF file paths.

Reuses the exact filename convention already validated end-to-end by
scripts/validate_vcf_mapping.py (see data/processed/vcf_mapping_report.json):
every isolate's VCF sits directly in data/raw/Afro_TB/AFRO_TB_VCF/ as
"<ID>_MT.vcf.gz", with a confirmed exact 1:1 mapping to sample_ids.csv (0
missing, 0 extra, 0 duplicates). This module does not re-derive that mapping;
it asserts the prior audit's result still holds before trusting the
convention, and fails loudly rather than silently re-checking 13,753 files
on every import.
"""

import json
from pathlib import Path

AFRO_TB_VCF_DIR = Path("data/raw/Afro_TB/AFRO_TB_VCF")
VCF_MAPPING_REPORT = Path("data/processed/vcf_mapping_report.json")


def assert_mapping_is_exact_1to1():
    """Raise RuntimeError unless the prior mapping audit confirmed exact 1:1."""
    with open(VCF_MAPPING_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)
    if not report.get("exact_1to1_mapping"):
        raise RuntimeError(
            f"{VCF_MAPPING_REPORT} does not confirm an exact 1:1 mapping -- "
            "re-run scripts/validate_vcf_mapping.py before trusting vcf_path_for()."
        )


def vcf_path_for(sample_id):
    """Return the VCF path for a canonical sample ID (does not check existence)."""
    return AFRO_TB_VCF_DIR / f"{sample_id}_MT.vcf.gz"
