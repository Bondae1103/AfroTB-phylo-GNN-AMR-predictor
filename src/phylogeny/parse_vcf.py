"""Stream a single per-isolate VCF and extract confidently-called SNPs.

Built on the exact streaming pattern already proven at full-dataset scale in
scripts/audit_vcf_content.py: text-mode gzip, per-line parsing, no VCF
library. SNP-only (indels excluded -- standard practice for bacterial
SNP-distance phylogenetics, since indel positions are ambiguous to align
across isolates). REF/ALT are uppercased before use (README Sec. 8: ~108K/109K
records use lowercase bases as a formatting quirk, not a validity issue).
FILTER is ignored entirely (confirmed 100% uninformative -- always ".").
"""

import gzip
import re

VALID_BASE_CHARS_CASE_INSENSITIVE = re.compile(r"^[ACGTNacgtn]+$")
MISSING_GT_TOKENS = {".", "./.", ".|.", ""}


def extract_snp_calls(vcf_path):
    """Return (confident_calls, missing_positions) for one isolate's VCF.

    confident_calls: set of (pos: int, alt: str) for SNP records with a
    confident non-reference genotype call (GT token "1"; ploidy is 1 in
    this dataset -- see README Sec. 4.3).
    missing_positions: set of pos (int) where the record's GT token was an
    explicit missing call (e.g. "./.").
    Non-SNP records (after splitting multiallelic ALT on comma) are skipped.
    """
    confident_calls = set()
    missing_positions = set()

    with gzip.open(vcf_path, "rt", encoding="utf-8", errors="strict") as f:
        header_columns = None
        gt_index = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_columns = line.split("\t")
                continue
            if header_columns is None:
                continue
            fields = line.split("\t")
            if len(fields) != len(header_columns):
                continue

            pos_str, ref, alt_raw = fields[1], fields[3], fields[4]
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            ref = ref.upper()

            format_field = fields[8] if len(fields) > 8 else None
            sample_field = fields[9] if len(fields) > 9 else None
            if format_field is None or sample_field is None:
                continue
            format_subfields = format_field.split(":")
            if gt_index is None:
                gt_index = format_subfields.index("GT") if "GT" in format_subfields else -1
            if gt_index == -1:
                continue
            sample_subfields = sample_field.split(":")
            if gt_index >= len(sample_subfields):
                continue
            gt_token = sample_subfields[gt_index]

            if gt_token in MISSING_GT_TOKENS:
                missing_positions.add(pos)
                continue
            if gt_token != "1":
                # Not a confident single-alt haploid call (unexpected format) -- skip.
                continue

            for alt in alt_raw.split(","):
                alt = alt.upper()
                if len(ref) == 1 and len(alt) == 1 and VALID_BASE_CHARS_CASE_INSENSITIVE.match(alt):
                    confident_calls.add((pos, alt))
                    break  # GT=1 refers to the first ALT allele only

    return confident_calls, missing_positions
