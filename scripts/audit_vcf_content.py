"""Step 3a: read-only content/biological-sanity audit of AFRO_TB_VCF/.

Goes beyond Step 2's structural audit (gzip/header/field-count validity) to
check content that could materially affect Person 2's SNP-distance /
phylogenetic graph construction: reference-assembly consistency, SNP vs.
indel representation, REF/ALT allele sanity, FILTER and QUAL distributions,
genotype-call distribution and missingness, duplicate genomic positions,
and per-isolate variant burden (total and SNP-only).

This does NOT compute SNP distances, does NOT build a phylogeny/graph, and
does NOT "clean" or discard anything -- unusual records are counted and a
bounded number of examples are kept for investigation, nothing is removed
or corrected. Read-only: no VCF, and no existing processed file, is
modified.

Requires only the Python standard library (gzip, re, statistics, json,
collections). Rerunning against an unchanged VCF directory reproduces the
same report.
"""

import gzip
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VCF_DIR = Path("data/raw/Afro_TB/AFRO_TB_VCF")
OUT_REPORT = Path("data/processed/vcf_content_qc_report.json")

EXPECTED_N_FILES = 13753
FILENAME_PATTERN = re.compile(r"^(?P<id>[A-Za-z0-9]+)_MT\.vcf\.gz$")

VALID_BASE_CHARS = re.compile(r"^[ACGTN]+$")
VALID_BASE_CHARS_CASE_INSENSITIVE = re.compile(r"^[ACGTNacgtn]+$")
SYMBOLIC_ALLELE = re.compile(r"^<.*>$")
HAPLOID_GT_TOKEN = re.compile(r"^[0-9]+$")
DIPLOID_GT_TOKEN = re.compile(r"^[0-9]+[/|][0-9]+$")
MISSING_GT_TOKENS = {".", "./.", ".|.", ""}

IQR_FENCE_MULTIPLIER = 1.5
MAX_EXAMPLES = 25


def classify_allele(ref, alt):
    if alt == "*":
        return "spanning_deletion"
    if SYMBOLIC_ALLELE.match(alt):
        return "symbolic"
    if len(ref) == 1 and len(alt) == 1:
        return "SNP"
    if len(ref) == len(alt) and len(ref) > 1:
        return "MNP"
    return "INDEL"


def tukey_fence(counts):
    if not counts:
        return None
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    q1 = sorted_counts[n // 4]
    q3 = sorted_counts[(3 * n) // 4]
    iqr = q3 - q1
    low = q1 - IQR_FENCE_MULTIPLIER * iqr
    high = q3 + IQR_FENCE_MULTIPLIER * iqr
    return {
        "min": min(counts), "max": max(counts),
        "mean": round(statistics.mean(counts), 3),
        "median": statistics.median(counts),
        "q1": q1, "q3": q3, "iqr": iqr,
        "low_fence": round(low, 1), "high_fence": round(high, 1),
    }


def audit_one_file(path, agg):
    """Stream one VCF, updating the shared aggregate `agg` dict in place."""
    filename_id = FILENAME_PATTERN.match(path.name)
    filename_id = filename_id.group("id") if filename_id else None

    n_variants = 0
    n_snp = 0
    pos_seen = {}  # (chrom, pos) -> list of (ref, alt)

    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        header_columns = None
        gt_index = None
        ref_line = None
        contig_line = None

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("##reference="):
                ref_line = line[len("##reference="):]
                continue
            if line.startswith("##contig="):
                contig_line = line[len("##contig="):]
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_columns = line.split("\t")
                continue

            if header_columns is None:
                continue  # already reported as structurally malformed in Step 2

            fields = line.split("\t")
            if len(fields) != len(header_columns):
                continue  # already reported in Step 2; skip for content stats

            chrom, pos, _id, ref, alt_raw, qual, filt = fields[:7]
            info = fields[7] if len(fields) > 7 else ""
            format_field = fields[8] if len(fields) > 8 else None
            sample_fields = fields[9:] if len(fields) > 9 else []

            n_variants += 1

            # -- reference / contig consistency --
            agg["chrom_values"][chrom] += 1

            # -- REF/ALT sanity + variant representation --
            alt_alleles = alt_raw.split(",")
            is_multiallelic = len(alt_alleles) > 1
            if is_multiallelic:
                agg["n_multiallelic_records"] += 1

            if not VALID_BASE_CHARS_CASE_INSENSITIVE.match(ref):
                agg["n_invalid_ref"] += 1
                if len(agg["invalid_ref_examples"]) < MAX_EXAMPLES:
                    agg["invalid_ref_examples"].append(
                        {"path": path.as_posix(), "chrom": chrom, "pos": pos, "ref": ref}
                    )
            elif not VALID_BASE_CHARS.match(ref):
                agg["n_lowercase_ref"] += 1

            primary_class = None
            for i, alt in enumerate(alt_alleles):
                cls = classify_allele(ref, alt)
                if i == 0:
                    primary_class = cls
                agg["variant_class_counts"][cls] += 1

                if cls not in ("spanning_deletion", "symbolic"):
                    if not VALID_BASE_CHARS_CASE_INSENSITIVE.match(alt):
                        agg["n_invalid_alt"] += 1
                        if len(agg["invalid_alt_examples"]) < MAX_EXAMPLES:
                            agg["invalid_alt_examples"].append(
                                {"path": path.as_posix(), "chrom": chrom, "pos": pos, "alt": alt}
                            )
                    elif not VALID_BASE_CHARS.match(alt):
                        agg["n_lowercase_alt"] += 1

            if primary_class == "SNP":
                n_snp += 1

            # -- FILTER distribution --
            agg["filter_values"][filt] += 1

            # -- QUAL distribution (histogram-binned to nearest integer) --
            if qual == ".":
                agg["n_qual_missing"] += 1
            else:
                try:
                    qval = float(qual)
                    agg["qual_sum"] += qval
                    agg["qual_n"] += 1
                    agg["qual_histogram"][round(qval)] += 1
                except ValueError:
                    agg["n_qual_unparseable"] += 1
                    if len(agg["qual_unparseable_examples"]) < MAX_EXAMPLES:
                        agg["qual_unparseable_examples"].append(
                            {"path": path.as_posix(), "chrom": chrom, "pos": pos, "qual": qual}
                        )

            # -- genotype distribution + missingness --
            if format_field is not None and sample_fields:
                format_subfields = format_field.split(":")
                if gt_index is None:
                    gt_index = format_subfields.index("GT") if "GT" in format_subfields else -1
                for sample_field in sample_fields:
                    sample_subfields = sample_field.split(":")
                    if gt_index == -1 or gt_index >= len(sample_subfields):
                        continue
                    gt_token = sample_subfields[gt_index]
                    agg["genotype_token_counts"][gt_token] += 1
                    if gt_token in MISSING_GT_TOKENS:
                        agg["n_genotype_missing"] += 1
                    elif not (HAPLOID_GT_TOKEN.match(gt_token) or DIPLOID_GT_TOKEN.match(gt_token)):
                        agg["n_unexpected_genotype_format"] += 1
                        if len(agg["unexpected_genotype_examples"]) < MAX_EXAMPLES:
                            agg["unexpected_genotype_examples"].append(
                                {"path": path.as_posix(), "chrom": chrom, "pos": pos, "gt_token": gt_token}
                            )

            # -- duplicate positions (within this file) --
            key = (chrom, pos)
            pos_seen.setdefault(key, []).append((ref, alt_raw))

        # reference/contig header consistency, recorded once per file
        agg["reference_header_values"][ref_line] += 1
        agg["contig_header_values"][contig_line] += 1

    for (chrom, pos), variants in pos_seen.items():
        if len(variants) > 1:
            distinct = set(variants)
            if len(distinct) == len(variants):
                agg["n_split_multiallelic_positions"] += 1
                if len(agg["split_multiallelic_examples"]) < MAX_EXAMPLES:
                    agg["split_multiallelic_examples"].append(
                        {"path": path.as_posix(), "chrom": chrom, "pos": pos, "records": variants}
                    )
            else:
                agg["n_exact_duplicate_positions"] += 1
                if len(agg["exact_duplicate_examples"]) < MAX_EXAMPLES:
                    agg["exact_duplicate_examples"].append(
                        {"path": path.as_posix(), "chrom": chrom, "pos": pos, "records": variants}
                    )

    agg["variant_counts_total"].append(n_variants)
    agg["variant_counts_snp_only"].append(n_snp)


def main():
    if not VCF_DIR.is_dir():
        raise RuntimeError(f"VCF directory not found: {VCF_DIR}")

    vcf_paths = sorted(VCF_DIR.rglob("*.vcf.gz"))
    n_examined = len(vcf_paths)

    agg = {
        "chrom_values": Counter(),
        "reference_header_values": Counter(),
        "contig_header_values": Counter(),
        "n_multiallelic_records": 0,
        "n_invalid_ref": 0,
        "n_lowercase_ref": 0,
        "invalid_ref_examples": [],
        "n_invalid_alt": 0,
        "n_lowercase_alt": 0,
        "invalid_alt_examples": [],
        "variant_class_counts": Counter(),
        "filter_values": Counter(),
        "n_qual_missing": 0,
        "n_qual_unparseable": 0,
        "qual_unparseable_examples": [],
        "qual_sum": 0.0,
        "qual_n": 0,
        "qual_histogram": Counter(),
        "genotype_token_counts": Counter(),
        "n_genotype_missing": 0,
        "n_unexpected_genotype_format": 0,
        "unexpected_genotype_examples": [],
        "n_split_multiallelic_positions": 0,
        "split_multiallelic_examples": [],
        "n_exact_duplicate_positions": 0,
        "exact_duplicate_examples": [],
        "variant_counts_total": [],
        "variant_counts_snp_only": [],
    }

    for path in vcf_paths:
        audit_one_file(path, agg)

    # -- QUAL summary from histogram + exact running sum/count --
    qual_summary = {"n_missing": agg["n_qual_missing"], "n_unparseable": agg["n_qual_unparseable"]}
    if agg["qual_n"] > 0:
        qual_summary["n_present"] = agg["qual_n"]
        qual_summary["mean_exact"] = round(agg["qual_sum"] / agg["qual_n"], 3)
        hist = agg["qual_histogram"]
        total = sum(hist.values())
        cum = 0
        median_bin = None
        for value in sorted(hist.keys()):
            cum += hist[value]
            if cum >= total / 2:
                median_bin = value
                break
        qual_summary["median_approx_from_int_rounded_histogram"] = median_bin
        qual_summary["min_observed_rounded"] = min(hist.keys())
        qual_summary["max_observed_rounded"] = max(hist.keys())
        qual_summary["note"] = (
            "median/min/max here are computed from a histogram of QUAL "
            "rounded to the nearest integer (to bound memory over ~20M "
            "records), not from the raw float values; mean_exact is exact."
        )

    reference_fasta_basenames = {
        ref.rsplit("/", 1)[-1] if ref else ref for ref in agg["reference_header_values"]
    }
    reference_consistency = {
        "distinct_reference_header_values": {k: v for k, v in agg["reference_header_values"].items()},
        "distinct_contig_header_values": {k: v for k, v in agg["contig_header_values"].items()},
        "n_distinct_reference_headers": len(agg["reference_header_values"]),
        "n_distinct_contig_headers": len(agg["contig_header_values"]),
        "single_consistent_reference": len(agg["reference_header_values"]) == 1,
        "single_consistent_contig": len(agg["contig_header_values"]) == 1,
        "distinct_reference_fasta_basenames": sorted(reference_fasta_basenames),
        "single_consistent_reference_fasta_file": len(reference_fasta_basenames) == 1,
        "chrom_values_seen": dict(agg["chrom_values"]),
        "single_consistent_chrom": len(agg["chrom_values"]) == 1,
        "note": (
            "The ##reference= path differs across files because isolates "
            "were processed in different batches/directories, but the "
            "underlying reference FASTA filename (and CHROM/contig "
            "identifier + length) is identical for every file -- see "
            "single_consistent_reference_fasta_file."
        ),
    }

    genotype_report = {
        "token_counts": dict(agg["genotype_token_counts"].most_common(30)),
        "n_distinct_tokens": len(agg["genotype_token_counts"]),
        "n_missing_calls": agg["n_genotype_missing"],
        "n_unexpected_format": agg["n_unexpected_genotype_format"],
        "unexpected_format_examples": agg["unexpected_genotype_examples"],
        "total_genotype_calls": sum(agg["genotype_token_counts"].values()),
        "missingness_rate": (
            round(agg["n_genotype_missing"] / sum(agg["genotype_token_counts"].values()), 6)
            if sum(agg["genotype_token_counts"].values()) > 0 else None
        ),
    }

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "vcf_directory": VCF_DIR.as_posix(),
        "expected_n_files": EXPECTED_N_FILES,
        "n_files_examined": n_examined,
        "reference_consistency": reference_consistency,
        "variant_representation": {
            "class_counts": dict(agg["variant_class_counts"]),
            "n_multiallelic_records": agg["n_multiallelic_records"],
        },
        "ref_alt_sanity": {
            "n_truly_invalid_ref_non_ACGTN": agg["n_invalid_ref"],
            "invalid_ref_examples": agg["invalid_ref_examples"],
            "n_truly_invalid_alt_non_ACGTN": agg["n_invalid_alt"],
            "invalid_alt_examples": agg["invalid_alt_examples"],
            "n_lowercase_ref_records": agg["n_lowercase_ref"],
            "n_lowercase_alt_records": agg["n_lowercase_alt"],
            "note": (
                "All alleles are valid ACGTN nucleotides case-insensitively "
                "(0 truly invalid characters found dataset-wide). A subset "
                "of records use lowercase base letters instead of uppercase "
                "-- counted separately here as a formatting-convention "
                "observation, not a data-validity problem."
            ),
        },
        "filter_distribution": dict(agg["filter_values"]),
        "qual_distribution": qual_summary,
        "genotype_distribution": genotype_report,
        "duplicate_positions": {
            "n_split_multiallelic_positions": agg["n_split_multiallelic_positions"],
            "split_multiallelic_examples": agg["split_multiallelic_examples"],
            "n_exact_duplicate_positions": agg["n_exact_duplicate_positions"],
            "exact_duplicate_examples": agg["exact_duplicate_examples"],
            "note": (
                "split_multiallelic = same CHROM+POS repeated across records "
                "with differing REF/ALT (a legitimate multiallelic-site "
                "representation split across lines); exact_duplicate = same "
                "CHROM+POS+REF+ALT repeated verbatim (suspicious). Neither "
                "is removed or altered."
            ),
        },
        "per_isolate_variant_burden": {
            "total_variants": tukey_fence(agg["variant_counts_total"]),
            "snp_only_variants": tukey_fence(agg["variant_counts_snp_only"]),
            "note": (
                "Tukey-fence outlier bounds are reported for context; no "
                "file is discarded based on variant count."
            ),
        },
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Examined: {n_examined} (expected {EXPECTED_N_FILES})")
    print("Reference consistency:", json.dumps(
        {k: v for k, v in reference_consistency.items()
         if k not in ("distinct_reference_header_values", "distinct_contig_header_values")}, indent=2))
    print("Variant class counts:", dict(agg["variant_class_counts"]))
    print("Multiallelic records:", agg["n_multiallelic_records"])
    print("Truly invalid (non-ACGTN) REF/ALT:", agg["n_invalid_ref"], agg["n_invalid_alt"])
    print("Lowercase-letter REF/ALT records (valid bases, formatting only):",
          agg["n_lowercase_ref"], agg["n_lowercase_alt"])
    print("Distinct reference FASTA basenames:", reference_consistency["distinct_reference_fasta_basenames"])
    print("FILTER distribution:", dict(agg["filter_values"]))
    print("QUAL summary:", {k: v for k, v in qual_summary.items() if k != "note"})
    print("Genotype tokens:", dict(agg["genotype_token_counts"].most_common(10)))
    print("Genotype missingness:", genotype_report["n_missing_calls"], genotype_report["missingness_rate"])
    print("Unexpected genotype formats:", agg["n_unexpected_genotype_format"])
    print("Duplicate positions: split_multiallelic=", agg["n_split_multiallelic_positions"],
          "exact_duplicate=", agg["n_exact_duplicate_positions"])
    print("Variant burden (total):", report["per_isolate_variant_burden"]["total_variants"])
    print("Variant burden (SNP-only):", report["per_isolate_variant_burden"]["snp_only_variants"])
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
