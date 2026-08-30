"""Step 3b: cross-check the 157-mutation catalog against VCF-annotated calls.

Goal: determine whether the resistance mutations encoded in features.csv
(from the AfroTB XLSX workbook) are actually observable in the
SnpEff-annotated VCFs in data/raw/Afro_TB/AFRO_TB_ANNOTATION_VCF/, giving
an independent, non-fabricated consistency check between X_mutations
(features.csv), y_amr (y_amr.csv), and the VCF-derived genomic data Person
2 will build the phylogeny/graph from.

Read-only. Does not modify features.csv, y_amr.csv, sample_ids.csv, or any
VCF. Does not compute SNP distances or build a graph. Does not invent
mutation positions or resistance associations -- amino-acid 3-letter <->
1-letter conversion is the standard, universal genetic-code table (not
dataset-specific information), used only to compare two already-existing
representations of the same catalogued mutation name.

Matching method (explicit, so results aren't overstated):
  - Substitutions, synonymous changes, single-residue del/dup, and
    del/ins over a residue *range* are matched by exact string equality
    against the annotation's HGVS.p field (after normalizing the AfroTB
    catalog name into the same "p.<Xxx><pos><...>" shape SnpEff uses).
  - Frameshift ("...fs") entries are matched only by gene + amino-acid
    position, because SnpEff's frameshift HGVS.p notation carries extra
    detail (e.g. a terminal-stop offset) the catalog name doesn't encode
    -- this is reported as "position-based approximate match", not exact.
  - A catalog entry whose name doesn't fit any known AfroTB naming
    pattern is reported as "unparseable" and excluded from matching
    rather than guessed at.

For every isolate, per catalog mutation, this script checks: is the
mutation present in features.csv (1/0), and is a matching variant present
in that isolate's annotated VCF? Aggregated, not per-isolate, in the
output report (per-isolate detail would be 13,753 x 157 cells -- too large
and not needed to assess dataset-level compatibility).

Writes data/processed/mutation_matrix_vcf_crosscheck_report.json.
"""

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SAMPLE_IDS_CSV = PROCESSED_DIR / "sample_ids.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
DATASET_METADATA_JSON = PROCESSED_DIR / "dataset_metadata.json"
ANNOTATION_VCF_DIR = Path("data/raw/Afro_TB/AFRO_TB_ANNOTATION_VCF")
OUT_REPORT = PROCESSED_DIR / "mutation_matrix_vcf_crosscheck_report.json"

EXPECTED_N_ISOLATES = 13753
ANNOTATION_FILENAME_PATTERN = re.compile(r"^(?P<id>[A-Za-z0-9]+)_library1\.vcf\.gz$")

AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

RE_1LETTER_SUB = re.compile(r"^([A-Z])(\d+)([A-Z])$")
RE_3LETTER_SUB = re.compile(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
RE_3LETTER_FS = re.compile(r"^([A-Z][a-z]{2})(\d+)fs$")
RE_3LETTER_SINGLE_DEL = re.compile(r"^([A-Z][a-z]{2})(\d+)del$")
RE_3LETTER_SINGLE_DUP = re.compile(r"^([A-Z][a-z]{2})(\d+)dup$")
RE_3LETTER_RANGE = re.compile(
    r"^([A-Z][a-z]{2}\d+) ([A-Z][a-z]{2}\d+)((?:del)|(?:ins[A-Za-z]+))$"
)


def parse_catalog_variation(variation):
    """Normalize an AfroTB mutation-name suffix (e.g. 'S315N', 'Val130fs')
    into a (match_type, normalized_hgvs_p_or_None, gene_position_or_None)
    tuple. match_type in {'exact', 'fs_position', 'unparseable'}.
    """
    m = RE_1LETTER_SUB.match(variation)
    if m:
        aa1, pos, aa2 = m.groups()
        if aa1 in AA_1TO3 and aa2 in AA_1TO3:
            return "exact", f"p.{AA_1TO3[aa1]}{pos}{AA_1TO3[aa2]}", None

    m = RE_3LETTER_SUB.match(variation)
    if m:
        return "exact", f"p.{variation}", None

    m = RE_3LETTER_SINGLE_DEL.match(variation)
    if m:
        return "exact", f"p.{variation}", None

    m = RE_3LETTER_SINGLE_DUP.match(variation)
    if m:
        return "exact", f"p.{variation}", None

    m = RE_3LETTER_RANGE.match(variation)
    if m:
        start, end, op = m.groups()
        return "exact", f"p.{start}_{end}{op}", None

    m = RE_3LETTER_FS.match(variation)
    if m:
        _aa, pos = m.groups()
        return "fs_position", None, int(pos)

    return "unparseable", None, None


def load_catalog():
    with open(DATASET_METADATA_JSON, encoding="utf-8") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]

    catalog = []
    for name in feature_names:
        gene, variation = name.split(" ", 1)
        match_type, norm_hgvs_p, fs_pos = parse_catalog_variation(variation)
        catalog.append({
            "feature_name": name,
            "gene": gene,
            "variation": variation,
            "match_type": match_type,
            "normalized_hgvs_p": norm_hgvs_p,
            "fs_position": fs_pos,
        })
    return catalog


def load_sample_ids():
    with open(SAMPLE_IDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


def load_features(canonical_ids):
    with open(FEATURES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[1:]
        rows = {row[0]: [int(v) for v in row[1:]] for row in reader}
    if set(rows.keys()) != set(canonical_ids):
        raise AssertionError("features.csv IDs do not match sample_ids.csv IDs.")
    return feature_names, rows


HGVS_P_POS_PATTERN = re.compile(r"^p\.[A-Za-z]{3}(\d+)")


def extract_isolate_observations(path, target_genes):
    """Scan one annotated VCF; return (exact_hgvs_p_by_gene, fs_positions_by_gene)."""
    exact_by_gene = defaultdict(set)
    fs_positions_by_gene = defaultdict(set)
    gene_markers = {gene: f"|{gene}|" for gene in target_genes}

    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "ANN=" not in line:
                continue
            info = line.split("\t", 8)[7] if line.count("\t") >= 7 else ""
            idx = info.find("ANN=")
            if idx == -1:
                continue
            ann_val = info[idx + 4:]
            semi = ann_val.find(";")
            if semi != -1:
                ann_val = ann_val[:semi]

            if not any(marker in ann_val for marker in gene_markers.values()):
                continue

            for eff in ann_val.split(","):
                parts = eff.split("|")
                if len(parts) <= 10:
                    continue
                gene = parts[3]
                if gene not in target_genes:
                    continue
                hgvs_p = parts[10]
                if not hgvs_p:
                    continue
                exact_by_gene[gene].add(hgvs_p)
                if "fs" in hgvs_p:
                    m = HGVS_P_POS_PATTERN.match(hgvs_p)
                    if m:
                        fs_positions_by_gene[gene].add(int(m.group(1)))

    return exact_by_gene, fs_positions_by_gene


def main():
    catalog = load_catalog()
    target_genes = sorted({c["gene"] for c in catalog})

    canonical_ids = load_sample_ids()
    if len(canonical_ids) != EXPECTED_N_ISOLATES:
        raise AssertionError(f"sample_ids.csv does not have {EXPECTED_N_ISOLATES} IDs.")
    feature_names, features_by_id = load_features(canonical_ids)

    feature_col_index = {name: i for i, name in enumerate(feature_names)}
    for c in catalog:
        if c["feature_name"] not in feature_col_index:
            raise AssertionError(f"Catalog entry {c['feature_name']!r} not found in features.csv header.")

    ann_paths = sorted(ANNOTATION_VCF_DIR.rglob("*.vcf.gz"))
    if len(ann_paths) != EXPECTED_N_ISOLATES:
        raise AssertionError(
            f"Expected {EXPECTED_N_ISOLATES} annotation VCFs, found {len(ann_paths)}."
        )

    id_to_path = {}
    for p in ann_paths:
        m = ANNOTATION_FILENAME_PATTERN.match(p.name)
        if not m:
            raise AssertionError(f"Annotation VCF filename does not match convention: {p.name}")
        id_to_path[m.group("id")] = p
    if set(id_to_path.keys()) != set(canonical_ids):
        raise AssertionError("Annotation VCF IDs do not match canonical sample_ids.csv IDs.")

    # per-catalog-entry aggregate counters
    stats = {
        c["feature_name"]: {
            "match_type": c["match_type"],
            "n_positive_in_features": 0,
            "n_positive_and_vcf_confirms": 0,
            "n_negative_in_features_but_vcf_shows_variant": 0,
        }
        for c in catalog
    }
    n_unparseable = sum(1 for c in catalog if c["match_type"] == "unparseable")

    exact_lookup = defaultdict(list)     # gene -> [(feature_name, normalized_hgvs_p)]
    fs_lookup = defaultdict(list)        # gene -> [(feature_name, fs_position)]
    for c in catalog:
        if c["match_type"] == "exact":
            exact_lookup[c["gene"]].append((c["feature_name"], c["normalized_hgvs_p"]))
        elif c["match_type"] == "fs_position":
            fs_lookup[c["gene"]].append((c["feature_name"], c["fs_position"]))

    n_isolates_processed = 0
    for name in canonical_ids:
        path = id_to_path[name]
        exact_by_gene, fs_positions_by_gene = extract_isolate_observations(path, set(target_genes))
        feature_row = features_by_id[name]

        for gene, entries in exact_lookup.items():
            observed = exact_by_gene.get(gene, set())
            for feature_name, norm_hgvs_p in entries:
                is_positive = bool(feature_row[feature_col_index[feature_name]])
                vcf_confirms = norm_hgvs_p in observed
                if is_positive:
                    stats[feature_name]["n_positive_in_features"] += 1
                    if vcf_confirms:
                        stats[feature_name]["n_positive_and_vcf_confirms"] += 1
                elif vcf_confirms:
                    stats[feature_name]["n_negative_in_features_but_vcf_shows_variant"] += 1

        for gene, entries in fs_lookup.items():
            observed_positions = fs_positions_by_gene.get(gene, set())
            for feature_name, fs_pos in entries:
                is_positive = bool(feature_row[feature_col_index[feature_name]])
                vcf_confirms = fs_pos in observed_positions
                if is_positive:
                    stats[feature_name]["n_positive_in_features"] += 1
                    if vcf_confirms:
                        stats[feature_name]["n_positive_and_vcf_confirms"] += 1
                elif vcf_confirms:
                    stats[feature_name]["n_negative_in_features_but_vcf_shows_variant"] += 1

        n_isolates_processed += 1

    per_mutation_report = {}
    for feature_name, s in stats.items():
        n_pos = s["n_positive_in_features"]
        concordance = (s["n_positive_and_vcf_confirms"] / n_pos) if n_pos > 0 else None
        per_mutation_report[feature_name] = {
            **s,
            "concordance_rate": round(concordance, 4) if concordance is not None else None,
        }

    matchable = [v for v in per_mutation_report.values() if v["match_type"] != "unparseable"]
    matchable_with_positives = [v for v in matchable if v["n_positive_in_features"] > 0]
    concordance_rates = [v["concordance_rate"] for v in matchable_with_positives if v["concordance_rate"] is not None]

    summary = {
        "n_catalog_mutations": len(catalog),
        "n_matchable_exact": sum(1 for c in catalog if c["match_type"] == "exact"),
        "n_matchable_fs_position_approx": sum(1 for c in catalog if c["match_type"] == "fs_position"),
        "n_unparseable": n_unparseable,
        "unparseable_feature_names": [c["feature_name"] for c in catalog if c["match_type"] == "unparseable"],
        "n_mutations_with_at_least_one_positive_isolate": len(matchable_with_positives),
        "mean_concordance_rate_across_matchable_mutations": (
            round(sum(concordance_rates) / len(concordance_rates), 4) if concordance_rates else None
        ),
        "n_mutations_zero_concordance": sum(1 for r in concordance_rates if r == 0.0),
        "n_mutations_full_concordance": sum(1 for r in concordance_rates if r == 1.0),
    }

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Cross-check whether the 157 resistance mutations encoded in "
            "features.csv (X_mutations) are independently observable in "
            "the SnpEff-annotated VCFs, as a consistency check between "
            "X_mutations, y_amr, and the VCF-derived genomic data."
        ),
        "matching_method": (
            "Substitutions/synonymous/single-residue del or dup/range "
            "del or ins are matched by exact string equality against the "
            "normalized HGVS.p annotation. Frameshift ('...fs') entries "
            "are matched only by gene + amino-acid position (labeled "
            "'fs_position' match_type) since SnpEff's frameshift notation "
            "carries extra detail the catalog name does not encode -- "
            "this is an approximate, position-based match, not exact. "
            "Entries not matching any known AfroTB naming pattern are "
            "'unparseable' and excluded from matching (not guessed at)."
        ),
        "n_isolates_processed": n_isolates_processed,
        "expected_n_isolates": EXPECTED_N_ISOLATES,
        "target_genes": target_genes,
        "summary": summary,
        "per_mutation": per_mutation_report,
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Isolates processed: {n_isolates_processed} (expected {EXPECTED_N_ISOLATES})")
    print("Summary:", json.dumps(summary, indent=2)[:2000])
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
