"""Derive data/processed/catalogue_sites.json by matching the 157 catalog
mutations against SnpEff-annotated VCFs and core SNP matrix coordinates.

Uses targeted lookup: instead of scanning all 13,753 VCFs, it inspects only
isolates known to carry each mutation (from features.csv).
"""

import csv
import gzip
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ANNOTATION_VCF_DIR = REPO_ROOT / "data" / "raw" / "Afro_TB" / "AFRO_TB_ANNOTATION_VCF"
OUTPUT_JSON = PROCESSED_DIR / "catalogue_sites.json"

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
RE_3LETTER_RANGE = re.compile(r"^([A-Z][a-z]{2}\d+) ([A-Z][a-z]{2}\d+)((?:del)|(?:ins[A-Za-z]+))$")
HGVS_P_POS_PATTERN = re.compile(r"^p\.[A-Za-z]{3}(\d+)")


def parse_variation(variation):
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


def load_core_snp_lookup():
    """Returns mapping (pos: int, alt: str) -> column_index in snp_matrix.npz."""
    sites_csv = PROCESSED_DIR / "snp_matrix_sites.csv"
    lookup = {}
    col_idx = 0
    with open(sites_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # pos,alt,global_call_count,missing_rate,kept_in_core
        for row in reader:
            pos = int(row[0])
            alt = row[1]
            kept = row[4].strip().lower() == "true"
            if kept:
                lookup[(pos, alt)] = col_idx
                col_idx += 1
    return lookup


def find_variant_site(vcf_path, gene, match_type, target_hgvs_p, target_fs_pos):
    """Scan one isolate's annotated VCF for a matching variant; return (pos, alt) or None."""
    if not vcf_path.exists():
        return None
    with gzip.open(vcf_path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "ANN=" not in line or gene not in line:
                continue
            fields = line.split("\t")
            pos = int(fields[1])
            alts = fields[4].upper().split(",")
            info = fields[7]
            idx = info.find("ANN=")
            if idx == -1:
                continue
            ann_val = info[idx + 4:]
            semi = ann_val.find(";")
            if semi != -1:
                ann_val = ann_val[:semi]

            for eff in ann_val.split(","):
                parts = eff.split("|")
                if len(parts) <= 10:
                    continue
                ann_gene = parts[3]
                if ann_gene != gene:
                    continue
                hgvs_p = parts[10]
                if not hgvs_p:
                    continue

                matched = False
                if match_type == "exact" and hgvs_p == target_hgvs_p:
                    matched = True
                elif match_type == "fs_position" and "fs" in hgvs_p:
                    m = HGVS_P_POS_PATTERN.match(hgvs_p)
                    if m and int(m.group(1)) == target_fs_pos:
                        matched = True

                if matched:
                    # Return pos and first matching alt
                    return pos, alts[0]
    return None


def main():
    print("Loading features and metadata...")
    with open(PROCESSED_DIR / "dataset_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    with open(PROCESSED_DIR / "features.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    sample_ids = [r[0] for r in rows]
    # Matrix of presence
    # map feature_name -> list of sample_ids with positive value
    pos_samples_by_feat = {}
    for col_idx, feat in enumerate(feature_names, start=1):
        pos_samples = [r[0] for r in rows if r[col_idx] == "1"]
        pos_samples_by_feat[feat] = pos_samples

    print("Loading core SNP index from snp_matrix_sites.csv...")
    core_lookup = load_core_snp_lookup()
    print(f"Loaded {len(core_lookup)} core SNP sites.")

    mutation_to_site = {}
    matched_core_indices = set()
    unmatched = []
    zero_positives = []
    not_in_core = []

    print(f"Resolving {len(feature_names)} catalogue mutations...")
    for feat in feature_names:
        gene, variation = feat.split(" ", 1)
        match_type, norm_hgvs_p, fs_pos = parse_variation(variation)
        pos_samples = pos_samples_by_feat[feat]

        if not pos_samples:
            zero_positives.append(feat)
            continue

        site_found = None
        # Try up to 5 positive samples to find coordinate
        for sample_id in pos_samples[:5]:
            vcf_path = ANNOTATION_VCF_DIR / f"{sample_id}_library1.vcf.gz"
            site = find_variant_site(vcf_path, gene, match_type, norm_hgvs_p, fs_pos)
            if site:
                site_found = site
                break

        if site_found:
            pos, alt = site_found
            if (pos, alt) in core_lookup:
                col_idx = core_lookup[(pos, alt)]
                mutation_to_site[feat] = col_idx
                matched_core_indices.add(col_idx)
            else:
                not_in_core.append({"feature": feat, "pos": pos, "alt": alt})
        else:
            unmatched.append(feat)

    site_indices = sorted(matched_core_indices)

    diagnostics = {
        "n_total_catalogue_mutations": len(feature_names),
        "n_resolved_in_core_snp_matrix": len(mutation_to_site),
        "n_unique_core_columns_mapped": len(site_indices),
        "n_zero_positive_isolates": len(zero_positives),
        "zero_positive_mutations": zero_positives,
        "n_not_in_core_snps": len(not_in_core),
        "not_in_core_mutations": not_in_core,
        "n_unmatched_vcf": len(unmatched),
        "unmatched_mutations": unmatched,
    }

    payload = {
        "source": "Afro-TB SnpEff-annotated VCFs cross-referenced with snp_matrix_sites.csv",
        "description": "snp_matrix.npz column index of each catalogue mutation",
        "mutation_to_site": mutation_to_site,
        "site_indices": site_indices,
        "diagnostics": diagnostics,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")
    print(f"Resolved in core SNP matrix: {len(mutation_to_site)} / {len(feature_names)}")
    print(f"Unique core SNP columns: {len(site_indices)}")
    if not_in_core:
        print(f"Sites not in core SNP matrix (e.g. indels or >10% missing): {len(not_in_core)}")
    if zero_positives:
        print(f"Mutations with 0 positives dataset-wide: {len(zero_positives)}")


if __name__ == "__main__":
    main()
