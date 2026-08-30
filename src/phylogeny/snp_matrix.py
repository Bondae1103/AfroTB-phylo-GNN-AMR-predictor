"""Build a core-genome SNP matrix from per-isolate VCFs, in two streaming passes.

Never materializes a dense (N_isolates x N_sites) array. Pass 1 tallies, per
candidate SNP site, how many isolates confidently call it and how many
isolates have an explicit missing ("./.") genotype at that position; sites
above MAX_SITE_MISSING_RATE are dropped from the "core" set used for distance
computation (but the full, unfiltered site list is still reported for
transparency -- see build_site_index's return value). Pass 2 re-streams and
fills a sparse matrix restricted to the kept sites.

Limitation, stated rather than hidden: these are per-isolate variant-call
VCFs, not joint/gVCF all-sites calls, so a site absent from an isolate's VCF
cannot be distinguished from "matches reference" versus "no coverage" without
depth/callable-sites data this dataset doesn't provide. The missingness
filter (default 10%) bounds the impact of collapsing those two cases to 0,
rather than pretending the collapse doesn't happen.
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp

from .parse_vcf import extract_snp_calls
from .vcf_paths import vcf_path_for

MAX_SITE_MISSING_RATE = 0.10


def build_site_index(sample_ids, max_site_missing_rate=MAX_SITE_MISSING_RATE):
    """Pass 1: tally global call/missing counts per SNP site across sample_ids.

    Returns a dict with:
      all_sites: sorted list of (pos, alt) tuples observed as a confident
        call in >=1 isolate (the complete, unfiltered variant-site set).
      site_missing_rate: {(pos, alt): missing_rate}, where missing_rate is
        based on POSITION-level missingness (missing_count[pos] / N), since
        "./." is recorded per-position, not per-allele.
      core_sites: sorted list of (pos, alt) with missing_rate <= threshold --
        this is what snp_matrix.py's pass 2 actually encodes.
      n_isolates, max_site_missing_rate: parameters used, for the report.
    """
    call_count = Counter()
    missing_count = Counter()
    n_isolates = 0

    for sample_id in sample_ids:
        confident_calls, missing_positions = extract_snp_calls(vcf_path_for(sample_id))
        call_count.update(confident_calls)
        missing_count.update(missing_positions)
        n_isolates += 1

    all_sites = sorted(call_count.keys())
    site_missing_rate = {
        site: missing_count[site[0]] / n_isolates for site in all_sites
    }
    core_sites = sorted(
        site for site in all_sites if site_missing_rate[site] <= max_site_missing_rate
    )

    return {
        "all_sites": all_sites,
        "site_missing_rate": site_missing_rate,
        "core_sites": core_sites,
        "call_count": call_count,
        "n_isolates": n_isolates,
        "max_site_missing_rate": max_site_missing_rate,
    }


def build_sparse_matrix(sample_ids, core_sites):
    """Pass 2: re-stream and build an (N, M_core) uint8 CSR matrix.

    Row order is exactly sample_ids (validated by the caller against
    sample_ids.csv); column order is exactly core_sites. 1 = confident ALT
    call at that site for that isolate; 0 = reference match or a site
    excluded from core_sites (see module docstring for that limitation).
    """
    site_col = {site: j for j, site in enumerate(core_sites)}
    rows, cols = [], []

    for i, sample_id in enumerate(sample_ids):
        confident_calls, _ = extract_snp_calls(vcf_path_for(sample_id))
        for site in confident_calls:
            j = site_col.get(site)
            if j is not None:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.uint8)
    X = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(sample_ids), len(core_sites)), dtype=np.uint8
    )
    return X
