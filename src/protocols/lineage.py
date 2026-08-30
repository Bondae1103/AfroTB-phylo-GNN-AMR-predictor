"""Clean labels.csv's Lineage column into a usable y_lineage target.

README Sec. 8 records two defects: the label is split across the spellings
BOV_AFRI and BOV-AFRI, and a small number of isolates carry "-" for missing.

What this module deliberately does NOT do
-----------------------------------------
1. It does not rename BOV_AFRI to "M. africanum" or "M. bovis". Those are
   distinct taxa; the dataset gives one combined label and CLAUDE.md forbids
   inventing lineage assignments. Only the hyphen/underscore spelling is
   harmonised -- a typographic fix, not a taxonomic claim.

2. It does not impute missing lineages from graph neighbours. The graph is
   built from SNP distance, and lineage is largely a function of SNP distance,
   so imputing a lineage from neighbours and then asking a graph model to
   predict lineage from those same neighbours is circular; it would inflate
   lineage accuracy without adding information. Missing values get
   IGNORE_INDEX instead, which CrossEntropyLoss drops natively while the
   isolate keeps its row, its AMR label, and its place in the graph.

3. It does not collapse sublineages to major clades by default. That decision
   loses information and should be made from the observed value counts, not
   guessed in advance. `collapse=True` is available and reports exactly what
   it merged, so the choice is visible instead of silent.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100
MISSING_TOKENS = {"-", "", "NA", "N/A", "nan", "None", "?"}


def normalise_label(raw):
    """Typographic harmonisation only. Returns None for a missing value.

    Applied rules, in order:
      * strip surrounding whitespace
      * treat the documented missing tokens as missing
      * collapse internal whitespace
      * unify '-' and '_' separators to '_' (this is what merges BOV-AFRI
        into BOV_AFRI)
      * uppercase, so case-only variants do not split a class
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s in MISSING_TOKENS:
        return None
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", "_").replace(" ", "_")
    return s.upper()


def major_clade(label):
    """Best-effort mapping of a normalised label to its major clade.

    Only used when collapse=True. Recognises the 'LINEAGE<N>' / 'L<N>' family
    and the animal-associated BOV/AFRI labels; anything else is returned
    unchanged rather than forced into a bucket it may not belong in.
    """
    if label is None:
        return None
    m = re.match(r"^(?:LINEAGE|L)_?(\d+)", label)
    if m:
        return "LINEAGE" + m.group(1)
    if label.startswith("BOV") or label.startswith("AFRI"):
        return label.split("_")[0]
    return label


def build_y_lineage(processed_dir, collapse=False):
    """Read labels.csv, return (y, classes, report).

    y is an int64 array aligned to sample_ids.csv order, holding a class index
    or IGNORE_INDEX. It is never reordered and never has rows dropped.
    """
    processed_dir = Path(processed_dir)
    with open(processed_dir / "sample_ids.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        sample_ids = [row[0] for row in r]

    with open(processed_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [r["Name"] for r in rows] != sample_ids:
        raise ValueError("labels.csv row order does not match sample_ids.csv")

    raw = [r["Lineage"] for r in rows]
    norm = [normalise_label(v) for v in raw]
    if collapse:
        norm = [major_clade(v) for v in norm]

    classes = sorted({v for v in norm if v is not None})
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[v] if v is not None else IGNORE_INDEX for v in norm],
                 dtype=np.int64)

    raw_counts = Counter(raw)
    merged = {}
    for r_label, count in raw_counts.items():
        n = normalise_label(r_label)
        if collapse:
            n = major_clade(n)
        key = n if n is not None else "<MISSING>"
        merged.setdefault(key, []).append((r_label, count))
    merges_performed = {k: v for k, v in merged.items() if len(v) > 1}

    report = {
        "n_isolates": len(sample_ids),
        "collapse_to_major_clade": collapse,
        "n_classes": len(classes),
        "classes": classes,
        "class_index": index,
        "class_counts": {c: int((y == i).sum()) for c, i in index.items()},
        "n_missing": int((y == IGNORE_INDEX).sum()),
        "missing_handling": (
            "encoded as %d (CrossEntropyLoss ignore_index); rows kept, never dropped"
            % IGNORE_INDEX
        ),
        "raw_label_counts": dict(raw_counts),
        "raw_labels_merged_by_normalisation": merges_performed,
        "not_done": [
            "no taxonomic renaming (BOV_AFRI left as the dataset's own label)",
            "no phylogenetic imputation of missing lineages (would be circular)",
        ],
    }
    return y, classes, report


def write_y_lineage(processed_dir, collapse=False):
    """Write y_lineage.csv and y_lineage_metadata.json into processed_dir."""
    processed_dir = Path(processed_dir)
    y, classes, report = build_y_lineage(processed_dir, collapse=collapse)

    with open(processed_dir / "sample_ids.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        sample_ids = [row[0] for row in r]

    with open(processed_dir / "y_lineage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Name", "lineage_encoded"])
        for sid, v in zip(sample_ids, y):
            w.writerow([sid, int(v)])

    with open(processed_dir / "y_lineage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return y, classes, report
