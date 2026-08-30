"""Entry point: clean labels.csv's Lineage column into y_lineage.

    .venv/Scripts/python.exe -m src.protocols.run_lineage
    .venv/Scripts/python.exe -m src.protocols.run_lineage --dir data/synthetic/full --collapse

Writes <dir>/y_lineage.csv and <dir>/y_lineage_metadata.json. Never drops a
row, never renames a taxon, never imputes a missing lineage -- see
src/protocols/lineage.py for why each of those is refused.
"""

import argparse
import json
from pathlib import Path

from .lineage import write_y_lineage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/processed")
    ap.add_argument("--collapse", action="store_true",
                    help="also collapse sublineages to major clades (reports every merge)")
    args = ap.parse_args()

    y, classes, report = write_y_lineage(Path(args.dir), collapse=args.collapse)

    print("isolates          : %d" % report["n_isolates"])
    print("classes           : %d  %s" % (report["n_classes"], classes))
    print("missing (ignored) : %d  -> encoded as -100" % report["n_missing"])
    print("\nclass counts:")
    for c, n in sorted(report["class_counts"].items(), key=lambda kv: -kv[1]):
        print("  %-24s %6d" % (c, n))

    merges = report["raw_labels_merged_by_normalisation"]
    if merges:
        print("\nraw labels merged by normalisation:")
        for norm, raws in merges.items():
            print("  %-24s <- %s" % (norm, ", ".join("%s (%d)" % (r, n) for r, n in raws)))
    else:
        print("\nno raw labels required merging")

    print("\ndeliberately NOT done:")
    for item in report["not_done"]:
        print("  - %s" % item)
    print("\nWrote %s/y_lineage.csv and y_lineage_metadata.json" % args.dir)


if __name__ == "__main__":
    main()
