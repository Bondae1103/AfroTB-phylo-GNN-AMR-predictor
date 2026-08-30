"""Where a run is allowed to write its metrics.

data_loader.processed_dir lets any entry point be redirected at another
processed-data directory (the synthetic replica, a subset run). That created a
hazard: a redirected run would still have written to results/, silently
overwriting the committed Afro-TB metrics with numbers from other data.

This guard makes the output path follow the INPUT. Only a run that actually
read data/processed may write to results/; anything else writes beside its own
input directory. Nothing can quietly replace a real result with a synthetic one.
"""

from pathlib import Path

REAL_PROCESSED = Path("data/processed")


def is_real(processed_dir):
    try:
        return Path(processed_dir).resolve() == REAL_PROCESSED.resolve()
    except OSError:
        return False


def resolve(processed_dir, results_path):
    """Return where to write, given the directory the data came from.

    results_path is used unchanged for a real-data run; otherwise the same
    filename is placed inside processed_dir.
    """
    results_path = Path(results_path)
    if is_real(processed_dir):
        return results_path
    return Path(processed_dir) / results_path.name
