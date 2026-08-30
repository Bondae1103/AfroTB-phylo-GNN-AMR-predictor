"""Shared fixtures. All tests run against the synthetic replica, never real data.

The real Afro-TB artifacts are gitignored and absent from a fresh checkout, so
tests that depended on them would be permanently skipped and would rot. These
build a small replica once per session instead.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def replica(tmp_path_factory):
    """A small generated dataset plus its k-NN graph."""
    from src.phylogeny.graph_from_matrix import build
    from src.synthetic.afrotb_replica import generate

    out = tmp_path_factory.mktemp("replica")
    manifest = generate(out, n_isolates=300, n_background_sites=1500,
                        n_subclades_per_lineage=4, seed=11)
    report, _ = build(out)
    return {"dir": out, "manifest": manifest, "graph_report": report}
