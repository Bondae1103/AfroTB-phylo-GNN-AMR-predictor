"""Integration guard: a redirected entry point must not write into results/.

This is deliberately an INTEGRATION test, not a unit test. An earlier unit test
checked results_path.resolve() in isolation and passed while both entry points
still wrote to the hardcoded RESULTS_PATH -- the helper was correct and simply
unused. Only actually invoking main() catches that, and it did: running
src.baselines.run_all against the synthetic replica overwrote the committed
results/baseline_metrics.json with synthetic numbers.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS = REPO_ROOT / "results"


def _snapshot(directory):
    if not directory.exists():
        return {}
    return {p.name: p.read_bytes() for p in directory.glob("*.json")}


@pytest.fixture
def in_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def test_run_all_redirected_writes_beside_input_not_results(replica, in_repo_root, monkeypatch):
    from src.baselines import run_all

    before = _snapshot(REAL_RESULTS)
    monkeypatch.setenv("AFROTB_PROCESSED_DIR", str(replica["dir"]))
    run_all.main()

    assert _snapshot(REAL_RESULTS) == before, \
        "a redirected run modified results/ -- the committed metrics are not safe"
    out = Path(replica["dir"]) / "baseline_metrics.json"
    assert out.exists(), "the redirected run must still write its own metrics"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_features"] == 157


def test_run_train_redirected_writes_beside_input_not_results(replica, in_repo_root, monkeypatch):
    from src.gnn import run_train

    before = _snapshot(REAL_RESULTS)
    monkeypatch.setenv("AFROTB_PROCESSED_DIR", str(replica["dir"]))
    monkeypatch.setattr(run_train, "MAX_EPOCHS", 5)
    monkeypatch.setattr(run_train, "PATIENCE", 5)
    run_train.main()

    assert _snapshot(REAL_RESULTS) == before, \
        "a redirected GNN run modified results/"
    assert (Path(replica["dir"]) / "gnn_metrics.json").exists()


def test_unredirected_default_still_targets_results():
    """The real-data path must be untouched by the guard."""
    from src.baselines.results_path import resolve
    assert resolve(Path("data/processed"), Path("results/baseline_metrics.json")) \
        == Path("results/baseline_metrics.json")
