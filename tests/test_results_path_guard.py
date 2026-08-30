"""A redirected run must never write into results/.

data_loader.processed_dir makes every entry point redirectable. Without a
guard, running any of them against the synthetic replica would overwrite the
committed Afro-TB metrics in results/ with numbers from other data. These
tests pin that guard down.
"""

from pathlib import Path

from src.baselines.results_path import is_real, resolve


def test_real_run_writes_to_results():
    assert resolve(Path("data/processed"), Path("results/baseline_metrics.json")) \
        == Path("results/baseline_metrics.json")


def test_redirected_run_writes_beside_its_own_input(tmp_path):
    out = resolve(tmp_path, Path("results/baseline_metrics.json"))
    assert out == tmp_path / "baseline_metrics.json"
    assert "results" not in out.parts[:-1] or out.parent == tmp_path


def test_synthetic_dir_is_not_treated_as_real(replica):
    assert is_real(replica["dir"]) is False
    out = resolve(replica["dir"], Path("results/gnn_metrics.json"))
    assert out.parent == replica["dir"]


def test_loaders_report_where_they_read_from(replica):
    from src.baselines.data_loader import load_aligned_data
    from src.gnn.data import load_gnn_data

    tab = load_aligned_data(replica["dir"])
    assert Path(tab["processed_dir"]) == Path(replica["dir"])

    gnn = load_gnn_data(replica["dir"])
    assert Path(gnn["processed_dir"]) == Path(replica["dir"])
