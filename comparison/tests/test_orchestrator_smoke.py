import json

from comparison.orchestrator.run import run_benchmark


def test_quick_run_produces_raw_results_and_gate(tmp_path):
    out = run_benchmark(
        frameworks=["bamengine"],
        results_dir=tmp_path,
        quick=True,
        gate_workers=2,
        budget_s=120,
    )
    raw = list((tmp_path / "raw").glob("*.json"))
    assert raw, "no raw result files written"
    rec = json.loads(raw[0].read_text())
    assert rec["status"] in {"ok", "skipped", "timeout"}
    assert (tmp_path / "raw").exists()
    assert "bamengine" in out["gate"]["frameworks"]


def test_quick_run_renders_report(tmp_path):
    run_benchmark(
        frameworks=["bamengine"],
        results_dir=tmp_path,
        quick=True,
        gate_workers=2,
        budget_s=120,
    )
    assert (tmp_path / "report.md").exists()
