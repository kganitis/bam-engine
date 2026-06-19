import json

import matplotlib

matplotlib.use("Agg")
from comparison.analysis.aggregate import load_results
from comparison.analysis.plots import scaling_curve
from comparison.analysis.report import write_report


def _raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for fw, sp in [("bamengine", 0.001), ("mesa", 0.05)]:
        for n in (100, 1000):
            for rep in range(2):
                (raw / f"{fw}__s{n}__rep{rep}.json").write_text(
                    json.dumps(
                        {
                            "run_id": f"{fw}__s{n}__rep{rep}",
                            "framework": fw,
                            "status": "ok",
                            "population": {
                                "n_firms": n,
                                "n_agents_total": n * 6 + n // 10,
                            },
                            "timing": {
                                "steady_state_per_period_seconds": sp * (n / 100),
                                "throughput_agent_steps_per_s": 1e6,
                                "init_seconds": 0.01,
                                "run_seconds": 1.0,
                            },
                            "process": {
                                "peak_rss_bytes": 10_000_000 * n // 100,
                                "startup_seconds": 0.2,
                                "wall_seconds": 1.3,
                            },
                        }
                    )
                )
    return raw


def test_load_results_shape(tmp_path):
    df = load_results(_raw(tmp_path))
    assert len(df) == 8
    assert {"framework", "n_firms", "steady_per_period"} <= set(df.columns)


def test_scaling_curve_writes_png(tmp_path):
    df = load_results(_raw(tmp_path))
    out = scaling_curve(df, tmp_path / "scaling.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_report_written(tmp_path):
    df = load_results(_raw(tmp_path))
    gate = {"frameworks": {"mesa": {"passed": True, "metrics": {}}}, "tolerances": {}}
    out = write_report(df, gate, {"cpu": "x", "n_cores": 8}, tmp_path / "report.md")
    text = out.read_text()
    assert "Benchmark" in text
    assert "—" not in text  # no em dash
