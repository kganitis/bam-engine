import math
import os
import subprocess
import sys

from comparison.orchestrator.contract import RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for
from comparison.orchestrator.run import RUNNER_CMD
from comparison.runners.netlogo.run import build_series, collect_run


class _FakeLink:
    """Minimal stand-in for a pyNetLogo NetLogoLink (no JVM).

    Records commands issued, and answers reporter queries from a scripted table
    that advances one row per 'go'. Lets us test collect_run's loop, timing dict,
    and output derivation entirely in CI.
    """

    def __init__(self, rows, production_final):
        self.rows = rows  # list of dicts keyed by reporter string
        self.production_final = production_final
        self.commands = []
        self._tick = -1

    def command(self, cmd):
        self.commands.append(cmd)
        if cmd == "go":
            self._tick += 1

    def report(self, rep):
        if rep.startswith("[production-Y]"):
            return list(self.production_final)
        return self.rows[self._tick][rep]

    def kill_workspace(self):
        self.commands.append("__kill__")


def _rows_for(reporters, n):
    # Build n scripted ticks with distinct, finite values per reporter.
    rows = []
    for t in range(n):
        rows.append(
            {
                reporters["unemployment"]: 0.1,
                reporters["price_inflation"]: 0.02,
                reporters["avg_wage"]: 1.0 + 0.01 * t,
                reporters["real_gdp"]: 100.0 + t,
                reporters["total_vacancies"]: 20.0,
            }
        )
    return rows


def _request() -> RunRequest:
    return RunRequest(
        run_id="netlogo__gate__seed0",
        framework="netlogo",
        model_params=canonical_params(),
        population=population_for(100),
        n_periods=60,
        warmup_periods=0,
        seed=0,
        collect_outputs=True,
        outputs_requested=["unemployment"],
    )


def test_netlogo_registered_in_runner_cmd():
    assert "netlogo" in RUNNER_CMD
    assert RUNNER_CMD["netlogo"][1:] == ["-m", "comparison.runners.netlogo.run"]


def test_netlogo_runner_skips_without_toolchain(tmp_path):
    # Run the stub with the CURRENT interpreter (which has no pynetlogo). It must
    # emit a valid, parseable RunResult with status "skipped", never crash.
    path = tmp_path / "req.json"
    path.write_text(_request().to_json())
    proc = subprocess.run(
        [sys.executable, "-m", "comparison.runners.netlogo.run", str(path)],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    res = RunResult.from_json(proc.stdout.strip().splitlines()[-1])
    assert res.framework == "netlogo"
    assert res.status == "skipped"


def test_build_series_derivations():
    raw = {
        "unemployment": [0.2, 0.1, 0.15],
        "price_inflation": [0.0, 0.03, -0.01],
        "avg_wage": [1.0, 1.1, 1.1],
        "real_gdp": [10.0, 0.0, 40.0],
        "total_vacancies": [50.0, 25.0, 0.0],
    }
    production_final = [4.0, 6.0]
    out = build_series(raw, production_final, n_workers=500)

    # Direct pass-through series.
    assert out["unemployment"] == [0.2, 0.1, 0.15]
    assert out["price_inflation"] == [0.0, 0.03, -0.01]

    # Wage inflation: period-over-period growth of avg_wage; first entry 0.0.
    assert out["wage_inflation"][0] == 0.0
    assert math.isclose(out["wage_inflation"][1], 0.1, rel_tol=1e-9)
    assert math.isclose(out["wage_inflation"][2], 0.0, abs_tol=1e-9)

    # log_gdp: log of real_gdp; non-positive (0.0) becomes 0.0 after nan_to_num.
    assert math.isclose(out["log_gdp"][0], math.log(10.0), rel_tol=1e-9)
    assert out["log_gdp"][1] == 0.0
    assert math.isclose(out["log_gdp"][2], math.log(40.0), rel_tol=1e-9)

    # Vacancy rate: total_vacancies / n_workers.
    assert out["vacancy_rate"] == [0.1, 0.05, 0.0]

    # Final per-firm production passes through, sanitized.
    assert out["production_final"] == [4.0, 6.0]


def test_build_series_all_series_present_and_finite():
    raw = {
        "unemployment": [0.1],
        "price_inflation": [0.0],
        "avg_wage": [0.0],  # zero previous wage must not divide-by-zero
        "real_gdp": [0.0],  # non-positive GDP -> 0.0
        "total_vacancies": [0.0],
    }
    out = build_series(raw, [0.0], n_workers=5)
    for key in (
        "unemployment",
        "price_inflation",
        "wage_inflation",
        "log_gdp",
        "vacancy_rate",
        "production_final",
    ):
        assert key in out
        assert all(math.isfinite(x) for x in out[key])


def test_collect_run_gate_mode_builds_outputs():
    from comparison.runners.netlogo.run import REPORTERS

    n = 5
    link = _FakeLink(_rows_for(REPORTERS, n), production_final=[1.0, 2.0, 3.0])
    req = _request()
    req.n_periods = n
    req.collect_outputs = True
    timing, outputs = collect_run(link, req)

    assert outputs is not None
    assert len(outputs["unemployment"]) == n
    assert len(outputs["production_final"]) == 3
    for key in (
        "init_seconds",
        "run_seconds",
        "steady_state_per_period_seconds",
        "throughput_agent_steps_per_s",
    ):
        assert key in timing
    # setup must have been called exactly once, and go exactly n times.
    assert link.commands.count("setup") == 1
    assert link.commands.count("go") == n


def test_collect_run_timing_mode_no_outputs():
    from comparison.runners.netlogo.run import REPORTERS

    total, warmup = 8, 3
    link = _FakeLink(_rows_for(REPORTERS, total), production_final=[1.0])
    req = _request()
    req.n_periods = total
    req.warmup_periods = warmup
    req.collect_outputs = False
    timing, outputs = collect_run(link, req)

    assert outputs is None
    # go called total times (warmup + timed); setup once.
    assert link.commands.count("go") == total
    assert timing["steady_state_per_period_seconds"] > 0
