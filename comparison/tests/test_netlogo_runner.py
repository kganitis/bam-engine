import math
import os
import subprocess
import sys

from comparison.orchestrator.contract import RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for
from comparison.orchestrator.run import RUNNER_CMD
from comparison.runners.netlogo.run import build_series


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
