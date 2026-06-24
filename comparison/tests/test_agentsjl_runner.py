"""End-to-end test for the Agents.jl subprocess runner.

Builds a RunRequest (100 firms, 50 periods, collect_outputs=True), invokes
the Julia runner as a subprocess, parses the RunResult JSON from stdout, and
asserts that all 6 output series are present, finite, and have the correct
length.

Skips automatically when ``julia`` is not on PATH so CI without Julia
passes without failures.

Run with:
    .venv/bin/python -m pytest comparison/tests/test_agentsjl_runner.py --no-cov -v
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from comparison.orchestrator.contract import STATUS_OK, RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for

# Path to the Agents.jl runner directory.
_AGENTSJL_DIR = Path(__file__).resolve().parent.parent / "runners" / "agentsjl"
_RUN_JL = _AGENTSJL_DIR / "run.jl"

# Skip the whole module if julia is not available.
pytestmark = pytest.mark.skipif(
    shutil.which("julia") is None,
    reason="julia not on PATH",
)

_N_FIRMS = 100
_N_PERIODS = 50


def _make_request(collect: bool, seed: int = 42) -> RunRequest:
    """Build a RunRequest for the Agents.jl runner."""
    return RunRequest(
        run_id=f"agentsjl__s{_N_FIRMS}__seed{seed}__rep0",
        framework="agentsjl",
        model_params=canonical_params(),
        population=population_for(_N_FIRMS),
        n_periods=_N_PERIODS,
        warmup_periods=10,
        seed=seed,
        collect_outputs=collect,
        outputs_requested=[
            "unemployment",
            "price_inflation",
            "wage_inflation",
            "log_gdp",
            "vacancy_rate",
            "production_final",
        ],
    )


def _run_julia(req: RunRequest, tmp_path: Path) -> RunResult:
    """Write the request to a temp file, invoke Julia, parse the RunResult."""
    req_file = tmp_path / "request.json"
    req_file.write_text(req.to_json())

    cmd = [
        "julia",
        f"--project={_AGENTSJL_DIR}",
        "--startup-file=no",
        str(_RUN_JL),
        str(req_file),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        timeout=300,  # allow up to 5 min for JIT compilation on first run
    )

    # Print stderr to aid debugging on failure.
    if proc.stderr:
        print("Julia stderr:\n", proc.stderr[-4000:])

    assert proc.returncode == 0, (
        f"Julia runner exited with code {proc.returncode}.\n"
        f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
    )

    last_line = proc.stdout.strip().splitlines()[-1]
    return RunResult.from_json(last_line)


def test_agentsjl_runner_collect(tmp_path):
    """Gate run: all 6 series present, finite, and correct length."""
    req = _make_request(collect=True)
    res = _run_julia(req, tmp_path)

    assert res.status == STATUS_OK, f"status={res.status!r}, error={res.error!r}"
    problems = res.validate()
    assert problems == [], f"RunResult validation failures: {problems}"

    assert res.timing["steady_state_per_period_seconds"] > 0

    outputs = res.outputs
    assert outputs is not None, "outputs must not be None when collect_outputs=True"

    expected_series = [
        "unemployment",
        "price_inflation",
        "wage_inflation",
        "log_gdp",
        "vacancy_rate",
        "production_final",
    ]
    for key in expected_series:
        assert key in outputs, f"Missing series: {key!r}"

    # Check lengths.
    for key in [
        "unemployment",
        "price_inflation",
        "wage_inflation",
        "log_gdp",
        "vacancy_rate",
    ]:
        series = outputs[key]
        assert len(series) == _N_PERIODS, (
            f"Series {key!r}: expected length {_N_PERIODS}, got {len(series)}"
        )
        # All values must be finite (no NaN or Inf after sanitization).
        bad = [v for v in series if not math.isfinite(v)]
        assert not bad, f"Series {key!r} contains non-finite values: {bad[:5]}"

    # production_final: one entry per firm.
    prod_final = outputs["production_final"]
    assert len(prod_final) == _N_FIRMS, (
        f"production_final: expected {_N_FIRMS} entries, got {len(prod_final)}"
    )
    bad = [v for v in prod_final if not math.isfinite(v)]
    assert not bad, f"production_final contains non-finite values: {bad[:5]}"

    # Unemployment should be in [0, 1].
    unemp = outputs["unemployment"]
    assert all(0.0 <= v <= 1.0 for v in unemp), (
        f"Unemployment values outside [0,1]: {[v for v in unemp if not (0 <= v <= 1)][:5]}"
    )

    # Vacancy rate should be non-negative.
    vr = outputs["vacancy_rate"]
    assert all(v >= 0.0 for v in vr), (
        f"Negative vacancy_rate values: {[v for v in vr if v < 0][:5]}"
    )

    # Check framework metadata.
    assert res.framework == "agentsjl"
    assert res.language == "julia"
    assert res.framework_version not in ("", "nothing", None), (
        f"framework_version should be set, got: {res.framework_version!r}"
    )


def test_agentsjl_runner_timing(tmp_path):
    """Timing run (no collection): status ok, timing fields present and positive."""
    req = _make_request(collect=False)
    res = _run_julia(req, tmp_path)

    assert res.status == STATUS_OK, f"status={res.status!r}, error={res.error!r}"
    problems = res.validate()
    assert problems == [], f"RunResult validation failures: {problems}"

    assert res.timing["init_seconds"] >= 0
    assert res.timing["run_seconds"] > 0
    assert res.timing["steady_state_per_period_seconds"] > 0
    assert res.timing["throughput_agent_steps_per_s"] > 0

    # No outputs in timing mode.
    assert res.outputs is None
