"""End-to-end test for the mesa-frames subprocess runner.

Invokes the runner as a subprocess via its dedicated venv (.venv-mf) and
asserts that it emits a valid RunResult JSON with all 6 series present,
finite, and the correct length.

The test is skipped if .venv-mf does not exist (CI without mesa-frames env).
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest

from comparison.orchestrator.contract import STATUS_OK, RunRequest, RunResult
from comparison.orchestrator.params import canonical_params, population_for

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MF_PYTHON = (
    _REPO_ROOT
    / "comparison"
    / "runners"
    / "mesa_frames"
    / ".venv-mf"
    / "bin"
    / "python"
)
_N_FIRMS = 20
_N_PERIODS = 50
_SEED = 7
_SIX_SERIES = [
    "unemployment",
    "price_inflation",
    "wage_inflation",
    "log_gdp",
    "vacancy_rate",
    "production_final",
]


@pytest.fixture
def run_request():
    return RunRequest(
        run_id="mf_e2e_test",
        framework="mesa_frames",
        model_params=canonical_params(),
        population=population_for(_N_FIRMS),
        n_periods=_N_PERIODS,
        warmup_periods=5,
        seed=_SEED,
        collect_outputs=True,
        outputs_requested=_SIX_SERIES,
    )


@pytest.fixture
def run_result(run_request, tmp_path):
    """Invoke the mesa-frames runner as a subprocess and parse its RunResult."""
    if not _MF_PYTHON.exists():
        pytest.skip(f"mesa-frames venv not found at {_MF_PYTHON}")

    req_path = tmp_path / "req.json"
    req_path.write_text(run_request.to_json())

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT)

    proc = subprocess.run(
        [str(_MF_PYTHON), "-m", "comparison.runners.mesa_frames.run", str(req_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"mesa_frames runner exited with code {proc.returncode}.\n"
        f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    return RunResult.from_json(last_line)


def test_mesa_frames_runner_status_ok(run_result):
    """Runner must report status=ok."""
    assert run_result.status == STATUS_OK, (
        f"Expected status=ok, got {run_result.status!r}. error: {run_result.error}"
    )


def test_mesa_frames_runner_schema_valid(run_result):
    """RunResult must pass schema validation."""
    problems = run_result.validate()
    assert problems == [], f"Schema validation failed: {problems}"


def test_mesa_frames_runner_all_six_series_present(run_result):
    """All 6 output series must be present in outputs."""
    assert run_result.outputs is not None, "outputs is None"
    for name in _SIX_SERIES:
        assert name in run_result.outputs, f"Missing series: {name!r}"


def test_mesa_frames_runner_series_length(run_result):
    """Per-period series must have length == n_periods."""
    outputs = run_result.outputs
    per_period = [
        "unemployment",
        "price_inflation",
        "wage_inflation",
        "log_gdp",
        "vacancy_rate",
    ]
    for name in per_period:
        series = outputs[name]
        assert len(series) == _N_PERIODS, (
            f"Series {name!r}: expected length {_N_PERIODS}, got {len(series)}"
        )


def test_mesa_frames_runner_production_final_length(run_result):
    """production_final must have length == n_firms."""
    series = run_result.outputs["production_final"]
    assert len(series) == _N_FIRMS, (
        f"production_final: expected length {_N_FIRMS}, got {len(series)}"
    )


def test_mesa_frames_runner_series_finite(run_result):
    """All series values must be finite (no NaN or inf after nan_to_num)."""
    outputs = run_result.outputs
    for name in _SIX_SERIES:
        series = outputs[name]
        non_finite = [v for v in series if not math.isfinite(v)]
        assert non_finite == [], (
            f"Series {name!r} contains non-finite values: {non_finite[:5]}"
        )


def test_mesa_frames_runner_timing_present(run_result):
    """Timing dict must contain all required keys with positive values."""
    timing = run_result.timing
    required = [
        "init_seconds",
        "run_seconds",
        "steady_state_per_period_seconds",
        "throughput_agent_steps_per_s",
    ]
    for key in required:
        assert key in timing, f"timing missing key: {key!r}"
        assert timing[key] >= 0, f"timing[{key!r}] is negative: {timing[key]}"
