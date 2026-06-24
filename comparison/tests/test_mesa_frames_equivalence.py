"""Committed equivalence-gate acceptance test for the mesa-frames port.

Runs the full gate phase (20 seeds × 1000 periods) for both ``bamengine`` and
``mesa_frames``, then asserts that mesa-frames passes all six behavioral
equivalence metrics.

Marked ``@pytest.mark.slow`` -- not included in the quick regression suite.
Expect several minutes to complete (20 mesa-frames simulations of 1000 periods
each run in parallel via ProcessPoolExecutor).
"""

from __future__ import annotations

import concurrent.futures

import pytest

import comparison.equivalence.gate as gate_mod
import comparison.equivalence.metrics as metrics_mod
import comparison.orchestrator.matrix as matrix_mod
import comparison.orchestrator.params as params_mod
from comparison.orchestrator.contract import RunRequest
from comparison.orchestrator.run import _execute_gate

# Per-job wall-clock budget: 1000-period mesa-frames runs at 100 firms take
# longer than Mesa; 600 s gives a generous safety margin.
_BUDGET_S = 600.0
_MAX_WORKERS = 8
_BURN_IN = 500


def _build_requests(frameworks: list[str]) -> list[RunRequest]:
    """Build gate RunRequests for the given frameworks (mirrors matrix.equivalence_jobs)."""
    params = params_mod.canonical_params()
    population = params_mod.population_for(100)
    reqs = []
    for fw in frameworks:
        for seed in matrix_mod.GATE_SEEDS:
            reqs.append(
                RunRequest(
                    run_id=f"{fw}__gate__seed{seed}",
                    framework=fw,
                    model_params=params,
                    population=population,
                    n_periods=matrix_mod.GATE_N_PERIODS,
                    warmup_periods=0,
                    seed=seed,
                    collect_outputs=True,
                    outputs_requested=list(matrix_mod.SERIES),
                )
            )
    return reqs


@pytest.mark.slow
def test_mesa_frames_passes_equivalence_gate():
    """mesa-frames must pass all six behavioral equivalence metrics at the full gate size.

    Gate config: 20 seeds, 1000 periods, burn-in 500 periods, 100 firms (canonical
    BAM ratio).  Tolerance per metric: max(validation_band, 2 * std_bamengine).
    """
    frameworks = ["bamengine", "mesa_frames"]
    reqs = _build_requests(frameworks)
    args = [(req, _BUDGET_S) for req in reqs]

    # Run all gate jobs in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        records = list(ex.map(_execute_gate, args))

    # Collect per-framework metric rows from successful runs.
    by_fw: dict[str, list[dict]] = {fw: [] for fw in frameworks}
    failed_runs: list[str] = []

    for req, rec in zip(reqs, records, strict=True):
        if rec.get("status") == "ok" and rec.get("outputs"):
            m = metrics_mod.compute_metrics(rec["outputs"], burn_in=_BURN_IN)
            by_fw[req.framework].append(m)
        else:
            failed_runs.append(
                f"{rec.get('run_id', req.run_id)}: "
                f"status={rec.get('status')} error={rec.get('error')}"
            )

    # Surface any run failures as part of the assertion context.
    run_failure_summary = (
        f"\n{len(failed_runs)} run(s) failed:\n" + "\n".join(failed_runs)
        if failed_runs
        else ""
    )

    assert by_fw["bamengine"], (
        f"No successful bamengine runs -- cannot evaluate gate.{run_failure_summary}"
    )
    assert by_fw["mesa_frames"], (
        f"No successful mesa_frames runs -- cannot evaluate gate.{run_failure_summary}"
    )

    gate = gate_mod.evaluate_gate(by_fw)
    mf_result = gate["frameworks"]["mesa_frames"]

    # Build a human-readable failure report showing per-metric deviations.
    failing_metrics = [
        f"  {m}: deviation={info['deviation']:.4f}  tolerance={info['tolerance']:.4f}"
        f"  (bamengine_mean={info['bamengine_mean']:.4f}  mesa_frames_mean={info['mean']:.4f})"
        for m, info in mf_result["metrics"].items()
        if info.get("within") is False
    ]

    failure_detail = (
        "\nFailing metrics:\n" + "\n".join(failing_metrics) if failing_metrics else ""
    )

    assert mf_result["passed"], (
        f"mesa_frames did NOT pass the equivalence gate.{failure_detail}{run_failure_summary}"
    )
