"""bamengine reference runner for the comparison harness.

This is the FIRST and REFERENCE runner: every other framework's output series
are compared against what this runner emits. It is a subprocess entry point that
reads a :class:`RunRequest` JSON file (path in ``sys.argv[1]``), runs bamengine,
and prints a :class:`RunResult` JSON as the FINAL line of stdout.

Usage
-----
    python -m comparison.runners.bamengine.run <request.json>

Two run modes (driven by ``RunRequest.collect_outputs``):

* ``collect_outputs=True`` (gate runs): one contiguous ``sim.run(n_periods,
  collect=...)`` producing FULL-LENGTH per-period series (length == n_periods).
  Burn-in is applied centrally by the orchestrator, not here.
* ``collect_outputs=False`` (timing runs): a warmup ``sim.run(warmup_periods,
  collect=False)`` followed by a TIMED ``sim.run(n_periods - warmup_periods,
  collect=False)``. No outputs are emitted.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any

import numpy as np

import bamengine as bam
from bamengine import BASELINE_COLLECT
from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_OK,
    RunRequest,
    RunResult,
)

# ─── Confirmed bamengine result keys (DISCOVERY, bamengine 0.9.1) ──────────────
#
# Verified via ``sorted(sim.collectables())`` / ``sorted(results.available())``
# with ``log_level="ERROR"``. The brief's assumed Economy.* aggregate keys mostly
# DO NOT exist; the six requested series are built as follows:
#
#   unemployment    : NOT an Economy key. Derived as
#                     ``1 - mean(Worker.employed, axis=1)``. ``Worker.employed``
#                     is a computed bool property (employer >= 0) and is only
#                     collected via the ``BASELINE_COLLECT`` config, which also
#                     pins its capture timing to ``firms_run_production`` (so the
#                     last period is not inflated by post-step contract expiry
#                     see CLAUDE.md "measurement timing"). This matches the
#                     canonical formula in validation/robustness/internal_validity.
#   price_inflation : ``Economy.inflation``  (exists, per-period scalar series).
#   wage_inflation  : NOT a key. Derived from the average wage of EMPLOYED workers
#                     per period (``Worker.wage`` masked by ``Worker.employed``),
#                     then period-over-period growth. Mirrors internal_validity.
#   log_gdp         : NOT a key. Derived as ``log(sum(Producer.production, axis=1))``
#                     (real GDP = total production; matches internal_validity).
#   vacancy_rate    : NOT an Economy key. Derived as
#                     ``sum(Employer.n_vacancies, axis=1) / n_households``
#                     (per-firm vacancies exist; rate uses households as the
#                     labor-force denominator, matching internal_validity).
#   production_final: ``Producer.production`` is (n_periods, n_firms); the final
#                     row is the per-firm production at the last period.
#
# All collected per-period arrays have shape (n_periods, n_agents); per-period
# scalar economy series have shape (n_periods,).


def build_series(results: Any, n_firms: int) -> dict[str, list]:
    """Build the six comparison output series from a collected results object.

    Parameters
    ----------
    results : SimulationResults
        Result of ``sim.run(..., collect=BASELINE_COLLECT)``.
    n_firms : int
        Number of firms (used only as a fallback; series lengths follow the
        collected arrays).

    Returns
    -------
    dict
        Mapping of series name to a JSON-safe list (no NaN/inf).
    """
    # Per-period scalar series.
    price_inflation = np.asarray(results["Economy.inflation"], dtype=float)

    # Per-agent / per-firm arrays: (n_periods, n_agents).
    employed = np.asarray(results["Worker.employed"], dtype=float)
    wage = np.asarray(results["Worker.wage"], dtype=float)
    production = np.asarray(results["Producer.production"], dtype=float)
    n_vacancies = np.asarray(results["Employer.n_vacancies"], dtype=float)

    n_households = employed.shape[1] if employed.ndim == 2 else 0

    # Unemployment: 1 - employment rate per period.
    unemployment = 1.0 - employed.mean(axis=1)

    # Average wage of employed workers per period (avoid divide-by-zero).
    employed_wage_sum = np.where(employed > 0, wage, 0.0).sum(axis=1)
    employed_count = (employed > 0).sum(axis=1)
    safe_count = np.where(employed_count > 0, employed_count, 1.0)
    avg_wage = np.where(employed_count > 0, employed_wage_sum / safe_count, 0.0)

    # Wage inflation: period-over-period growth of the average employed wage.
    wage_inflation = np.zeros_like(avg_wage)
    if avg_wage.size > 1:
        prev = avg_wage[:-1]
        safe_prev = np.where(prev != 0, prev, 1.0)
        wage_inflation[1:] = np.where(prev != 0, avg_wage[1:] / safe_prev - 1.0, 0.0)

    # Real GDP = total production; log_gdp on a strictly-positive denominator.
    gdp = production.sum(axis=1)
    safe_gdp = np.where(gdp > 0, gdp, np.nan)
    log_gdp = np.log(safe_gdp)

    # Vacancy rate: total open vacancies over the labor force (households).
    total_vacancies = n_vacancies.sum(axis=1)
    denom = n_households if n_households > 0 else max(1, n_firms)
    vacancy_rate = total_vacancies / float(denom)

    # Per-firm production at the final period.
    production_final = production[-1] if production.ndim == 2 else production

    return {
        "unemployment": np.nan_to_num(unemployment).tolist(),
        "price_inflation": np.nan_to_num(price_inflation).tolist(),
        "wage_inflation": np.nan_to_num(wage_inflation).tolist(),
        "log_gdp": np.nan_to_num(log_gdp).tolist(),
        "vacancy_rate": np.nan_to_num(vacancy_rate).tolist(),
        "production_final": np.nan_to_num(np.asarray(production_final, float)).tolist(),
    }


def main(request_path: str) -> None:
    """Run bamengine for one request and print a RunResult JSON to stdout."""
    with open(request_path) as fh:
        req = RunRequest.from_json(fh.read())

    framework_version = getattr(bam, "__version__", "unknown")
    language_version = sys.version.split()[0]
    n_agents = sum(int(v) for v in req.population.values())

    try:
        # ── init timing wraps ONLY Simulation.init ──────────────────────────
        t_init0 = time.perf_counter()
        sim = bam.Simulation.init(
            seed=req.seed,
            log_level="ERROR",
            **req.population,
            **req.model_params,
        )
        t_init1 = time.perf_counter()
        init_seconds = t_init1 - t_init0

        outputs: dict[str, list] | None = None

        if req.collect_outputs:
            # Gate run: ONE contiguous run producing FULL-LENGTH series
            # (length == n_periods). Burn-in is applied centrally later.
            t_run0 = time.perf_counter()
            results = sim.run(n_periods=req.n_periods, collect=BASELINE_COLLECT)
            t_run1 = time.perf_counter()
            run_seconds = t_run1 - t_run0
            timed_periods = req.n_periods
            outputs = build_series(results, int(req.population["n_firms"]))
        else:
            # Timing run: warmup (untimed) then a timed steady-state segment.
            warmup = max(0, req.warmup_periods)
            timed_periods = max(1, req.n_periods - warmup)
            if warmup > 0:
                sim.run(n_periods=warmup, collect=False)
            t_run0 = time.perf_counter()
            sim.run(n_periods=timed_periods, collect=False)
            t_run1 = time.perf_counter()
            run_seconds = t_run1 - t_run0

        steady = run_seconds / timed_periods if timed_periods else run_seconds
        throughput = n_agents * timed_periods / run_seconds if run_seconds > 0 else 0.0

        res = RunResult(
            schema_version=SCHEMA_VERSION,
            run_id=req.run_id,
            framework="bamengine",
            framework_version=framework_version,
            language="python",
            language_version=language_version,
            status=STATUS_OK,
            error=None,
            population={**req.population, "n_agents_total": n_agents},
            n_periods=req.n_periods,
            warmup_periods=req.warmup_periods,
            seed=req.seed,
            timing={
                "init_seconds": init_seconds,
                "run_seconds": run_seconds,
                "steady_state_per_period_seconds": steady,
                "throughput_agent_steps_per_s": throughput,
            },
            outputs=outputs,
        )
    except Exception:
        res = RunResult(
            schema_version=SCHEMA_VERSION,
            run_id=req.run_id,
            framework="bamengine",
            framework_version=framework_version,
            language="python",
            language_version=language_version,
            status=STATUS_ERROR,
            error=traceback.format_exc(),
            population=req.population,
            n_periods=req.n_periods,
            warmup_periods=req.warmup_periods,
            seed=req.seed,
            timing={},
            outputs=None,
        )

    print(res.to_json())


if __name__ == "__main__":
    main(sys.argv[1])
