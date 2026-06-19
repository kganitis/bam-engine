"""Mesa reference runner for the comparison harness.

This is the Mesa subprocess runner: reads a :class:`RunRequest` JSON file
(path in ``sys.argv[1]``), runs the Mesa BAM model, and prints a
:class:`RunResult` JSON as the FINAL line of stdout.

Usage
-----
    python -m comparison.runners.mesa.run <request.json>

Two run modes (driven by ``RunRequest.collect_outputs``):

* ``collect_outputs=True`` (gate runs): one contiguous run of ``n_periods``
  with collection on, producing FULL-LENGTH per-period series
  (length == n_periods). Burn-in is applied centrally by the orchestrator.
* ``collect_outputs=False`` (timing runs): a warmup run (collect off) then
  a TIMED run (collect off). No outputs are emitted.
"""

from __future__ import annotations

import sys
import time
import traceback

import mesa
import numpy as np

from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_OK,
    RunRequest,
    RunResult,
)
from comparison.runners.mesa.model import BamModel


def build_series(model: BamModel) -> dict[str, list]:
    """Build the six comparison output series from a collected BamModel.

    Parameters
    ----------
    model : BamModel
        A model that has been run with ``collect=True``.

    Returns
    -------
    dict
        Mapping of series name to a JSON-safe list (no NaN/inf).
    """
    unemployment = np.array(model._c_unemployment, dtype=float)
    avg_wage = np.array(model._c_avg_employed_wage, dtype=float)
    total_production = np.array(model._c_total_production, dtype=float)
    total_vacancies = np.array(model._c_total_vacancies, dtype=float)
    price_inflation = np.array(model._c_inflation, dtype=float)

    # Wage inflation: period-over-period growth of the average employed wage.
    wage_inflation = np.zeros_like(avg_wage)
    if avg_wage.size > 1:
        prev = avg_wage[:-1]
        safe_prev = np.where(prev != 0, prev, 1.0)
        wage_inflation[1:] = np.where(prev != 0, avg_wage[1:] / safe_prev - 1.0, 0.0)

    # Real GDP = total production; log_gdp on a strictly-positive denominator.
    safe_gdp = np.where(total_production > 0, total_production, np.nan)
    log_gdp = np.log(safe_gdp)

    # Vacancy rate: total open vacancies over the labor force (households).
    n_households = (
        model.n_households if model.n_households > 0 else max(1, model.n_firms)
    )
    vacancy_rate = total_vacancies / float(n_households)

    # Per-firm production at the final period.
    production_final = np.array(model._c_production_final, dtype=float)

    return {
        "unemployment": np.nan_to_num(unemployment).tolist(),
        "price_inflation": np.nan_to_num(price_inflation).tolist(),
        "wage_inflation": np.nan_to_num(wage_inflation).tolist(),
        "log_gdp": np.nan_to_num(log_gdp).tolist(),
        "vacancy_rate": np.nan_to_num(vacancy_rate).tolist(),
        "production_final": np.nan_to_num(production_final).tolist(),
    }


def main(request_path: str) -> None:
    """Run the Mesa BAM model for one request and print a RunResult JSON to stdout."""
    with open(request_path) as fh:
        req = RunRequest.from_json(fh.read())

    framework_version = getattr(mesa, "__version__", "unknown")
    language_version = sys.version.split()[0]
    n_agents = sum(int(v) for v in req.population.values())

    try:
        n_firms = int(req.population["n_firms"])
        n_households = int(req.population["n_households"])
        n_banks = int(req.population["n_banks"])

        outputs: dict[str, list] | None = None

        if req.collect_outputs:
            # Gate run: ONE contiguous run producing FULL-LENGTH series
            # (length == n_periods). Burn-in is applied centrally later.
            t_init0 = time.perf_counter()
            model = BamModel(
                n_firms,
                n_households,
                n_banks,
                req.model_params,
                seed=req.seed,
                collect=True,
            )
            t_init1 = time.perf_counter()
            init_seconds = t_init1 - t_init0

            t_run0 = time.perf_counter()
            for _ in range(req.n_periods):
                model.step()
            t_run1 = time.perf_counter()
            run_seconds = t_run1 - t_run0
            timed_periods = req.n_periods
            outputs = build_series(model)
        else:
            # Timing run: warmup (untimed) then a timed steady-state segment.
            warmup = max(0, req.warmup_periods)
            timed_periods = max(1, req.n_periods - warmup)

            t_init0 = time.perf_counter()
            model = BamModel(
                n_firms,
                n_households,
                n_banks,
                req.model_params,
                seed=req.seed,
                collect=False,
            )
            t_init1 = time.perf_counter()
            init_seconds = t_init1 - t_init0

            if warmup > 0:
                for _ in range(warmup):
                    model.step()

            t_run0 = time.perf_counter()
            for _ in range(timed_periods):
                model.step()
            t_run1 = time.perf_counter()
            run_seconds = t_run1 - t_run0

        steady = run_seconds / timed_periods if timed_periods else run_seconds
        throughput = n_agents * timed_periods / run_seconds if run_seconds > 0 else 0.0

        res = RunResult(
            schema_version=SCHEMA_VERSION,
            run_id=req.run_id,
            framework="mesa",
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
            framework="mesa",
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
