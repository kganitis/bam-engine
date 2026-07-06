"""NetLogo runner for the cross-framework comparison harness.

Drives the third-party Platas ``DelliBAM_.nlogo`` model through pyNetLogo and
prints a RunResult JSON as the FINAL line of stdout. Reads a RunRequest JSON
file whose path is ``sys.argv[1]`` (contract in
``comparison/orchestrator/contract.py``).

The pyNetLogo import is guarded: when the toolchain is absent (for example in
CI), the runner emits a ``status = "skipped"`` RunResult and the rest of the
harness proceeds. The real run path is added in later tasks.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_SKIPPED,
    RunRequest,
    RunResult,
)


def _skeleton(req: RunRequest, status: str, error: object) -> RunResult:
    """Build a RunResult carrying only request echo fields (no outputs/timing)."""
    return RunResult(
        schema_version=SCHEMA_VERSION,
        run_id=req.run_id,
        framework="netlogo",
        framework_version="unknown",
        language="netlogo",
        language_version="unknown",
        status=status,
        error=error,
        population=req.population,
        n_periods=req.n_periods,
        warmup_periods=req.warmup_periods,
        seed=req.seed,
        timing={},
        outputs=None,
    )


def build_series(
    raw: dict[str, list], production_final: list, n_workers: int
) -> dict[str, list]:
    """Derive the six comparison series from raw per-tick NetLogo reporter values.

    Mirrors ``comparison/runners/bamengine/run.py::build_series`` exactly so the
    NetLogo reference is derived identically to every other framework.

    Parameters
    ----------
    raw : dict
        Per-tick reporter lists (all length T): ``unemployment``,
        ``price_inflation``, ``avg_wage``, ``real_gdp``, ``total_vacancies``.
    production_final : list
        Per-firm production read at the final tick.
    n_workers : int
        Labor-force denominator for the vacancy rate (households).

    Returns
    -------
    dict
        Six-series mapping, each a JSON-safe list with no NaN/inf.
    """
    unemployment = np.asarray(raw["unemployment"], dtype=float)
    price_inflation = np.asarray(raw["price_inflation"], dtype=float)
    avg_wage = np.asarray(raw["avg_wage"], dtype=float)
    real_gdp = np.asarray(raw["real_gdp"], dtype=float)
    total_vacancies = np.asarray(raw["total_vacancies"], dtype=float)

    # Wage inflation: period-over-period growth of the average employed wage.
    wage_inflation = np.zeros_like(avg_wage)
    if avg_wage.size > 1:
        prev = avg_wage[:-1]
        safe_prev = np.where(prev != 0, prev, 1.0)
        wage_inflation[1:] = np.where(prev != 0, avg_wage[1:] / safe_prev - 1.0, 0.0)

    # Real GDP = total production; log on a strictly-positive denominator.
    safe_gdp = np.where(real_gdp > 0, real_gdp, np.nan)
    log_gdp = np.log(safe_gdp)

    # Vacancy rate: total open vacancies over the labor force.
    denom = float(n_workers) if n_workers > 0 else 1.0
    vacancy_rate = total_vacancies / denom

    production_final_arr = np.asarray(production_final, dtype=float)

    return {
        "unemployment": np.nan_to_num(unemployment).tolist(),
        "price_inflation": np.nan_to_num(price_inflation).tolist(),
        "wage_inflation": np.nan_to_num(wage_inflation).tolist(),
        "log_gdp": np.nan_to_num(log_gdp).tolist(),
        "vacancy_rate": np.nan_to_num(vacancy_rate).tolist(),
        "production_final": np.nan_to_num(production_final_arr).tolist(),
    }


def main(request_path: str) -> None:
    """Run one request and print a RunResult JSON to stdout."""
    with open(request_path) as fh:
        req = RunRequest.from_json(fh.read())

    try:
        import pynetlogo  # noqa: F401
    except ImportError:
        print(
            _skeleton(
                req, STATUS_SKIPPED, "pynetlogo/NetLogo toolchain not available"
            ).to_json()
        )
        return

    # Real run path is implemented in Task 4. Until then, emit a clear error so a
    # partially-installed environment does not look like a silent success.
    try:
        raise NotImplementedError("NetLogo run path not yet implemented (Task 4)")
    except Exception:
        print(_skeleton(req, STATUS_ERROR, traceback.format_exc()).to_json())


if __name__ == "__main__":
    main(sys.argv[1])
