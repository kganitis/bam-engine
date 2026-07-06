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

import contextlib
import sys
import time
import traceback

import numpy as np

from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED,
    RunRequest,
    RunResult,
)

# EXPECTED NetLogo reporter strings -- these MUST be confirmed against the
# fetched DelliBAM_.nlogo model in operator Task 3 (NOT yet verified). The
# breed name for firms is assumed to be ``firms``; if Task 3 finds a
# different breed, update the two ``of firms`` reporters below accordingly.
REPORTERS = {
    "unemployment": "fn-unemployment-rate",
    "price_inflation": "annualized-inflation",
    "avg_wage": "mean [wage-offered-Wb] of firms",
    "real_gdp": "real-GDP",
    "total_vacancies": "sum [number-of-vacancies-offered-V] of firms",
    "production_final": "[production-Y] of firms",
}

# bamengine canonical param key -> NetLogo global. Only these economic params are
# overridden; all others stay at Platas defaults (documented residuals).
GLOBAL_MAP = {
    "max_M": "labor-market-M",
    "max_H": "credit-market-H",
    "max_Z": "goods-market-Z",
}

# Per-tick series collected in gate mode (production_final is read once at end).
_TICK_SERIES = (
    "unemployment",
    "price_inflation",
    "avg_wage",
    "real_gdp",
    "total_vacancies",
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


def collect_run(link, req: RunRequest) -> tuple[dict, dict | None]:
    """Drive a loaded NetLogo model for one request; return (timing, outputs).

    ``link`` must expose ``command(str)``, ``report(str)``, ``kill_workspace()``.
    Globals and seed are set by ``main`` before ``setup``; this function issues
    ``setup`` then the run loop. In gate mode it collects per-tick reporters and
    a final per-firm production snapshot; in timing mode it runs an untimed
    warmup loop then a timed loop and returns no outputs.
    """
    n_workers = int(req.population["n_households"])
    n_agents = sum(int(v) for v in req.population.values())

    # init timing wraps setup (model construction for this run).
    t_init0 = time.perf_counter()
    link.command("setup")
    t_init1 = time.perf_counter()
    init_seconds = t_init1 - t_init0

    outputs: dict | None = None

    if req.collect_outputs:
        raw: dict[str, list] = {k: [] for k in _TICK_SERIES}
        t_run0 = time.perf_counter()
        for _ in range(req.n_periods):
            link.command("go")
            for key in _TICK_SERIES:
                raw[key].append(float(link.report(REPORTERS[key])))
        t_run1 = time.perf_counter()
        run_seconds = t_run1 - t_run0
        timed_periods = req.n_periods
        production_final = list(link.report(REPORTERS["production_final"]))
        outputs = build_series(raw, production_final, n_workers)
    else:
        warmup = max(0, req.warmup_periods)
        timed_periods = max(1, req.n_periods - warmup)
        for _ in range(warmup):  # untimed: absorb JVM JIT + interpreter warmup
            link.command("go")
        t_run0 = time.perf_counter()
        for _ in range(timed_periods):
            link.command("go")
        t_run1 = time.perf_counter()
        run_seconds = t_run1 - t_run0

    steady = run_seconds / timed_periods if timed_periods else run_seconds
    throughput = n_agents * timed_periods / run_seconds if run_seconds > 0 else 0.0

    timing = {
        "init_seconds": init_seconds,
        "run_seconds": run_seconds,
        "steady_state_per_period_seconds": steady,
        "throughput_agent_steps_per_s": throughput,
    }
    return timing, outputs


def main(request_path: str) -> None:
    """Run one request and print a RunResult JSON to stdout."""
    with open(request_path) as fh:
        req = RunRequest.from_json(fh.read())

    try:
        import pynetlogo
    except ImportError:
        print(
            _skeleton(
                req, STATUS_SKIPPED, "pynetlogo/NetLogo toolchain not available"
            ).to_json()
        )
        return

    import os
    from importlib.metadata import PackageNotFoundError, version

    try:
        pynetlogo_version = version("pynetlogo")
    except PackageNotFoundError:
        pynetlogo_version = "unknown"

    n_agents = sum(int(v) for v in req.population.values())
    netlogo_home = os.environ.get("NETLOGO_HOME") or None
    link = None
    try:
        # JVM boot + link construction is startup (captured by the orchestrator as
        # wall - init - run), not counted in init_seconds.
        link = pynetlogo.NetLogoLink(gui=False, netlogo_home=netlogo_home)
        link.load_model(
            os.path.join(os.path.dirname(__file__), "model", "DelliBAM_.nlogo")
        )

        # Set only the cleanly-mappable globals, then the seed.
        link.command(f"set number-of-firms {int(req.population['n_firms'])}")
        for key, nl_global in GLOBAL_MAP.items():
            if key in req.model_params:
                link.command(f"set {nl_global} {int(req.model_params[key])}")
        link.command(f"random-seed {int(req.seed)}")

        try:
            netlogo_version = str(link.netlogo_version)
        except Exception:
            netlogo_version = "unknown"

        timing, outputs = collect_run(link, req)

        res = RunResult(
            schema_version=SCHEMA_VERSION,
            run_id=req.run_id,
            framework="netlogo",
            framework_version=pynetlogo_version,
            language="netlogo",
            language_version=netlogo_version,
            status=STATUS_OK,
            error=None,
            population={**req.population, "n_agents_total": n_agents},
            n_periods=req.n_periods,
            warmup_periods=req.warmup_periods,
            seed=req.seed,
            timing=timing,
            outputs=outputs,
        )
    except Exception:
        res = _skeleton(req, STATUS_ERROR, traceback.format_exc())
    finally:
        if link is not None:
            with contextlib.suppress(Exception):
                link.kill_workspace()

    print(res.to_json())


if __name__ == "__main__":
    main(sys.argv[1])
