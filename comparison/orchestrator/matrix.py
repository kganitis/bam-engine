from __future__ import annotations

from comparison.orchestrator.contract import RunRequest
from comparison.orchestrator.params import canonical_params, population_for

SCALE_FIRMS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000)
GATE_SEEDS = tuple(range(20))
TIMING_REPS = 7
GATE_N_PERIODS, GATE_WARMUP = 1000, 0
TIMING_N_PERIODS, TIMING_WARMUP = 1000, 100
SERIES = [
    "unemployment",
    "price_inflation",
    "wage_inflation",
    "log_gdp",
    "vacancy_rate",
    "production_final",
]


def equivalence_jobs(frameworks: list[str]) -> list[RunRequest]:
    """Build gate (equivalence) run requests for the given frameworks.

    One job per (framework, seed) combination. All jobs use the smallest
    population size (100 firms) so they finish quickly in parallel.

    Parameters
    ----------
    frameworks : list[str]
        Framework names to include (e.g. ``["bamengine"]``).

    Returns
    -------
    list[RunRequest]
        One :class:`RunRequest` per framework × seed pair.
    """
    params = canonical_params()
    out = []
    for fw in frameworks:
        for seed in GATE_SEEDS:
            out.append(
                RunRequest(
                    run_id=f"{fw}__gate__seed{seed}",
                    framework=fw,
                    model_params=params,
                    population=population_for(100),
                    n_periods=GATE_N_PERIODS,
                    warmup_periods=GATE_WARMUP,
                    seed=seed,
                    collect_outputs=True,
                    outputs_requested=list(SERIES),
                )
            )
    return out


def timing_jobs(frameworks: list[str]) -> list[RunRequest]:
    """Build timing run requests for the given frameworks.

    One job per (framework, n_firms, rep) combination across all sizes and reps.
    These jobs do not collect outputs (timing mode only).

    Parameters
    ----------
    frameworks : list[str]
        Framework names to include.

    Returns
    -------
    list[RunRequest]
        One :class:`RunRequest` per framework × size × rep triple.
    """
    params = canonical_params()
    out = []
    for fw in frameworks:
        for n_firms in SCALE_FIRMS:
            for rep in range(TIMING_REPS):
                out.append(
                    RunRequest(
                        run_id=f"{fw}__s{n_firms}__rep{rep}",
                        framework=fw,
                        model_params=params,
                        population=population_for(n_firms),
                        n_periods=TIMING_N_PERIODS,
                        warmup_periods=TIMING_WARMUP,
                        seed=1000 + rep,
                        collect_outputs=False,
                        outputs_requested=[],
                    )
                )
    return out
