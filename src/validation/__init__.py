"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Subpackages:
    scenarios/: Scenario definitions and visualizations
        - baseline/: Baseline scenario (Section 3.9.1)
        - growth_plus/: Growth+ scenario (Section 3.9.2)
        - buffer_stock/: Buffer-stock scenario (Section 3.9.4)
    robustness/: Robustness analysis (Section 3.10)
        - Internal validity (multi-seed stability)
        - Sensitivity analysis (univariate parameter sweeps)

Modules:
    types: Core types, dataclasses, and enums
    scoring: Scoring and status check functions
    engine: Generic validation engine
    reporting: Report printing functions

Usage:
    from validation import run_validation, run_stability_test

    # Compare different configurations (baseline)
    score_a = run_validation(seed=0)
    score_b = run_validation(seed=0, h_rho=0.15)
    print(f"Default: {score_a.total_score:.3f}")
    print(f"Modified: {score_b.total_score:.3f}")

    # Test stability across seeds
    result = run_stability_test(seeds=[0, 42, 123, 456, 789])
    print(f"Mean score: {result.mean_score:.3f} ± {result.std_score:.3f}")

    # Growth+ scenario
    from validation import run_growth_plus_validation
    score = run_growth_plus_validation(seed=0)
    print(f"Growth+ score: {score.total_score:.3f}")

    # Buffer-stock scenario
    from validation import run_buffer_stock_validation
    score = run_buffer_stock_validation(seed=0)
    print(f"Buffer-stock score: {score.total_score:.3f}")

    # Run scenarios with visualization
    from validation import run_baseline_scenario, run_growth_plus_scenario
    run_baseline_scenario(seed=0, show_plot=True)
    run_growth_plus_scenario(seed=2, show_plot=True)

    # For extensions, import from extensions package:
    from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG
    from extensions.buffer_stock import BufferStock, BUFFER_STOCK_EVENTS, BUFFER_STOCK_CONFIG
"""

from __future__ import annotations

from functools import partial
from typing import Any

# Engine functions
from validation.engine import evaluate_metric, load_targets, stability_test, validate

# Reporting functions (generic)
from validation.reporting import print_report, print_stability_report

# Scenario registry
from validation.scenarios import get_scenario

# Scoring functions
from validation.scoring import (
    check_improvement,
    check_mean_tolerance,
    check_outlier_penalty,
    check_pct_within_target,
    check_range,
    compute_combined_score,
    score_improvement,
    score_mean_tolerance,
    score_outlier_penalty,
    score_pct_within_target,
    score_range,
)
from validation.types import (
    DEFAULT_STABILITY_SEEDS,
    BufferStockValidationScore,
    CheckType,
    MetricFormat,
    MetricGroup,
    MetricResult,
    MetricSpec,
    MetricStats,
    Scenario,
    StabilityResult,
    Status,
    ValidationScore,
)

# =============================================================================
# Weights (derived from MetricSpecs for backwards compatibility)
# =============================================================================


def _derive_weights(specs: list[MetricSpec]) -> dict[str, float]:
    """Derive weights dictionary from MetricSpecs."""
    return {spec.name: spec.weight for spec in specs}


BASELINE_WEIGHTS = _derive_weights(get_scenario("baseline").metric_specs)
GROWTH_PLUS_WEIGHTS = _derive_weights(get_scenario("growth_plus").metric_specs)
BUFFER_STOCK_WEIGHTS = _derive_weights(get_scenario("buffer_stock").metric_specs)


# =============================================================================
# Factory-generated wrapper functions (backwards-compatible)
# =============================================================================


def _make_validate(scenario_name: str):  # type: ignore[no-untyped-def]
    def _validate(
        *, seed: int = 0, n_periods: int = 1000, **config_overrides: Any
    ) -> ValidationScore:
        return validate(
            get_scenario(scenario_name),
            seed=seed,
            n_periods=n_periods,
            **config_overrides,
        )

    _validate.__doc__ = f"Run {scenario_name} validation and return scored result."
    return _validate


def _make_stability(scenario_name: str):  # type: ignore[no-untyped-def]
    def _stability(
        seeds: list[int] | int = 5,
        n_periods: int = 1000,
        n_workers: int = 1,
        **config_overrides: Any,
    ) -> StabilityResult:
        return stability_test(
            get_scenario(scenario_name),
            seeds=seeds,
            n_periods=n_periods,
            n_workers=n_workers,
            **config_overrides,
        )

    _stability.__doc__ = f"Run {scenario_name} validation across multiple seeds."
    return _stability


def _make_scenario_runner(scenario_name: str):  # type: ignore[no-untyped-def]
    def _runner(**kwargs: Any) -> Any:
        import importlib

        mod = importlib.import_module(f"validation.scenarios.{scenario_name}")
        return mod.run_scenario(**kwargs)

    _runner.__doc__ = (
        f"Run {scenario_name} scenario with visualization.\n\n"
        f"See validation.scenarios.{scenario_name}.run_scenario for parameters."
    )
    return _runner


# Backwards-compatible names
run_validation = _make_validate("baseline")
run_stability_test = _make_stability("baseline")
run_growth_plus_validation = _make_validate("growth_plus")
run_growth_plus_stability_test = _make_stability("growth_plus")


# =============================================================================
# Buffer-stock validation (custom — not factory-generated)
# =============================================================================


def run_buffer_stock_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    growth_plus_result: ValidationScore | None = None,
    **config_overrides: Any,
) -> BufferStockValidationScore:
    """Run buffer-stock validation for a single seed.

    Per-seed PASS/FAIL is determined by 8 unique metrics only.
    Improvement deltas vs Growth+ are stored for informational use.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    growth_plus_result : ValidationScore or None
        Optional pre-computed Growth+ result for the same seed.
        If ``None``, Growth+ validation is run internally.
    **config_overrides
        Simulation config overrides.

    Returns
    -------
    BufferStockValidationScore
        Result with 8 unique metrics (PASS/FAIL) and improvement deltas
        (informational).
    """
    from validation.scenarios.buffer_stock import validate_buffer_stock

    return validate_buffer_stock(
        seed=seed,
        n_periods=n_periods,
        growth_plus_result=growth_plus_result,
        **config_overrides,
    )


def run_buffer_stock_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    n_workers: int = 1,
    growth_plus_results: dict[int, ValidationScore] | None = None,
    **config_overrides: Any,
) -> StabilityResult:
    """Run buffer-stock validation across multiple seeds.

    Parameters
    ----------
    seeds : list[int] or int
        List of seeds or number of seeds to test.
    n_periods : int
        Number of simulation periods per seed.
    n_workers : int
        Number of parallel workers.
    growth_plus_results : dict or None
        Optional dict mapping seed -> pre-computed Growth+ ValidationScore.
        Seeds not found in the dict are run internally.
    **config_overrides
        Simulation config overrides.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    import math

    import numpy as np

    from validation.scenarios.buffer_stock import validate_buffer_stock

    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = seeds

    gp_lookup = growth_plus_results or {}

    seed_results: list[ValidationScore] = []

    if n_workers > 1 and len(seed_list) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        overrides = dict(config_overrides)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _validate_buffer_stock_seed,
                    seed,
                    n_periods,
                    gp_lookup.get(seed),
                    overrides,
                ): seed
                for seed in seed_list
            }
            for future in as_completed(futures):
                result = future.result()
                seed_results.append(result)
        seed_results.sort(key=lambda r: r.config.get("seed", 0))
    else:
        for seed in seed_list:
            result = validate_buffer_stock(
                seed=seed,
                n_periods=n_periods,
                growth_plus_result=gp_lookup.get(seed),
                **config_overrides,
            )
            seed_results.append(result)

    # Filter out collapsed seeds (NaN total_score)
    valid_results = [r for r in seed_results if not math.isnan(r.total_score)]

    if not valid_results:
        return StabilityResult(
            seed_results=seed_results,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            pass_rate=0.0,
            n_seeds=len(seed_results),
            metric_stats={},
        )

    scores = [r.total_score for r in valid_results]
    n_passed = sum(1 for r in valid_results if r.passed)

    # Per-metric stats (only from valid results)
    metric_names = [m.name for m in valid_results[0].metric_results]
    metric_stats: dict[str, MetricStats] = {}
    for idx, name in enumerate(metric_names):
        values = [r.metric_results[idx].actual for r in valid_results]
        scores_for = [r.metric_results[idx].score for r in valid_results]
        statuses = [r.metric_results[idx].status for r in valid_results]
        metric_stats[name] = MetricStats(
            name=name,
            mean_value=float(np.mean(values)),
            std_value=float(np.std(values)),
            mean_score=float(np.mean(scores_for)),
            std_score=float(np.std(scores_for)),
            pass_rate=sum(1 for s in statuses if s != "FAIL") / len(statuses),
            format=valid_results[0].metric_results[idx].format,
        )

    # Aggregate improvement check: compute mean deltas across all seeds
    # and check for systematic degradation (much more stable than per-seed).
    bs_results = [r for r in valid_results if isinstance(r, BufferStockValidationScore)]
    aggregate_degraded: list[str] = []
    if bs_results:
        # Load threshold config
        import yaml

        from validation.scoring import check_improvement

        targets_path = get_scenario("buffer_stock").targets_path
        with open(targets_path) as f:
            bs_targets = yaml.safe_load(f)
        imp_config = bs_targets.get("improvement", {})
        max_deg_base = imp_config.get("max_degradation_base", 0.25)

        # Collect all per-seed deltas and compute mean
        all_delta_keys = bs_results[0].improvement_deltas.keys()
        gp_specs = {s.name: s.weight for s in get_scenario("growth_plus").metric_specs}

        for metric_name in all_delta_keys:
            seed_deltas = [r.improvement_deltas[metric_name] for r in bs_results]
            mean_delta = float(np.mean(seed_deltas))
            weight = gp_specs.get(metric_name, 1.0)
            status = check_improvement(mean_delta, weight, max_deg_base)
            if status == "FAIL":
                aggregate_degraded.append(metric_name)

        # Store aggregate degradation on each seed result for reporting
        for r in bs_results:
            r.degraded_metrics = aggregate_degraded

    stability = StabilityResult(
        seed_results=seed_results,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        min_score=float(np.min(scores)),
        max_score=float(np.max(scores)),
        pass_rate=n_passed / len(valid_results),
        n_seeds=len(seed_results),
        metric_stats=metric_stats,
    )

    if aggregate_degraded:
        import warnings

        warnings.warn(
            f"Aggregate improvement check: {len(aggregate_degraded)} Growth+ "
            f"metric(s) systematically degraded: {aggregate_degraded}",
            stacklevel=2,
        )

    return stability


def _validate_buffer_stock_seed(
    seed: int,
    n_periods: int,
    growth_plus_result: ValidationScore | None,
    config_overrides: dict[str, Any],
) -> BufferStockValidationScore:
    """Run buffer-stock validation for a single seed. Module-level for pickling."""
    from validation.scenarios.buffer_stock import validate_buffer_stock

    return validate_buffer_stock(
        seed=seed,
        n_periods=n_periods,
        growth_plus_result=growth_plus_result,
        **config_overrides,
    )


# =============================================================================
# Report printers (partial from Scenario.title)
# =============================================================================

print_validation_report = partial(print_report, title=get_scenario("baseline").title)
print_baseline_stability_report = partial(
    print_stability_report, title=get_scenario("baseline").stability_title
)
print_growth_plus_report = partial(
    print_report, title=get_scenario("growth_plus").title
)
print_growth_plus_stability_report = partial(
    print_stability_report, title=get_scenario("growth_plus").stability_title
)
print_buffer_stock_report = partial(
    print_report, title=get_scenario("buffer_stock").title
)
print_buffer_stock_stability_report = partial(
    print_stability_report, title=get_scenario("buffer_stock").stability_title
)

# =============================================================================
# Scenario Runner Functions (with visualization — lazy imports)
# =============================================================================

run_baseline_scenario = _make_scenario_runner("baseline")
run_growth_plus_scenario = _make_scenario_runner("growth_plus")
run_buffer_stock_scenario = _make_scenario_runner("buffer_stock")


# =============================================================================
# Calibration Package Support
# =============================================================================


def get_validation_funcs(
    scenario: str = "baseline",
) -> tuple[Any, Any, Any, Any]:
    """Get validation functions for a scenario (for calibration).

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. "baseline", "growth_plus", "buffer_stock").

    Returns
    -------
    tuple
        (run_validation_func, run_stability_func, print_report_func, print_stability_func)
    """
    s = get_scenario(scenario)
    return (
        _make_validate(scenario),
        _make_stability(scenario),
        partial(print_report, title=s.title),
        partial(print_stability_report, title=s.stability_title),
    )


def get_validation_func(scenario: str = "baseline") -> Any:
    """Get the validation function for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. "baseline", "growth_plus", "buffer_stock").

    Returns
    -------
    Callable
        The run_validation function for the scenario.
    """
    return get_validation_funcs(scenario)[0]


__all__ = [
    # Core types
    "Status",
    "CheckType",
    "MetricGroup",
    "MetricFormat",
    "MetricSpec",
    "MetricResult",
    "ValidationScore",
    "BufferStockValidationScore",
    "MetricStats",
    "StabilityResult",
    "Scenario",
    # Constants
    "DEFAULT_STABILITY_SEEDS",
    "BASELINE_WEIGHTS",
    "GROWTH_PLUS_WEIGHTS",
    "BUFFER_STOCK_WEIGHTS",
    # Scoring functions
    "score_mean_tolerance",
    "score_range",
    "score_pct_within_target",
    "score_outlier_penalty",
    "score_improvement",
    "check_mean_tolerance",
    "check_range",
    "check_pct_within_target",
    "check_outlier_penalty",
    "check_improvement",
    "compute_combined_score",
    # Engine functions
    "validate",
    "stability_test",
    "evaluate_metric",
    "load_targets",
    # Report functions
    "print_report",
    "print_stability_report",
    "print_validation_report",
    "print_baseline_stability_report",
    "print_growth_plus_report",
    "print_growth_plus_stability_report",
    "print_buffer_stock_report",
    "print_buffer_stock_stability_report",
    # Wrapper functions
    "run_validation",
    "run_stability_test",
    "run_growth_plus_validation",
    "run_growth_plus_stability_test",
    "run_buffer_stock_validation",
    "run_buffer_stock_stability_test",
    # Scenario visualization
    "run_baseline_scenario",
    "run_growth_plus_scenario",
    "run_buffer_stock_scenario",
    # Calibration support
    "get_validation_funcs",
    "get_validation_func",
    # Registry
    "get_scenario",
]
