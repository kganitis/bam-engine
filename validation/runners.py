"""Validation runner functions.

This module provides functions to run validation tests and generate reports
for both baseline and Growth+ scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

import bamengine as bam
from validation.core import (
    BASELINE_WEIGHTS,
    GROWTH_PLUS_WEIGHTS,
    MetricResult,
    MetricStats,
    StabilityResult,
    ValidationScore,
    check_mean_tolerance,
    check_range,
    score_mean_tolerance,
    score_range,
)
from validation.metrics import (
    BASELINE_COLLECT_CONFIG,
    GROWTH_PLUS_COLLECT_CONFIG,
    compute_baseline_metrics,
    compute_growth_plus_metrics,
)

# =============================================================================
# Baseline Scenario Runners
# =============================================================================


def run_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    weights: dict[str, float] | None = None,
    **config_overrides: Any,
) -> ValidationScore:
    """Run validation and return a scored result.

    This function allows programmatic comparison of different configurations.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    weights : dict, optional
        Custom weights for metrics. Defaults to BASELINE_WEIGHTS.
    **config_overrides
        Any simulation config parameters to override (e.g., h_rho=0.15).

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.

    Example
    -------
    >>> score_a = run_validation(seed=0)
    >>> score_b = run_validation(seed=0, h_rho=0.15)
    >>> print(f"Default: {score_a.total_score:.3f}")
    >>> print(f"Modified: {score_b.total_score:.3f}")
    """
    if weights is None:
        weights = BASELINE_WEIGHTS

    # Load targets
    targets_path = Path(__file__).parent / "targets/baseline.yaml"
    with open(targets_path) as f:
        targets = yaml.safe_load(f)

    # Build config
    config = {
        "n_firms": 300,
        "n_households": 3000,
        "n_banks": 10,
        "n_periods": n_periods,
        "seed": seed,
        "logging": {"default_level": "ERROR"},
        **config_overrides,
    }

    # Run simulation
    sim = bam.Simulation.init(**config)
    results = sim.run(collect=BASELINE_COLLECT_CONFIG)

    # Compute metrics
    burn_in = targets["metadata"]["validation"]["burn_in_periods"]
    firm_threshold = targets["distributions"]["firm_size"]["targets"]["threshold_small"]
    metrics = compute_baseline_metrics(
        sim, results, burn_in=burn_in, firm_size_threshold=firm_threshold
    )

    # Validate each metric
    validation_results: list[MetricResult] = []
    ts = targets["time_series"]
    curves = targets["curves"]

    # --- Time Series Metrics ---

    # Unemployment
    u = ts["unemployment_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "unemployment_rate_mean",
            status,
            metrics.unemployment_mean,
            f"target: {u['mean_target']:.4f} ± {u['mean_tolerance']:.4f}",
            score,
            weights.get("unemployment_rate_mean", 1.0),
        )
    )

    # Inflation
    i = ts["inflation_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "inflation_rate_mean",
            status,
            metrics.inflation_mean,
            f"target: {i['mean_target']:.4f} ± {i['mean_tolerance']:.4f}",
            score,
            weights.get("inflation_rate_mean", 1.0),
        )
    )

    # Log GDP
    g = ts["log_gdp"]["targets"]
    status = check_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "log_gdp_mean",
            status,
            metrics.log_gdp_mean,
            f"target: {g['mean_target']:.4f} ± {g['mean_tolerance']:.4f}",
            score,
            weights.get("log_gdp_mean", 1.0),
        )
    )

    # Real wage
    w = ts["real_wage"]["targets"]
    status = check_mean_tolerance(
        metrics.real_wage_mean, w["mean_target"], w["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.real_wage_mean, w["mean_target"], w["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "real_wage_mean",
            status,
            metrics.real_wage_mean,
            f"target: {w['mean_target']:.4f} ± {w['mean_tolerance']:.4f}",
            score,
            weights.get("real_wage_mean", 1.0),
        )
    )

    # Vacancy rate
    v = targets["distributions"]["vacancy_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "vacancy_rate_mean",
            status,
            metrics.vacancy_rate_mean,
            f"target: {v['mean_target']:.4f} ± {v['mean_tolerance']:.4f}",
            score,
            weights.get("vacancy_rate_mean", 1.0),
        )
    )

    # --- Curve Correlations ---

    # Phillips
    p = curves["phillips"]["targets"]
    status = check_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    score = score_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "phillips_correlation",
            status,
            metrics.phillips_corr,
            f"range: [{p['correlation_min']:.2f}, {p['correlation_max']:.2f}]",
            score,
            weights.get("phillips_correlation", 1.0),
        )
    )

    # Okun
    o = curves["okun"]["targets"]
    status = check_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    score = score_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    validation_results.append(
        MetricResult(
            "okun_correlation",
            status,
            metrics.okun_corr,
            f"range: [{o['correlation_min']:.2f}, {o['correlation_max']:.2f}]",
            score,
            weights.get("okun_correlation", 1.0),
        )
    )

    # Beveridge
    b = curves["beveridge"]["targets"]
    status = check_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    score = score_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "beveridge_correlation",
            status,
            metrics.beveridge_corr,
            f"range: [{b['correlation_min']:.2f}, {b['correlation_max']:.2f}]",
            score,
            weights.get("beveridge_correlation", 1.0),
        )
    )

    # --- Distribution Metrics ---
    d = targets["distributions"]["firm_size"]["targets"]

    # Skewness
    status = check_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    score = score_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    validation_results.append(
        MetricResult(
            "firm_size_skewness",
            status,
            metrics.firm_size_skewness,
            f"range: [{d['skewness_min']:.1f}, {d['skewness_max']:.1f}]",
            score,
            weights.get("firm_size_skewness", 1.0),
        )
    )

    # Percentile threshold
    status = check_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    score = score_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    validation_results.append(
        MetricResult(
            "firm_size_pct_below",
            status,
            metrics.firm_size_pct_below_threshold,
            f"range: [{d['pct_below_small_min'] * 100:.0f}%, {d['pct_below_small_max'] * 100:.0f}%]",
            score,
            weights.get("firm_size_pct_below", 1.0),
        )
    )

    # Compute totals
    n_pass = sum(1 for r in validation_results if r.status == "PASS")
    n_warn = sum(1 for r in validation_results if r.status == "WARN")
    n_fail = sum(1 for r in validation_results if r.status == "FAIL")

    # Compute weighted total score
    total_weight = sum(r.weight for r in validation_results)
    total_score = sum(r.score * r.weight for r in validation_results) / total_weight

    return ValidationScore(
        metric_results=validation_results,
        total_score=total_score,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        config=config,
    )


def print_validation_report(result: ValidationScore) -> None:
    """Print formatted validation report to stdout."""
    print("\n" + "=" * 78)
    print("BASELINE SCENARIO VALIDATION")
    print("=" * 78)

    print("\nTIME SERIES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[:5]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nCURVES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[5:8]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nDISTRIBUTION:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[8:]:
        if "pct" in r.name:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual * 100:>7.1f}%  {r.score:>6.3f}  ({r.target_desc})"
            )
        else:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
            )

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {result.n_pass} PASS, {result.n_warn} WARN, {result.n_fail} FAIL  |  "
        f"TOTAL SCORE: {result.total_score:.3f}"
    )
    print("=" * 78 + "\n")


def run_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    weights: dict[str, float] | None = None,
    **config_overrides: Any,
) -> StabilityResult:
    """Run validation across multiple seeds and measure consistency.

    Parameters
    ----------
    seeds : list[int] or int
        List of specific seeds to test, or number of seeds to generate.
        If int, uses seeds [0, 1, 2, ..., seeds-1].
    n_periods : int
        Number of simulation periods per seed.
    weights : dict, optional
        Custom weights for metrics.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.

    Example
    -------
    >>> result = run_stability_test(seeds=[0, 42, 123, 456, 789])
    >>> print(f"Mean score: {result.mean_score:.3f} ± {result.std_score:.3f}")
    >>> print(f"Pass rate: {result.pass_rate:.0%}")
    """
    # Handle seeds parameter
    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = seeds

    # Run validation for each seed
    seed_results: list[ValidationScore] = []
    for seed in seed_list:
        result = run_validation(
            seed=seed,
            n_periods=n_periods,
            weights=weights,
            **config_overrides,
        )
        seed_results.append(result)

    # Compute aggregate score metrics
    scores = [r.total_score for r in seed_results]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    # Compute pass rate
    n_passed = sum(1 for r in seed_results if r.passed)
    pass_rate = n_passed / len(seed_results)

    # Compute per-metric statistics
    metric_names = [m.name for m in seed_results[0].metric_results]
    metric_stats: dict[str, MetricStats] = {}

    for idx, name in enumerate(metric_names):
        values = [r.metric_results[idx].actual for r in seed_results]
        scores_for_metric = [r.metric_results[idx].score for r in seed_results]
        statuses = [r.metric_results[idx].status for r in seed_results]

        metric_stats[name] = MetricStats(
            name=name,
            mean_value=float(np.mean(values)),
            std_value=float(np.std(values)),
            mean_score=float(np.mean(scores_for_metric)),
            std_score=float(np.std(scores_for_metric)),
            pass_rate=sum(1 for s in statuses if s != "FAIL") / len(statuses),
        )

    return StabilityResult(
        seed_results=seed_results,
        mean_score=mean_score,
        std_score=std_score,
        min_score=min_score,
        max_score=max_score,
        pass_rate=pass_rate,
        n_seeds=len(seed_results),
        metric_stats=metric_stats,
    )


def print_stability_report(result: StabilityResult) -> None:
    """Print formatted stability test report to stdout."""
    print("\n" + "=" * 78)
    print(f"SEED STABILITY TEST ({result.n_seeds} seeds)")
    print("=" * 78)

    print("\nAGGREGATE SCORES:")
    print(f"  Mean Score:    {result.mean_score:.3f} ± {result.std_score:.3f}")
    print(f"  Score Range:   [{result.min_score:.3f}, {result.max_score:.3f}]")
    n_passed = int(result.pass_rate * result.n_seeds)
    print(
        f"  Pass Rate:     {result.pass_rate:.0%} ({n_passed}/{result.n_seeds} seeds passed)"
    )

    print("\nPER-METRIC STABILITY:")
    print(f"  {'Metric':<28} {'Mean':>10} {'Std':>8} {'Score':>7} {'Pass%':>7}")
    print("  " + "-" * 62)

    for name, stats in result.metric_stats.items():
        # Flag unstable metrics
        flag = ""
        if stats.pass_rate < 0.8:
            flag = " <- unstable"
        elif stats.std_score > 0.1:
            flag = " <- variable"

        if "pct" in name:
            print(
                f"  {name:<28} {stats.mean_value * 100:>9.1f}% {stats.std_value * 100:>7.2f}% "
                f"{stats.mean_score:>7.3f} {stats.pass_rate:>6.0%}{flag}"
            )
        else:
            print(
                f"  {name:<28} {stats.mean_value:>10.4f} {stats.std_value:>8.4f} "
                f"{stats.mean_score:>7.3f} {stats.pass_rate:>6.0%}{flag}"
            )

    print("\n" + "=" * 78)
    stability_status = "PASS" if result.is_stable else "WARN"
    print(
        f"STABILITY: {stability_status} "
        f"(pass_rate={result.pass_rate:.0%}, std={result.std_score:.3f})"
    )
    print("=" * 78 + "\n")


# =============================================================================
# Growth+ Scenario Runners
# =============================================================================


def _import_rnd_role() -> Any:
    """Import RnD role from example without sys.path pollution.

    Uses importlib to load the module directly from file path,
    avoiding global sys.path modifications.
    """
    import importlib.util

    example_path = (
        Path(__file__).parent.parent
        / "examples"
        / "extensions"
        / "example_growth_plus.py"
    )
    spec = importlib.util.spec_from_file_location("example_growth_plus", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RnD


def run_growth_plus_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    weights: dict[str, float] | None = None,
    **config_overrides: Any,
) -> ValidationScore:
    """Run Growth+ validation and return a scored result.

    This function allows programmatic comparison of different configurations.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    weights : dict, optional
        Custom weights for metrics. Defaults to GROWTH_PLUS_WEIGHTS.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.
    """
    if weights is None:
        weights = GROWTH_PLUS_WEIGHTS

    # Load targets
    targets_path = Path(__file__).parent / "targets/growth_plus.yaml"
    with open(targets_path) as f:
        targets = yaml.safe_load(f)

    # Build config with Growth+ extension parameters
    config = {
        "n_firms": 100,  # Book values (no scaling)
        "n_households": 500,
        "n_banks": 10,
        "n_periods": n_periods,
        "seed": seed,
        "logging": {"default_level": "ERROR"},
        "new_firm_size_factor": 0.5,
        "new_firm_production_factor": 0.5,
        "new_firm_wage_factor": 0.5,
        "new_firm_price_markup": 1.5,
        # R&D extension parameters
        "sigma_min": targets["metadata"]["extension_params"]["sigma_min"],
        "sigma_max": targets["metadata"]["extension_params"]["sigma_max"],
        "sigma_decay": -1.0,
        **config_overrides,
    }

    # Import RnD role BEFORE creating simulation so @event(after=...) hooks work
    RnD = _import_rnd_role()

    # Run simulation with RnD extension
    sim = bam.Simulation.init(**config)
    sim.use_role(RnD)  # Attach custom RnD role
    results = sim.run(collect=GROWTH_PLUS_COLLECT_CONFIG)

    # Compute metrics
    burn_in = targets["metadata"]["validation"]["burn_in_periods"]
    # Adjust burn_in if n_periods is too short (e.g., for quick tests)
    if burn_in >= n_periods:
        burn_in = max(0, n_periods // 2)
    firm_threshold = targets["distributions"]["firm_size"]["targets"]["threshold_small"]
    metrics = compute_growth_plus_metrics(
        sim, results, burn_in=burn_in, firm_size_threshold=firm_threshold
    )

    # Validate each metric
    validation_results: list[MetricResult] = []
    ts = targets["time_series"]
    curves = targets["curves"]

    # --- Time Series Metrics ---

    # Unemployment
    u = ts["unemployment_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "unemployment_rate_mean",
            status,
            metrics.unemployment_mean,
            f"target: {u['mean_target']:.4f} ± {u['mean_tolerance']:.4f}",
            score,
            weights.get("unemployment_rate_mean", 1.0),
        )
    )

    # Inflation
    i = ts["inflation_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "inflation_rate_mean",
            status,
            metrics.inflation_mean,
            f"target: {i['mean_target']:.4f} ± {i['mean_tolerance']:.4f}",
            score,
            weights.get("inflation_rate_mean", 1.0),
        )
    )

    # Log GDP
    g = ts["log_gdp"]["targets"]
    status = check_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "log_gdp_mean",
            status,
            metrics.log_gdp_mean,
            f"target: {g['mean_target']:.4f} ± {g['mean_tolerance']:.4f}",
            score,
            weights.get("log_gdp_mean", 1.0),
        )
    )

    # Vacancy rate
    v = targets["distributions"]["vacancy_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "vacancy_rate_mean",
            status,
            metrics.vacancy_rate_mean,
            f"target: {v['mean_target']:.4f} ± {v['mean_tolerance']:.4f}",
            score,
            weights.get("vacancy_rate_mean", 1.0),
        )
    )

    # --- Curve Correlations ---

    # Phillips (stronger in Growth+: -0.19)
    p = curves["phillips"]["targets"]
    status = check_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    score = score_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "phillips_correlation",
            status,
            metrics.phillips_corr,
            f"range: [{p['correlation_min']:.2f}, {p['correlation_max']:.2f}]",
            score,
            weights.get("phillips_correlation", 1.0),
        )
    )

    # Okun
    o = curves["okun"]["targets"]
    status = check_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    score = score_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    validation_results.append(
        MetricResult(
            "okun_correlation",
            status,
            metrics.okun_corr,
            f"range: [{o['correlation_min']:.2f}, {o['correlation_max']:.2f}]",
            score,
            weights.get("okun_correlation", 1.0),
        )
    )

    # Beveridge
    b = curves["beveridge"]["targets"]
    status = check_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    score = score_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "beveridge_correlation",
            status,
            metrics.beveridge_corr,
            f"range: [{b['correlation_min']:.2f}, {b['correlation_max']:.2f}]",
            score,
            weights.get("beveridge_correlation", 1.0),
        )
    )

    # --- Distribution Metrics ---
    d = targets["distributions"]["firm_size"]["targets"]

    # Skewness
    status = check_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    score = score_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    validation_results.append(
        MetricResult(
            "firm_size_skewness",
            status,
            metrics.firm_size_skewness,
            f"range: [{d['skewness_min']:.1f}, {d['skewness_max']:.1f}]",
            score,
            weights.get("firm_size_skewness", 1.0),
        )
    )

    # Percentile threshold
    status = check_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    score = score_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    validation_results.append(
        MetricResult(
            "firm_size_pct_below",
            status,
            metrics.firm_size_pct_below_threshold,
            f"range: [{d['pct_below_small_min'] * 100:.0f}%, {d['pct_below_small_max'] * 100:.0f}%]",
            score,
            weights.get("firm_size_pct_below", 1.0),
        )
    )

    # --- Growth+ Specific Metrics ---

    # Productivity growth
    prod = ts["productivity"]["targets"]
    status = check_range(
        metrics.total_productivity_growth,
        prod["total_growth_min"],
        prod["total_growth_max"],
    )
    score = score_range(
        metrics.total_productivity_growth,
        prod["total_growth_min"],
        prod["total_growth_max"],
    )
    validation_results.append(
        MetricResult(
            "productivity_growth",
            status,
            metrics.total_productivity_growth,
            f"range: [{prod['total_growth_min'] * 100:.0f}%, {prod['total_growth_max'] * 100:.0f}%]",
            score,
            weights.get("productivity_growth", 1.0),
        )
    )

    # Real wage growth
    wage = ts["real_wage"]["targets"]
    status = check_range(
        metrics.total_real_wage_growth,
        wage["total_growth_min"],
        wage["total_growth_max"],
    )
    score = score_range(
        metrics.total_real_wage_growth,
        wage["total_growth_min"],
        wage["total_growth_max"],
    )
    validation_results.append(
        MetricResult(
            "real_wage_growth",
            status,
            metrics.total_real_wage_growth,
            f"range: [{wage['total_growth_min'] * 100:.0f}%, {wage['total_growth_max'] * 100:.0f}%]",
            score,
            weights.get("real_wage_growth", 1.0),
        )
    )

    # Productivity trend coefficient
    status = check_range(
        metrics.productivity_trend_coefficient,
        prod["trend_coefficient_min"],
        prod["trend_coefficient_max"],
    )
    score = score_range(
        metrics.productivity_trend_coefficient,
        prod["trend_coefficient_min"],
        prod["trend_coefficient_max"],
    )
    validation_results.append(
        MetricResult(
            "productivity_trend",
            status,
            metrics.productivity_trend_coefficient,
            f"range: [{prod['trend_coefficient_min']:.4f}, {prod['trend_coefficient_max']:.4f}]",
            score,
            weights.get("productivity_trend", 1.0),
        )
    )

    # Compute totals
    n_pass = sum(1 for r in validation_results if r.status == "PASS")
    n_warn = sum(1 for r in validation_results if r.status == "WARN")
    n_fail = sum(1 for r in validation_results if r.status == "FAIL")

    # Compute weighted total score
    total_weight = sum(r.weight for r in validation_results)
    total_score = sum(r.score * r.weight for r in validation_results) / total_weight

    return ValidationScore(
        metric_results=validation_results,
        total_score=total_score,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        config=config,
    )


def print_growth_plus_report(result: ValidationScore) -> None:
    """Print formatted Growth+ validation report to stdout."""
    print("\n" + "=" * 78)
    print("GROWTH+ SCENARIO VALIDATION")
    print("=" * 78)

    print("\nTIME SERIES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[:4]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nCURVES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[4:7]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nDISTRIBUTION:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[7:9]:
        if "pct" in r.name:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual * 100:>7.1f}%  {r.score:>6.3f}  ({r.target_desc})"
            )
        else:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
            )

    print("\nGROWTH METRICS:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[9:]:
        if "growth" in r.name:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual * 100:>7.1f}%  {r.score:>6.3f}  ({r.target_desc})"
            )
        else:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual:>8.6f}  {r.score:>6.3f}  ({r.target_desc})"
            )

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {result.n_pass} PASS, {result.n_warn} WARN, {result.n_fail} FAIL  |  "
        f"TOTAL SCORE: {result.total_score:.3f}"
    )
    print("=" * 78 + "\n")


def run_growth_plus_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    weights: dict[str, float] | None = None,
    **config_overrides: Any,
) -> StabilityResult:
    """Run Growth+ validation across multiple seeds and measure consistency.

    Parameters
    ----------
    seeds : list[int] or int
        List of specific seeds to test, or number of seeds to generate.
    n_periods : int
        Number of simulation periods per seed.
    weights : dict, optional
        Custom weights for metrics.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = seeds

    seed_results: list[ValidationScore] = []
    for seed in seed_list:
        result = run_growth_plus_validation(
            seed=seed,
            n_periods=n_periods,
            weights=weights,
            **config_overrides,
        )
        seed_results.append(result)

    scores = [r.total_score for r in seed_results]
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    n_passed = sum(1 for r in seed_results if r.passed)
    pass_rate = n_passed / len(seed_results)

    metric_names = [m.name for m in seed_results[0].metric_results]
    metric_stats: dict[str, MetricStats] = {}

    for idx, name in enumerate(metric_names):
        values = [r.metric_results[idx].actual for r in seed_results]
        scores_for_metric = [r.metric_results[idx].score for r in seed_results]
        statuses = [r.metric_results[idx].status for r in seed_results]

        metric_stats[name] = MetricStats(
            name=name,
            mean_value=float(np.mean(values)),
            std_value=float(np.std(values)),
            mean_score=float(np.mean(scores_for_metric)),
            std_score=float(np.std(scores_for_metric)),
            pass_rate=sum(1 for s in statuses if s != "FAIL") / len(statuses),
        )

    return StabilityResult(
        seed_results=seed_results,
        mean_score=mean_score,
        std_score=std_score,
        min_score=min_score,
        max_score=max_score,
        pass_rate=pass_rate,
        n_seeds=len(seed_results),
        metric_stats=metric_stats,
    )


def print_growth_plus_stability_report(result: StabilityResult) -> None:
    """Print formatted Growth+ stability test report to stdout."""
    print("\n" + "=" * 78)
    print(f"GROWTH+ SEED STABILITY TEST ({result.n_seeds} seeds)")
    print("=" * 78)

    print("\nAGGREGATE SCORES:")
    print(f"  Mean Score:    {result.mean_score:.3f} ± {result.std_score:.3f}")
    print(f"  Score Range:   [{result.min_score:.3f}, {result.max_score:.3f}]")
    n_passed = int(result.pass_rate * result.n_seeds)
    print(
        f"  Pass Rate:     {result.pass_rate:.0%} ({n_passed}/{result.n_seeds} seeds passed)"
    )

    print("\nPER-METRIC STABILITY:")
    print(f"  {'Metric':<28} {'Mean':>10} {'Std':>8} {'Score':>7} {'Pass%':>7}")
    print("  " + "-" * 62)

    for name, stats in result.metric_stats.items():
        flag = ""
        if stats.pass_rate < 0.8:
            flag = " <- unstable"
        elif stats.std_score > 0.1:
            flag = " <- variable"

        if "pct" in name or "growth" in name:
            print(
                f"  {name:<28} {stats.mean_value * 100:>9.1f}% {stats.std_value * 100:>7.2f}% "
                f"{stats.mean_score:>7.3f} {stats.pass_rate:>6.0%}{flag}"
            )
        else:
            print(
                f"  {name:<28} {stats.mean_value:>10.4f} {stats.std_value:>8.4f} "
                f"{stats.mean_score:>7.3f} {stats.pass_rate:>6.0%}{flag}"
            )

    print("\n" + "=" * 78)
    stability_status = "PASS" if result.is_stable else "WARN"
    print(
        f"STABILITY: {stability_status} "
        f"(pass_rate={result.pass_rate:.0%}, std={result.std_score:.3f})"
    )
    print("=" * 78 + "\n")
