"""Generic validation engine.

This module provides the core validation logic that works with any scenario.
The engine takes a Scenario configuration and executes validation using
MetricSpec definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

import bamengine as bam
from validation.scenarios._utils import adjust_burn_in
from validation.scoring import (
    check_mean_tolerance,
    check_outlier_penalty,
    check_pct_within_target,
    check_range,
    fail_escalation_multiplier,
    score_mean_tolerance,
    score_outlier_penalty,
    score_pct_within_target,
    score_range,
)
from validation.types import (
    CheckType,
    MetricResult,
    MetricSpec,
    MetricStats,
    Scenario,
    StabilityResult,
    ValidationScore,
)


def load_targets(scenario: Scenario) -> dict[str, Any]:
    """Load validation targets from YAML for a scenario.

    Parameters
    ----------
    scenario : Scenario
        The scenario to load targets for.

    Returns
    -------
    dict
        Nested dictionary of targets from YAML file.
    """
    with open(scenario.targets_path) as f:
        return yaml.safe_load(f)


def _get_nested_value(data: dict, path: str) -> Any:
    """Get a nested value from a dictionary using dot-separated path.

    Parameters
    ----------
    data : dict
        Nested dictionary.
    path : str
        Dot-separated path like "time_series.unemployment.mean".

    Returns
    -------
    Any
        Value at the path.

    Raises
    ------
    KeyError
        If path not found.
    """
    keys = path.split(".")
    value = data
    for key in keys:
        value = value[key]
    return value


def evaluate_metric(
    spec: MetricSpec, metrics: Any, targets: dict[str, Any]
) -> MetricResult:
    """Evaluate a single metric against targets.

    Uses standardized keys in the YAML:
    - MEAN_TOLERANCE: 'target', 'tolerance'
    - RANGE: 'min', 'max'
    - PCT_WITHIN: 'target', 'min'
    - OUTLIER: 'max_outlier', 'penalty_weight' (optional, default 2.0)

    Parameters
    ----------
    spec : MetricSpec
        Specification for the metric.
    metrics : Any
        Metrics dataclass with computed values.
    targets : dict
        Full targets dictionary from YAML.

    Returns
    -------
    MetricResult
        Validation result for this metric.
    """
    # Get actual value from metrics
    actual = getattr(metrics, spec.field)

    # Get target values from YAML
    target_section = _get_nested_value(targets, spec.target_path)

    # Weight-based fail escalation (BOOLEAN checks are exempt)
    escalation = fail_escalation_multiplier(spec.weight)

    if spec.check_type == CheckType.MEAN_TOLERANCE:
        target = target_section["target"]
        tolerance = target_section["tolerance"]
        status = check_mean_tolerance(actual, target, tolerance, escalation=escalation)
        score = score_mean_tolerance(actual, target, tolerance)
        min_val = target_section.get("min")
        max_val = target_section.get("max")
        if min_val is not None or max_val is not None:
            if (min_val is not None and actual < min_val) or (
                max_val is not None and actual > max_val
            ):
                status = "FAIL"
            elif status == "FAIL":
                status = "WARN"
        if spec.target_desc is not None:
            target_desc = spec.target_desc
        else:
            target_desc = f"target: {target:.4f} \u00b1 {tolerance:.4f}"

    elif spec.check_type == CheckType.RANGE:
        min_val = target_section["min"]
        max_val = target_section["max"]
        status = check_range(actual, min_val, max_val, escalation=escalation)
        score = score_range(actual, min_val, max_val)
        if spec.target_desc is not None:
            target_desc = spec.target_desc
        else:
            target_desc = f"range: [{min_val:.2f}, {max_val:.2f}]"

    elif spec.check_type == CheckType.PCT_WITHIN:
        target_pct = target_section["target"]
        min_pct = target_section["min"]
        status = check_pct_within_target(
            actual, target_pct, min_pct, escalation=escalation
        )
        score = score_pct_within_target(actual, target_pct, min_pct)
        if spec.target_desc is not None:
            target_desc = spec.target_desc
        else:
            target_desc = f"target: {target_pct:.0%} (min: {min_pct:.0%})"

    elif spec.check_type == CheckType.OUTLIER:
        max_outlier = target_section["max_outlier"]
        penalty_weight = target_section.get("penalty_weight", 2.0)
        status = check_outlier_penalty(actual, max_outlier, escalation=escalation)
        score = score_outlier_penalty(actual, max_outlier, penalty_weight)
        if spec.target_desc is not None:
            target_desc = spec.target_desc
        else:
            target_desc = f"max: {max_outlier:.0%}"

    elif spec.check_type == CheckType.BOOLEAN:
        if spec.invert:
            passed = actual < spec.threshold
        else:
            passed = actual > spec.threshold
        status = "PASS" if passed else "FAIL"
        score = 1.0 if passed else 0.0
        if spec.target_desc is not None:
            target_desc = spec.target_desc
        else:
            op = "<" if spec.invert else ">"
            target_desc = f"{op} {spec.threshold}"

    else:
        raise ValueError(f"Unknown check type: {spec.check_type}")

    return MetricResult(
        name=spec.name,
        status=status,
        actual=float(actual),
        target_desc=target_desc,
        score=score,
        weight=spec.weight,
        group=spec.group,
        format=spec.format,
    )


def validate(
    scenario: Scenario,
    *,
    seed: int = 0,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> ValidationScore:
    """Run validation for a scenario and return scored result.

    Parameters
    ----------
    scenario : Scenario
        Scenario configuration.
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.
    """
    # Load targets
    targets = load_targets(scenario)

    # Build config
    config = {
        **scenario.default_config,
        "n_periods": n_periods,
        "seed": seed,
        "logging": {"default_level": "ERROR"},
        **config_overrides,
    }

    # Run setup hook if provided (e.g., for RnD import)
    if scenario.setup_hook:
        scenario.setup_hook(None)  # Pre-import call

    # Run simulation
    sim = bam.Simulation.init(**config)

    # Run setup hook on simulation instance if needed
    if scenario.setup_hook:
        scenario.setup_hook(sim)

    results = sim.run(collect=scenario.collect_config)

    # Compute metrics
    burn_in = targets["metadata"]["validation"]["burn_in_periods"]
    burn_in = adjust_burn_in(burn_in, n_periods)

    metrics = scenario.compute_metrics(sim, results, burn_in)

    # Validate each metric
    validation_results: list[MetricResult] = []
    for spec in scenario.metric_specs:
        result = evaluate_metric(spec, metrics, targets)
        validation_results.append(result)

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


def stability_test(
    scenario: Scenario,
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> StabilityResult:
    """Run validation across multiple seeds and measure consistency.

    Parameters
    ----------
    scenario : Scenario
        Scenario configuration.
    seeds : list[int] or int
        List of specific seeds to test, or number of seeds to generate.
        If int, uses seeds [0, 1, 2, ..., seeds-1].
    n_periods : int
        Number of simulation periods per seed.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    # Handle seeds parameter
    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = seeds

    # Run validation for each seed
    seed_results: list[ValidationScore] = []
    for seed in seed_list:
        result = validate(
            scenario,
            seed=seed,
            n_periods=n_periods,
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
            format=seed_results[0].metric_results[idx].format,
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
