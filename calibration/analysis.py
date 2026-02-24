"""Result types, parameter pattern analysis, config export, and comparison.

This module provides:
- Core result types (CalibrationResult, ComparisonResult)
- Progress formatting helpers
- Parameter pattern analysis for identifying best values
- Config export (YAML) and before/after comparison
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from validation import StabilityResult, get_validation_func

# Output directory for calibration results
OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class CalibrationResult:
    """Result from calibration optimization.

    Attributes
    ----------
    params : dict
        Parameter configuration.
    single_score : float
        Validation score from single-seed run.
    n_pass : int
        Number of metrics that passed.
    n_warn : int
        Number of metrics with warnings.
    n_fail : int
        Number of metrics that failed.
    mean_score : float, optional
        Mean score across stability seeds.
    std_score : float, optional
        Standard deviation of scores across seeds.
    pass_rate : float, optional
        Fraction of seeds that passed (no FAIL metrics).
    combined_score : float, optional
        Combined score balancing accuracy and stability.
    stability_result : StabilityResult, optional
        Full stability test result.
    seed_scores : list[float], optional
        Individual seed scores (for incremental stability).
    """

    params: dict[str, Any]
    single_score: float
    n_pass: int
    n_warn: int
    n_fail: int
    mean_score: float | None = None
    std_score: float | None = None
    pass_rate: float | None = None
    combined_score: float | None = None
    stability_result: StabilityResult | None = None
    seed_scores: list[float] | None = None


@dataclass
class ComparisonResult:
    """Result from before/after config comparison."""

    scenario: str
    default_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    default_score: float
    calibrated_score: float
    improvements: list[
        tuple[str, float, float, float]
    ]  # (name, default, calibrated, pct_change)


# =============================================================================
# Progress tracking helpers
# =============================================================================


def format_eta(remaining: int, avg_time: float, n_workers: int) -> str:
    """Format an ETA string from remaining items and average time.

    Parameters
    ----------
    remaining : int
        Number of remaining items.
    avg_time : float
        Average seconds per item.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    str
        Formatted ETA string (e.g., "5m 30s").
    """
    if avg_time <= 0 or n_workers <= 0:
        return "unknown"
    eta_seconds = avg_time * remaining / n_workers
    if eta_seconds < 60:
        return f"{eta_seconds:.0f}s"
    minutes = int(eta_seconds // 60)
    seconds = int(eta_seconds % 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes:02d}m"


def format_progress(completed: int, total: int, remaining: int, eta: str) -> str:
    """Format a progress line.

    Parameters
    ----------
    completed : int
        Number of completed items.
    total : int
        Total number of items.
    remaining : int
        Number of remaining items.
    eta : str
        ETA string.

    Returns
    -------
    str
        Formatted progress line.
    """
    pct = 100.0 * completed / total if total > 0 else 0.0
    return f"Screened {completed}/{total} ({pct:.1f}%) | {remaining} remaining | ETA: {eta}"


# =============================================================================
# Parameter pattern analysis
# =============================================================================


def analyze_parameter_patterns(
    results: list[CalibrationResult],
    top_n: int = 50,
) -> dict[str, dict[Any, int]]:
    """Analyze which parameter values consistently appear in top configs.

    Parameters
    ----------
    results : list[CalibrationResult]
        Screening results sorted by score (best first).
    top_n : int
        Number of top configs to analyze.

    Returns
    -------
    dict[str, dict[Any, int]]
        For each parameter, a dict mapping value -> count in top configs.
    """
    top = results[:top_n]
    if not top:
        return {}

    # Collect all parameter names
    all_params = set()
    for r in top:
        all_params.update(r.params.keys())

    patterns: dict[str, dict[Any, int]] = {}
    for param in sorted(all_params):
        counter: Counter[Any] = Counter()
        for r in top:
            if param in r.params:
                counter[r.params[param]] += 1
        patterns[param] = dict(counter.most_common())

    return patterns


def print_parameter_patterns(
    patterns: dict[str, dict[Any, int]], top_n: int = 50
) -> None:
    """Print parameter pattern analysis.

    Parameters
    ----------
    patterns : dict
        Output from analyze_parameter_patterns().
    top_n : int
        Number of top configs used for display.
    """
    print(f"\nParameter Patterns (top {top_n} configs):")
    print("-" * 70)
    for param, counts in patterns.items():
        parts = []
        for val, count in counts.items():
            pct = 100.0 * count / top_n
            parts.append(f"{val}={count} ({pct:.0f}%)")
        line = " | ".join(parts[:4])
        if len(parts) > 4:
            line += f" | +{len(parts) - 4} more"

        # Detect strong preferences
        top_count = max(counts.values())
        top_pct = 100.0 * top_count / top_n
        hint = ""
        if top_pct >= 80:
            hint = " -> strongly prefer"
        elif top_pct >= 60:
            hint = " -> slight preference"

        print(f"  {param:<30} {line}{hint}")
    print()


# =============================================================================
# Config export
# =============================================================================


def export_best_config(
    result: CalibrationResult, scenario: str, path: Path | None = None
) -> Path:
    """Export best calibration result as a ready-to-use YAML config.

    Parameters
    ----------
    result : CalibrationResult
        Best calibration result.
    scenario : str
        Scenario name.
    path : Path, optional
        Output path. Defaults to output/{scenario}_best_config.yml.

    Returns
    -------
    Path
        Path to exported config file.
    """
    if path is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{scenario}_best_config.yml"

    # Build config: only include params that differ from engine defaults
    config = dict(result.params)

    # Add header comment
    header = (
        f"# Best calibration config for {scenario} scenario\n"
        f"# Score: {result.combined_score or result.single_score:.4f}\n"
    )
    if result.mean_score is not None:
        header += f"# Mean: {result.mean_score:.3f} +/- {result.std_score:.3f}\n"
    if result.pass_rate is not None:
        header += f"# Pass rate: {result.pass_rate:.0%}\n"
    header += f"#\n# Usage: sim = bam.Simulation.init(config='{path}')\n\n"

    with open(path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

    return path


# =============================================================================
# Before/after comparison
# =============================================================================


def compare_configs(
    default: dict[str, Any],
    calibrated: dict[str, Any],
    scenario: str,
    seed: int = 0,
    n_periods: int = 1000,
) -> ComparisonResult:
    """Run default and calibrated configs side-by-side and compare.

    Parameters
    ----------
    default : dict
        Default config overrides (can be empty for engine defaults).
    calibrated : dict
        Calibrated config params.
    scenario : str
        Scenario name.
    seed : int
        Random seed.
    n_periods : int
        Simulation periods.

    Returns
    -------
    ComparisonResult
        Side-by-side comparison of metrics.
    """
    validate = get_validation_func(scenario)

    default_result = validate(seed=seed, n_periods=n_periods, **default)
    calibrated_result = validate(seed=seed, n_periods=n_periods, **calibrated)

    default_metrics = {mr.name: mr.actual for mr in default_result.metric_results}
    calibrated_metrics = {mr.name: mr.actual for mr in calibrated_result.metric_results}

    improvements = []
    for mr_cal in calibrated_result.metric_results:
        name = mr_cal.name
        cal_val = mr_cal.actual
        def_val = default_metrics.get(name, 0.0)
        if def_val != 0:
            pct_change = 100.0 * (cal_val - def_val) / abs(def_val)
        else:
            pct_change = 0.0
        improvements.append((name, def_val, cal_val, pct_change))

    return ComparisonResult(
        scenario=scenario,
        default_metrics=default_metrics,
        calibrated_metrics=calibrated_metrics,
        default_score=default_result.total_score,
        calibrated_score=calibrated_result.total_score,
        improvements=improvements,
    )


def print_comparison(result: ComparisonResult) -> None:
    """Print before/after comparison table.

    Parameters
    ----------
    result : ComparisonResult
        Output from compare_configs().
    """
    print("\n" + "=" * 80)
    print(f"BEFORE/AFTER COMPARISON ({result.scenario})")
    print("=" * 80)
    print(
        f"\nTotal Score: {result.default_score:.4f} -> {result.calibrated_score:.4f} "
        f"({'improved' if result.calibrated_score > result.default_score else 'unchanged'})"
    )
    print(f"\n{'Metric':<35} {'Default':>10} {'Calibrated':>10} {'Change':>10}")
    print("-" * 70)

    for name, def_val, cal_val, pct in result.improvements:
        if abs(pct) < 1.0:
            change_str = "(unchanged)"
        elif pct > 0:
            change_str = f"+{pct:.1f}%"
        else:
            change_str = f"{pct:.1f}%"
        print(f"  {name:<33} {def_val:>10.4f} {cal_val:>10.4f} {change_str:>10}")

    print("=" * 80 + "\n")
