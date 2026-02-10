"""Report printing functions for validation results.

This module provides functions to print formatted validation reports
to stdout.
"""

from __future__ import annotations

from validation.types import (
    MetricFormat,
    MetricGroup,
    MetricResult,
    MetricStats,
    StabilityResult,
    ValidationScore,
)


def _format_actual(result: MetricResult) -> str:
    """Format the actual value based on metric format."""
    if result.format == MetricFormat.PERCENT:
        return f"{result.actual * 100:>7.1f}%"
    elif result.format == MetricFormat.TREND:
        return f"{result.actual:>8.6f}"
    elif result.format == MetricFormat.INTEGER:
        return f"{result.actual:>8.0f}"
    else:
        return f"{result.actual:>8.4f}"


def _format_stability_values(stats: MetricStats) -> tuple[str, str]:
    """Format mean and std values for a stability report row."""
    if stats.format == MetricFormat.PERCENT:
        return f"{stats.mean_value * 100:>9.1f}%", f"{stats.std_value * 100:>7.2f}%"
    elif stats.format == MetricFormat.TREND:
        return f"{stats.mean_value:>10.6f}", f"{stats.std_value:>8.6f}"
    elif stats.format == MetricFormat.INTEGER:
        return f"{stats.mean_value:>10.0f}", f"{stats.std_value:>8.2f}"
    else:
        return f"{stats.mean_value:>10.4f}", f"{stats.std_value:>8.4f}"


def _print_metric_section(title: str, metrics: list[MetricResult]) -> None:
    """Print a section of metrics."""
    if not metrics:
        return

    print(f"\n{title}:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in metrics:
        actual_str = _format_actual(r)
        print(
            f"  {r.name:<28} {r.status:<6} {actual_str}  "
            f"{r.score:>6.3f}  ({r.target_desc})"
        )


def print_report(result: ValidationScore, title: str = "VALIDATION") -> None:
    """Print formatted validation report to stdout.

    Parameters
    ----------
    result : ValidationScore
        Validation result to print.
    title : str
        Title for the report header.
    """
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    # Group metrics by their group attribute
    groups: dict[MetricGroup, list[MetricResult]] = {}
    for metric in result.metric_results:
        group = metric.group
        if group not in groups:
            groups[group] = []
        groups[group].append(metric)

    # Define group display order and titles
    group_titles = {
        MetricGroup.TIME_SERIES: "TIME SERIES",
        MetricGroup.CURVES: "CURVES",
        MetricGroup.DISTRIBUTION: "DISTRIBUTION",
        MetricGroup.GROWTH: "GROWTH METRICS",
        MetricGroup.FINANCIAL: "FINANCIAL DYNAMICS",
        MetricGroup.GROWTH_RATE_DIST: "GROWTH RATE DISTRIBUTIONS",
    }

    group_order = [
        MetricGroup.TIME_SERIES,
        MetricGroup.CURVES,
        MetricGroup.DISTRIBUTION,
        MetricGroup.GROWTH,
        MetricGroup.FINANCIAL,
        MetricGroup.GROWTH_RATE_DIST,
    ]

    for group in group_order:
        if group in groups:
            _print_metric_section(group_titles.get(group, str(group)), groups[group])

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {result.n_pass} PASS, {result.n_warn} WARN, {result.n_fail} FAIL  |  "
        f"TOTAL SCORE: {result.total_score:.3f}"
    )
    print("=" * 78 + "\n")


def print_stability_report(
    result: StabilityResult, title: str = "STABILITY TEST"
) -> None:
    """Print formatted stability test report to stdout.

    Parameters
    ----------
    result : StabilityResult
        Stability result to print.
    title : str
        Title for the report header.
    """
    print("\n" + "=" * 78)
    print(f"{title} ({result.n_seeds} seeds)")
    print("=" * 78)

    print("\nAGGREGATE SCORES:")
    print(f"  Mean Score:    {result.mean_score:.3f} \u00b1 {result.std_score:.3f}")
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

        mean_str, std_str = _format_stability_values(stats)
        print(
            f"  {name:<28} {mean_str} {std_str} "
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
# Legacy-compatible report functions
# =============================================================================


def print_validation_report(result: ValidationScore) -> None:
    """Print baseline validation report (legacy API)."""
    print_report(result, "BASELINE SCENARIO VALIDATION")


def print_baseline_stability_report(result: StabilityResult) -> None:
    """Print baseline stability report (legacy API)."""
    print_stability_report(result, "SEED STABILITY TEST")


def print_growth_plus_report(result: ValidationScore) -> None:
    """Print Growth+ validation report (legacy API)."""
    print_report(result, "GROWTH+ SCENARIO VALIDATION")


def print_growth_plus_stability_report(result: StabilityResult) -> None:
    """Print Growth+ stability report (legacy API)."""
    print_stability_report(result, "GROWTH+ SEED STABILITY TEST")


def print_buffer_stock_report(result: ValidationScore) -> None:
    """Print buffer-stock validation report."""
    print_report(result, "BUFFER-STOCK SCENARIO VALIDATION")


def print_buffer_stock_stability_report(result: StabilityResult) -> None:
    """Print buffer-stock stability report."""
    print_stability_report(result, "BUFFER-STOCK SEED STABILITY TEST")
