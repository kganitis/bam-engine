"""Auto-generated markdown reports for calibration results.

Each phase of the calibration pipeline generates a markdown report
alongside its JSON results in the timestamped output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calibration.analysis import CalibrationResult, ComparisonResult
from calibration.sensitivity import SensitivityResult


def generate_sensitivity_report(
    result: SensitivityResult,
    method: str,
    path: Path,
) -> None:
    """Generate markdown report for sensitivity phase.

    Parameters
    ----------
    result : SensitivityResult
        Sensitivity analysis result.
    method : str
        Sensitivity method used ("morris" or "oat").
    path : Path
        Output path for the markdown report.
    """
    lines = [
        f"# Sensitivity Analysis Report ({result.scenario})",
        "",
        f"**Method:** {method.upper()}",
        f"**Baseline score:** {result.baseline_score:.4f}",
        f"**Seeds per eval:** {result.n_seeds}",
        f"**Avg time/run:** {result.avg_time_per_run:.1f}s",
        "",
        "## Parameter Ranking",
        "",
        "| Parameter | Sensitivity | Best Value | Best Score |",
        "|-----------|------------|------------|------------|",
    ]

    for p in result.ranked:
        lines.append(
            f"| {p.name} | {p.sensitivity:.4f} | {p.best_value} | {p.best_score:.4f} |"
        )

    # Classification
    included, fixed = result.get_important(0.02)
    lines.extend(
        [
            "",
            "## Classification (threshold = 0.02)",
            "",
            f"**INCLUDE ({len(included)}):** {', '.join(included) or 'None'}",
            "",
            f"**FIX ({len(fixed)}):** {', '.join(fixed) or 'None'}",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def generate_screening_report(
    results: list[CalibrationResult],
    grid: dict[str, list[Any]],
    fixed: dict[str, Any],
    patterns: dict[str, dict[Any, int]],
    sensitivity: SensitivityResult,
    scenario: str,
    path: Path,
    top_n: int = 50,
) -> None:
    """Generate markdown report for grid screening phase.

    Parameters
    ----------
    results : list[CalibrationResult]
        Screening results (sorted by score).
    grid : dict
        Grid parameters searched.
    fixed : dict
        Fixed parameter values.
    patterns : dict
        Parameter patterns from top configs.
    sensitivity : SensitivityResult
        Sensitivity result used for grid building.
    scenario : str
        Scenario name.
    path : Path
        Output path for the markdown report.
    top_n : int
        Number of top configs used for pattern analysis.
    """
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    lines = [
        f"# Grid Screening Report ({scenario})",
        "",
        f"**Combinations tested:** {len(results)} / {n_combos}",
        f"**Grid parameters:** {', '.join(grid.keys())}",
        f"**Fixed parameters:** {', '.join(fixed.keys())}",
        "",
    ]

    # Top results
    lines.extend(
        [
            "## Top 10 Results",
            "",
            "| Rank | Score | Pass | Warn | Fail |",
            "|------|-------|------|------|------|",
        ]
    )

    for i, r in enumerate(results[:10]):
        lines.append(
            f"| {i + 1} | {r.single_score:.4f} | {r.n_pass} | {r.n_warn} | {r.n_fail} |"
        )

    # Parameter patterns
    if patterns:
        lines.extend(["", f"## Parameter Patterns (top {top_n})", ""])
        for param, counts in patterns.items():
            parts = []
            for val, count in list(counts.items())[:4]:
                pct = 100.0 * count / top_n
                parts.append(f"{val}={count} ({pct:.0f}%)")
            lines.append(f"- **{param}:** {' | '.join(parts)}")

    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def generate_stability_report(
    results: list[CalibrationResult],
    scenario: str,
    tiers: list[tuple[int, int]],
    comparison: ComparisonResult | None,
    path: Path,
) -> None:
    """Generate markdown report for stability phase.

    Parameters
    ----------
    results : list[CalibrationResult]
        Stability test results (sorted by ranking).
    scenario : str
        Scenario name.
    tiers : list[tuple[int, int]]
        Stability tiers used.
    comparison : ComparisonResult or None
        Before/after comparison (if available).
    path : Path
        Output path for the markdown report.
    """
    lines = [
        f"# Stability Testing Report ({scenario})",
        "",
        f"**Tiers:** {' -> '.join(f'{c}x{s}seeds' for c, s in tiers)}",
        f"**Final candidates:** {len(results)}",
        "",
    ]

    # Top results
    if results:
        lines.extend(
            [
                "## Top Results",
                "",
                "| Rank | Combined | Mean | Std | Seeds |",
                "|------|----------|------|-----|-------|",
            ]
        )

        for i, r in enumerate(results[:10]):
            n_seeds = len(r.seed_scores or [])
            lines.append(
                f"| {i + 1} | {r.combined_score or 0:.4f} "
                f"| {r.mean_score or 0:.4f} | {r.std_score or 0:.4f} "
                f"| {n_seeds} |"
            )

        # Best config
        best = results[0]
        lines.extend(["", "## Best Configuration", "", "```yaml"])
        for k, v in sorted(best.params.items()):
            lines.append(f"{k}: {v}")
        lines.append("```")

    # Comparison
    if comparison:
        lines.extend(
            [
                "",
                "## Before/After Comparison",
                "",
                f"**Default score:** {comparison.default_score:.4f}",
                f"**Calibrated score:** {comparison.calibrated_score:.4f}",
                "",
            ]
        )
        improved = [
            (name, pct)
            for name, _, _, pct in comparison.improvements
            if abs(pct) >= 1.0
        ]
        if improved:
            lines.extend(
                [
                    "### Notable Changes",
                    "",
                    "| Metric | Change |",
                    "|--------|--------|",
                ]
            )
            for name, pct in improved:
                sign = "+" if pct > 0 else ""
                lines.append(f"| {name} | {sign}{pct:.1f}% |")

    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def generate_full_report(
    sensitivity: SensitivityResult,
    screening_results: list[CalibrationResult],
    stability_results: list[CalibrationResult],
    comparison: ComparisonResult | None,
    scenario: str,
    tiers: list[tuple[int, int]],
    path: Path,
) -> None:
    """Generate comprehensive calibration report combining all phases.

    Parameters
    ----------
    sensitivity : SensitivityResult
        Sensitivity analysis result.
    screening_results : list[CalibrationResult]
        Grid screening results.
    stability_results : list[CalibrationResult]
        Stability testing results.
    comparison : ComparisonResult or None
        Before/after comparison.
    scenario : str
        Scenario name.
    tiers : list[tuple[int, int]]
        Stability tiers used.
    path : Path
        Output path for the markdown report.
    """
    lines = [
        f"# Calibration Report ({scenario})",
        "",
        "## Summary",
        "",
    ]

    # Sensitivity summary
    included, fixed = sensitivity.get_important(0.02)
    lines.extend(
        [
            "### Sensitivity Phase",
            "",
            f"- **INCLUDE params:** {len(included)} ({', '.join(included[:5])}{'...' if len(included) > 5 else ''})",
            f"- **FIX params:** {len(fixed)}",
            f"- **Baseline score:** {sensitivity.baseline_score:.4f}",
            "",
        ]
    )

    # Screening summary
    if screening_results:
        lines.extend(
            [
                "### Screening Phase",
                "",
                f"- **Combinations tested:** {len(screening_results)}",
                f"- **Best single-seed score:** {screening_results[0].single_score:.4f}",
                "",
            ]
        )

    # Stability summary
    if stability_results:
        best = stability_results[0]
        lines.extend(
            [
                "### Stability Phase",
                "",
                f"- **Tiers:** {' -> '.join(f'{c}x{s}seeds' for c, s in tiers)}",
                f"- **Best combined score:** {best.combined_score or 0:.4f}",
                f"- **Best mean +/- std:** {best.mean_score or 0:.4f} +/- {best.std_score or 0:.4f}",
                "",
            ]
        )

    # Comparison
    if comparison:
        delta = comparison.calibrated_score - comparison.default_score
        lines.extend(
            [
                "### Improvement",
                "",
                f"- **Default score:** {comparison.default_score:.4f}",
                f"- **Calibrated score:** {comparison.calibrated_score:.4f}",
                f"- **Delta:** {delta:+.4f}",
                "",
            ]
        )

    # Best config
    if stability_results:
        best = stability_results[0]
        lines.extend(["## Best Configuration", "", "```yaml"])
        for k, v in sorted(best.params.items()):
            lines.append(f"{k}: {v}")
        lines.extend(["```", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
