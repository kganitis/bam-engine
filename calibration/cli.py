"""Command-line interface for calibration.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)

Usage:
    # Run all phases sequentially (baseline)
    python -m calibration --scenario baseline --workers 10

    # Run individual phases
    python -m calibration --phase sensitivity --scenario baseline
    python -m calibration --phase grid --scenario baseline
    python -m calibration --phase stability --scenario baseline

    # Run pairwise interaction analysis
    python -m calibration --phase pairwise --scenario baseline

    # Resume interrupted grid search
    python -m calibration --phase grid --scenario baseline --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibration.optimizer import (
    CalibrationResult,
    analyze_parameter_patterns,
    build_focused_grid,
    compare_configs,
    export_best_config,
    parse_stability_tiers,
    print_comparison,
    print_parameter_patterns,
    run_screening,
    run_tiered_stability,
)
from calibration.parameter_space import (
    count_combinations,
    generate_combinations,
    get_default_values,
    get_parameter_grid,
)
from calibration.sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
    print_pairwise_report,
    print_sensitivity_report,
    run_pairwise_analysis,
    run_sensitivity_analysis,
)

# Output directory for calibration results
OUTPUT_DIR = Path(__file__).parent / "output"


def print_results(results: list[CalibrationResult], top_n: int = 10) -> None:
    """Print formatted calibration results.

    Parameters
    ----------
    results : list[CalibrationResult]
        Calibration results sorted by combined_score.
    top_n : int
        Number of top results to display.
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} CALIBRATION RESULTS")
    print("=" * 80)

    for i, r in enumerate(results[:top_n]):
        score_label = r.combined_score or r.single_score
        print(f"\n#{i + 1}: Combined Score = {score_label:.4f}")
        if r.mean_score is not None:
            print(f"    Mean Score:  {r.mean_score:.3f} ± {r.std_score:.3f}")
        if r.pass_rate is not None:
            print(f"    Pass Rate:   {r.pass_rate:.0%}")
        if r.seed_scores:
            print(f"    Seeds tested: {len(r.seed_scores)}")
        print("    Parameters:")
        for k, v in r.params.items():
            print(f"      {k}: {v}")

    print("\n" + "=" * 80)


# =============================================================================
# Sensitivity JSON serialization
# =============================================================================


def _save_sensitivity(sensitivity: SensitivityResult, path: Path) -> None:
    """Save sensitivity result to JSON."""
    data = {
        "scenario": sensitivity.scenario,
        "baseline_score": sensitivity.baseline_score,
        "avg_time_per_run": sensitivity.avg_time_per_run,
        "n_seeds": sensitivity.n_seeds,
        "parameters": {
            p.name: {
                "sensitivity": p.sensitivity,
                "best_value": p.best_value,
                "best_score": p.best_score,
                "values": p.values,
                "scores": p.scores,
                "group_scores": p.group_scores,
            }
            for p in sensitivity.parameters
        },
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Sensitivity results saved to {path}")


def _load_sensitivity(path: Path) -> SensitivityResult:
    """Load sensitivity result from JSON."""
    with open(path) as f:
        data = json.load(f)

    parameters = []
    for name, pdata in data["parameters"].items():
        parameters.append(
            ParameterSensitivity(
                name=name,
                values=pdata["values"],
                scores=pdata["scores"],
                best_value=pdata["best_value"],
                best_score=pdata["best_score"],
                sensitivity=pdata["sensitivity"],
                group_scores=pdata.get("group_scores", {}),
            )
        )

    return SensitivityResult(
        parameters=parameters,
        baseline_score=data["baseline_score"],
        scenario=data.get("scenario", "baseline"),
        avg_time_per_run=data.get("avg_time_per_run", 0.0),
        n_seeds=data.get("n_seeds", 1),
    )


# =============================================================================
# Screening JSON serialization
# =============================================================================


def _save_screening(
    results: list[CalibrationResult],
    sensitivity: SensitivityResult,
    grid: dict[str, list[Any]],
    fixed: dict[str, Any],
    patterns: dict[str, dict[Any, int]],
    scenario: str,
    path: Path,
) -> None:
    """Save screening results to JSON."""
    data = {
        "scenario": scenario,
        "avg_time_per_run": sensitivity.avg_time_per_run,
        "sensitivity": {
            p.name: {"sensitivity": p.sensitivity, "best_value": p.best_value}
            for p in sensitivity.parameters
        },
        "grid_params": grid,
        "fixed_params": fixed,
        "patterns": {
            param: {str(v): c for v, c in counts.items()}
            for param, counts in patterns.items()
        },
        "results": [
            {
                "rank": i + 1,
                "params": r.params,
                "single_score": r.single_score,
                "n_pass": r.n_pass,
                "n_warn": r.n_warn,
                "n_fail": r.n_fail,
            }
            for i, r in enumerate(results)
        ],
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Screening results saved to {path}")


def _load_screening(
    path: Path,
) -> tuple[list[CalibrationResult], float]:
    """Load screening results from JSON. Returns (results, avg_time_per_run)."""
    with open(path) as f:
        data = json.load(f)

    results = [
        CalibrationResult(
            params=r["params"],
            single_score=r["single_score"],
            n_pass=r["n_pass"],
            n_warn=r["n_warn"],
            n_fail=r["n_fail"],
        )
        for r in data["results"]
    ]
    return results, data.get("avg_time_per_run", 0.0)


# =============================================================================
# Phase execution
# =============================================================================


def _run_sensitivity_phase(args: argparse.Namespace) -> SensitivityResult:
    """Run Phase 1: Sensitivity analysis."""
    print("=" * 70)
    print(f"PHASE 1: SENSITIVITY ANALYSIS ({args.scenario})")
    print("=" * 70)

    sensitivity = run_sensitivity_analysis(
        scenario=args.scenario,
        n_workers=args.workers,
        n_periods=args.periods,
        n_seeds=args.sensitivity_seeds,
    )
    print_sensitivity_report(sensitivity, args.sensitivity_threshold)

    # Save
    output_path = OUTPUT_DIR / f"{args.scenario}_sensitivity.json"
    _save_sensitivity(sensitivity, output_path)

    return sensitivity


def _run_grid_phase(
    args: argparse.Namespace,
    sensitivity: SensitivityResult | None = None,
) -> list[CalibrationResult]:
    """Run Phase 2: Grid screening."""
    # Load sensitivity if not provided
    if sensitivity is None:
        sens_path = OUTPUT_DIR / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            raise FileNotFoundError(
                f"Sensitivity results not found at {sens_path}. "
                f"Run --phase sensitivity first."
            )
        sensitivity = _load_sensitivity(sens_path)
        print(f"Loaded sensitivity from {sens_path}")

    print("\n" + "=" * 70)
    print(f"PHASE 2: GRID SCREENING ({args.scenario})")
    print("=" * 70)

    pruning = _resolve_pruning_threshold(
        args.pruning_threshold, args.sensitivity_threshold
    )

    grid, fixed = build_focused_grid(
        sensitivity,
        scenario=args.scenario,
        sensitivity_threshold=args.sensitivity_threshold,
        pruning_threshold=pruning,
    )

    n_combinations = count_combinations(grid)
    print(f"Grid parameters: {list(grid.keys())}")
    print(f"Fixed parameters: {list(fixed.keys())}")
    print(f"Combinations to test: {n_combinations}")
    if pruning is not None:
        print(f"Pruning threshold: {pruning} (values within {pruning} of best kept)")
    else:
        print("Pruning: disabled")

    if n_combinations == 0:
        print("\nNo parameters require grid search - all fixed at best values.")
        print("Best configuration:")
        for k, v in fixed.items():
            print(f"  {k}: {v}")
        return []

    # Generate combinations
    combinations = []
    for combo in generate_combinations(grid):
        full_params = {**fixed, **combo}
        combinations.append(full_params)

    # Screen
    results = run_screening(
        combinations,
        args.scenario,
        n_workers=args.workers,
        n_periods=args.periods,
        avg_time_per_run=sensitivity.avg_time_per_run,
        resume=args.resume,
    )

    # Parameter pattern analysis
    patterns = analyze_parameter_patterns(results, top_n=50)
    print_parameter_patterns(patterns, top_n=50)

    # Save
    output_path = OUTPUT_DIR / f"{args.scenario}_screening.json"
    _save_screening(
        results, sensitivity, grid, fixed, patterns, args.scenario, output_path
    )

    return results


def _run_stability_phase(
    args: argparse.Namespace,
    screening_results: list[CalibrationResult] | None = None,
) -> list[CalibrationResult]:
    """Run Phase 3: Stability testing."""
    avg_time = 0.0

    # Load screening if not provided
    if screening_results is None:
        screen_path = OUTPUT_DIR / f"{args.scenario}_screening.json"
        if not screen_path.exists():
            raise FileNotFoundError(
                f"Screening results not found at {screen_path}. Run --phase grid first."
            )
        screening_results, avg_time = _load_screening(screen_path)
        print(f"Loaded {len(screening_results)} screening results from {screen_path}")

    print("\n" + "=" * 70)
    print(f"PHASE 3: STABILITY TESTING ({args.scenario})")
    print("=" * 70)

    tiers = parse_stability_tiers(args.stability_tiers)
    results = run_tiered_stability(
        screening_results,
        args.scenario,
        tiers=tiers,
        n_workers=args.workers,
        n_periods=args.periods,
        avg_time_per_run=avg_time,
    )

    print_results(results)

    # Export best config
    if results:
        config_path = export_best_config(results[0], args.scenario)
        print(f"Best config exported to {config_path}")

        # Before/after comparison
        default_config = get_default_values(args.scenario)
        comparison = compare_configs(
            default_config,
            results[0].params,
            args.scenario,
            n_periods=args.periods,
        )
        print_comparison(comparison)

    # Save final results
    output_data = {
        "scenario": args.scenario,
        "results": [
            {
                "rank": i + 1,
                "params": r.params,
                "combined_score": r.combined_score,
                "mean_score": r.mean_score,
                "std_score": r.std_score,
                "pass_rate": r.pass_rate,
                "seed_scores": r.seed_scores,
                "single_score": r.single_score,
            }
            for i, r in enumerate(results)
        ],
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.scenario}_calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nFinal results saved to {output_path}")

    return results


def _run_pairwise_phase(
    args: argparse.Namespace,
    sensitivity: SensitivityResult | None = None,
) -> None:
    """Run pairwise interaction analysis."""
    # Load sensitivity if not provided
    if sensitivity is None:
        sens_path = OUTPUT_DIR / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            raise FileNotFoundError(
                f"Sensitivity results not found at {sens_path}. "
                f"Run --phase sensitivity first."
            )
        sensitivity = _load_sensitivity(sens_path)

    included, _ = sensitivity.get_important(args.sensitivity_threshold)

    if len(included) < 2:
        print("Need at least 2 INCLUDE params for pairwise analysis.")
        print(f"Found INCLUDE: {included}")
        return

    print("=" * 70)
    print(f"PAIRWISE INTERACTION ANALYSIS ({args.scenario})")
    print("=" * 70)
    print(f"INCLUDE params: {included}")

    grid = get_parameter_grid(args.scenario)

    pruning = _resolve_pruning_threshold(
        args.pruning_threshold, args.sensitivity_threshold
    )
    grid = sensitivity.prune_grid(grid, pruning)

    best_values = {p.name: p.best_value for p in sensitivity.parameters}

    result = run_pairwise_analysis(
        params=included,
        grid=grid,
        best_values=best_values,
        scenario=args.scenario,
        n_seeds=args.sensitivity_seeds,
        n_periods=args.periods,
        n_workers=args.workers,
    )
    print_pairwise_report(result)

    # Save
    output_data = {
        "scenario": args.scenario,
        "baseline_score": result.baseline_score,
        "interactions": [
            {
                "param_a": ix.param_a,
                "value_a": ix.value_a,
                "param_b": ix.param_b,
                "value_b": ix.value_b,
                "combined_score": ix.combined_score,
                "individual_a_score": ix.individual_a_score,
                "individual_b_score": ix.individual_b_score,
                "interaction_strength": ix.interaction_strength,
            }
            for ix in result.ranked
        ],
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.scenario}_pairwise.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Pairwise results saved to {output_path}")


# =============================================================================
# Pruning threshold resolver
# =============================================================================


def _resolve_pruning_threshold(raw: str, sensitivity_threshold: float) -> float | None:
    """Resolve pruning threshold from CLI string.

    Parameters
    ----------
    raw : str
        Raw CLI value: "auto", "none", "0", or a float string.
    sensitivity_threshold : float
        Current sensitivity threshold (used for "auto" calculation).

    Returns
    -------
    float or None
        Resolved pruning threshold, or None if disabled.
    """
    if raw.lower() == "auto":
        return 2.0 * sensitivity_threshold
    if raw.lower() == "none" or raw == "0":
        return None
    return float(raw)


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    """Main entry point for calibration CLI."""
    parser = argparse.ArgumentParser(
        description="BAM Parameter Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases (baseline)
  python -m calibration --scenario baseline --workers 10

  # Run individual phases
  python -m calibration --phase sensitivity --scenario baseline
  python -m calibration --phase grid --scenario baseline
  python -m calibration --phase stability --scenario baseline
  python -m calibration --phase pairwise --scenario baseline

  # Resume interrupted grid search
  python -m calibration --phase grid --scenario baseline --resume

  # Custom stability tiers
  python -m calibration --phase stability --stability-tiers "50:10,20:30,5:100"
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "growth_plus", "buffer_stock"],
        default="baseline",
        help="Scenario to calibrate (default: baseline)",
    )
    parser.add_argument(
        "--phase",
        choices=["sensitivity", "grid", "stability", "pairwise"],
        default=None,
        help="Run a single phase (default: all phases sequentially)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Simulation periods (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=0.02,
        help="Minimum Δ for INCLUDE in grid search (default: 0.02)",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=str,
        default="auto",
        help=(
            'Max score gap for keeping grid values: "auto" (2× sensitivity threshold), '
            '"none" (disabled), or a float (default: "auto")'
        ),
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=3,
        help="Number of seeds per sensitivity evaluation (default: 3)",
    )
    parser.add_argument(
        "--stability-tiers",
        type=str,
        default="100:10,50:20,10:100",
        help='Stability tiers as "configs:seeds,..." (default: "100:10,50:20,10:100")',
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (grid/stability phases)",
    )

    args = parser.parse_args()

    # Dispatch to phase
    if args.phase == "sensitivity":
        _run_sensitivity_phase(args)
    elif args.phase == "grid":
        _run_grid_phase(args)
    elif args.phase == "stability":
        _run_stability_phase(args)
    elif args.phase == "pairwise":
        _run_pairwise_phase(args)
    else:
        # Run all phases sequentially
        sensitivity = _run_sensitivity_phase(args)
        screening = _run_grid_phase(args, sensitivity=sensitivity)
        if screening:
            _run_stability_phase(args, screening_results=screening)


if __name__ == "__main__":
    main()
