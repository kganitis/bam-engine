"""Command-line interface for calibration.

Thin dispatch layer: parses arguments, delegates to phase modules,
saves results via io.py, generates reports via reporting.py.

Usage:
    # Run all phases sequentially (baseline, Morris default)
    python -m calibration --scenario baseline --workers 10

    # Run individual phases
    python -m calibration --phase sensitivity --scenario baseline
    python -m calibration --phase grid --scenario baseline
    python -m calibration --phase stability --scenario baseline

    # Use OAT instead of Morris for sensitivity
    python -m calibration --phase sensitivity --method oat --scenario baseline

    # Load custom grid from YAML
    python -m calibration --phase grid --grid custom_grid.yaml

    # Custom ranking strategy
    python -m calibration --phase stability --rank-by stability --k-factor 1.5

    # Resume interrupted grid search
    python -m calibration --phase grid --scenario baseline --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from calibration.analysis import (
    CalibrationResult,
    analyze_parameter_patterns,
    compare_configs,
    export_best_config,
    print_comparison,
    print_parameter_patterns,
)
from calibration.grid import (
    build_focused_grid,
    count_combinations,
    generate_combinations,
)
from calibration.io import (
    create_run_dir,
    load_screening,
    load_sensitivity,
    save_morris,
    save_pairwise,
    save_screening,
    save_sensitivity,
    save_stability,
)
from calibration.morris import print_morris_report, run_morris_screening
from calibration.parameter_space import get_default_values, get_parameter_grid
from calibration.reporting import (
    generate_full_report,
    generate_screening_report,
    generate_sensitivity_report,
    generate_stability_report,
)
from calibration.screening import run_screening
from calibration.sensitivity import (
    SensitivityResult,
    print_pairwise_report,
    print_sensitivity_report,
    run_pairwise_analysis,
    run_sensitivity_analysis,
)
from calibration.stability import parse_stability_tiers, run_tiered_stability

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
            print(f"    Mean Score:  {r.mean_score:.3f} +/- {r.std_score:.3f}")
        if r.pass_rate is not None:
            print(f"    Pass Rate:   {r.pass_rate:.0%}")
        if r.seed_scores:
            print(f"    Seeds tested: {len(r.seed_scores)}")
        print("    Parameters:")
        for k, v in r.params.items():
            print(f"      {k}: {v}")

    print("\n" + "=" * 80)


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
# Phase execution
# =============================================================================


def _run_oat_phase(
    args: argparse.Namespace, run_dir: Path | None = None
) -> SensitivityResult:
    """Run Phase 1 using OAT sensitivity analysis."""
    print("=" * 70)
    print(f"PHASE 1: OAT SENSITIVITY ANALYSIS ({args.scenario})")
    print("=" * 70)

    sensitivity = run_sensitivity_analysis(
        scenario=args.scenario,
        n_workers=args.workers,
        n_periods=args.periods,
        n_seeds=args.sensitivity_seeds,
    )
    print_sensitivity_report(sensitivity, args.sensitivity_threshold)

    # Save
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    save_sensitivity(sensitivity, out / f"{args.scenario}_sensitivity.json")
    print(f"Sensitivity results saved to {out}")

    # Generate report
    if run_dir:
        generate_sensitivity_report(
            sensitivity, "oat", run_dir / "sensitivity_report.md"
        )

    return sensitivity


def _run_morris_phase(
    args: argparse.Namespace, run_dir: Path | None = None
) -> SensitivityResult:
    """Run Phase 1 using Morris Method screening."""
    print("=" * 70)
    print(f"PHASE 1: MORRIS METHOD SCREENING ({args.scenario})")
    print("=" * 70)

    morris = run_morris_screening(
        scenario=args.scenario,
        n_trajectories=args.morris_trajectories,
        n_seeds=args.sensitivity_seeds,
        n_periods=args.periods,
        n_workers=args.workers,
    )
    print_morris_report(morris, args.sensitivity_threshold, args.sensitivity_threshold)

    # Save detailed Morris results
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    save_morris(morris, out / f"{args.scenario}_morris.json")

    # Convert to SensitivityResult for downstream compatibility
    sensitivity = morris.to_sensitivity_result()

    # Save standard sensitivity JSON (for grid phase compatibility)
    save_sensitivity(sensitivity, out / f"{args.scenario}_sensitivity.json")
    print(f"Results saved to {out}")

    # Generate report
    if run_dir:
        generate_sensitivity_report(
            sensitivity, "morris", run_dir / "sensitivity_report.md"
        )

    return sensitivity


def _run_sensitivity_phase(
    args: argparse.Namespace, run_dir: Path | None = None
) -> SensitivityResult:
    """Run Phase 1: Sensitivity analysis (Morris or OAT)."""
    if args.method == "oat":
        return _run_oat_phase(args, run_dir)
    return _run_morris_phase(args, run_dir)


def _run_grid_phase(
    args: argparse.Namespace,
    sensitivity: SensitivityResult | None = None,
    run_dir: Path | None = None,
) -> list[CalibrationResult]:
    """Run Phase 2: Grid screening."""
    out = run_dir or OUTPUT_DIR

    # Load sensitivity if not provided
    if sensitivity is None:
        sens_path = out / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            # Fall back to non-timestamped output
            sens_path = OUTPUT_DIR / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            raise FileNotFoundError(
                "Sensitivity results not found. Run --phase sensitivity first."
            )
        sensitivity = load_sensitivity(sens_path)
        print(f"Loaded sensitivity from {sens_path}")

    print("\n" + "=" * 70)
    print(f"PHASE 2: GRID SCREENING ({args.scenario})")
    print("=" * 70)

    pruning = _resolve_pruning_threshold(
        args.pruning_threshold, args.sensitivity_threshold
    )

    # Load custom grid or build from sensitivity
    if args.grid:
        from calibration.grid import load_grid

        custom_grid = load_grid(Path(args.grid))
        grid = custom_grid
        fixed: dict[str, Any] = {}
        # Apply any --fixed params
        for kv in args.fixed or []:
            key, val = kv.split("=", 1)
            fixed[key] = _parse_value(val)
    else:
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
    combinations = list(generate_combinations(grid, fixed=fixed))

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
    out.mkdir(parents=True, exist_ok=True)
    save_screening(
        results,
        sensitivity,
        grid,
        fixed,
        patterns,
        args.scenario,
        out / f"{args.scenario}_screening.json",
    )
    print(f"Screening results saved to {out}")

    # Generate report
    if run_dir:
        generate_screening_report(
            results,
            grid,
            fixed,
            patterns,
            sensitivity,
            args.scenario,
            run_dir / "screening_report.md",
        )

    return results


def _run_stability_phase(
    args: argparse.Namespace,
    screening_results: list[CalibrationResult] | None = None,
    run_dir: Path | None = None,
) -> list[CalibrationResult]:
    """Run Phase 3: Stability testing."""
    avg_time = 0.0
    out = run_dir or OUTPUT_DIR

    # Load screening if not provided
    if screening_results is None:
        screen_path = out / f"{args.scenario}_screening.json"
        if not screen_path.exists():
            screen_path = OUTPUT_DIR / f"{args.scenario}_screening.json"
        if not screen_path.exists():
            raise FileNotFoundError(
                "Screening results not found. Run --phase grid first."
            )
        screening_results, avg_time = load_screening(screen_path)
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
        rank_by=args.rank_by,
        k_factor=args.k_factor,
    )

    print_results(results)

    # Export best config and run comparison
    comparison = None
    if results:
        config_path = export_best_config(results[0], args.scenario)
        print(f"Best config exported to {config_path}")

        default_config = get_default_values(args.scenario)
        comparison = compare_configs(
            default_config,
            results[0].params,
            args.scenario,
            n_periods=args.periods,
        )
        print_comparison(comparison)

    # Save
    out.mkdir(parents=True, exist_ok=True)
    save_stability(results, args.scenario, out / f"{args.scenario}_stability.json")
    print(f"Stability results saved to {out}")

    # Generate report
    if run_dir:
        generate_stability_report(
            results,
            args.scenario,
            tiers,
            comparison,
            run_dir / "stability_report.md",
        )

    return results


def _run_pairwise_phase(
    args: argparse.Namespace,
    sensitivity: SensitivityResult | None = None,
    run_dir: Path | None = None,
) -> None:
    """Run pairwise interaction analysis."""
    out = run_dir or OUTPUT_DIR

    # Load sensitivity if not provided
    if sensitivity is None:
        sens_path = out / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            sens_path = OUTPUT_DIR / f"{args.scenario}_sensitivity.json"
        if not sens_path.exists():
            raise FileNotFoundError(
                "Sensitivity results not found. Run --phase sensitivity first."
            )
        sensitivity = load_sensitivity(sens_path)

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
    out.mkdir(parents=True, exist_ok=True)
    save_pairwise(result, args.scenario, out / f"{args.scenario}_pairwise.json")
    print(f"Pairwise results saved to {out}")


# =============================================================================
# CLI helpers
# =============================================================================


def _parse_value(s: str) -> int | float | str:
    """Parse a string value to int, float, or str."""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


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
  # Run all phases (baseline, Morris default)
  python -m calibration --scenario baseline --workers 10

  # Run individual phases
  python -m calibration --phase sensitivity --scenario baseline
  python -m calibration --phase grid --scenario baseline
  python -m calibration --phase stability --scenario baseline
  python -m calibration --phase pairwise --scenario baseline

  # Use OAT instead of Morris
  python -m calibration --phase sensitivity --method oat --scenario baseline

  # Load custom grid from YAML
  python -m calibration --phase grid --grid custom_grid.yaml

  # Custom ranking strategy
  python -m calibration --phase stability --rank-by stability

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
        choices=["sensitivity", "morris", "grid", "stability", "pairwise", "all"],
        default=None,
        help="Run a single phase (default: all phases sequentially)",
    )
    parser.add_argument(
        "--method",
        choices=["morris", "oat"],
        default="morris",
        help='Sensitivity method: "morris" (default) or "oat"',
    )
    parser.add_argument(
        "--morris-trajectories",
        type=int,
        default=10,
        help="Number of Morris trajectories (default: 10)",
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

    # Grid input
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Load custom grid from YAML/JSON file",
    )
    parser.add_argument(
        "--fixed",
        type=str,
        action="append",
        default=None,
        help="Fix a parameter: KEY=VALUE (repeatable)",
    )

    # Ranking
    parser.add_argument(
        "--rank-by",
        choices=["combined", "stability", "mean"],
        default="combined",
        help='Ranking strategy for stability (default: "combined")',
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=1.0,
        help="k in mean-k*std formula for combined ranking (default: 1.0)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: timestamped subdir)",
    )

    # Sensitivity
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=0.02,
        help="Minimum delta for INCLUDE in grid search (default: 0.02)",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=str,
        default="auto",
        help=(
            'Max score gap for keeping grid values: "auto" (2x sensitivity threshold), '
            '"none" (disabled), or a float (default: "auto")'
        ),
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=3,
        help="Number of seeds per sensitivity evaluation (default: 3)",
    )

    # Stability
    parser.add_argument(
        "--stability-tiers",
        type=str,
        default="100:10,50:20,10:100",
        help='Stability tiers as "configs:seeds,..." (default: "100:10,50:20,10:100")',
    )

    # Execution
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (grid/stability phases)",
    )

    args = parser.parse_args()

    # Create timestamped run directory
    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(args.scenario)
    print(f"Output directory: {run_dir}")

    # Dispatch to phase
    if args.phase == "sensitivity":
        _run_sensitivity_phase(args, run_dir)
    elif args.phase == "morris":
        _run_morris_phase(args, run_dir)
    elif args.phase == "grid":
        _run_grid_phase(args, run_dir=run_dir)
    elif args.phase == "stability":
        _run_stability_phase(args, run_dir=run_dir)
    elif args.phase == "pairwise":
        _run_pairwise_phase(args, run_dir=run_dir)
    else:
        # Run all phases sequentially
        sensitivity = _run_sensitivity_phase(args, run_dir)
        screening = _run_grid_phase(args, sensitivity=sensitivity, run_dir=run_dir)
        if screening:
            stability = _run_stability_phase(
                args, screening_results=screening, run_dir=run_dir
            )

            # Generate full report
            comparison = None
            if stability:
                default_config = get_default_values(args.scenario)
                comparison = compare_configs(
                    default_config,
                    stability[0].params,
                    args.scenario,
                    n_periods=args.periods,
                )

            tiers = parse_stability_tiers(args.stability_tiers)
            generate_full_report(
                sensitivity,
                screening,
                stability,
                comparison,
                args.scenario,
                tiers,
                run_dir / "full_report.md",
            )
            print(f"\nFull report: {run_dir / 'full_report.md'}")


if __name__ == "__main__":
    main()
