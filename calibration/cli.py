"""Command-line interface for calibration.

Usage:
    # Run sensitivity analysis only
    python -m calibration --sensitivity-only --workers 10

    # Run full calibration
    python -m calibration --workers 10 --periods 1000

    # Custom thresholds
    python -m calibration --high-threshold 0.08 --medium-threshold 0.04
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.optimizer import (
    CalibrationResult,
    build_focused_grid,
    run_focused_calibration,
)
from calibration.parameter_space import count_combinations
from calibration.sensitivity import print_sensitivity_report, run_sensitivity_analysis

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
        print(f"\n#{i + 1}: Combined Score = {r.combined_score:.4f}")
        print(f"    Mean Score:  {r.mean_score:.3f} ± {r.std_score:.3f}")
        print(f"    Pass Rate:   {r.pass_rate:.0%}")
        print("    Parameters:")
        for k, v in r.params.items():
            print(f"      {k}: {v}")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point for calibration CLI."""
    parser = argparse.ArgumentParser(
        description="BAM Parameter Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sensitivity analysis only (~15-20 minutes)
  python -m calibration --sensitivity-only --workers 10

  # Run full calibration (runtime depends on sensitivity results)
  python -m calibration --workers 10 --periods 1000

  # Use stricter thresholds for smaller grid
  python -m calibration --high-threshold 0.08 --medium-threshold 0.04
        """,
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Run only sensitivity analysis (Phase 1-2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top configs for stability testing (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10, optimized for M4 Pro)",
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
        default="calibration_results.json",
        help="Output file for results (default: calibration_results.json)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.05,
        help="Sensitivity threshold for HIGH importance (default: 0.05)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.02,
        help="Sensitivity threshold for MEDIUM importance (default: 0.02)",
    )

    args = parser.parse_args()

    # Phase 1-2: Sensitivity Analysis
    print("=" * 70)
    print("PHASE 1: SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Testing 29 parameter variations (10 parameters x ~3 values each)")

    sensitivity = run_sensitivity_analysis(
        n_workers=args.workers,
        n_periods=args.periods,
    )
    print_sensitivity_report(sensitivity)

    if args.sensitivity_only:
        # Save sensitivity results
        sensitivity_data = {
            "baseline_score": sensitivity.baseline_score,
            "parameters": {
                p.name: {
                    "sensitivity": p.sensitivity,
                    "best_value": p.best_value,
                    "best_score": p.best_score,
                    "values": p.values,
                    "scores": p.scores,
                }
                for p in sensitivity.parameters
            },
        }
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = OUTPUT_DIR / args.output.replace(".json", "_sensitivity.json")
        with open(output_file, "w") as f:
            json.dump(sensitivity_data, f, indent=2)
        print(f"Sensitivity results saved to {output_file}")
        print("Run without --sensitivity-only to continue with grid search.")
        return

    # Phase 3-4: Focused Grid Search + Stability Testing
    print("\n" + "=" * 70)
    print("PHASE 2: BUILD FOCUSED GRID")
    print("=" * 70)

    grid, fixed = build_focused_grid(
        sensitivity,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )

    n_combinations = count_combinations(grid)
    print(f"Grid parameters: {list(grid.keys())}")
    print(f"Fixed parameters: {list(fixed.keys())}")
    print(f"Combinations to test: {n_combinations}")

    if n_combinations == 0:
        print("\nNo parameters require grid search - all fixed at best values.")
        print("Best configuration:")
        for k, v in fixed.items():
            print(f"  {k}: {v}")
        return

    results = run_focused_calibration(
        grid=grid,
        fixed_params=fixed,
        top_k=args.top_k,
        n_workers=args.workers,
        n_periods=args.periods,
    )

    print_results(results)

    # Save results
    output_data = {
        "sensitivity": {
            p.name: {"sensitivity": p.sensitivity, "best_value": p.best_value}
            for p in sensitivity.parameters
        },
        "fixed_params": fixed,
        "grid_params": {k: v for k, v in grid.items()},
        "results": [
            {
                "rank": i + 1,
                "params": r.params,
                "combined_score": r.combined_score,
                "mean_score": r.mean_score,
                "std_score": r.std_score,
                "pass_rate": r.pass_rate,
            }
            for i, r in enumerate(results)
        ],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / args.output
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
