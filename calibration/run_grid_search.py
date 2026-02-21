"""Run constrained grid search for parameter calibration.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)

Usage:
    # Run grid search for baseline scenario
    python calibration/run_grid_search.py

    # Run grid search for growth_plus scenario
    python calibration/run_grid_search.py --scenario growth_plus

    # Run grid search for buffer_stock scenario
    python calibration/run_grid_search.py --scenario buffer_stock

    # Customize workers and periods
    python calibration/run_grid_search.py --workers 8 --periods 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.optimizer import format_eta  # noqa: E402
from calibration.parameter_space import get_parameter_grid  # noqa: E402
from validation import compute_combined_score, get_validation_funcs  # noqa: E402

# Output directory for calibration results
OUTPUT_DIR = Path(__file__).parent / "output"


def generate_combinations(scenario: str = "baseline"):
    """Generate valid parameter combinations for a scenario.

    For new firm parameters, applies constraint: production_factor >= size_factor
    (a new firm cannot produce more than its size allows).

    Parameters
    ----------
    scenario : str
        Scenario name.

    Returns
    -------
    list[dict]
        List of valid parameter combinations.
    """
    from itertools import product

    grid = get_parameter_grid(scenario)
    param_names = list(grid.keys())
    param_values = list(grid.values())

    combinations = []
    for values in product(*param_values):
        params = dict(zip(param_names, values, strict=True))

        # Constraint: production_factor >= size_factor
        size = params.get("new_firm_size_factor", 0)
        prod = params.get("new_firm_production_factor", 1)
        if prod >= size:
            combinations.append(params)

    return combinations


def screen_single(args: tuple) -> tuple:
    """Screen a single parameter combination.

    Parameters
    ----------
    args : tuple
        (params, scenario, n_periods) tuple for imap compatibility.

    Returns
    -------
    tuple
        (params, score, n_pass, n_warn, n_fail, elapsed)
    """
    params, scenario, n_periods = args
    validate, _, _, _ = get_validation_funcs(scenario)
    t0 = time.monotonic()
    result = validate(seed=0, n_periods=n_periods, **params)
    elapsed = time.monotonic() - t0
    return (
        params,
        result.total_score,
        result.n_pass,
        result.n_warn,
        result.n_fail,
        elapsed,
    )


def main():
    """Main entry point for grid search."""
    parser = argparse.ArgumentParser(
        description="Run constrained grid search for parameter calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run grid search for baseline scenario
  python calibration/run_grid_search.py

  # Run grid search for growth_plus scenario
  python calibration/run_grid_search.py --scenario growth_plus

  # Customize workers and periods
  python calibration/run_grid_search.py --workers 8 --periods 500
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "growth_plus", "buffer_stock"],
        default="baseline",
        help="Scenario to calibrate (default: baseline)",
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
        "--top-k",
        type=int,
        default=20,
        help="Number of top configs for stability testing (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file name (default: {scenario}_grid_search_results.json)",
    )
    parser.add_argument(
        "--stability-seeds",
        type=int,
        default=10,
        help="Number of seeds for stability testing (default: 10)",
    )

    args = parser.parse_args()

    # Get validation functions for scenario
    _, run_stability, _, _ = get_validation_funcs(args.scenario)

    combinations = generate_combinations(args.scenario)
    print(f"Scenario: {args.scenario}")
    print(f"Total combinations: {len(combinations)}")
    print()

    # Phase 1: Screen all combinations
    print("=" * 70)
    print(f"PHASE 1: SCREENING [{args.scenario}] (single seed)")
    print("=" * 70)

    results = []
    run_times: list[float] = []
    # Prepare arguments for imap_unordered
    task_args = [(p, args.scenario, args.periods) for p in combinations]

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(screen_single, task_args, chunksize=1)
        ):
            params, score, n_pass, n_warn, n_fail, elapsed = result
            results.append((params, score, n_pass, n_warn, n_fail))
            run_times.append(elapsed)

            completed = i + 1
            remaining = len(combinations) - completed
            avg_time = sum(run_times) / len(run_times) if run_times else 0
            eta = format_eta(remaining, avg_time, args.workers)
            pct = 100.0 * completed / len(combinations)
            print(
                f"  Screened {completed}/{len(combinations)} ({pct:.1f}%) "
                f"| {remaining} remaining | ETA: {eta}",
                flush=True,
            )

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    print()
    print("=" * 70)
    print(f"TOP {args.top_k} SCREENING RESULTS")
    print("=" * 70)
    for i, (params, score, n_pass, n_warn, n_fail) in enumerate(results[: args.top_k]):
        print(f"#{i + 1}: score={score:.4f} (P:{n_pass}/W:{n_warn}/F:{n_fail})")
        for key, value in params.items():
            print(f"    {key}: {value}")

    # Phase 2: Stability test top k
    print()
    print("=" * 70)
    print(
        f"PHASE 2: STABILITY TESTING [{args.scenario}] ({args.stability_seeds} seeds)"
    )
    print("=" * 70)

    # Use specified number of seeds for stability testing
    stability_seeds = list(range(args.stability_seeds))
    stability_results = []
    for i, (params, score, _, _, _) in enumerate(results[: args.top_k]):
        remaining = args.top_k - i - 1
        avg_time = sum(run_times) / len(run_times) if run_times else 0
        eta = format_eta(remaining * args.stability_seeds, avg_time, args.workers)
        print(f"  Testing {i + 1}/{args.top_k}: score={score:.3f} | ETA: {eta}")
        stability = run_stability(
            seeds=stability_seeds, n_periods=args.periods, **params
        )
        combined = compute_combined_score(stability)
        stability_results.append(
            (
                params,
                combined,
                stability.mean_score,
                stability.std_score,
                stability.pass_rate,
            )
        )

    stability_results.sort(key=lambda x: x[1], reverse=True)

    print()
    print("=" * 70)
    print(f"FINAL RESULTS [{args.scenario}] (by combined score)")
    print("=" * 70)
    for i, (params, combined, mean, std, pass_rate) in enumerate(
        stability_results[:10]
    ):
        print()
        print(
            f"#{i + 1}: Combined={combined:.4f} "
            f"(mean={mean:.3f}±{std:.3f}, pass={pass_rate:.0%})"
        )
        for key, value in params.items():
            print(f"    {key}: {value}")

    # Save results
    output = {
        "scenario": args.scenario,
        "n_periods": args.periods,
        "top_results": [
            {
                "rank": i + 1,
                "params": {k: v for k, v in params.items()},
                "combined_score": combined,
                "mean_score": mean,
                "std_score": std,
                "pass_rate": pass_rate,
            }
            for i, (params, combined, mean, std, pass_rate) in enumerate(
                stability_results
            )
        ],
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / (
        args.output or f"{args.scenario}_grid_search_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
