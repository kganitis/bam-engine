"""Run constrained grid search for parameter calibration.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)

Usage:
    # Run grid search for baseline scenario
    python calibration/run_grid_search.py

    # Run grid search for growth_plus scenario
    python calibration/run_grid_search.py --scenario growth_plus

    # Customize workers and periods
    python calibration/run_grid_search.py --workers 8 --periods 500
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    list[dict]
        List of valid parameter combinations.
    """
    grid = get_parameter_grid(scenario)

    # Extract new firm parameters (common to both scenarios)
    size_factors = grid.get("new_firm_size_factor", [0.5])
    prod_factors = grid.get("new_firm_production_factor", [0.5])
    wage_factors = grid.get("new_firm_wage_factor", [0.5])
    price_markups = grid.get("new_firm_price_markup", [1.0])

    # Extract scenario-specific parameters
    sigma_decays = grid.get("sigma_decay", [None])  # Only for growth_plus

    combinations = []
    for size in size_factors:
        for prod in prod_factors:
            if prod >= size:  # Constraint: production >= size
                for wage in wage_factors:
                    for markup in price_markups:
                        for sigma in sigma_decays:
                            params = {
                                "new_firm_size_factor": size,
                                "new_firm_production_factor": prod,
                                "new_firm_wage_factor": wage,
                                "new_firm_price_markup": markup,
                            }
                            if sigma is not None:
                                params["sigma_decay"] = sigma
                            combinations.append(params)
    return combinations


def screen_single(params: dict, scenario: str, n_periods: int) -> tuple:
    """Screen a single parameter combination.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name ("baseline" or "growth_plus").
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    tuple
        (params, score, n_pass, n_warn, n_fail)
    """
    validate, _ = get_validation_funcs(scenario)
    result = validate(seed=0, n_periods=n_periods, **params)
    return params, result.total_score, result.n_pass, result.n_warn, result.n_fail


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
        choices=["baseline", "growth_plus"],
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
    _, run_stability = get_validation_funcs(args.scenario)

    combinations = generate_combinations(args.scenario)
    print(f"Scenario: {args.scenario}")
    print(f"Total combinations: {len(combinations)}")
    print(
        f"Estimated time: ~{len(combinations) * 35 / 60 / args.workers:.1f} minutes "
        f"with {args.workers} workers"
    )
    print()

    # Phase 1: Screen all combinations
    print("=" * 70)
    print(f"PHASE 1: SCREENING [{args.scenario}] (single seed)")
    print("=" * 70)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(screen_single, p, args.scenario, args.periods)
            for p in combinations
        ]
        for i, future in enumerate(as_completed(futures)):
            params, score, n_pass, n_warn, n_fail = future.result()
            results.append((params, score, n_pass, n_warn, n_fail))
            if (i + 1) % 100 == 0 or (i + 1) == len(combinations):
                print(f"  Screened {i + 1}/{len(combinations)}")

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    print()
    print("=" * 70)
    print(f"TOP {args.top_k} SCREENING RESULTS")
    print("=" * 70)
    for i, (params, score, n_pass, n_warn, n_fail) in enumerate(results[: args.top_k]):
        print(f"#{i + 1}: score={score:.4f} (P:{n_pass}/W:{n_warn}/F:{n_fail})")
        print(
            f"    size={params['new_firm_size_factor']}, "
            f"prod={params['new_firm_production_factor']}"
        )
        print(
            f"    wage={params['new_firm_wage_factor']}, "
            f"markup={params['new_firm_price_markup']}"
        )
        if "sigma_decay" in params:
            print(f"    sigma_decay={params['sigma_decay']}")

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
        print(f"  Testing {i + 1}/{args.top_k}: score={score:.3f}")
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
        print(f"    new_firm_size_factor: {params['new_firm_size_factor']}")
        print(f"    new_firm_production_factor: {params['new_firm_production_factor']}")
        print(f"    new_firm_wage_factor: {params['new_firm_wage_factor']}")
        print(f"    new_firm_price_markup: {params['new_firm_price_markup']}")
        if "sigma_decay" in params:
            print(f"    sigma_decay: {params['sigma_decay']}")

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
