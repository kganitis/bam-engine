"""Run constrained grid search for parameter calibration."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from tests.validation.test_baseline_scenario import (  # noqa: E402
    run_stability_test,
    run_validation,
)

# Output directory for calibration results
OUTPUT_DIR = Path(__file__).parent / "output"


# Fixed parameters (from sensitivity analysis)
FIXED = {
    "job_search_method": "vacancies_only",
    "max_leverage": 100,
    "max_loan_to_net_worth": 100,
    "matching_method": "sequential",
}

# Grid parameters
SIZE_FACTORS = [0.5, 0.7, 0.8, 0.9]
PRODUCTION_FACTORS = [0.5, 0.7, 0.8, 0.9]
WAGE_FACTORS = [0.5, 0.7, 0.8, 0.9, 1.0]
PRICE_MARKUPS = [1.0, 1.05, 1.1, 1.25, 1.50]
FIRING_METHODS = ["expensive", "random"]
CAP_FACTORS = [1.1, 100]


def generate_combinations():
    """Generate valid combinations where production >= size."""
    combinations = []
    for size in SIZE_FACTORS:
        for prod in PRODUCTION_FACTORS:
            if prod >= size:  # Constraint: production >= size
                for wage in WAGE_FACTORS:
                    for markup in PRICE_MARKUPS:
                        for firing in FIRING_METHODS:
                            for cap in CAP_FACTORS:
                                params = {
                                    **FIXED,
                                    "new_firm_size_factor": size,
                                    "new_firm_production_factor": prod,
                                    "new_firm_wage_factor": wage,
                                    "new_firm_price_markup": markup,
                                    "firing_method": firing,
                                    "cap_factor": cap,
                                }
                                combinations.append(params)
    return combinations


def screen_single(params: dict) -> tuple:
    """Screen a single parameter combination."""
    result = run_validation(seed=0, n_periods=1000, **params)
    return params, result.total_score, result.n_pass, result.n_warn, result.n_fail


def compute_combined(stability) -> float:
    """Compute combined score from stability result."""
    return stability.mean_score * stability.pass_rate * (1 - stability.std_score)


def main():
    combinations = generate_combinations()
    print(f"Total combinations: {len(combinations)}")
    print(
        f"Estimated time: ~{len(combinations) * 35 / 60 / 10:.1f} minutes with 10 workers"
    )
    print()

    # Phase 1: Screen all combinations
    print("=" * 70)
    print("PHASE 1: SCREENING (single seed)")
    print("=" * 70)

    results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(screen_single, p) for p in combinations]
        for i, future in enumerate(as_completed(futures)):
            params, score, n_pass, n_warn, n_fail = future.result()
            results.append((params, score, n_pass, n_warn, n_fail))
            if (i + 1) % 100 == 0 or (i + 1) == len(combinations):
                print(f"  Screened {i + 1}/{len(combinations)}")

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    print()
    print("=" * 70)
    print("TOP 20 SCREENING RESULTS")
    print("=" * 70)
    for i, (params, score, n_pass, n_warn, n_fail) in enumerate(results[:20]):
        print(f"#{i + 1}: score={score:.4f} (P:{n_pass}/W:{n_warn}/F:{n_fail})")
        print(
            f"    size={params['new_firm_size_factor']}, prod={params['new_firm_production_factor']}, wage={params['new_firm_wage_factor']}"
        )
        print(
            f"    markup={params['new_firm_price_markup']}, firing={params['firing_method']}, cap={params['cap_factor']}"
        )

    # Phase 2: Stability test top 20
    print()
    print("=" * 70)
    print("PHASE 2: STABILITY TESTING (5 seeds)")
    print("=" * 70)

    stability_results = []
    for i, (params, score, _, _, _) in enumerate(results[:20]):
        print(f"  Testing {i + 1}/20: score={score:.3f}")
        stability = run_stability_test(
            seeds=[0, 42, 123, 456, 789], n_periods=1000, **params
        )
        combined = compute_combined(stability)
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
    print("FINAL RESULTS (by combined score)")
    print("=" * 70)
    for i, (params, combined, mean, std, pass_rate) in enumerate(
        stability_results[:10]
    ):
        print()
        print(
            f"#{i + 1}: Combined={combined:.4f} (mean={mean:.3f}±{std:.3f}, pass={pass_rate:.0%})"
        )
        print(f"    new_firm_size_factor: {params['new_firm_size_factor']}")
        print(f"    new_firm_production_factor: {params['new_firm_production_factor']}")
        print(f"    new_firm_wage_factor: {params['new_firm_wage_factor']}")
        print(f"    new_firm_price_markup: {params['new_firm_price_markup']}")
        print(f"    firing_method: {params['firing_method']}")
        print(f"    cap_factor: {params['cap_factor']}")

    # Save results
    output = {
        "fixed_params": FIXED,
        "top_results": [
            {
                "rank": i + 1,
                "params": {k: v for k, v in params.items() if k not in FIXED},
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
    output_file = OUTPUT_DIR / "grid_search_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
