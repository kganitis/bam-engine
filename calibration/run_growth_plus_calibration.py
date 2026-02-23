#!/usr/bin/env python3
"""Growth+ calibration: grid screening + tiered stability testing.

Run with:
    python calibration/run_growth_plus_calibration.py [--phase screening|stability|both]
"""

import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("calibration/output")

# ─── Grid definition ──────────────────────────────────────────────────────────
# 9 INCLUDE params (varied in grid)
GRID = {
    "new_firm_price_markup": [1.0, 1.05, 1.10, 1.25, 1.50],
    "new_firm_wage_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
    "new_firm_size_factor": [0.5, 0.7, 0.8, 0.9],
    "new_firm_production_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
    "price_cut_allow_increase": [True, False],
    "inflation_method": ["yoy", "annualized"],
    "min_wage_ratchet": [True, False],
    "job_search_method": ["vacancies_only", "all_firms"],
    "max_M": [2, 4],
}

# 13 FIX params (held at defaults.yml values)
FIXED = {
    "price_init": 1.5,
    "savings_init": 3.0,
    "net_worth_ratio": 1.0,
    "equity_base_init": 5.0,
    "pricing_phase": "planning",
    "min_wage_ratio": 1.0,
    "beta": 2.5,
    "labor_matching": "interleaved",
    "max_leverage": 5,
    "sigma_decay": -1.0,
    "max_loan_to_net_worth": 2,
    "credit_matching": "interleaved",
    "matching_method": "sequential",
}


def build_combinations():
    """Build all grid combinations with constraint filtering."""
    keys = list(GRID.keys())
    combos = []
    for vals in product(*GRID.values()):
        params = dict(zip(keys, vals, strict=True))
        # Constraint: production_factor >= size_factor
        if params["new_firm_production_factor"] < params["new_firm_size_factor"]:
            continue
        combos.append({**FIXED, **params})
    return combos


def save_screening_results(results, path):
    """Save all screening results to JSON."""
    output = {
        "scenario": "growth_plus",
        "n_combinations": len(results),
        "n_include_params": len(GRID),
        "n_fix_params": len(FIXED),
        "include_params": list(GRID.keys()),
        "fix_params": FIXED,
        "results": [],
    }
    for r in results:
        output["results"].append(
            {
                "params": r.params,
                "single_score": r.single_score,
                "n_pass": r.n_pass,
                "n_warn": r.n_warn,
                "n_fail": r.n_fail,
            }
        )
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScreening results saved to {path} ({len(results)} configs)")


def print_top_results(results, n=20):
    """Print top N screening results."""
    print(f"\n{'=' * 72}")
    print(f"TOP {n} SCREENING RESULTS")
    print(f"{'=' * 72}")
    for i, r in enumerate(results[:n]):
        p = r.params
        include_params = {k: p[k] for k in GRID}
        print(
            f"#{i + 1}: score={r.single_score:.4f} (F:{r.n_fail}/W:{r.n_warn}) {include_params}"
        )


def run_screening_phase():
    """Phase 2: Grid search screening."""
    from calibration.optimizer import run_screening

    combos = build_combinations()
    print(f"Grid: {len(combos)} combinations ({len(GRID)} INCLUDE, {len(FIXED)} FIX)")

    results = run_screening(
        combos,
        scenario="growth_plus",
        n_workers=10,
        n_periods=1000,
        avg_time_per_run=2.9,
        checkpoint_every=50,
        resume=True,  # Resume from checkpoint if available
    )

    # Save results IMMEDIATELY after screening
    save_screening_results(results, OUTPUT_DIR / "growth_plus_screening.json")
    print_top_results(results)

    # Print score distribution stats
    scores = [r.single_score for r in results]
    zero_fail = [r for r in results if r.n_fail == 0]
    print(
        f"\nScore stats: min={min(scores):.4f} max={max(scores):.4f} "
        f"mean={sum(scores) / len(scores):.4f}"
    )
    print(
        f"Zero-fail: {len(zero_fail)}/{len(results)} ({100 * len(zero_fail) / len(results):.1f}%)"
    )
    print(f"Score >= 0.85: {sum(1 for s in scores if s >= 0.85)}")
    print(f"Score >= 0.86: {sum(1 for s in scores if s >= 0.86)}")

    return results


def run_stability_phase(screening_results=None):
    """Phase 3: Tiered stability testing."""
    from calibration.optimizer import CalibrationResult, run_tiered_stability

    # Load screening results if not provided
    if screening_results is None:
        path = OUTPUT_DIR / "growth_plus_screening.json"
        print(f"Loading screening results from {path}")
        with open(path) as f:
            data = json.load(f)
        screening_results = []
        for r in data["results"]:
            cr = CalibrationResult(
                params=r["params"],
                single_score=r["single_score"],
                n_pass=r["n_pass"],
                n_warn=r["n_warn"],
                n_fail=r["n_fail"],
                seed_scores=[r["single_score"]],
            )
            screening_results.append(cr)

    # Take top 110 candidates (score >= 0.84, top ~1%)
    candidates = screening_results[:110]
    for c in candidates:
        if c.seed_scores is None:
            c.seed_scores = [c.single_score]

    print(f"\nLoaded {len(candidates)} candidates for stability testing")
    print(
        f"Score range: {candidates[-1].single_score:.4f} - {candidates[0].single_score:.4f}"
    )

    # 4-tiered stability: 110→50→20→10
    tiers = [(110, 10), (50, 20), (20, 50), (10, 100)]
    tier1_runs = 110 * 9  # already have seed 0
    tier2_runs = 50 * 10
    tier3_runs = 20 * 30
    tier4_runs = 10 * 50
    total_runs = tier1_runs + tier2_runs + tier3_runs + tier4_runs
    print("\nTier structure:")
    print("  Tier 1: 110 configs × 10 seeds → keep top 50")
    print("  Tier 2:  50 configs × 20 seeds → keep top 20")
    print("  Tier 3:  20 configs × 50 seeds → keep top 10")
    print("  Tier 4:  10 configs × 100 seeds → final ranking")
    print(f"  Estimated runs: {total_runs}, ~{total_runs * 2.9 / 10 / 60:.0f} min")

    print(f"\n{'=' * 72}")
    print("STARTING TIERED STABILITY TESTING")
    print(f"{'=' * 72}")

    results = run_tiered_stability(
        candidates,
        scenario="growth_plus",
        tiers=tiers,
        n_workers=10,
        n_periods=1000,
        avg_time_per_run=2.9,
    )

    # Print final results
    print(f"\n{'=' * 72}")
    print("FINAL STABILITY RESULTS")
    print(f"{'=' * 72}")
    for i, r in enumerate(results):
        p = r.params
        print(
            f"\n#{i + 1}: combined={r.combined_score:.4f} "
            f"mean={r.mean_score:.4f} ± {r.std_score:.4f} "
            f"({len(r.seed_scores)} seeds)"
        )
        include_params = {k: p.get(k) for k in GRID}
        print(f"  {include_params}")

    # Save results
    output = {
        "scenario": "growth_plus",
        "tiers": [list(t) for t in tiers],
        "results": [],
    }
    for i, r in enumerate(results):
        output["results"].append(
            {
                "rank": i + 1,
                "params": r.params,
                "combined_score": r.combined_score,
                "mean_score": r.mean_score,
                "std_score": r.std_score,
                "single_score": r.single_score,
                "n_fail": r.n_fail,
                "n_warn": r.n_warn,
                "seed_scores": r.seed_scores,
            }
        )

    path = OUTPUT_DIR / "growth_plus_calibration_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nStability results saved to {path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Growth+ calibration")
    parser.add_argument(
        "--phase",
        choices=["screening", "stability", "both"],
        default="both",
        help="Which phase to run (default: both)",
    )
    args = parser.parse_args()

    t0 = time.monotonic()

    if args.phase in ("screening", "both"):
        screening = run_screening_phase()
    else:
        screening = None

    if args.phase in ("stability", "both"):
        run_stability_phase(screening)

    elapsed = time.monotonic() - t0
    print(f"\nTotal elapsed: {elapsed / 60:.1f} minutes")
