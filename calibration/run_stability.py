#!/usr/bin/env python3
"""Phase 3: Tiered stability testing for Growth+ calibration."""

import json
import sys

sys.path.insert(0, ".")
from calibration.optimizer import CalibrationResult, run_tiered_stability

# Load top 20 from screening
with open("calibration/output/growth_plus_screening.json") as f:
    data = json.load(f)

# Reconstruct CalibrationResult objects
candidates = []
for r in data["top_results"]:
    cr = CalibrationResult(
        params={**data["fix_params"], **r["params"]},
        single_score=r["single_score"],
        n_pass=66 - r["n_fail"] - r["n_warn"],  # approximate
        n_warn=r["n_warn"],
        n_fail=r["n_fail"],
        seed_scores=[r["single_score"]],  # seed 0 already done
    )
    candidates.append(cr)

print(f"Loaded {len(candidates)} candidates for stability testing")
print(
    f"Score range: {candidates[-1].single_score:.4f} - {candidates[0].single_score:.4f}"
)

# Adjusted tiers for 20 candidates:
# Tier 1: 20 configs × 10 seeds → keep top 10
# Tier 2: 10 configs × 20 seeds → keep top 5
# Tier 3:  5 configs × 100 seeds → final ranking
tiers = [(20, 10), (10, 20), (5, 100)]

print("\nTier structure:")
print(f"  Tier 1: {tiers[0][0]} configs × {tiers[0][1]} seeds → keep top {tiers[1][0]}")
print(f"  Tier 2: {tiers[1][0]} configs × {tiers[1][1]} seeds → keep top {tiers[2][0]}")
print(f"  Tier 3: {tiers[2][0]} configs × {tiers[2][1]} seeds → final ranking")

# Estimated cost
tier1_runs = 20 * (10 - 1)  # already have seed 0
tier2_runs = 10 * (20 - 10)
tier3_runs = 5 * (100 - 20)
total_runs = tier1_runs + tier2_runs + tier3_runs
print(f"\nEstimated runs: {tier1_runs} + {tier2_runs} + {tier3_runs} = {total_runs}")
print(
    f"Estimated time: ~{total_runs * 2.9 / 10 / 60:.0f} minutes (10 workers, ~2.9s/run)"
)

print("\n" + "=" * 72)
print("STARTING TIERED STABILITY TESTING")
print("=" * 72)

results = run_tiered_stability(
    candidates,
    scenario="growth_plus",
    tiers=tiers,
    n_workers=10,
    n_periods=1000,
    avg_time_per_run=2.9,
)

# Print final results
print("\n" + "=" * 72)
print("FINAL STABILITY RESULTS")
print("=" * 72)
for i, r in enumerate(results):
    p = r.params
    print(
        f"\n#{i + 1}: combined={r.combined_score:.4f} "
        f"mean={r.mean_score:.4f} ± {r.std_score:.4f} "
        f"({len(r.seed_scores)} seeds)"
    )
    print(
        f"  nfpm={p.get('new_firm_price_markup')}, "
        f"nfwf={p.get('new_firm_wage_factor')}, "
        f"nfsf={p.get('new_firm_size_factor')}, "
        f"nfpf={p.get('new_firm_production_factor')}"
    )
    print(
        f"  pca={p.get('price_cut_allow_increase')}, "
        f"im={p.get('inflation_method')}, "
        f"mwr={p.get('min_wage_ratchet')}, "
        f"js={p.get('job_search_method')}, "
        f"mM={p.get('max_M')}"
    )

# Save results
output = {"scenario": "growth_plus", "tiers": [list(t) for t in tiers], "results": []}
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

with open("calibration/output/growth_plus_calibration_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to calibration/output/growth_plus_calibration_results.json")
