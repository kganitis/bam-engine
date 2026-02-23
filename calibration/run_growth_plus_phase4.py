#!/usr/bin/env python3
"""Growth+ Phase 4: Second-pass calibration of structural/initial-condition params.

Phase 2-3 calibrated the 9 most sensitive parameters (new-firm entry, labor market,
inflation). This second pass fixes those 9 at their current default values and
optimizes the 8 structural/initial-condition params that were held fixed.

Run with:
    python calibration/run_growth_plus_phase4.py --phase morris|screening|stability|all
"""

import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("calibration/output")

# ─── Grid definition (8 params to optimize) ─────────────────────────────────
GRID = {
    "price_init": [0.5, 1.0, 1.5, 2.0, 3.0],
    "min_wage_ratio": [0.50, 0.67, 0.80, 1.0],
    "net_worth_ratio": [0.5, 1.0, 2.0, 3.0, 5.0],
    "equity_base_init": [1.0, 3.0, 5.0, 10.0],
    "savings_init": [1.0, 3.0, 5.0, 10.0],
    "beta": [0.5, 1.0, 2.5, 5.0, 10.0],
    "max_loan_to_net_worth": [0, 2, 5, 10],
    "max_leverage": [0, 5, 10, 20],
}

# ─── Fixed parameters (14 total) ────────────────────────────────────────────
# 9 calibrated INCLUDE params from Phase 2-3 (at current defaults)
FIXED_CALIBRATED = {
    "new_firm_size_factor": 0.5,
    "new_firm_production_factor": 0.5,
    "new_firm_wage_factor": 0.5,
    "new_firm_price_markup": 1.0,
    "max_M": 4,
    "job_search_method": "all_firms",
    "price_cut_allow_increase": True,
    "inflation_method": "yoy",
    "min_wage_ratchet": False,
}

# 5 invariant params (excluded from both searches)
FIXED_INVARIANT = {
    "sigma_decay": -1.0,
    "pricing_phase": "planning",
    "labor_matching": "interleaved",
    "credit_matching": "interleaved",
    "matching_method": "sequential",
}

FIXED = {**FIXED_CALIBRATED, **FIXED_INVARIANT}


def build_combinations():
    """Build all grid combinations with fixed params merged in."""
    keys = list(GRID.keys())
    combos = []
    for vals in product(*GRID.values()):
        params = dict(zip(keys, vals, strict=True))
        combos.append({**FIXED, **params})
    return combos


def save_screening_results(results, path):
    """Save all screening results to JSON."""
    output = {
        "scenario": "growth_plus",
        "phase": "phase4",
        "n_combinations": len(results),
        "n_grid_params": len(GRID),
        "n_fixed_params": len(FIXED),
        "grid_params": list(GRID.keys()),
        "fixed_params": FIXED,
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
        grid_params = {k: p[k] for k in GRID}
        print(
            f"#{i + 1}: score={r.single_score:.4f} (F:{r.n_fail}/W:{r.n_warn}) {grid_params}"
        )


# ─── Phase 4A: Morris screening ─────────────────────────────────────────────


def run_morris_phase():
    """Phase 4A: Morris Method screening on the 8 structural params."""
    from calibration.morris import print_morris_report, run_morris_screening

    print("=" * 72)
    print("PHASE 4A: MORRIS SCREENING (structural/initial-condition params)")
    print("=" * 72)

    # Build Morris grid: single-value entries for FIXED, multi-value for GRID.
    # Single-value params produce zero elementary effects → auto-classified FIX.
    morris_grid = {**{k: [v] for k, v in FIXED.items()}, **GRID}

    n_active = sum(1 for v in morris_grid.values() if len(v) > 1)
    n_traj = 20
    n_seeds = 5
    n_configs_est = (n_active + 1) * n_traj
    n_runs_est = n_configs_est * n_seeds
    print(f"\nGrid: {len(morris_grid)} params ({n_active} active, {len(FIXED)} fixed)")
    print(
        f"Estimated: {n_configs_est} unique configs × {n_seeds} seeds = {n_runs_est} runs"
    )
    print(
        f"Estimated time: ~{n_runs_est * 2.9 / 10 / 60:.0f} min (10 workers, ~2.9s/run)"
    )

    morris = run_morris_screening(
        scenario="growth_plus",
        grid=morris_grid,
        n_trajectories=n_traj,
        n_seeds=n_seeds,
        n_periods=1000,
        n_workers=10,
    )

    print_morris_report(morris)

    # Save Morris results
    _save_morris(morris, OUTPUT_DIR / "growth_plus_p4_morris.json")

    # Print grid size estimate after pruning
    sensitivity = morris.to_sensitivity_result()
    _print_pruning_preview(sensitivity)

    return morris


def _save_morris(morris, path):
    """Save Morris result to JSON (same format as cli.py)."""
    data = {
        "scenario": morris.scenario,
        "phase": "phase4",
        "n_trajectories": morris.n_trajectories,
        "n_evaluations": morris.n_evaluations,
        "avg_time_per_run": morris.avg_time_per_run,
        "n_seeds": morris.n_seeds,
        "effects": {
            e.name: {
                "mu": e.mu,
                "mu_star": e.mu_star,
                "sigma": e.sigma,
                "elementary_effects": e.elementary_effects,
                "value_scores": {
                    str(v): scores for v, scores in e.value_scores.items()
                },
            }
            for e in morris.effects
        },
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Morris results saved to {path}")


def _print_pruning_preview(sensitivity):
    """Show grid size before/after Morris-based pruning."""
    from calibration.optimizer import build_focused_grid
    from calibration.parameter_space import count_combinations

    raw_count = count_combinations(GRID)

    # Use build_focused_grid to see what Morris recommends
    grid_pruned, fixed_from_morris = build_focused_grid(
        sensitivity,
        full_grid=GRID,
        scenario="growth_plus",
    )

    pruned_count = count_combinations(grid_pruned)

    print(f"\n{'=' * 72}")
    print("GRID SIZE PREVIEW (after Morris pruning)")
    print(f"{'=' * 72}")
    print(f"Raw grid:    {raw_count:,} combinations ({len(GRID)} params)")
    print(
        f"Pruned grid: {pruned_count:,} combinations ({len(grid_pruned)} INCLUDE params)"
    )
    if fixed_from_morris:
        print(f"Fixed by Morris: {fixed_from_morris}")
    print(f"Reduction: {100 * (1 - pruned_count / raw_count):.1f}%")
    print(f"Estimated screening time: ~{pruned_count * 2.9 / 10 / 60:.0f} min")
    print(f"{'=' * 72}")


# ─── Phase 4B: Grid search screening ────────────────────────────────────────


def run_screening_phase(morris=None):
    """Phase 4B: Grid search screening with Morris-pruned grid."""
    from calibration.optimizer import build_focused_grid, run_screening

    print("\n" + "=" * 72)
    print("PHASE 4B: GRID SEARCH SCREENING")
    print("=" * 72)

    # Load Morris results if not provided
    if morris is None:
        morris = _load_morris()

    sensitivity = morris.to_sensitivity_result()

    # Build pruned grid from Morris results
    grid_pruned, fixed_from_morris = build_focused_grid(
        sensitivity,
        full_grid=GRID,
        scenario="growth_plus",
    )

    # Merge all fixed params: FIXED (14) + Morris-classified FIX params
    all_fixed = {**FIXED, **fixed_from_morris}

    # Build combinations
    keys = list(grid_pruned.keys())
    combos = []
    for vals in product(*grid_pruned.values()):
        params = dict(zip(keys, vals, strict=True))
        combos.append({**all_fixed, **params})

    from calibration.parameter_space import count_combinations

    print(f"Pruned grid: {count_combinations(grid_pruned):,} combinations")
    print(f"INCLUDE params: {list(grid_pruned.keys())}")
    for k, v in grid_pruned.items():
        print(f"  {k}: {v}")
    print(
        f"Fixed params: {len(all_fixed)} ({len(FIXED)} original + {len(fixed_from_morris)} Morris FIX)"
    )
    if fixed_from_morris:
        print(f"  Morris FIX: {fixed_from_morris}")

    results = run_screening(
        combos,
        scenario="growth_plus",
        n_workers=10,
        n_periods=1000,
        avg_time_per_run=2.9,
        checkpoint_every=50,
        resume=True,
    )

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_screening_results(results, OUTPUT_DIR / "growth_plus_p4_screening.json")
    print_top_results(results)

    # Score distribution
    scores = [r.single_score for r in results]
    zero_fail = [r for r in results if r.n_fail == 0]
    print(
        f"\nScore stats: min={min(scores):.4f} max={max(scores):.4f} "
        f"mean={sum(scores) / len(scores):.4f}"
    )
    print(
        f"Zero-fail: {len(zero_fail)}/{len(results)} ({100 * len(zero_fail) / len(results):.1f}%)"
    )
    for threshold in [0.84, 0.85, 0.86, 0.87]:
        n = sum(1 for s in scores if s >= threshold)
        print(f"Score >= {threshold}: {n}")

    return results


def _load_morris():
    """Load Morris results from JSON."""
    from calibration.morris import MorrisParameterEffect, MorrisResult

    path = OUTPUT_DIR / "growth_plus_p4_morris.json"
    print(f"Loading Morris results from {path}")
    with open(path) as f:
        data = json.load(f)

    effects = []
    for name, edata in data["effects"].items():
        # Convert string keys back to native types for value_scores
        value_scores = {}
        for k, v in edata.get("value_scores", {}).items():
            try:
                key = json.loads(k)
            except (json.JSONDecodeError, ValueError):
                key = k
            value_scores[key] = v

        effects.append(
            MorrisParameterEffect(
                name=name,
                mu=edata["mu"],
                mu_star=edata["mu_star"],
                sigma=edata["sigma"],
                elementary_effects=edata["elementary_effects"],
                value_scores=value_scores,
            )
        )

    return MorrisResult(
        effects=effects,
        n_trajectories=data["n_trajectories"],
        n_evaluations=data["n_evaluations"],
        scenario=data["scenario"],
        avg_time_per_run=data.get("avg_time_per_run", 0.0),
        n_seeds=data.get("n_seeds", 1),
    )


# ─── Phase 4C: Tiered stability testing ─────────────────────────────────────


def run_stability_phase(screening_results=None):
    """Phase 4C: Tiered stability testing."""
    from calibration.optimizer import CalibrationResult, run_tiered_stability

    print("\n" + "=" * 72)
    print("PHASE 4C: TIERED STABILITY TESTING")
    print("=" * 72)

    # Load screening results if not provided
    if screening_results is None:
        path = OUTPUT_DIR / "growth_plus_p4_screening.json"
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

    # Take top candidates (adapt count to screening results)
    n_total = len(screening_results)
    n_candidates = min(110, n_total)
    candidates = screening_results[:n_candidates]
    for c in candidates:
        if c.seed_scores is None:
            c.seed_scores = [c.single_score]

    print(f"\nLoaded {n_candidates} candidates for stability testing")
    print(
        f"Score range: {candidates[-1].single_score:.4f} - {candidates[0].single_score:.4f}"
    )

    # 4-tiered stability: adapt to available candidates
    if n_candidates >= 110:
        tiers = [(110, 10), (50, 20), (20, 50), (10, 100)]
    elif n_candidates >= 50:
        tiers = [(n_candidates, 10), (30, 20), (15, 50), (10, 100)]
    elif n_candidates >= 20:
        tiers = [(n_candidates, 10), (15, 20), (10, 50), (5, 100)]
    else:
        tiers = [
            (n_candidates, 10),
            (min(10, n_candidates), 20),
            (min(5, n_candidates), 100),
        ]

    # Print tier structure
    print("\nTier structure:")
    for i, (n_cfg, n_seeds) in enumerate(tiers):
        next_n = tiers[i + 1][0] if i + 1 < len(tiers) else "final"
        print(f"  Tier {i + 1}: {n_cfg} configs × {n_seeds} seeds → keep top {next_n}")

    # Estimate cost
    total_runs = 0
    prev_seeds = 1  # seed 0 already done
    for n_cfg, n_seeds in tiers:
        total_runs += n_cfg * (n_seeds - prev_seeds)
        prev_seeds = n_seeds
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
        grid_params = {k: p.get(k) for k in GRID}
        print(f"  {grid_params}")

    # Save results
    output = {
        "scenario": "growth_plus",
        "phase": "phase4",
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

    path = OUTPUT_DIR / "growth_plus_p4_calibration_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nStability results saved to {path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Growth+ Phase 4 calibration")
    parser.add_argument(
        "--phase",
        choices=["morris", "screening", "stability", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    t0 = time.monotonic()

    morris = None
    screening = None

    if args.phase in ("morris", "all"):
        morris = run_morris_phase()

    if args.phase in ("screening", "all"):
        screening = run_screening_phase(morris)

    if args.phase in ("stability", "all"):
        run_stability_phase(screening)

    elapsed = time.monotonic() - t0
    print(f"\nTotal elapsed: {elapsed / 60:.1f} minutes")
