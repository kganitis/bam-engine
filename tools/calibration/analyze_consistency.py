"""
Analyze Configuration Consistency Across Seeds
===============================================

This script runs top configurations with multiple seeds to evaluate
which configurations are most consistent (low variance across seeds).

Supports both original and re-scored checkpoints:
- Original checkpoint (calibration_checkpoint.json): Contains grid_results
- Re-scored checkpoint (calibration_checkpoint_rescored.json): Contains results

Usage:
    python tools/calibration/analyze_consistency.py [--top N] [--seeds S]

Examples:
    python tools/calibration/analyze_consistency.py --top 20 --seeds 10
    python tools/calibration/analyze_consistency.py --checkpoint output/calibration_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from calibration import OUTPUT_DIR, run_single_simulation, visualize_configuration

# Default sweep parameters (for deduplicating original checkpoint)
SWEEP_PARAM_DEFAULTS = {
    "v": 0.10,
    "savings_init": 3,
    "new_firm_scale_factor": 0.8,
    "price_cut_allow_increase": False,
    "loanbook_clear_on_repay": True,
    "firing_method": "random",
    "fragility_cap_method": "none",
}


def normalize_params(params: dict) -> tuple:
    """Normalize params by removing sweep params that match defaults."""
    normalized = {}
    for k, v in params.items():
        if k not in SWEEP_PARAM_DEFAULTS or v != SWEEP_PARAM_DEFAULTS[k]:
            normalized[k] = v
    return tuple(sorted(normalized.items()))


def get_unique_realistic_configs(
    results: list[dict], min_unemployment: float = 0.01
) -> list[dict]:
    """Get unique, realistic configurations from original checkpoint results."""
    sorted_results = sorted(results, key=lambda x: x["scores"]["total"])
    seen = set()
    candidates = []
    for r in sorted_results:
        unemp = r["scores"].get("_unemployment_mean", 0)
        if unemp < min_unemployment:
            continue
        norm = normalize_params(r["params"])
        if norm not in seen:
            seen.add(norm)
            candidates.append(r)
    return candidates


@dataclass
class ConsistencyResult:
    """Results from multi-seed consistency analysis."""

    rank: int
    params: dict

    # Checkpoint scores
    checkpoint_score: float  # Original total or rescored_total
    deflation_pct_checkpoint: float  # From checkpoint (if available)

    # Multi-seed score results
    scores: list[float]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    consistency_score: float  # Lower is better (mean + std penalty)

    # Okun correlation
    okun_corrs: list[float]
    mean_okun: float
    std_okun: float

    # Deflation metrics
    deflation_pcts: list[float]
    mean_deflation_pct: float


def analyze_config_consistency(
    params: dict,
    n_seeds: int = 10,
    n_periods: int = 1000,
    burn_in: int = 500,
) -> dict:
    """Run a configuration with multiple seeds and compute statistics."""
    scores = []
    okun_corrs = []
    unemp_means = []
    deflation_pcts = []

    for seed in range(n_seeds):
        result = run_single_simulation(params, seed, n_periods, burn_in)
        scores.append(result["total"])
        okun_corrs.append(result.get("_okun_corr", 0))
        unemp_means.append(result.get("_unemployment_mean", 0))

        # Get deflation info from the result
        positive_pct = result.get("_inflation_positive_pct", 1.0)
        deflation_pct = (1 - positive_pct) * 100
        deflation_pcts.append(deflation_pct)

    return {
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "okun_corrs": okun_corrs,
        "mean_okun": np.mean(okun_corrs),
        "std_okun": np.std(okun_corrs),
        "unemp_means": unemp_means,
        "mean_unemp": np.mean(unemp_means),
        "deflation_pcts": deflation_pcts,
        "mean_deflation_pct": np.mean(deflation_pcts),
    }


def load_checkpoint(checkpoint_path: str) -> tuple[list[dict], bool]:
    """
    Load checkpoint and return (configs, is_rescored).

    Returns:
        configs: List of config dicts with 'params' and score info
        is_rescored: True if this is a re-scored checkpoint
    """
    with open(checkpoint_path) as f:
        data = json.load(f)

    # Detect checkpoint type
    if "results" in data:
        # Re-scored checkpoint format
        return data["results"], True
    elif "grid_results" in data:
        # Original checkpoint format - need to deduplicate
        unique = get_unique_realistic_configs(data["grid_results"])
        return unique, False
    else:
        raise ValueError(f"Unknown checkpoint format: {list(data.keys())}")


def extract_config_info(config: dict, is_rescored: bool) -> tuple[dict, float, float]:
    """
    Extract params, checkpoint_score, and deflation_pct from config.

    Returns:
        params: Parameter dictionary
        checkpoint_score: Total score from checkpoint
        deflation_pct: Deflation percentage (0 if not available)
    """
    params = config["params"]

    if is_rescored:
        rescored = config["rescored"]
        checkpoint_score = rescored["rescored_total"]
        deflation_pct = rescored.get("deflation_pct", 0)
    else:
        checkpoint_score = config["scores"]["total"]
        positive_pct = config["scores"].get("_inflation_positive_pct", 1.0)
        deflation_pct = (1 - positive_pct) * 100

    return params, checkpoint_score, deflation_pct


def main():
    parser = argparse.ArgumentParser(
        description="Analyze configuration consistency across seeds",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top configurations to analyze (default: 20)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds to test per configuration (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: calibration_checkpoint_rescored.json)",
    )
    parser.add_argument(
        "--visualize-top",
        type=int,
        default=0,
        help="Number of top consistent configs to visualize (default: 0)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Number of periods (default: 1000)",
    )

    args = parser.parse_args()

    # Default to re-scored checkpoint
    checkpoint_path = args.checkpoint or os.path.join(
        OUTPUT_DIR, "calibration_checkpoint_rescored.json"
    )

    # Determine output directory based on checkpoint name
    checkpoint_name = Path(checkpoint_path).stem
    output_dir = os.path.join(OUTPUT_DIR, f"consistency_{checkpoint_name}")
    os.makedirs(output_dir, exist_ok=True)

    burn_in = args.periods // 2

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    configs, is_rescored = load_checkpoint(checkpoint_path)

    checkpoint_type = "re-scored" if is_rescored else "original"
    print(f"Checkpoint type: {checkpoint_type}")
    print(f"Total configs: {len(configs):,}")
    print(f"Analyzing top {args.top} with {args.seeds} seeds each...")
    print(f"Total simulations: {args.top * args.seeds}")
    print()

    # Analyze each config
    consistency_results = []

    for i, config in enumerate(configs[: args.top], 1):
        params, checkpoint_score, deflation_pct_checkpoint = extract_config_info(
            config, is_rescored
        )

        print(
            f"[{i}/{args.top}] Analyzing config "
            f"(score: {checkpoint_score:.2f}, deflation: {deflation_pct_checkpoint:.1f}%)..."
        )

        stats = analyze_config_consistency(
            params,
            n_seeds=args.seeds,
            n_periods=args.periods,
            burn_in=burn_in,
        )

        # Consistency score: mean + penalty for variance
        # Lower is better
        consistency_score = stats["mean_score"] + stats["std_score"] * 2

        result = ConsistencyResult(
            rank=i,
            params=params,
            checkpoint_score=checkpoint_score,
            deflation_pct_checkpoint=deflation_pct_checkpoint,
            scores=stats["scores"],
            mean_score=stats["mean_score"],
            std_score=stats["std_score"],
            min_score=stats["min_score"],
            max_score=stats["max_score"],
            consistency_score=consistency_score,
            okun_corrs=stats["okun_corrs"],
            mean_okun=stats["mean_okun"],
            std_okun=stats["std_okun"],
            deflation_pcts=stats["deflation_pcts"],
            mean_deflation_pct=stats["mean_deflation_pct"],
        )
        consistency_results.append(result)

        print(f"    Scores: {[f'{s:.1f}' for s in stats['scores']]}")
        print(f"    Mean: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
        print(f"    Okun: {stats['mean_okun']:.3f} ± {stats['std_okun']:.3f}")
        print(
            f"    Deflation: {stats['mean_deflation_pct']:.1f}% "
            f"(checkpoint: {deflation_pct_checkpoint:.1f}%)"
        )
        print()

    # Sort by consistency score (lower is better)
    consistency_results.sort(key=lambda x: x.consistency_score)

    # Print summary table
    print("=" * 100)
    print("CONSISTENCY ANALYSIS RESULTS (sorted by consistency score)")
    print("=" * 100)
    print()
    print(
        f"{'Rank':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} "
        f"{'Consist.':<10} {'Okun':<10} {'Defl%':<8}"
    )
    print("-" * 100)

    for r in consistency_results:
        print(
            f"{r.rank:<6} {r.mean_score:<8.2f} {r.std_score:<8.2f} "
            f"{r.min_score:<8.2f} {r.max_score:<8.2f} {r.consistency_score:<10.2f} "
            f"{r.mean_okun:<+10.3f} {r.mean_deflation_pct:<8.1f}"
        )

    print()
    print("=" * 100)
    print("TOP 3 MOST CONSISTENT CONFIGURATIONS")
    print("=" * 100)

    for i, r in enumerate(consistency_results[:3], 1):
        print(f"\n#{i} (Original Rank {r.rank})")
        print(f"  Checkpoint Score: {r.checkpoint_score:.2f}")
        print(f"  Consistency Score: {r.consistency_score:.2f}")
        print(f"  Mean Score: {r.mean_score:.2f} ± {r.std_score:.2f}")
        print(f"  Score Range: [{r.min_score:.2f}, {r.max_score:.2f}]")
        print(f"  Mean Okun: {r.mean_okun:+.3f} ± {r.std_okun:.3f}")
        print(
            f"  Mean Deflation: {r.mean_deflation_pct:.1f}% "
            f"(checkpoint: {r.deflation_pct_checkpoint:.1f}%)"
        )
        print(f"  Individual scores: {[f'{s:.1f}' for s in r.scores]}")
        print("  Parameters:")
        for k, v in r.params.items():
            print(f"    {k}: {v}")

    # Save results to JSON
    results_file = os.path.join(output_dir, "consistency_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "analysis_params": {
                    "checkpoint": checkpoint_path,
                    "checkpoint_type": checkpoint_type,
                    "n_configs": args.top,
                    "n_seeds": args.seeds,
                    "n_periods": args.periods,
                },
                "results": [
                    {
                        "original_rank": r.rank,
                        "params": r.params,
                        "checkpoint_score": r.checkpoint_score,
                        "deflation_pct_checkpoint": r.deflation_pct_checkpoint,
                        "scores": r.scores,
                        "mean_score": r.mean_score,
                        "std_score": r.std_score,
                        "min_score": r.min_score,
                        "max_score": r.max_score,
                        "consistency_score": r.consistency_score,
                        "mean_okun": r.mean_okun,
                        "std_okun": r.std_okun,
                        "mean_deflation_pct": r.mean_deflation_pct,
                    }
                    for r in consistency_results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")

    # Visualize top consistent configs with multiple seeds
    if args.visualize_top > 0:
        print(f"\n{'=' * 100}")
        print(
            f"GENERATING VISUALIZATIONS FOR TOP {args.visualize_top} CONSISTENT CONFIGS"
        )
        print(f"{'=' * 100}")

        for i, r in enumerate(consistency_results[: args.visualize_top], 1):
            print(
                f"\nVisualizing #{i} (Original Rank {r.rank}) with {args.seeds} seeds..."
            )

            for seed in range(args.seeds):
                filename = (
                    f"consistent{i:02d}_origrank{r.rank:02d}_"
                    f"seed{seed}_score{r.scores[seed]:.1f}.png"
                )
                save_path = os.path.join(output_dir, filename)

                title = (
                    f"Consistent #{i} (Rank {r.rank}) - "
                    f"Seed {seed} (Score: {r.scores[seed]:.1f})"
                )

                try:
                    visualize_configuration(
                        params=r.params,
                        seed=seed,
                        n_periods=args.periods,
                        burn_in=burn_in,
                        title=title,
                        save_path=save_path,
                    )
                    print(f"  Saved: {filename}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    print(f"\n{'=' * 100}")
    print(f"Analysis complete! Output directory: {output_dir}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
