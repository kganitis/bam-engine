"""
Visualize Top Calibration Configurations
=========================================

This script loads the calibration checkpoint, deduplicates configurations,
and generates visualizations for the top N unique, realistic configurations.

Usage:
    python tools/calibration/visualize_top_configs.py [--top N] [--output-dir DIR]

Examples:
    python tools/calibration/visualize_top_configs.py --top 20
    python tools/calibration/visualize_top_configs.py --top 10 --output-dir my_visualizations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add this directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from analyze_consistency import SWEEP_PARAM_DEFAULTS, get_unique_realistic_configs
from calibration import OUTPUT_DIR, visualize_configuration


def generate_config_filename(rank: int, config: dict) -> str:
    """Generate a descriptive filename for a configuration."""
    p = config["params"]
    score = config["scores"]["total"]

    # Key params for filename
    prod = p.get("production_init", "?")
    price_off = p.get("price_init_offset", "?")
    min_wage = p.get("min_wage_ratio", "?")

    # Non-default sweep params
    non_default = []
    for k in SWEEP_PARAM_DEFAULTS:
        if k in p and p[k] != SWEEP_PARAM_DEFAULTS[k]:
            # Shorten param names for filename
            short_names = {
                "v": "v",
                "savings_init": "savs",
                "new_firm_scale_factor": "nfsf",
                "price_cut_allow_increase": "pcai",
                "loanbook_clear_on_repay": "lbcor",
                "firing_method": "fire",
                "fragility_cap_method": "fcm",
            }
            short = short_names.get(k, k[:4])
            val = p[k]
            if isinstance(val, bool):
                val = "T" if val else "F"
            elif isinstance(val, float):
                val = f"{val:.2f}".rstrip("0").rstrip(".")
            non_default.append(f"{short}={val}")

    sweep_str = "_".join(non_default) if non_default else "defaults"

    return f"rank{rank:02d}_score{score:.2f}_prod{prod}_poff{price_off}_mw{min_wage}_{sweep_str}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top calibration configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top configurations to visualize (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: output/top_configs)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: output/calibration_checkpoint.json)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seeds to use for visualization (default: 0). Use multiple seeds to see variability.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Number of periods to simulate (default: 1000)",
    )

    args = parser.parse_args()

    # Set paths
    checkpoint_path = args.checkpoint or os.path.join(
        OUTPUT_DIR, "calibration_checkpoint.json"
    )
    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, "top_configs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path) as f:
        data = json.load(f)

    results = data["grid_results"]
    print(f"Total results in checkpoint: {len(results):,}")

    # Get unique, realistic configs
    unique_configs = get_unique_realistic_configs(results)
    print(f"Unique realistic configs: {len(unique_configs):,}")

    # Select top N
    top_n = min(args.top, len(unique_configs))
    selected = unique_configs[:top_n]
    print(f"Visualizing top {top_n} configurations...")
    print(f"Output directory: {output_dir}")
    print()

    # Generate visualizations
    burn_in = args.periods // 2

    for i, config in enumerate(selected, 1):
        params = config["params"]
        score = config["scores"]["total"]
        okun = config["scores"].get("_okun_corr", 0)
        unemp = config["scores"].get("_unemployment_mean", 0) * 100

        print(
            f"[{i}/{top_n}] Rank {i}: Score={score:.2f}, Okun={okun:+.3f}, Unemp={unemp:.1f}%"
        )

        for seed in args.seeds:
            filename = generate_config_filename(i, config)
            if len(args.seeds) > 1:
                # Add seed to filename if multiple seeds
                filename = filename.replace(".png", f"_seed{seed}.png")

            save_path = os.path.join(output_dir, filename)

            title = f"Rank {i} (Score: {score:.2f}, Okun: {okun:+.3f})"

            try:
                visualize_configuration(
                    params=params,
                    seed=seed,
                    n_periods=args.periods,
                    burn_in=burn_in,
                    title=title,
                    save_path=save_path,
                )
                print(f"  Saved: {filename}")
            except Exception as e:
                print(f"  ERROR: {e}")

        print()

    print("=" * 60)
    print(f"Done! Generated {top_n} visualizations in: {output_dir}")
    print("=" * 60)

    # Print summary table
    print("\n=== Configuration Summary ===\n")
    print(f"{'Rank':<5} {'Score':<7} {'Okun':<8} {'Unemp':<7} {'Key Params'}")
    print("-" * 70)
    for i, config in enumerate(selected, 1):
        score = config["scores"]["total"]
        okun = config["scores"].get("_okun_corr", 0)
        unemp = config["scores"].get("_unemployment_mean", 0) * 100
        p = config["params"]
        key = f"prod={p.get('production_init')}, poff={p.get('price_init_offset')}, mw={p.get('min_wage_ratio')}"
        print(f"{i:<5} {score:<7.2f} {okun:<+8.3f} {unemp:<7.1f} {key}")


if __name__ == "__main__":
    main()
