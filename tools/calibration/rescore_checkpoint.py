"""
Re-score Calibration Checkpoint
===============================

Re-scores the calibration checkpoint with:
- Okun correlation weight reduced to 25%
- Deflation bonus for configs with 5-10% deflation episodes

Usage:
    python tools/calibration/rescore_checkpoint.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


def rescore(
    result: dict, okun_weight: float = 25, deflation_bonus_weight: float = 50
) -> dict:
    """
    Recalculate scores with:
    - Reduced Okun weight (25%)
    - Bonus for deflation episodes (reward 5-10% deflation)

    Returns dict with new scores and the rescored total.
    """
    s = result["scores"]

    # Base score components (without Okun and inflation_positive)
    base = (
        s.get("collapse_penalty", 0)
        + s.get("log_gdp_level", 0)
        + s.get("log_gdp_stability", 0)
        + s.get("unemployment_range", 0)
        + s.get("unemployment_mean", 0)
        + s.get("unemployment_stability", 0)
        + s.get("inflation_mean", 0)
        +
        # Skip inflation_positive - replaced with deflation scoring
        s.get("prod_wage_stability", 0)
        + s.get("prod_wage_drift", 0)
        + s.get("phillips_shape", 0)
        + s.get("phillips_positive", 0)
        + s.get("beveridge_shape", 0)
        + s.get("beveridge_vacancy_range", 0)
        + s.get("firm_size_dist", 0)
        + s.get("production_level", 0)
    )

    # Okun penalty (reduced weight)
    okun_corr = s.get("_okun_corr", 0)
    if okun_corr > -0.5:
        okun_penalty = (okun_weight / 100) * ((-0.5 - okun_corr) ** 2) * 1000
    else:
        okun_penalty = 0

    # Deflation scoring
    # Target: 5-10% deflation periods (positive_pct around 0.90-0.95)
    positive_pct = s.get("_inflation_positive_pct", 1.0)
    deflation_pct = 1 - positive_pct

    if 0.05 <= deflation_pct <= 0.10:
        # Sweet spot: 5-10% deflation - no penalty
        deflation_penalty = 0
    elif deflation_pct < 0.05:
        # Not enough deflation - penalize
        deflation_penalty = (0.05 - deflation_pct) * deflation_bonus_weight * 10
    elif deflation_pct > 0.15:
        # Too much deflation (>15%) - penalize
        deflation_penalty = (deflation_pct - 0.15) * deflation_bonus_weight * 10
    else:
        # Between 10-15% - small penalty
        deflation_penalty = (deflation_pct - 0.10) * deflation_bonus_weight * 5

    new_total = base + okun_penalty + deflation_penalty

    return {
        "base_score": base,
        "okun_penalty": okun_penalty,
        "deflation_penalty": deflation_penalty,
        "rescored_total": new_total,
        "deflation_pct": deflation_pct * 100,
    }


def main():
    # Load original checkpoint
    checkpoint_path = OUTPUT_DIR / "calibration_checkpoint.json"
    print(f"Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path) as f:
        data = json.load(f)

    print(f"Total results in checkpoint: {len(data['grid_results'])}")

    # Re-score all results
    rescored_results = []

    for r in data["grid_results"]:
        s = r["scores"]
        unemp = s.get("_unemployment_mean", 1.0)
        original_total = s.get("total", 9999)

        # Only include realistic configs (3-15% unemployment, original score < 200)
        if 0.03 < unemp < 0.15 and original_total < 200:
            rescore_info = rescore(r)

            rescored_results.append(
                {
                    "params": r["params"],
                    "original_scores": r["scores"],
                    "rescored": rescore_info,
                }
            )

    # Sort by rescored total (lower is better)
    rescored_results.sort(key=lambda x: x["rescored"]["rescored_total"])

    print(f"Realistic configs after filtering: {len(rescored_results)}")

    # Save to new checkpoint
    output_path = OUTPUT_DIR / "calibration_checkpoint_rescored.json"

    output_data = {
        "description": "Re-scored checkpoint with Okun=25% and deflation bonus for 5-10%",
        "scoring_params": {
            "okun_weight": 25,
            "deflation_bonus_weight": 50,
            "deflation_target_min": 5,
            "deflation_target_max": 10,
        },
        "total_configs": len(rescored_results),
        "results": rescored_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved re-scored checkpoint to: {output_path}")

    # Print top 10 summary
    print("\n" + "=" * 80)
    print("TOP 10 RE-SCORED CONFIGS")
    print("=" * 80)

    for i, r in enumerate(rescored_results[:10], 1):
        rs = r["rescored"]
        os = r["original_scores"]
        p = r["params"]

        print(
            f"\n#{i} Rescored: {rs['rescored_total']:.2f} (Original: {os['total']:.2f})"
        )
        print(
            f"   Deflation: {rs['deflation_pct']:.1f}% | Okun: {os.get('_okun_corr', 0):.3f} | Unemp: {os.get('_unemployment_mean', 0) * 100:.1f}%"
        )
        print(
            f"   min_wage_ratio: {p.get('min_wage_ratio')} | unemployment_calc: {p.get('unemployment_calc_method')}"
        )
        other = {
            k: p.get(k)
            for k in [
                "v",
                "production_init",
                "price_init_offset",
                "zero_production_bankrupt",
            ]
            if k in p
        }
        print(f"   Other: {other}")


if __name__ == "__main__":
    main()
