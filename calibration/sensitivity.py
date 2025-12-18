"""
One-at-a-Time Sensitivity Analysis
==================================

Identifies which parameters have the most impact on calibration scores.
Run separately before full calibration to inform parameter grid design.

Usage::

    python -m calibration --sensitivity

After running, review the results and update CALIBRATION_PARAM_GRID in
config.py to expand the search space for sensitive parameters.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import OAT_PARAM_GRID, get_default_value
from .runner import run_ensemble


@dataclass
class ParameterResult:
    """Results for a single parameter value test."""

    value: Any
    score: float
    impact: float  # |score - baseline_score|
    scores: dict = field(default_factory=dict)  # Full score breakdown


@dataclass
class SensitivityResult:
    """Aggregated sensitivity results for a parameter."""

    param_name: str
    default_value: Any
    baseline_score: float
    value_results: list[ParameterResult] = field(default_factory=list)
    score_spread: float = 0.0  # max_score - min_score
    max_impact: float = 0.0
    best_value: Any = None
    best_score: float = float("inf")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "param_name": self.param_name,
            "default_value": self.default_value,
            "baseline_score": self.baseline_score,
            "score_spread": self.score_spread,
            "max_impact": self.max_impact,
            "best_value": self.best_value,
            "best_score": self.best_score,
            "value_results": [
                {
                    "value": r.value,
                    "score": r.score,
                    "impact": r.impact,
                }
                for r in self.value_results
            ],
        }


def run_oat_sensitivity_analysis(
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    output_dir: Path | None = None,
) -> dict[str, SensitivityResult]:
    """
    Run one-at-a-time sensitivity analysis.

    Starting from defaults.yml values, changes ONE parameter at a time
    to its test values and measures impact on total score. This identifies
    which parameters have the most influence on model outcomes.

    After running, review the results and manually update CALIBRATION_PARAM_GRID
    in config.py to expand the search space for sensitive parameters.

    Parameters
    ----------
    n_seeds : int
        Number of seeds per configuration.
    n_periods : int
        Simulation length.
    burn_in : int
        Burn-in period to exclude.
    output_dir : Path, optional
        Directory to save results. Defaults to current directory.

    Returns
    -------
    dict[str, SensitivityResult]
        Results for each parameter showing impact scores.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Seeds per config: {n_seeds}")
    print(f"  Periods: {n_periods}")
    print(f"  Burn-in: {burn_in}")
    print(f"  Parameters to test: {len(OAT_PARAM_GRID)}")

    # Count total configurations
    total_configs = 1 + sum(len(values) for values in OAT_PARAM_GRID.values())
    print(f"  Total configurations: {total_configs}")
    print(f"  Estimated time: ~{total_configs} seconds\n")

    start_time = time.time()

    # Run baseline first (all defaults)
    print("Running baseline configuration...")
    baseline_scores, _ = run_ensemble({}, n_seeds, n_periods, burn_in)
    baseline_total = baseline_scores["total"]
    print(f"Baseline total score: {baseline_total:.2f}\n")

    results: dict[str, SensitivityResult] = {}
    completed = 1  # Baseline done

    for param_name, param_values in OAT_PARAM_GRID.items():
        default_value = get_default_value(param_name)

        result = SensitivityResult(
            param_name=param_name,
            default_value=default_value,
            baseline_score=baseline_total,
        )

        print(f"Testing {param_name} ({len(param_values)} values)...")

        for value in param_values:
            # Test single parameter change
            test_params = {param_name: value}
            scores, _ = run_ensemble(test_params, n_seeds, n_periods, burn_in)
            total = scores["total"]
            impact = abs(total - baseline_total)

            result.value_results.append(
                ParameterResult(
                    value=value,
                    score=total,
                    impact=impact,
                    scores=scores,
                )
            )

            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_configs - completed) / rate if rate > 0 else 0

            print(
                f"  {param_name}={value}: score={total:.2f}, "
                f"impact={impact:.2f} [{completed}/{total_configs}, "
                f"ETA: {eta:.0f}s]"
            )

        # Calculate aggregates
        scores_list = [r.score for r in result.value_results]
        result.score_spread = max(scores_list) - min(scores_list)
        result.max_impact = max(r.impact for r in result.value_results)
        best_result = min(result.value_results, key=lambda r: r.score)
        result.best_value = best_result.value
        result.best_score = best_result.score

        results[param_name] = result

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nBaseline score: {baseline_total:.2f}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Rate: {total_configs / elapsed:.1f} configs/second")

    # Rank by score spread (most sensitive first)
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].score_spread,
        reverse=True,
    )

    print("\n" + "-" * 70)
    print("PARAMETER SENSITIVITY RANKING (by score spread)")
    print("-" * 70)
    print(
        f"{'Rank':<5} {'Parameter':<30} {'Spread':<10} {'Max Impact':<12} {'Best Value':<15}"
    )
    print("-" * 70)

    for rank, (param_name, result) in enumerate(ranked, 1):
        print(
            f"{rank:<5} {param_name:<30} {result.score_spread:<10.2f} "
            f"{result.max_impact:<12.2f} {result.best_value!s:<15}"
        )

    # Save results to JSON
    output_path = output_dir / "sensitivity_results.json"
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "baseline_score": baseline_total,
            "n_seeds": n_seeds,
            "n_periods": n_periods,
            "burn_in": burn_in,
            "total_configs": total_configs,
            "elapsed_seconds": elapsed,
        },
        "results": {name: result.to_dict() for name, result in results.items()},
        "ranking": [name for name, _ in ranked],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print guidance
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review the sensitivity ranking above
2. For parameters with HIGH score spread (most sensitive):
   - Consider expanding their value range in CALIBRATION_PARAM_GRID
   - Add more intermediate values to explore the space
3. For parameters with LOW score spread (not sensitive):
   - Use their best value as a fixed parameter
   - Or keep minimal values in the grid
4. Edit calibration/config.py with your updated grid
5. Run full calibration: python -m calibration --calibrate
""")

    return results


def load_sensitivity_results(path: Path) -> dict:
    """
    Load sensitivity results from JSON file.

    Parameters
    ----------
    path : Path
        Path to sensitivity_results.json file.

    Returns
    -------
    dict
        Loaded results data.
    """
    with open(path) as f:
        return json.load(f)
