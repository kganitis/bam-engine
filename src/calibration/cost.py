"""Targeted cost analysis -- measure the cost of swapping values into a base config.

Evaluates the impact of substituting preferred parameter values into an
optimized base configuration. Classifies each swap by cost:
FREE (<0.002), CHEAP (<0.005), MODERATE (<0.010), EXPENSIVE (>=0.010).
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibration.grid import generate_combinations
from calibration.io import OUTPUT_DIR, load_stability, save_stability
from calibration.screening import run_screening
from calibration.stability import (
    _evaluate_single_seed,
    parse_stability_tiers,
    run_tiered_stability,
)


@dataclass
class SwapResult:
    """Result of swapping a single parameter value into the base config.

    Attributes
    ----------
    param : str
        Parameter name.
    value : Any
        Swapped value.
    base_combined : float
        Base config's combined score.
    swap_combined : float
        Combined score with this value swapped in.
    delta : float
        Score change (swap - base). Negative = worse.
    classification : str
        Cost classification: FREE, CHEAP, MODERATE, or EXPENSIVE.
    pass_rate : float
        Pass rate with swapped value.
    """

    param: str
    value: Any
    base_combined: float
    swap_combined: float
    delta: float
    classification: str
    pass_rate: float


def classify_cost(delta_abs: float) -> str:
    """Classify the absolute cost of a swap.

    Parameters
    ----------
    delta_abs : float
        Absolute combined score difference.

    Returns
    -------
    str
        "FREE", "CHEAP", "MODERATE", or "EXPENSIVE".
    """
    if delta_abs < 0.002:
        return "FREE"
    if delta_abs < 0.005:
        return "CHEAP"
    if delta_abs < 0.010:
        return "MODERATE"
    return "EXPENSIVE"


def parse_swaps(swap_args: list[str]) -> dict[str, list[Any]]:
    """Parse swap arguments from CLI.

    Parameters
    ----------
    swap_args : list[str]
        List of "param=v1,v2,v3" strings.

    Returns
    -------
    dict[str, list]
        Parameter -> list of values to try.
    """
    swaps: dict[str, list[Any]] = {}
    for arg in swap_args:
        param, values_str = arg.split("=", 1)
        values: list[Any] = []
        for v in values_str.split(","):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
        swaps[param.strip()] = values
    return swaps


def run_cost_analysis(
    base_params: dict[str, Any],
    swaps: dict[str, list[Any]],
    scenario: str,
    n_seeds: int = 20,
    n_periods: int = 1000,
    n_workers: int = 10,
    base_combined: float | None = None,
) -> list[SwapResult]:
    """Run targeted cost analysis for parameter swaps.

    Parameters
    ----------
    base_params : dict
        Base configuration (the stability winner).
    swaps : dict[str, list]
        Parameters to swap and their candidate values.
    scenario : str
        Scenario name.
    n_seeds : int
        Seeds per evaluation.
    n_periods : int
        Simulation periods.
    n_workers : int
        Parallel workers.
    base_combined : float, optional
        Pre-computed base combined score. If None, evaluates the base.

    Returns
    -------
    list[SwapResult]
        Results for each swap, sorted by absolute delta.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _eval_seeds(
        params: dict[str, Any],
        scenario: str,
        n_seeds: int,
        n_periods: int,
        n_workers: int,
    ) -> tuple[list[float], list[int]]:
        """Evaluate a config across seeds, returning (scores, fails)."""
        scores: list[float] = []
        fails: list[int] = []
        seeds = list(range(n_seeds))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        _evaluate_single_seed, params, scenario, s, n_periods
                    )
                    for s in seeds
                ]
                for future in as_completed(futures):
                    _, _, score, n_fail = future.result()
                    scores.append(score)
                    fails.append(n_fail)
        else:
            for seed in seeds:
                _, _, score, n_fail = _evaluate_single_seed(
                    params, scenario, seed, n_periods
                )
                scores.append(score)
                fails.append(n_fail)
        return scores, fails

    # Evaluate base if needed
    if base_combined is None:
        print(f"  Evaluating base config ({n_seeds} seeds)...")
        scores, fails = _eval_seeds(
            base_params, scenario, n_seeds, n_periods, n_workers
        )
        base_mean = statistics.mean(scores)
        base_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        n_passed = sum(1 for nf in fails if nf == 0)
        base_pr = n_passed / len(fails)
        base_combined = base_mean * base_pr * (1.0 - base_std)
        print(f"  Base combined: {base_combined:.4f}")

    # Evaluate each swap
    results: list[SwapResult] = []
    total_swaps = sum(len(vals) for vals in swaps.values())
    done = 0

    for param, values in swaps.items():
        for value in values:
            done += 1
            swap_params = {**base_params, param: value}
            scores, fails = _eval_seeds(
                swap_params, scenario, n_seeds, n_periods, n_workers
            )

            swap_mean = statistics.mean(scores)
            swap_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            n_passed = sum(1 for nf in fails if nf == 0)
            swap_pr = n_passed / len(fails)
            swap_combined = swap_mean * swap_pr * (1.0 - swap_std)

            delta = swap_combined - base_combined
            classification = classify_cost(abs(delta))

            results.append(
                SwapResult(
                    param=param,
                    value=value,
                    base_combined=base_combined,
                    swap_combined=swap_combined,
                    delta=delta,
                    classification=classification,
                    pass_rate=swap_pr,
                )
            )
            print(
                f"  [{done}/{total_swaps}] {param}={value}: "
                f"delta={delta:+.4f} ({classification})"
            )

    results.sort(key=lambda r: abs(r.delta))
    return results


def save_cost_results(
    results: list[SwapResult],
    scenario: str,
    path: Path,
) -> None:
    """Save cost analysis results to JSON."""
    data = {
        "scenario": scenario,
        "results": [
            {
                "param": r.param,
                "value": r.value,
                "base_combined": r.base_combined,
                "swap_combined": r.swap_combined,
                "delta": r.delta,
                "classification": r.classification,
                "pass_rate": r.pass_rate,
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_base_config(
    base_path: Path,
) -> tuple[dict[str, Any], float | None]:
    """Load base config from stability JSON or YAML config file.

    Returns
    -------
    tuple[dict, float | None]
        (base_params, base_combined_score_or_None)
    """
    if base_path.suffix in (".yml", ".yaml"):
        import yaml

        with open(base_path) as f:
            params = yaml.safe_load(f) or {}
        return params, None  # no pre-computed combined score
    else:
        results = load_stability(base_path)
        if not results:
            raise ValueError(f"No results in {base_path}")
        return dict(results[0].params), results[0].combined_score


def run_cost_phase(args: argparse.Namespace, run_dir: Path | None = None) -> None:
    """CLI entry point for cost phase."""
    if not args.base:
        raise SystemExit("--base is required for cost phase")
    if not args.swaps:
        raise SystemExit("--swaps is required for cost phase")

    base_path = Path(args.base)
    base_params, base_combined = _load_base_config(base_path)

    swaps = parse_swaps(args.swaps)

    print(f"[cost] Base config from {base_path}")
    print(f"[cost] Testing {sum(len(v) for v in swaps.values())} swaps")

    results = run_cost_analysis(
        base_params=base_params,
        swaps=swaps,
        scenario=args.scenario,
        n_seeds=args.sensitivity_seeds,
        n_periods=args.periods,
        n_workers=args.workers,
        base_combined=base_combined,
    )

    # Print summary table
    print(f"\n  {'Param':<25} {'Value':>8} {'Delta':>8} {'Class':>10} {'Pass%':>6}")
    print("  " + "-" * 59)
    for r in results:
        print(
            f"  {r.param:<25} {r.value!s:>8} {r.delta:>+8.4f} "
            f"{r.classification:>10} {r.pass_rate:>5.0%}"
        )

    # Combo grid: run all FREE+CHEAP swaps combined
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if args.combo_grid:
        cheap_swaps: dict[str, list[Any]] = {}
        for r in results:
            if r.classification in ("FREE", "CHEAP"):
                cheap_swaps.setdefault(r.param, []).append(r.value)

        if cheap_swaps:
            print(f"\n[cost] Combo grid: {cheap_swaps}")
            # Exclude swap keys from fixed to avoid overlap ValueError
            fixed_for_combo = {
                k: v for k, v in base_params.items() if k not in cheap_swaps
            }
            combos = list(generate_combinations(cheap_swaps, fixed=fixed_for_combo))
            print(f"[cost] {len(combos)} combinations")

            screening = run_screening(
                combos, args.scenario, n_workers=args.workers, n_periods=args.periods
            )
            tiers = parse_stability_tiers(args.stability_tiers)
            combo_results = run_tiered_stability(
                screening,
                args.scenario,
                tiers=tiers,
                n_workers=args.workers,
                n_periods=args.periods,
            )
            if combo_results:
                winner = combo_results[0]
                print(f"[cost] Combo winner: combined={winner.combined_score:.4f}")
                save_stability(
                    combo_results,
                    args.scenario,
                    out / f"{args.scenario}_cost_combo.json",
                )
        else:
            print("\n[cost] No FREE/CHEAP swaps found -- skipping combo grid")

    # Save
    save_cost_results(results, args.scenario, out / f"{args.scenario}_cost.json")
    print(f"\nCost results saved to {out}")
