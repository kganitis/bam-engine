"""Cross-scenario evaluation -- run configs across multiple scenarios.

Evaluates parameter configurations on all specified scenarios simultaneously
and ranks using cross-scenario criteria.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from calibration.analysis import CalibrationResult, ScenarioResult
from calibration.io import OUTPUT_DIR, save_stability
from calibration.stability import _evaluate_single_seed


def rank_cross_scenario(
    results: list[CalibrationResult],
    strategy: str = "stability-first",
) -> list[CalibrationResult]:
    """Rank configs using cross-scenario criteria.

    Parameters
    ----------
    results : list[CalibrationResult]
        Results with ``scenario_results`` populated.
    strategy : str
        Ranking strategy:
        - "stability-first": min(pass_rates) -> total fails -> min(combined)
        - "score-first": min(combined) -> total fails
        - "balanced": geometric mean of combined scores

    Returns
    -------
    list[CalibrationResult]
        Sorted results (best first).

    Raises
    ------
    ValueError
        If strategy is not recognized.
    """
    if strategy == "stability-first":
        results.sort(
            key=lambda r: (
                min(sr.pass_rate for sr in (r.scenario_results or {}).values()),
                -sum(sr.n_fail for sr in (r.scenario_results or {}).values()),
                min(sr.combined_score for sr in (r.scenario_results or {}).values()),
            ),
            reverse=True,
        )
    elif strategy == "score-first":
        results.sort(
            key=lambda r: (
                min(sr.combined_score for sr in (r.scenario_results or {}).values()),
                -sum(sr.n_fail for sr in (r.scenario_results or {}).values()),
            ),
            reverse=True,
        )
    elif strategy == "balanced":

        def geomean(r: CalibrationResult) -> float:
            scores = [sr.combined_score for sr in (r.scenario_results or {}).values()]
            if not scores:
                return 0.0
            return math.exp(sum(math.log(max(s, 1e-10)) for s in scores) / len(scores))

        results.sort(key=geomean, reverse=True)
    else:
        raise ValueError(
            f"Unknown ranking strategy: '{strategy}'. "
            f"Available: stability-first, score-first, balanced"
        )
    return results


def evaluate_cross_scenario(
    configs: list[dict[str, Any]],
    scenarios: list[str],
    n_seeds: int = 100,
    n_periods: int = 1000,
    n_workers: int = 10,
) -> list[CalibrationResult]:
    """Evaluate configs across multiple scenarios.

    Parameters
    ----------
    configs : list[dict]
        Parameter configurations to evaluate.
    scenarios : list[str]
        Scenario names to evaluate on.
    n_seeds : int
        Seeds per scenario per config.
    n_periods : int
        Simulation periods.
    n_workers : int
        Parallel workers.

    Returns
    -------
    list[CalibrationResult]
        Results with scenario_results populated.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: list[CalibrationResult] = []

    for cfg_idx, config in enumerate(configs):
        scenario_results: dict[str, ScenarioResult] = {}

        for scenario in scenarios:
            seeds = list(range(n_seeds))
            scores: list[float] = []
            fails: list[int] = []

            print(
                f"  Config {cfg_idx + 1}/{len(configs)}, "
                f"scenario={scenario}, {n_seeds} seeds..."
            )

            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(
                            _evaluate_single_seed, config, scenario, seed, n_periods
                        )
                        for seed in seeds
                    ]
                    for future in as_completed(futures):
                        _, _, score, n_fail = future.result()
                        scores.append(score)
                        fails.append(n_fail)
            else:
                for seed in seeds:
                    _, _, score, n_fail = _evaluate_single_seed(
                        config, scenario, seed, n_periods
                    )
                    scores.append(score)
                    fails.append(n_fail)

            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            n_passed = sum(1 for nf in fails if nf == 0)
            pass_rate = n_passed / len(fails)
            combined = mean * pass_rate * (1.0 - std)
            total_fails = sum(1 for nf in fails if nf > 0)

            scenario_results[scenario] = ScenarioResult(
                mean_score=mean,
                std_score=std,
                combined_score=combined,
                pass_rate=pass_rate,
                n_fail=total_fails,
                seed_scores=scores,
            )

        results.append(
            CalibrationResult.from_cross_eval(
                params=config,
                scenario_results=scenario_results,
            )
        )

    return results


def _load_configs(path: Path) -> list[dict[str, Any]]:
    """Load configs from screening/stability JSON or YAML grid file.

    Auto-detects format:
    - JSON with ``results`` key: screening or stability result file
    - YAML: treated as a parameter grid (generates all combinations)
    """
    if path.suffix in (".yml", ".yaml"):
        import yaml

        with open(path) as f:
            grid = yaml.safe_load(f) or {}
        from calibration.grid import generate_combinations

        return list(generate_combinations(grid))

    import json

    with open(path) as f:
        data = json.load(f)

    return [r["params"] for r in data["results"]]


def compute_scenario_tension(
    results: list[CalibrationResult],
    scenarios: list[str],
) -> dict[str, dict[str, Any]]:
    """Analyze parameter tensions between scenarios.

    Identifies params where the optimal value differs between scenarios,
    indicating a fundamental trade-off.

    Parameters
    ----------
    results : list[CalibrationResult]
        Results with ``scenario_results`` populated.
    scenarios : list[str]
        Scenario names to compare.

    Returns
    -------
    dict[str, dict]
        Per-parameter tension info: which value each scenario prefers,
        and the score gap.
    """
    if len(results) < 2 or len(scenarios) < 2:
        return {}

    # For each scenario, find the best config (by combined score for that scenario)
    best_per_scenario: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        best = max(
            results,
            key=lambda r: (
                (r.scenario_results or {})
                .get(scenario, ScenarioResult(0, 0, 0, 0, 0, []))
                .combined_score
            ),
        )
        best_per_scenario[scenario] = dict(best.params)

    # Identify params that differ
    tension: dict[str, dict[str, Any]] = {}
    all_params: set[str] = set()
    for params in best_per_scenario.values():
        all_params.update(params.keys())

    for param in sorted(all_params):
        values = {s: best_per_scenario[s].get(param) for s in scenarios}
        unique_values = set(values.values())
        if len(unique_values) > 1:
            tension[param] = {"preferred_by": values}

    return tension


def run_cross_eval_phase(args: argparse.Namespace, run_dir: Path | None = None) -> None:
    """CLI entry point for cross-eval phase."""
    if not args.scenarios:
        raise SystemExit("--scenarios is required for cross-eval phase")
    if not args.configs:
        raise SystemExit("--configs is required for cross-eval phase")

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    configs = _load_configs(Path(args.configs))

    print(f"[cross-eval] {len(configs)} configs x {len(scenarios)} scenarios")
    print(f"[cross-eval] Scenarios: {scenarios}")

    results = evaluate_cross_scenario(
        configs=configs,
        scenarios=scenarios,
        n_seeds=args.sensitivity_seeds,
        n_periods=args.periods,
        n_workers=args.workers,
    )

    # Default to stability-first for cross-eval; argparse validates choices
    rank_by = (
        args.rank_by
        if args.rank_by in ("stability-first", "score-first", "balanced")
        else "stability-first"
    )
    results = rank_cross_scenario(results, rank_by)

    # Print summary
    print(f"\n  Cross-Scenario Ranking ({rank_by}):")
    for i, r in enumerate(results[:10]):
        srs = r.scenario_results or {}
        min_pr = min((sr.pass_rate for sr in srs.values()), default=0)
        total_f = sum(sr.n_fail for sr in srs.values())
        min_c = min((sr.combined_score for sr in srs.values()), default=0)
        print(
            f"  #{i + 1}: min_pass={min_pr:.0%} total_fails={total_f} "
            f"min_combined={min_c:.4f}"
        )

    # Scenario tension analysis
    tension = compute_scenario_tension(results, scenarios)
    if tension:
        print("\n  Scenario Tension Analysis:")
        for param, info in tension.items():
            prefs = ", ".join(f"{s}={v}" for s, v in info["preferred_by"].items())
            print(f"    {param}: {prefs}")

    # Save
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    save_stability(results, "cross_eval", out / "cross_eval_results.json")
    print(f"\nCross-eval results saved to {out}")
