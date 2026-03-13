"""Structured parameter sweep by category, carrying forward winners.

Each stage runs a grid of its parameters while holding everything else
fixed from the base config (plus winners from prior stages). Optionally
cross-evaluates against other scenarios at each stage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibration.grid import count_combinations, generate_combinations
from calibration.io import OUTPUT_DIR, save_stability
from calibration.screening import run_screening
from calibration.stability import parse_stability_tiers, run_tiered_stability


def parse_stage(stage_str: str) -> tuple[str, dict[str, list[Any]]]:
    """Parse a single stage definition.

    Format: "LABEL:param1=v1,v2,v3 param2=v4,v5"

    Parameters
    ----------
    stage_str : str
        Stage definition string.

    Returns
    -------
    tuple[str, dict]
        (label, param_grid)
    """
    label, params_str = stage_str.split(":", 1)
    label = label.strip()

    grid: dict[str, list[Any]] = {}
    for part in params_str.strip().split():
        param, values_str = part.split("=", 1)
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
        grid[param.strip()] = values

    return label, grid


def parse_stages(
    stage_args: list[str],
) -> list[tuple[str, dict[str, list[Any]]]]:
    """Parse multiple stage definitions.

    Parameters
    ----------
    stage_args : list[str]
        List of stage definition strings.

    Returns
    -------
    list[tuple[str, dict]]
        List of (label, param_grid) tuples.
    """
    return [parse_stage(s) for s in stage_args]


@dataclass
class StageResult:
    """Result of a single sweep stage.

    Attributes
    ----------
    label : str
        Stage label.
    winner_params : dict[str, Any]
        Winner's parameters.
    combined_score : float
        Winner's combined score.
    mean_score : float
        Winner's mean score.
    pass_rate : float
        Winner's pass rate.
    n_candidates : int
        Number of grid combinations tested.
    improved : bool
        Whether the stage found an improvement over the base.
    """

    label: str
    winner_params: dict[str, Any]
    combined_score: float
    mean_score: float
    pass_rate: float
    n_candidates: int
    improved: bool


def run_sweep(
    base_params: dict[str, Any],
    stages: list[tuple[str, dict[str, list[Any]]]],
    scenario: str,
    n_workers: int = 10,
    n_periods: int = 1000,
    stability_tiers: list[tuple[int, int]] | None = None,
    rank_by: str = "combined",
    k_factor: float = 1.0,
    cross_scenario: str | None = None,
) -> list[StageResult]:
    """Run structured multi-stage parameter sweep.

    Parameters
    ----------
    base_params : dict
        Starting configuration.
    stages : list[tuple[str, dict]]
        List of (label, param_grid) stages to run in order.
    scenario : str
        Scenario name.
    n_workers : int
        Parallel workers.
    n_periods : int
        Simulation periods.
    stability_tiers : list[tuple[int, int]], optional
        Tiers for stability testing. Defaults to [(100, 10), (50, 20), (10, 100)].
    rank_by : str
        Ranking strategy for stability.
    k_factor : float
        k in combined score formula.
    cross_scenario : str, optional
        If set, cross-evaluate the stage winner against this scenario.

    Returns
    -------
    list[StageResult]
        Per-stage results with winner params and scores.
    """
    if stability_tiers is None:
        stability_tiers = [(100, 10), (50, 20), (10, 100)]

    current_base = dict(base_params)
    stage_results: list[StageResult] = []

    for stage_idx, (label, grid) in enumerate(stages):
        print(f"\n{'=' * 70}")
        print(f"SWEEP STAGE {stage_idx + 1}: {label}")
        print(f"{'=' * 70}")

        n_combos = count_combinations(grid)
        print(f"  Grid: {list(grid.keys())} ({n_combos} combinations)")

        # Fixed = current base minus the stage's grid params
        fixed = {k: v for k, v in current_base.items() if k not in grid}

        combinations = list(generate_combinations(grid, fixed=fixed))

        # Screen
        screening = run_screening(
            combinations, scenario, n_workers=n_workers, n_periods=n_periods
        )

        # Stability
        results = run_tiered_stability(
            screening,
            scenario,
            tiers=stability_tiers,
            n_workers=n_workers,
            n_periods=n_periods,
            rank_by=rank_by,
            k_factor=k_factor,
        )

        if results:
            winner = results[0]
            current_base = dict(winner.params)
            print(
                f"\n  Stage {label} winner: combined={winner.combined_score:.4f} "
                f"mean={winner.mean_score:.4f}"
            )

            # Cross-scenario check
            if cross_scenario:
                from calibration.cross_eval import evaluate_cross_scenario

                cross_results = evaluate_cross_scenario(
                    configs=[dict(winner.params)],
                    scenarios=[scenario, cross_scenario],
                    n_seeds=20,
                    n_periods=n_periods,
                    n_workers=n_workers,
                )
                if cross_results:
                    cr = cross_results[0]
                    for sname, sr in (cr.scenario_results or {}).items():
                        print(
                            f"    {sname}: pass={sr.pass_rate:.0%} "
                            f"combined={sr.combined_score:.4f}"
                        )

            stage_results.append(
                StageResult(
                    label=label,
                    winner_params=dict(winner.params),
                    combined_score=winner.combined_score or 0.0,
                    mean_score=winner.mean_score or 0.0,
                    pass_rate=winner.pass_rate or 0.0,
                    n_candidates=n_combos,
                    improved=True,
                )
            )
        else:
            stage_results.append(
                StageResult(
                    label=label,
                    winner_params=dict(current_base),
                    combined_score=0.0,
                    mean_score=0.0,
                    pass_rate=0.0,
                    n_candidates=n_combos,
                    improved=False,
                )
            )
            print(f"\n  Stage {label}: no improvement found, keeping base")

    return stage_results


def run_sweep_phase(args: argparse.Namespace, run_dir: Path | None = None) -> None:
    """CLI entry point for sweep phase."""
    if not args.stages:
        raise SystemExit("--stages is required for sweep phase")

    stages = parse_stages(args.stages)

    # Load base params
    if args.base:
        base_path = Path(args.base)
        if base_path.suffix in (".yml", ".yaml"):
            import yaml

            with open(base_path) as f:
                base_params = yaml.safe_load(f) or {}
        else:
            from calibration.io import load_stability

            results = load_stability(base_path)
            base_params = dict(results[0].params) if results else {}
    else:
        base_params = {}

    tiers = parse_stability_tiers(args.stability_tiers)

    print(f"[sweep] {len(stages)} stages, scenario={args.scenario}")
    for label, grid in stages:
        print(
            f"  Stage '{label}': {list(grid.keys())} "
            f"({count_combinations(grid)} combos)"
        )

    cross_scenario = getattr(args, "cross_scenario", None)
    stage_results = run_sweep(
        base_params=base_params,
        stages=stages,
        scenario=args.scenario,
        n_workers=args.workers,
        n_periods=args.periods,
        stability_tiers=tiers,
        rank_by=args.rank_by,
        k_factor=args.k_factor,
        cross_scenario=cross_scenario,
    )

    # Print cumulative summary
    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 70}")
    for sr in stage_results:
        status = "improved" if sr.improved else "no change"
        print(
            f"  {sr.label}: combined={sr.combined_score:.4f} "
            f"pass={sr.pass_rate:.0%} ({status}, {sr.n_candidates} combos)"
        )

    # Save per-stage results and final winner
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if stage_results:
        from calibration.analysis import CalibrationResult

        # Save each stage result
        for sr in stage_results:
            stage_cr = CalibrationResult(
                params=sr.winner_params,
                single_score=0.0,
                n_pass=0,
                n_warn=0,
                n_fail=0,
                mean_score=sr.mean_score,
                combined_score=sr.combined_score,
                pass_rate=sr.pass_rate,
            )
            save_stability(
                [stage_cr],
                args.scenario,
                out / f"{args.scenario}_sweep_stage_{sr.label}.json",
            )

        # Save final winner (last stage)
        final = CalibrationResult(
            params=stage_results[-1].winner_params,
            single_score=0.0,
            n_pass=0,
            n_warn=0,
            n_fail=0,
            mean_score=stage_results[-1].mean_score,
            combined_score=stage_results[-1].combined_score,
            pass_rate=stage_results[-1].pass_rate,
        )
        save_stability([final], args.scenario, out / f"{args.scenario}_sweep.json")
    print(f"\nSweep results saved to {out}")
