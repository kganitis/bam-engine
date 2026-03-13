"""Second-pass Morris screening after locking optimized params.

Delegates to ``run_morris_screening(fixed_params=...)`` and computes
the sensitivity collapse between Phase 1 and Phase 2 Morris results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from calibration.io import OUTPUT_DIR, load_morris, load_stability, save_morris
from calibration.morris import MorrisResult, print_morris_report, run_morris_screening
from calibration.parameter_space import PARAM_GROUPS, get_parameter_grid


def resolve_params(params_str: str) -> list[str]:
    """Resolve a param group name or comma-separated param names.

    Parameters
    ----------
    params_str : str
        Either a PARAM_GROUPS key (e.g., "entry", "behavioral") or
        comma-separated full parameter names (e.g., "beta,max_M").

    Returns
    -------
    list[str]
        List of parameter names.

    Raises
    ------
    ValueError
        If the string is not a known group and doesn't look like param names.
    """
    if params_str in PARAM_GROUPS:
        return list(PARAM_GROUPS[params_str])
    if "," in params_str:
        return [p.strip() for p in params_str.split(",")]
    # Single param name or unknown group
    if params_str in {p for group in PARAM_GROUPS.values() for p in group}:
        return [params_str]
    raise ValueError(
        f"Unknown parameter group: '{params_str}'. "
        f"Available groups: {list(PARAM_GROUPS.keys())}"
    )


def load_fixed_from_result(path: Path) -> dict[str, Any]:
    """Load the #1-ranked result's params from a stability result file.

    Parameters
    ----------
    path : Path
        Path to stability result JSON.

    Returns
    -------
    dict[str, Any]
        Parameter dict from the top-ranked result.
    """
    results = load_stability(path)
    if not results:
        raise ValueError(f"No results in {path}")
    return dict(results[0].params)


def compute_sensitivity_collapse(
    phase1: MorrisResult,
    phase2: MorrisResult,
) -> dict[str, dict[str, float]]:
    """Compute sensitivity collapse between two Morris screenings.

    Parameters
    ----------
    phase1 : MorrisResult
        First-pass Morris result (before locking params).
    phase2 : MorrisResult
        Second-pass Morris result (after locking params).

    Returns
    -------
    dict[str, dict]
        Per-parameter dict with phase1_mu_star, phase2_mu_star, collapse_pct.
    """
    p1_map = {e.name: e.mu_star for e in phase1.effects}
    p2_map = {e.name: e.mu_star for e in phase2.effects}

    collapse: dict[str, dict[str, float]] = {}
    for name in p2_map:
        p1_val = p1_map.get(name, 0.0)
        p2_val = p2_map[name]
        pct = 100.0 * (1.0 - p2_val / p1_val) if p1_val > 0 else 0.0
        collapse[name] = {
            "phase1_mu_star": p1_val,
            "phase2_mu_star": p2_val,
            "collapse_pct": pct,
        }
    return collapse


def run_rescreen(
    scenario: str,
    fix_from: Path,
    params: list[str],
    n_trajectories: int = 20,
    n_seeds: int = 5,
    n_periods: int = 1000,
    n_workers: int = 10,
    phase1_morris: MorrisResult | None = None,
) -> tuple[MorrisResult, dict[str, dict[str, float]]]:
    """Run second-pass Morris screening on a subset of params.

    Parameters
    ----------
    scenario : str
        Scenario name.
    fix_from : Path
        Path to stability result JSON to load fixed params from.
    params : list[str]
        Parameter names to screen (the rest are fixed).
    n_trajectories : int
        Number of Morris trajectories.
    n_seeds : int
        Seeds per evaluation.
    n_periods : int
        Simulation periods.
    n_workers : int
        Parallel workers.
    phase1_morris : MorrisResult, optional
        Phase 1 Morris result for collapse comparison.

    Returns
    -------
    tuple[MorrisResult, dict]
        (phase2_result, collapse_table)
    """
    fixed = load_fixed_from_result(fix_from)

    # Build grid: only include the specified params with full grid values
    full_grid = get_parameter_grid(scenario)
    unknown = [p for p in params if p not in full_grid]
    if unknown:
        import warnings

        warnings.warn(
            f"Params not in {scenario} grid (ignored): {unknown}",
            stacklevel=2,
        )
    rescreen_grid = {p: full_grid[p] for p in params if p in full_grid}

    # Remaining params from the winner's config become fixed
    fixed_params = {k: v for k, v in fixed.items() if k not in rescreen_grid}

    print(
        f"[rescreen] Screening {len(rescreen_grid)} params: {list(rescreen_grid.keys())}"
    )
    print(f"[rescreen] Fixed from winner: {len(fixed_params)} params")

    result = run_morris_screening(
        scenario=scenario,
        grid=rescreen_grid,
        n_trajectories=n_trajectories,
        n_seeds=n_seeds,
        n_periods=n_periods,
        n_workers=n_workers,
        fixed_params=fixed_params,
    )

    collapse: dict[str, dict[str, float]] = {}
    if phase1_morris is not None:
        collapse = compute_sensitivity_collapse(phase1_morris, result)
        print("\n  Sensitivity Collapse:")
        print(
            f"  {'Parameter':<30} {'Phase1 mu*':>10} {'Phase2 mu*':>10} {'Collapse':>10}"
        )
        print("  " + "-" * 62)
        for name, data in sorted(collapse.items(), key=lambda x: -x[1]["collapse_pct"]):
            print(
                f"  {name:<30} {data['phase1_mu_star']:>10.4f} "
                f"{data['phase2_mu_star']:>10.4f} {data['collapse_pct']:>9.1f}%"
            )

    return result, collapse


def run_rescreen_phase(args: argparse.Namespace, run_dir: Path | None = None) -> None:
    """CLI entry point for rescreen phase."""
    if not args.fix_from:
        raise SystemExit("--fix-from is required for rescreen phase")
    if not args.params:
        raise SystemExit("--params is required for rescreen phase")

    params = resolve_params(args.params)
    fix_from = Path(args.fix_from)

    # Try to load phase 1 Morris for collapse comparison
    phase1 = None
    out = run_dir or OUTPUT_DIR
    morris_path = out / f"{args.scenario}_morris.json"
    if morris_path.exists():
        phase1 = load_morris(morris_path)
        print(f"Loaded Phase 1 Morris from {morris_path}")

    result, _collapse = run_rescreen(
        scenario=args.scenario,
        fix_from=fix_from,
        params=params,
        n_trajectories=args.morris_trajectories,
        n_seeds=args.sensitivity_seeds,
        n_periods=args.periods,
        n_workers=args.workers,
        phase1_morris=phase1,
    )

    print_morris_report(result, args.sensitivity_threshold, args.sensitivity_threshold)

    # Save
    out = run_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    save_morris(result, out / f"{args.scenario}_rescreen_morris.json")
    print(f"Rescreen results saved to {out}")
