"""Single-seed grid screening with progress tracking and checkpointing.

This module handles the grid screening phase of calibration: testing
many parameter combinations quickly using a single seed, with progress
reporting and checkpoint-based resumption.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from calibration.analysis import CalibrationResult, format_eta, format_progress
from calibration.io import OUTPUT_DIR
from validation import get_validation_funcs


def screen_single_seed(
    params: dict[str, Any],
    scenario: str,
    seed: int,
    n_periods: int,
) -> CalibrationResult:
    """Run single-seed validation for quick screening.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name.
    seed : int
        Random seed.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    CalibrationResult
        Result with single-seed metrics.
    """
    validate, _, _, _ = get_validation_funcs(scenario)
    result = validate(seed=seed, n_periods=n_periods, **params)
    return CalibrationResult(
        params=params,
        single_score=result.total_score,
        n_pass=result.n_pass,
        n_warn=result.n_warn,
        n_fail=result.n_fail,
    )


# =============================================================================
# Checkpointing
# =============================================================================


def _params_key(params: dict[str, Any]) -> str:
    """Create a deterministic string key from params dict."""
    return json.dumps(dict(sorted(params.items())), sort_keys=True, default=str)


def save_checkpoint(
    results: list[CalibrationResult],
    scenario: str,
    phase: str = "screening",
) -> Path:
    """Save intermediate results to a checkpoint file.

    Parameters
    ----------
    results : list[CalibrationResult]
        Results to checkpoint.
    scenario : str
        Scenario name.
    phase : str
        Phase name for filename.

    Returns
    -------
    Path
        Path to checkpoint file.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{scenario}_{phase}_checkpoint.json"
    data = [
        {
            "params": r.params,
            "single_score": r.single_score,
            "n_pass": r.n_pass,
            "n_warn": r.n_warn,
            "n_fail": r.n_fail,
            "seed_scores": r.seed_scores,
            "mean_score": r.mean_score,
            "std_score": r.std_score,
            "pass_rate": r.pass_rate,
            "combined_score": r.combined_score,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_checkpoint(
    scenario: str,
    phase: str = "screening",
) -> list[CalibrationResult] | None:
    """Load checkpoint if it exists.

    Parameters
    ----------
    scenario : str
        Scenario name.
    phase : str
        Phase name.

    Returns
    -------
    list[CalibrationResult] or None
        Previously checkpointed results, or None if no checkpoint.
    """
    path = OUTPUT_DIR / f"{scenario}_{phase}_checkpoint.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [
        CalibrationResult(
            params=d["params"],
            single_score=d["single_score"],
            n_pass=d["n_pass"],
            n_warn=d["n_warn"],
            n_fail=d["n_fail"],
            seed_scores=d.get("seed_scores"),
            mean_score=d.get("mean_score"),
            std_score=d.get("std_score"),
            pass_rate=d.get("pass_rate"),
            combined_score=d.get("combined_score"),
        )
        for d in data
    ]


def delete_checkpoint(scenario: str, phase: str = "screening") -> None:
    """Delete checkpoint file if it exists."""
    path = OUTPUT_DIR / f"{scenario}_{phase}_checkpoint.json"
    if path.exists():
        path.unlink()


# =============================================================================
# Grid screening with progress and checkpointing
# =============================================================================


def run_screening(
    combinations: list[dict[str, Any]],
    scenario: str,
    n_workers: int = 10,
    n_periods: int = 1000,
    avg_time_per_run: float = 0.0,
    checkpoint_every: int = 50,
    resume: bool = False,
) -> list[CalibrationResult]:
    """Screen parameter combinations with progress tracking and checkpointing.

    Parameters
    ----------
    combinations : list[dict]
        Parameter combinations to test.
    scenario : str
        Scenario name.
    n_workers : int
        Parallel workers.
    n_periods : int
        Simulation periods.
    avg_time_per_run : float
        Estimated time per run (from sensitivity). 0 = measure during warmup.
    checkpoint_every : int
        Save checkpoint every N completions.
    resume : bool
        If True, load checkpoint and skip already-evaluated configs.

    Returns
    -------
    list[CalibrationResult]
        Results sorted by single_score (best first).
    """
    total = len(combinations)
    results: list[CalibrationResult] = []
    done_keys: set[str] = set()

    # Resume from checkpoint if available
    if resume:
        checkpoint = load_checkpoint(scenario, "screening")
        if checkpoint:
            results = checkpoint
            done_keys = {_params_key(r.params) for r in results}
            print(f"  Resumed from checkpoint: {len(results)} already evaluated")

    remaining_combos = [c for c in combinations if _params_key(c) not in done_keys]

    if not remaining_combos:
        print("  All combinations already evaluated")
        results.sort(key=lambda r: r.single_score, reverse=True)
        return results

    completed = len(results)
    run_times: list[float] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(screen_single_seed, p, scenario, 0, n_periods): p
            for p in remaining_combos
        }
        for future in as_completed(futures):
            t0 = time.monotonic()
            result = future.result()
            elapsed = time.monotonic() - t0
            run_times.append(elapsed)

            results.append(result)
            completed += 1

            # Compute ETA
            if avg_time_per_run > 0:
                est_time = avg_time_per_run
            elif len(run_times) >= 5:
                est_time = sum(run_times) / len(run_times)
            else:
                est_time = 0.0

            remaining = total - completed
            eta = format_eta(remaining, est_time, n_workers)
            print(f"  {format_progress(completed, total, remaining, eta)}")

            # Checkpoint periodically
            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                save_checkpoint(results, scenario, "screening")

    results.sort(key=lambda r: r.single_score, reverse=True)

    # Clean up checkpoint on completion
    delete_checkpoint(scenario, "screening")

    return results
