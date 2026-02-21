"""Main optimization logic using validation infrastructure.

This module provides the core calibration functionality, including
grid building based on sensitivity analysis, focused grid search with
tiered stability testing, parameter pattern analysis, config export,
and before/after comparison.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from calibration.parameter_space import (
    count_combinations,
    generate_combinations,
    get_parameter_grid,
)
from calibration.sensitivity import SensitivityResult
from validation import (
    StabilityResult,
    compute_combined_score,
    get_validation_func,
    get_validation_funcs,
)

# Output directory for calibration results
OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class CalibrationResult:
    """Result from calibration optimization.

    Attributes
    ----------
    params : dict
        Parameter configuration.
    single_score : float
        Validation score from single-seed run.
    n_pass : int
        Number of metrics that passed.
    n_warn : int
        Number of metrics with warnings.
    n_fail : int
        Number of metrics that failed.
    mean_score : float, optional
        Mean score across stability seeds.
    std_score : float, optional
        Standard deviation of scores across seeds.
    pass_rate : float, optional
        Fraction of seeds that passed (no FAIL metrics).
    combined_score : float, optional
        Combined score balancing accuracy and stability.
    stability_result : StabilityResult, optional
        Full stability test result.
    seed_scores : list[float], optional
        Individual seed scores (for incremental stability).
    """

    params: dict[str, Any]
    single_score: float
    n_pass: int
    n_warn: int
    n_fail: int
    mean_score: float | None = None
    std_score: float | None = None
    pass_rate: float | None = None
    combined_score: float | None = None
    stability_result: StabilityResult | None = None
    seed_scores: list[float] | None = None


@dataclass
class ComparisonResult:
    """Result from before/after config comparison."""

    scenario: str
    default_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    default_score: float
    calibrated_score: float
    improvements: list[
        tuple[str, float, float, float]
    ]  # (name, default, calibrated, pct_change)


# =============================================================================
# Progress tracking helpers
# =============================================================================


def format_eta(remaining: int, avg_time: float, n_workers: int) -> str:
    """Format an ETA string from remaining items and average time.

    Parameters
    ----------
    remaining : int
        Number of remaining items.
    avg_time : float
        Average seconds per item.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    str
        Formatted ETA string (e.g., "5m 30s").
    """
    if avg_time <= 0 or n_workers <= 0:
        return "unknown"
    eta_seconds = avg_time * remaining / n_workers
    if eta_seconds < 60:
        return f"{eta_seconds:.0f}s"
    minutes = int(eta_seconds // 60)
    seconds = int(eta_seconds % 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes:02d}m"


def format_progress(completed: int, total: int, remaining: int, eta: str) -> str:
    """Format a progress line.

    Parameters
    ----------
    completed : int
        Number of completed items.
    total : int
        Total number of items.
    remaining : int
        Number of remaining items.
    eta : str
        ETA string.

    Returns
    -------
    str
        Formatted progress line.
    """
    pct = 100.0 * completed / total if total > 0 else 0.0
    return f"Screened {completed}/{total} ({pct:.1f}%) | {remaining} remaining | ETA: {eta}"


# =============================================================================
# Core calibration functions
# =============================================================================


def build_focused_grid(
    sensitivity: SensitivityResult,
    full_grid: dict[str, list[Any]] | None = None,
    scenario: str = "baseline",
    sensitivity_threshold: float = 0.02,
    pruning_threshold: float | None = 0.04,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Build focused grid from sensitivity analysis.

    Parameters
    ----------
    sensitivity : SensitivityResult
        Result from run_sensitivity_analysis().
    full_grid : dict, optional
        Full parameter grid. Defaults to scenario-specific grid.
    scenario : str
        Scenario name.
    sensitivity_threshold : float
        Minimum sensitivity (Δ) for inclusion in grid search.
    pruning_threshold : float or None
        Maximum score gap from best value for keeping a grid value.
        ``None`` disables pruning.

    Returns
    -------
    tuple[dict, dict]
        (grid_to_search, fixed_params)
        - INCLUDE params (Δ > threshold): all grid values (pruned if enabled)
        - FIX params (Δ ≤ threshold): fix at best value
    """
    if full_grid is None:
        full_grid = get_parameter_grid(scenario)

    included, _ = sensitivity.get_important(sensitivity_threshold)
    param_best = {p.name: p.best_value for p in sensitivity.parameters}

    grid_to_search: dict[str, list[Any]] = {}
    fixed_params: dict[str, Any] = {}

    for name, values in full_grid.items():
        if name in included:
            grid_to_search[name] = values
        else:
            fixed_params[name] = param_best[name]

    grid_to_search = sensitivity.prune_grid(grid_to_search, pruning_threshold)

    return grid_to_search, fixed_params


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
    validate, _ = get_validation_funcs(scenario)
    result = validate(seed=seed, n_periods=n_periods, **params)
    return CalibrationResult(
        params=params,
        single_score=result.total_score,
        n_pass=result.n_pass,
        n_warn=result.n_warn,
        n_fail=result.n_fail,
    )


def _evaluate_single_seed(
    params: dict[str, Any],
    scenario: str,
    seed: int,
    n_periods: int,
) -> tuple[dict[str, Any], int, float]:
    """Evaluate a single seed for stability testing. Returns (params, seed, score)."""
    validate = get_validation_func(scenario)
    result = validate(seed=seed, n_periods=n_periods, **params)
    return dict(params), seed, result.total_score


def evaluate_stability(
    params: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> CalibrationResult:
    """Run multi-seed stability test for full evaluation.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name.
    seeds : list[int]
        List of random seeds to test.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    CalibrationResult
        Result with stability metrics and combined score.
    """
    _, run_stability = get_validation_funcs(scenario)
    stability = run_stability(seeds=seeds, n_periods=n_periods, **params)
    combined = compute_combined_score(stability)
    return CalibrationResult(
        params=params,
        single_score=stability.seed_results[0].total_score,
        n_pass=stability.seed_results[0].n_pass,
        n_warn=stability.seed_results[0].n_warn,
        n_fail=stability.seed_results[0].n_fail,
        mean_score=stability.mean_score,
        std_score=stability.std_score,
        pass_rate=stability.pass_rate,
        combined_score=combined,
        stability_result=stability,
        seed_scores=[sr.total_score for sr in stability.seed_results],
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


# =============================================================================
# Tiered stability testing
# =============================================================================


def parse_stability_tiers(tiers_str: str) -> list[tuple[int, int]]:
    """Parse stability tiers from CLI string.

    Parameters
    ----------
    tiers_str : str
        Format: "100:10,50:20,10:100" meaning
        (top 100 × 10 seeds, top 50 × 20 seeds, top 10 × 100 seeds)

    Returns
    -------
    list[tuple[int, int]]
        List of (n_configs, total_seeds) tuples.
    """
    tiers = []
    for part in tiers_str.split(","):
        configs, seeds = part.strip().split(":")
        tiers.append((int(configs), int(seeds)))
    return tiers


def run_tiered_stability(
    candidates: list[CalibrationResult],
    scenario: str,
    tiers: list[tuple[int, int]],
    n_workers: int = 10,
    n_periods: int = 1000,
    avg_time_per_run: float = 0.0,
) -> list[CalibrationResult]:
    """Run incremental tiered stability testing.

    Each tier runs only NEW seeds (not previously tested ones) and
    accumulates all seed scores for ranking.

    Parameters
    ----------
    candidates : list[CalibrationResult]
        Screening results to stability-test.
    scenario : str
        Scenario name.
    tiers : list[tuple[int, int]]
        List of (n_configs, total_seeds) — each tier tests the top n_configs
        using enough new seeds to reach total_seeds cumulative.
    n_workers : int
        Parallel workers.
    n_periods : int
        Simulation periods.
    avg_time_per_run : float
        Estimated time per run for ETA.

    Returns
    -------
    list[CalibrationResult]
        Final results sorted by combined_score (best first).
    """
    # Initialize seed_scores from screening (seed 0)
    for c in candidates:
        if c.seed_scores is None:
            c.seed_scores = [c.single_score]

    current = candidates

    for tier_idx, (n_configs, total_seeds) in enumerate(tiers):
        tier_num = tier_idx + 1
        # Take top n_configs (or all if fewer available)
        top = current[:n_configs]

        print(f"\n  Tier {tier_num}: {len(top)} configs × {total_seeds} total seeds")

        # Determine which new seeds each config needs
        for c in top:
            existing_n = len(c.seed_scores or [])
            new_seeds_needed = max(0, total_seeds - existing_n)

            if new_seeds_needed > 0:
                new_seed_ids = list(range(existing_n, total_seeds))
                print(
                    f"    Testing {c.single_score:.3f} config: "
                    f"+{new_seeds_needed} seeds ({existing_n}→{total_seeds})"
                )

                # Run new seeds in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(
                            _evaluate_single_seed,
                            c.params,
                            scenario,
                            seed,
                            n_periods,
                        )
                        for seed in new_seed_ids
                    ]
                    new_scores = []
                    for i, future in enumerate(as_completed(futures)):
                        _, _seed, score = future.result()
                        new_scores.append(score)
                        done = i + 1
                        remaining = new_seeds_needed - done
                        eta = format_eta(remaining, avg_time_per_run, n_workers)
                        print(
                            f"    Tier {tier_num}: "
                            f"Testing {done}/{new_seeds_needed} "
                            f"({100 * done / new_seeds_needed:.0f}%) "
                            f"| {remaining} remaining | ETA: {eta}"
                        )

                c.seed_scores = (c.seed_scores or []) + new_scores

            # Update aggregate metrics
            scores = c.seed_scores or []
            if scores:
                import statistics

                c.mean_score = statistics.mean(scores)
                c.std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
                # pass_rate not available without full ValidationScore — approximate
                c.combined_score = c.mean_score * (1.0 - c.std_score)

        # Rank by combined_score
        top.sort(key=lambda r: r.combined_score or 0, reverse=True)
        current = top

        # Print tier results
        print(f"\n  Tier {tier_num} results (top 5):")
        for i, r in enumerate(current[:5]):
            print(
                f"    #{i + 1}: combined={r.combined_score:.4f} "
                f"mean={r.mean_score:.3f} ± {r.std_score:.3f} "
                f"({len(r.seed_scores or [])} seeds)"
            )

        # Checkpoint after each tier
        save_checkpoint(current, scenario, "stability")

    delete_checkpoint(scenario, "stability")
    return current


# =============================================================================
# Focused calibration (orchestrates screening + stability)
# =============================================================================


def run_focused_calibration(
    grid: dict[str, list[Any]],
    fixed_params: dict[str, Any],
    scenario: str = "baseline",
    n_workers: int = 10,
    n_periods: int = 1000,
    stability_tiers: list[tuple[int, int]] | None = None,
    avg_time_per_run: float = 0.0,
    resume: bool = False,
) -> list[CalibrationResult]:
    """Run calibration on focused grid with fixed params.

    Parameters
    ----------
    grid : dict
        Parameter grid to search (from build_focused_grid).
    fixed_params : dict
        Fixed parameter values (from build_focused_grid).
    scenario : str
        Scenario name.
    n_workers : int
        Number of parallel workers.
    n_periods : int
        Number of simulation periods.
    stability_tiers : list[tuple[int, int]], optional
        Tiered stability config. Defaults to [(100, 10), (50, 20), (10, 100)].
    avg_time_per_run : float
        Average time per simulation run (from sensitivity).
    resume : bool
        If True, resume from checkpoint.

    Returns
    -------
    list[CalibrationResult]
        Results sorted by combined_score (best first).
    """
    if stability_tiers is None:
        stability_tiers = [(100, 10), (50, 20), (10, 100)]

    total = count_combinations(grid)
    print(f"\n[{scenario}] Focused Grid Search: {total} combinations")
    print(f"Fixed params: {fixed_params}")

    # Generate all combinations (merging fixed params)
    combinations = []
    for combo in generate_combinations(grid):
        full_params = {**fixed_params, **combo}
        combinations.append(full_params)

    # Screening phase
    print(f"\nScreening {total} combinations with {n_workers} workers...")
    screening_results = run_screening(
        combinations,
        scenario,
        n_workers=n_workers,
        n_periods=n_periods,
        avg_time_per_run=avg_time_per_run,
        resume=resume,
    )

    # Stability testing phase
    print("\nTiered stability testing...")
    return run_tiered_stability(
        screening_results,
        scenario,
        tiers=stability_tiers,
        n_workers=n_workers,
        n_periods=n_periods,
        avg_time_per_run=avg_time_per_run,
    )


# =============================================================================
# Parameter pattern analysis
# =============================================================================


def analyze_parameter_patterns(
    results: list[CalibrationResult],
    top_n: int = 50,
) -> dict[str, dict[Any, int]]:
    """Analyze which parameter values consistently appear in top configs.

    Parameters
    ----------
    results : list[CalibrationResult]
        Screening results sorted by score (best first).
    top_n : int
        Number of top configs to analyze.

    Returns
    -------
    dict[str, dict[Any, int]]
        For each parameter, a dict mapping value → count in top configs.
    """
    top = results[:top_n]
    if not top:
        return {}

    # Collect all parameter names
    all_params = set()
    for r in top:
        all_params.update(r.params.keys())

    patterns: dict[str, dict[Any, int]] = {}
    for param in sorted(all_params):
        counter: Counter[Any] = Counter()
        for r in top:
            if param in r.params:
                counter[r.params[param]] += 1
        patterns[param] = dict(counter.most_common())

    return patterns


def print_parameter_patterns(
    patterns: dict[str, dict[Any, int]], top_n: int = 50
) -> None:
    """Print parameter pattern analysis.

    Parameters
    ----------
    patterns : dict
        Output from analyze_parameter_patterns().
    top_n : int
        Number of top configs used for display.
    """
    print(f"\nParameter Patterns (top {top_n} configs):")
    print("-" * 70)
    for param, counts in patterns.items():
        parts = []
        for val, count in counts.items():
            pct = 100.0 * count / top_n
            parts.append(f"{val}={count} ({pct:.0f}%)")
        line = " | ".join(parts[:4])
        if len(parts) > 4:
            line += f" | +{len(parts) - 4} more"

        # Detect strong preferences
        top_count = max(counts.values())
        top_pct = 100.0 * top_count / top_n
        hint = ""
        if top_pct >= 80:
            hint = " → strongly prefer"
        elif top_pct >= 60:
            hint = " → slight preference"

        print(f"  {param:<30} {line}{hint}")
    print()


# =============================================================================
# Config export
# =============================================================================


def export_best_config(
    result: CalibrationResult, scenario: str, path: Path | None = None
) -> Path:
    """Export best calibration result as a ready-to-use YAML config.

    Parameters
    ----------
    result : CalibrationResult
        Best calibration result.
    scenario : str
        Scenario name.
    path : Path, optional
        Output path. Defaults to output/{scenario}_best_config.yml.

    Returns
    -------
    Path
        Path to exported config file.
    """
    if path is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{scenario}_best_config.yml"

    # Build config: only include params that differ from engine defaults
    config = dict(result.params)

    # Add header comment
    header = (
        f"# Best calibration config for {scenario} scenario\n"
        f"# Score: {result.combined_score or result.single_score:.4f}\n"
    )
    if result.mean_score is not None:
        header += f"# Mean: {result.mean_score:.3f} ± {result.std_score:.3f}\n"
    if result.pass_rate is not None:
        header += f"# Pass rate: {result.pass_rate:.0%}\n"
    header += f"#\n# Usage: sim = bam.Simulation.init(config='{path}')\n\n"

    with open(path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

    return path


# =============================================================================
# Before/after comparison
# =============================================================================


def compare_configs(
    default: dict[str, Any],
    calibrated: dict[str, Any],
    scenario: str,
    seed: int = 0,
    n_periods: int = 1000,
) -> ComparisonResult:
    """Run default and calibrated configs side-by-side and compare.

    Parameters
    ----------
    default : dict
        Default config overrides (can be empty for engine defaults).
    calibrated : dict
        Calibrated config params.
    scenario : str
        Scenario name.
    seed : int
        Random seed.
    n_periods : int
        Simulation periods.

    Returns
    -------
    ComparisonResult
        Side-by-side comparison of metrics.
    """
    validate = get_validation_func(scenario)

    default_result = validate(seed=seed, n_periods=n_periods, **default)
    calibrated_result = validate(seed=seed, n_periods=n_periods, **calibrated)

    default_metrics = {mr.name: mr.actual for mr in default_result.metric_results}
    calibrated_metrics = {mr.name: mr.actual for mr in calibrated_result.metric_results}

    improvements = []
    for mr_cal in calibrated_result.metric_results:
        name = mr_cal.name
        cal_val = mr_cal.actual
        def_val = default_metrics.get(name, 0.0)
        if def_val != 0:
            pct_change = 100.0 * (cal_val - def_val) / abs(def_val)
        else:
            pct_change = 0.0
        improvements.append((name, def_val, cal_val, pct_change))

    return ComparisonResult(
        scenario=scenario,
        default_metrics=default_metrics,
        calibrated_metrics=calibrated_metrics,
        default_score=default_result.total_score,
        calibrated_score=calibrated_result.total_score,
        improvements=improvements,
    )


def print_comparison(result: ComparisonResult) -> None:
    """Print before/after comparison table.

    Parameters
    ----------
    result : ComparisonResult
        Output from compare_configs().
    """
    print("\n" + "=" * 80)
    print(f"BEFORE/AFTER COMPARISON ({result.scenario})")
    print("=" * 80)
    print(
        f"\nTotal Score: {result.default_score:.4f} → {result.calibrated_score:.4f} "
        f"({'improved' if result.calibrated_score > result.default_score else 'unchanged'})"
    )
    print(f"\n{'Metric':<35} {'Default':>10} {'Calibrated':>10} {'Change':>10}")
    print("-" * 70)

    for name, def_val, cal_val, pct in result.improvements:
        if abs(pct) < 1.0:
            change_str = "(unchanged)"
        elif pct > 0:
            change_str = f"+{pct:.1f}%"
        else:
            change_str = f"{pct:.1f}%"
        print(f"  {name:<33} {def_val:>10.4f} {cal_val:>10.4f} {change_str:>10}")

    print("=" * 80 + "\n")
