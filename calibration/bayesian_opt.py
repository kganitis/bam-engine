"""
Bayesian Optimization
=====================

Fine-tune continuous parameters using Gaussian Process optimization.

This module focuses on continuous parameters only. Categorical/boolean parameters
are better explored in the local sweep phase.

Usage::

    python -m calibration --bayesian

Requires scikit-optimize::

    pip install scikit-optimize
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .progress import ProgressTracker
from .runner import run_ensemble

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager


# Check if scikit-optimize is available
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


# Bayesian Optimization search space (continuous parameters only)
# Categorical/boolean params are better explored in local sweep
BO_SPACE_CONFIG: list[dict[str, Any]] = [
    {"name": "price_init_offset", "type": "real", "low": 0.05, "high": 0.60},
    {"name": "min_wage_ratio", "type": "real", "low": 0.45, "high": 0.75},
    {"name": "net_worth_init", "type": "real", "low": 0.5, "high": 15.0},
    {"name": "new_firm_price_markup", "type": "real", "low": 0.9, "high": 1.4},
    {"name": "new_firm_production_factor", "type": "real", "low": 0.7, "high": 1.1},
    {"name": "new_firm_wage_factor", "type": "real", "low": 0.4, "high": 1.1},
    {"name": "equity_base_init", "type": "real", "low": 3.0, "high": 15.0},
]


def _build_skopt_space() -> list:
    """Build scikit-optimize space from configuration."""
    if not SKOPT_AVAILABLE:
        return []

    space = []
    for param in BO_SPACE_CONFIG:
        if param["type"] == "real":
            space.append(Real(param["low"], param["high"], name=param["name"]))
        elif param["type"] == "integer":
            space.append(Integer(param["low"], param["high"], name=param["name"]))
    return space


def _decode_params(x: list) -> dict[str, Any]:
    """Convert optimizer output to parameter dict."""
    params = {}
    for i, param_config in enumerate(BO_SPACE_CONFIG):
        params[param_config["name"]] = x[i]
    return params


def _encode_params(params: dict[str, Any]) -> list:
    """Convert parameter dict to optimizer input."""
    x = []
    for param_config in BO_SPACE_CONFIG:
        name = param_config["name"]
        default = (param_config["low"] + param_config["high"]) / 2
        value = params.get(name, default)
        # Clamp to bounds
        value = max(param_config["low"], min(param_config["high"], value))
        x.append(value)
    return x


def print_bo_info(n_calls: int = 300):
    """Print information about Bayesian Optimization configuration."""
    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION CONFIGURATION")
    print("=" * 70)

    if not SKOPT_AVAILABLE:
        print("\nWARNING: scikit-optimize not installed!")
        print("Install with: pip install scikit-optimize")
        print("=" * 70)
        return

    print(f"\nParameters in search space: {len(BO_SPACE_CONFIG)}")
    print("-" * 70)

    for param in BO_SPACE_CONFIG:
        bounds = f"[{param['low']}, {param['high']}]"
        print(f"  {param['name']}: {bounds} ({param['type']})")

    print("-" * 70)
    print(f"\nNumber of iterations: {n_calls}")
    print("=" * 70)


def run_bayesian_optimization(
    checkpoint: CheckpointManager,
    top_configs: list[dict],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_calls: int = 300,
    n_workers: int = 1,
) -> dict[str, Any]:
    """
    Run Bayesian optimization to fine-tune continuous parameters.

    Uses Gaussian Process optimization (gp_minimize from scikit-optimize)
    to explore the continuous parameter space.

    Parameters
    ----------
    checkpoint : CheckpointManager
        Checkpoint manager for saving results.
    top_configs : list[dict]
        Top configurations from grid search/local sweep to initialize.
    n_seeds : int
        Number of seeds per configuration.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.
    n_calls : int
        Number of BO iterations.
    n_workers : int
        Number of parallel workers for objective evaluation.

    Returns
    -------
    dict
        Best parameters and score found.
    """
    if not SKOPT_AVAILABLE:
        print("\n" + "=" * 70)
        print("BAYESIAN OPTIMIZATION (SKIPPED - scikit-optimize not installed)")
        print("=" * 70)
        print("\nInstall with: pip install scikit-optimize")
        return {}

    # Check if already completed
    if checkpoint.data["metadata"].get("bayesian_optimization_completed", False):
        print("\n" + "=" * 70)
        print("BAYESIAN OPTIMIZATION (SKIPPED - already completed)")
        print("=" * 70)
        bo_results = checkpoint.data.get("bo_results", {})
        if bo_results:
            print(f"Best score: {bo_results.get('best_score', 'N/A'):.2f}")
            print(f"Best params: {bo_results.get('best_params', {})}")
        return bo_results

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION")
    print("=" * 70)
    checkpoint.set_metadata("stage", "bayesian_optimization")

    # Print configuration info
    print_bo_info(n_calls)

    # Build search space
    space = _build_skopt_space()

    # Get best categorical/boolean values from top configs to use as fixed values
    # (BO only optimizes continuous params)
    categorical_params = {}
    if top_configs:
        best_params = top_configs[0]["params"]
        # Extract categorical params not in BO space
        bo_param_names = {p["name"] for p in BO_SPACE_CONFIG}
        for k, v in best_params.items():
            if k not in bo_param_names:
                categorical_params[k] = v
        print(f"\nUsing categorical params from best config: {categorical_params}")

    # Get initial best score
    initial_best = float("inf")
    if top_configs:
        initial_best = top_configs[0]["scores"]["total"]
        print(f"Initial best score: {initial_best:.2f}")

    # Initialize with top configs (convert to BO space)
    x0 = []
    y0 = []
    for cfg in top_configs[: min(20, len(top_configs))]:
        try:
            x = _encode_params(cfg["params"])
            x0.append(x)
            y0.append(cfg["scores"]["total"])
        except (KeyError, TypeError):
            continue

    print(f"\nInitializing with {len(x0)} configurations from previous stages")
    print(f"Running {n_calls} BO iterations with {n_workers} worker(s)...\n")

    # Track results for later saving (thread-safe list append)
    all_results: list[tuple[dict, dict, list]] = []

    if n_workers == 1:
        # Sequential mode: use progress tracker
        progress = ProgressTracker(
            total=n_calls,
            stage="Bayesian Opt",
            update_interval=5,
        )
        progress.stats.best_score = initial_best

        def objective(x: list) -> float:
            """Objective function for optimization."""
            params = _decode_params(x)
            params.update(categorical_params)
            scores, _seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
            all_results.append((params.copy(), scores, _seed_totals))
            progress.update(score=scores["total"], params=params)
            if progress.stats.completed % 50 == 0:
                checkpoint.save()
            return scores["total"]
    else:
        # Parallel mode: simple objective (no progress updates)
        print("  (Progress bar disabled in parallel mode)")

        def objective(x: list) -> float:
            """Objective function for optimization."""
            params = _decode_params(x)
            params.update(categorical_params)
            scores, _seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
            return scores["total"]

        progress = None

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_jobs=n_workers,
        x0=x0 if x0 else None,
        y0=y0 if y0 else None,
        random_state=42,
    )

    # Close progress tracker if used
    if progress is not None:
        progress.close()

    # Extract best results
    best_params = _decode_params(result.x)
    best_params.update(categorical_params)
    best_bo_score = result.fun

    # Store results
    checkpoint.data["bo_results"] = {
        "best_params": best_params,
        "best_score": best_bo_score,
        "n_iterations": n_calls,
        "categorical_params_used": categorical_params,
    }
    checkpoint.data["metadata"]["bayesian_optimization_completed"] = True
    checkpoint.save()

    if best_bo_score < initial_best:
        improvement = initial_best - best_bo_score
        print(
            f"\nScore improvement: {initial_best:.2f} -> {best_bo_score:.2f} ({improvement:.2f})"
        )
    else:
        print("\nNo improvement over previous best.")

    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return {"best_params": best_params, "best_score": best_bo_score}
