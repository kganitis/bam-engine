"""
Simulation Runner
=================

Simulation execution and data collection for calibration.
Patterns match example_baseline_scenario.py for consistency.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import bamengine as bam
from bamengine import ops

from .config import FIXED_PARAMS, apply_config_offsets
from .scoring import compute_all_scores

# Suppress NumPy warnings for division by zero and invalid values.
# These occur when simulations collapse (100% unemployment, zero prices, etc.)
# which is expected during calibration.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

# Suppress config validation warnings (e.g., min_wage >= wage_offer_init)
# which are expected when sweeping parameter combinations.
warnings.filterwarnings("ignore", category=UserWarning, module="bamengine.simulation")


def apply_unemployment_smoothing(
    unemployment_raw: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """
    Apply moving average smoothing to unemployment data.

    This matches the baseline scenario implementation which uses a 4-quarter
    moving average with proper padding at the start (raw values for periods
    without enough history for the full window).

    Parameters
    ----------
    unemployment_raw : np.ndarray
        Raw unemployment rate data.
    window : int, default=4
        Window size for moving average (4 = quarterly).

    Returns
    -------
    np.ndarray
        Smoothed unemployment data with same length as input.
    """
    if len(unemployment_raw) < window:
        return unemployment_raw

    kernel = np.ones(window) / window
    smoothed = np.convolve(unemployment_raw, kernel, mode="valid")

    # Pad beginning with raw values (not enough history for MA)
    return np.concatenate([unemployment_raw[: window - 1], smoothed])


def run_single_simulation(
    params: dict[str, Any],
    seed: int,
    n_periods: int = 1000,
    burn_in: int = 500,
) -> dict[str, Any]:
    """
    Run a single simulation with given parameters.

    Data collection and metric calculation follows the patterns established
    in example_baseline_scenario.py for consistency.

    Parameters
    ----------
    params : dict
        Parameter overrides (merged with FIXED_PARAMS).
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    burn_in : int
        Burn-in periods to exclude from scoring.

    Returns
    -------
    dict
        All scoring metrics and diagnostic values.
    """
    # Build full config
    config = {**FIXED_PARAMS, **params}
    config["seed"] = seed
    config["n_periods"] = n_periods
    config["logging"] = {"default_level": "ERROR"}

    # Extract capture timing params before applying offsets
    employed_capture_event = config.pop(
        "employed_capture_event", "workers_update_contracts"
    )
    vacancies_capture_event = config.pop(
        "vacancies_capture_event", "firms_fire_workers"
    )

    # Convert offset-based params to absolute values
    config = apply_config_offsets(config)

    try:
        sim = bam.Simulation.init(**config)

        # Build capture_timing dict (matching baseline scenario)
        capture_timing: dict[str, str] = {"Worker.wage": "workers_receive_wage"}
        if employed_capture_event is not None:
            capture_timing["Worker.employed"] = employed_capture_event
        if vacancies_capture_event is not None:
            capture_timing["Employer.n_vacancies"] = vacancies_capture_event

        # Data collection (matching baseline scenario)
        results = sim.run(
            collect={
                "Producer": ["production", "labor_productivity"],
                "Worker": ["wage", "employed"],
                "Employer": ["n_vacancies"],
                "Economy": True,
                "aggregate": None,
                "capture_timing": capture_timing,
            }
        )

        # Extract economy data
        inflation = np.array(results.economy_data["inflation"])
        avg_price = np.array(results.economy_data["avg_price"])

        # Extract role data
        production = np.array(results.role_data["Producer"]["production"])
        labor_productivity = np.array(
            results.role_data["Producer"]["labor_productivity"]
        )
        wages = np.array(results.role_data["Worker"]["wage"])
        employed = np.array(results.role_data["Worker"]["employed"])
        n_vacancies = np.array(results.role_data["Employer"]["n_vacancies"])

        # Calculate unemployment from Worker.employed (matching baseline)
        unemployment_raw = 1 - ops.mean(employed.astype(float), axis=1)

        # Apply 4-quarter MA smoothing (matching baseline)
        unemployment = apply_unemployment_smoothing(unemployment_raw)

        # GDP as sum of production (matching baseline)
        gdp = ops.sum(production, axis=1)

        # Production-weighted productivity (matching baseline)
        weighted_productivity = ops.sum(
            ops.multiply(labor_productivity, production), axis=1
        )
        avg_productivity = ops.divide(weighted_productivity, gdp)

        # Average wage for employed workers only (matching baseline)
        employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
        employed_count = ops.sum(employed, axis=1)
        avg_employed_wage = ops.where(
            ops.greater(employed_count, 0),
            ops.divide(employed_wages_sum, employed_count),
            0.0,
        )

        # Real wage calculation (matching baseline)
        real_wage = ops.divide(avg_employed_wage, avg_price)

        # Total vacancies
        total_vacancies = ops.sum(n_vacancies, axis=1)

        # Final production distribution
        final_production = production[-1]

        # Check for collapse
        destroyed = sim.ec.destroyed if hasattr(sim.ec, "destroyed") else False

        # Compute all scores
        scores = compute_all_scores(
            unemployment=unemployment,
            unemployment_raw=unemployment_raw,
            inflation=inflation,
            gdp=gdp,
            avg_productivity=avg_productivity,
            avg_employed_wage=avg_employed_wage,
            avg_price=avg_price,
            real_wage=real_wage,
            total_vacancies=total_vacancies,
            n_households=config["n_households"],
            final_production=final_production,
            burn_in=burn_in,
            destroyed=destroyed,
        )

        return scores

    except Exception as e:
        # Simulation crashed - maximum penalty
        return {"total": 10000.0, "_error": str(e), "collapse_penalty": 10000.0}


def run_ensemble(
    params: dict[str, Any],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
) -> tuple[dict[str, float], list[float]]:
    """
    Run simulation with multiple seeds and average scores.

    Parameters
    ----------
    params : dict
        Parameter overrides.
    n_seeds : int
        Number of seeds to run.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.

    Returns
    -------
    tuple[dict, list]
        Averaged scores and list of total scores per seed.
    """
    all_scores: list[dict] = []

    for seed in range(n_seeds):
        scores = run_single_simulation(params, seed, n_periods, burn_in)
        all_scores.append(scores)

    # Average scores across seeds
    avg_scores: dict[str, float] = {}

    # Collect all keys from all scores
    all_keys = set()
    for s in all_scores:
        all_keys.update(s.keys())

    for key in all_keys:
        if key == "_error":
            # Preserve error message if any seed crashed
            errors = [s.get("_error") for s in all_scores if "_error" in s]
            if errors:
                avg_scores["_error"] = errors[0]
            continue

        values = [s.get(key, 0) for s in all_scores]
        avg_scores[key] = float(np.mean(values))
        if not key.startswith("_"):
            avg_scores[f"{key}_std"] = float(np.std(values))

    seed_totals = [s["total"] for s in all_scores]

    return avg_scores, seed_totals


# Worker function for parallel execution
def _run_config_worker(
    args: tuple[dict, int, int, int],
) -> tuple[dict, dict, list]:
    """
    Worker function for parallel config evaluation.

    Parameters
    ----------
    args : tuple
        (params, n_seeds, n_periods, burn_in)

    Returns
    -------
    tuple
        (params, scores, seed_totals)
    """
    params, n_seeds, n_periods, burn_in = args
    scores, seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
    return params, scores, seed_totals
