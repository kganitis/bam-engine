"""
===============================
BAM Parameter Calibration
===============================

This script performs comprehensive parameter calibration to match the 8 target
figures from Delli Gatti et al. (2011), Section 3.9.1.

The calibration uses a four-stage approach:
1. Sensitivity Analysis: Identify most impactful lower-priority parameters
2. Grid Search: Full factorial search on high-impact + sensitive parameters
3. Local Sensitivity Sweep: Test alternative values on top configurations
4. Bayesian Optimization: Refine search around best configurations

Usage
-----
Baseline mode (single run with current defaults):
    python tools/calibration.py --baseline

Test mode (quick validation, ~1-2 minutes):
    python tools/calibration.py --test

Full mode (extended calibration, ~8 hours with parallelization):
    python tools/calibration.py

Resume from checkpoint:
    python tools/calibration.py --resume

Visualize best configuration:
    python tools/calibration.py --visualize
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any

# Disable all logging - calibration runs many simulations that may collapse,
# which is expected behavior and would generate excessive log messages
logging.disable(logging.CRITICAL)

# Suppress NumPy warnings for division by zero and invalid values.
# These occur when simulations collapse (100% unemployment, zero prices, etc.)
# which is expected during calibration. The np.where fallback values are still
# returned correctly; these warnings are just noise.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

# Suppress config validation warnings (e.g., min_wage >= wage_offer_init)
# which are expected when sweeping parameter combinations during calibration.
warnings.filterwarnings("ignore", category=UserWarning, module="bamengine.simulation")

import numpy as np  # noqa: E402

import bamengine as bam  # noqa: E402

# Output directory for all calibration results (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional imports
try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ==============================================================================
# PROGRESS MONITORING
# ==============================================================================


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""

    completed: int = 0
    total: int = 0
    start_time: float = 0.0
    best_score: float = float("inf")
    best_params: dict = field(default_factory=dict)
    stage: str = ""
    current_config: dict = field(default_factory=dict)


class ProgressTracker:
    """
    Track and display progress for long-running calibration.

    Features:
    - Percentage complete with progress bar
    - Elapsed time and ETA
    - Processing rate (configs/minute)
    - Best score found so far
    - Current configuration being evaluated
    """

    def __init__(
        self,
        total: int,
        stage: str = "",
        use_tqdm: bool = True,
        update_interval: int = 1,
    ):
        self.stats = ProgressStats(
            total=total,
            start_time=time.time(),
            stage=stage,
        )
        self.update_interval = update_interval
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar = None
        self._last_print_time = 0.0

        if self.use_tqdm:
            self._pbar = tqdm(
                total=total,
                desc=stage,
                unit="cfg",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Best: {postfix}",
            )
            self._pbar.set_postfix_str("score=-.--")

    def update(
        self,
        n: int = 1,
        score: float | None = None,
        params: dict | None = None,
        force_print: bool = False,
    ):
        """Update progress with optional score tracking."""
        self.stats.completed += n

        # Track best score
        if score is not None and score < self.stats.best_score:
            self.stats.best_score = score
            self.stats.best_params = params or {}

        if params:
            self.stats.current_config = params

        if self.use_tqdm and self._pbar:
            self._pbar.update(n)
            if score is not None:
                self._pbar.set_postfix_str(f"score={self.stats.best_score:.2f}")
        else:
            # Fallback: print progress at intervals
            now = time.time()
            should_print = (
                force_print
                or self.stats.completed == self.stats.total
                or self.stats.completed % self.update_interval == 0
                or (now - self._last_print_time) >= 10.0  # At least every 10 seconds
            )
            if should_print:
                self._print_progress()
                self._last_print_time = now

    def _print_progress(self):
        """Print progress to console (fallback when tqdm unavailable)."""
        stats = self.stats
        elapsed = time.time() - stats.start_time
        pct = 100 * stats.completed / stats.total if stats.total > 0 else 0

        # Calculate rate and ETA
        rate = stats.completed / elapsed if elapsed > 0 else 0
        remaining = stats.total - stats.completed
        eta_seconds = remaining / rate if rate > 0 else 0

        # Format time strings
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)

        # Build progress bar
        bar_width = 30
        filled = (
            int(bar_width * stats.completed / stats.total) if stats.total > 0 else 0
        )
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build status line
        status = (
            f"\r[{stats.stage}] {bar} {pct:5.1f}% "
            f"({stats.completed}/{stats.total}) | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"Rate: {rate * 60:.1f} cfg/min | "
            f"Best: {stats.best_score:.2f}"
        )

        # Print with carriage return for in-place update
        print(status, end="", flush=True)

        # Print newline when complete
        if stats.completed >= stats.total:
            print()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def close(self):
        """Close progress bar and print summary."""
        if self.use_tqdm and self._pbar:
            self._pbar.close()

        # Print final summary
        elapsed = time.time() - self.stats.start_time
        print(f"\n{self.stats.stage} completed:")
        print(f"  Total configs: {self.stats.completed}")
        print(f"  Total time: {self._format_time(elapsed)}")
        print(f"  Avg rate: {self.stats.completed / elapsed * 60:.1f} configs/min")
        if self.stats.best_score < float("inf"):
            print(f"  Best score: {self.stats.best_score:.2f}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ==============================================================================
# TARGET VALUES FROM BOOK (Delli Gatti et al., 2011, Section 3.9.1)
# ==============================================================================


@dataclass(frozen=True)
class BookTargets:
    """Target values from the BAM book for scoring simulations."""

    # Log GDP targets (from Figure 3.9.1a)
    log_gdp_min: float = 5.40
    log_gdp_max: float = 5.50
    log_gdp_std_max: float = 0.03

    # Unemployment rate targets (from Figure 3.9.1b)
    unemployment_min: float = 0.03  # 3%
    unemployment_max: float = 0.11  # 11%
    unemployment_mean_min: float = 0.05  # 5%
    unemployment_mean_max: float = 0.07  # 7%

    # Inflation rate targets (from Figure 3.9.1c)
    inflation_mean_min: float = 0.03  # 3%
    inflation_mean_max: float = 0.07  # 7%
    inflation_positive_pct: float = 0.90  # >90% should be positive

    # Productivity/Real Wage Ratio (Figure 3.9.1d) - match TREND, not magnitude
    # Book shows ~0.35, our model produces ~1.05
    prod_wage_cv_max: float = 0.10  # Coefficient of variation < 10%
    prod_wage_drift_max: float = 0.15  # Max drift from mean

    # Phillips Curve (Figure 3.3a) - weak negative correlation
    phillips_corr_target: float = -0.10
    phillips_corr_tolerance: float = 0.15  # Range: [-0.25, 0.05]
    phillips_wage_infl_positive_pct: float = 0.85

    # Okun Curve (Figure 3.3b) - STRONG negative correlation (main issue)
    okun_corr_threshold: float = -0.70

    # Beveridge Curve (Figure 3.3c) - weak negative correlation
    beveridge_corr_target: float = -0.27
    beveridge_corr_tolerance: float = 0.15  # Range: [-0.42, -0.12]
    beveridge_vacancy_min: float = 0.08  # 8%
    beveridge_vacancy_max: float = 0.20  # 20%

    # Firm Size Distribution (Figure 3.3d) - highly right-skewed
    firm_size_pct_below_3: float = 0.90  # >90% produce <3 units
    firm_size_mean_max: float = 3.0


TARGETS = BookTargets()


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================


def score_log_gdp(log_gdp: np.ndarray) -> dict[str, float]:
    """
    Score log GDP against book targets.

    Target: Range 5.40-5.50, mean ~5.46, std ~0.02-0.03
    """
    mean_val = float(np.nanmean(log_gdp))
    std_val = float(np.nanstd(log_gdp))

    # Level score: 0 if in target range, proportional penalty otherwise
    if TARGETS.log_gdp_min <= mean_val <= TARGETS.log_gdp_max:
        level_score = 0.0
    else:
        dist = min(
            abs(mean_val - TARGETS.log_gdp_min), abs(mean_val - TARGETS.log_gdp_max)
        )
        level_score = dist * 50

    # Stability score: 0 if std <= target, penalty otherwise
    if std_val <= TARGETS.log_gdp_std_max:
        stability_score = 0.0
    else:
        stability_score = (std_val - TARGETS.log_gdp_std_max) * 100

    return {
        "log_gdp_level": level_score,
        "log_gdp_stability": stability_score,
        "_log_gdp_mean": mean_val,
        "_log_gdp_std": std_val,
    }


def score_unemployment(unemployment: np.ndarray) -> dict[str, float]:
    """
    Score unemployment rate against book targets.

    Target: Range 3-11%, mean 5-7%, stable (low volatility in visualized data)

    Note: Stability is scored on the visualized data (after seasonal adjustment
    if simple_ma is used). This means smooth unemployment will score better
    even if the underlying raw data is volatile.
    """
    mean_val = float(np.nanmean(unemployment))
    std_val = float(np.nanstd(unemployment))
    min_val = float(np.nanmin(unemployment))
    max_val = float(np.nanmax(unemployment))

    # Range violations
    violations = np.sum(unemployment < TARGETS.unemployment_min) + np.sum(
        unemployment > TARGETS.unemployment_max
    )
    range_score = (violations / len(unemployment)) * 10

    # Mean score
    if TARGETS.unemployment_mean_min <= mean_val <= TARGETS.unemployment_mean_max:
        mean_score = 0.0
    else:
        dist = min(
            abs(mean_val - TARGETS.unemployment_mean_min),
            abs(mean_val - TARGETS.unemployment_mean_max),
        )
        mean_score = dist * 100

    # Stability score: lower volatility is better (target std < 3%)
    # The visualized (seasonally adjusted) data should be stable
    target_std = 0.03  # 3% std is considered stable
    if std_val <= target_std:
        stability_score = 0.0
    else:
        stability_score = (std_val - target_std) * 50  # Penalty for excess volatility

    return {
        "unemployment_range": range_score,
        "unemployment_mean": mean_score,
        "unemployment_stability": stability_score,
        "_unemployment_mean": mean_val,
        "_unemployment_std": std_val,
        "_unemployment_min": min_val,
        "_unemployment_max": max_val,
    }


def score_inflation(inflation: np.ndarray) -> dict[str, float]:
    """
    Score inflation rate against book targets.

    Target: Mean ~5%, >90% positive
    """
    mean_val = float(np.nanmean(inflation))
    positive_pct = float(np.mean(inflation > 0))

    # Mean score
    if TARGETS.inflation_mean_min <= mean_val <= TARGETS.inflation_mean_max:
        mean_score = 0.0
    else:
        dist = min(
            abs(mean_val - TARGETS.inflation_mean_min),
            abs(mean_val - TARGETS.inflation_mean_max),
        )
        mean_score = dist * 50

    # Positive percentage score
    if positive_pct >= TARGETS.inflation_positive_pct:
        positive_score = 0.0
    else:
        positive_score = (TARGETS.inflation_positive_pct - positive_pct) * 50

    return {
        "inflation_mean": mean_score,
        "inflation_positive": positive_score,
        "_inflation_mean": mean_val,
        "_inflation_positive_pct": positive_pct,
    }


def score_prod_wage_ratio(prod_wage_ratio: np.ndarray) -> dict[str, float]:
    """
    Score productivity/real wage ratio against book targets.

    Target: Match TREND stability (CV < 10%, drift < 15%), ignore magnitude
    """
    valid = prod_wage_ratio[prod_wage_ratio > 0]
    if len(valid) < 100:
        return {
            "prod_wage_stability": 5.0,
            "prod_wage_drift": 5.0,
            "_prod_wage_cv": 1.0,
            "_prod_wage_drift": 1.0,
            "_prod_wage_mean": 0.0,
        }

    mean_val = float(np.nanmean(valid))
    std_val = float(np.nanstd(valid))
    cv = std_val / mean_val if mean_val > 0 else 1.0

    # CV score (stability)
    if cv <= TARGETS.prod_wage_cv_max:
        cv_score = 0.0
    else:
        cv_score = (cv - TARGETS.prod_wage_cv_max) * 30

    # Drift score (compare first 100 vs last 100 periods)
    first_100 = float(np.nanmean(valid[:100]))
    last_100 = float(np.nanmean(valid[-100:]))
    drift = abs(last_100 - first_100) / mean_val if mean_val > 0 else 1.0

    if drift <= TARGETS.prod_wage_drift_max:
        drift_score = 0.0
    else:
        drift_score = (drift - TARGETS.prod_wage_drift_max) * 20

    return {
        "prod_wage_stability": cv_score,
        "prod_wage_drift": drift_score,
        "_prod_wage_cv": cv,
        "_prod_wage_drift": drift,
        "_prod_wage_mean": mean_val,
    }


def score_phillips_curve(
    unemployment: np.ndarray, wage_inflation: np.ndarray
) -> dict[str, float]:
    """
    Score Phillips curve against book targets.

    Target: Weak negative correlation r ~ -0.10, wage inflation mostly positive
    Weight: 2.0x (DOUBLE)
    """
    min_len = min(len(unemployment) - 1, len(wage_inflation))
    if min_len < 10:
        return {
            "phillips_shape": 10.0,
            "phillips_positive": 10.0,
            "_phillips_corr": 0.0,
            "_wage_inflation_positive_pct": 0.0,
        }

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = float(
                np.corrcoef(unemployment[1 : min_len + 1], wage_inflation[:min_len])[
                    0, 1
                ]
            )
        if np.isnan(corr):
            corr = 0.0
    except (ValueError, RuntimeWarning):
        corr = 0.0

    # Correlation score (target r ~ -0.10, tolerance +/- 0.15)
    # Double weighted via 40 multiplier instead of 20
    corr_distance = abs(corr - TARGETS.phillips_corr_target)
    if corr_distance <= TARGETS.phillips_corr_tolerance:
        shape_score = 0.0
    else:
        shape_score = (corr_distance - TARGETS.phillips_corr_tolerance) * 40

    # Wage inflation positive percentage (double weighted via 50)
    positive_pct = float(np.mean(wage_inflation > 0))
    if positive_pct >= TARGETS.phillips_wage_infl_positive_pct:
        positive_score = 0.0
    else:
        positive_score = (TARGETS.phillips_wage_infl_positive_pct - positive_pct) * 50

    return {
        "phillips_shape": shape_score,
        "phillips_positive": positive_score,
        "_phillips_corr": corr,
        "_wage_inflation_positive_pct": positive_pct,
    }


def score_okun_curve(unemployment: np.ndarray, gdp: np.ndarray) -> dict[str, float]:
    """
    Score Okun curve against book targets.

    Target: STRONG negative correlation r < -0.70 (this is the main issue)
    Weight: 2.0x (DOUBLE, CRITICAL)
    """
    if len(unemployment) < 10 or len(gdp) < 10:
        return {"okun_shape": 100.0, "_okun_corr": 0.0}

    # Calculate growth rates
    unemp_growth = np.diff(unemployment) / np.maximum(unemployment[:-1], 1e-10)
    gdp_growth = np.diff(gdp) / np.maximum(gdp[:-1], 1e-10)

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = float(np.corrcoef(unemp_growth, gdp_growth)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    except (ValueError, RuntimeWarning):
        corr = 0.0

    # Target: strong negative correlation r < -0.70
    # Heavy penalty for weak/positive correlation (DOUBLE WEIGHTED via 100)
    if corr < TARGETS.okun_corr_threshold:
        shape_score = 0.0
    else:
        shape_score = (corr - TARGETS.okun_corr_threshold) * 100

    # Extra penalty if correlation is positive (wrong sign)
    if corr > 0:
        shape_score += corr * 50

    return {
        "okun_shape": shape_score,
        "_okun_corr": corr,
    }


def score_beveridge_curve(
    unemployment: np.ndarray, vacancy_rate: np.ndarray
) -> dict[str, float]:
    """
    Score Beveridge curve against book targets.

    Target: Weak negative correlation r ~ -0.27, vacancy rate 8-20%
    Weight: 2.0x (DOUBLE)
    """
    if len(unemployment) < 10 or len(vacancy_rate) < 10:
        return {
            "beveridge_shape": 10.0,
            "beveridge_vacancy_range": 10.0,
            "_beveridge_corr": 0.0,
            "_vacancy_rate_mean": 0.0,
        }

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = float(np.corrcoef(unemployment, vacancy_rate)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    except (ValueError, RuntimeWarning):
        corr = 0.0

    # Correlation score (target r ~ -0.27, tolerance +/- 0.15)
    # Double weighted via 40
    corr_distance = abs(corr - TARGETS.beveridge_corr_target)
    if corr_distance <= TARGETS.beveridge_corr_tolerance:
        shape_score = 0.0
    else:
        shape_score = (corr_distance - TARGETS.beveridge_corr_tolerance) * 40

    # Vacancy rate range score
    mean_vr = float(np.mean(vacancy_rate))
    if TARGETS.beveridge_vacancy_min <= mean_vr <= TARGETS.beveridge_vacancy_max:
        vr_score = 0.0
    else:
        dist = min(
            abs(mean_vr - TARGETS.beveridge_vacancy_min),
            abs(mean_vr - TARGETS.beveridge_vacancy_max),
        )
        vr_score = dist * 100

    return {
        "beveridge_shape": shape_score,
        "beveridge_vacancy_range": vr_score,
        "_beveridge_corr": corr,
        "_vacancy_rate_mean": mean_vr,
    }


def score_firm_size_distribution(final_production: np.ndarray) -> dict[str, float]:
    """
    Score firm size distribution against book targets.

    Target: Highly right-skewed, >90% produce <3 units
    """
    if len(final_production) == 0:
        return {
            "firm_size_dist": 10.0,
            "production_level": 10.0,
            "_firm_pct_below_3": 0.0,
            "_production_mean": 0.0,
        }

    pct_below_3 = float(np.mean(final_production < TARGETS.firm_size_mean_max))
    mean_prod = float(np.mean(final_production))

    # Percentage below 3 score
    if pct_below_3 >= TARGETS.firm_size_pct_below_3:
        pct_score = 0.0
    else:
        pct_score = (TARGETS.firm_size_pct_below_3 - pct_below_3) * 20

    # Mean production score
    if mean_prod < TARGETS.firm_size_mean_max:
        mean_score = 0.0
    else:
        mean_score = (mean_prod - TARGETS.firm_size_mean_max) * 10

    return {
        "firm_size_dist": pct_score,
        "production_level": mean_score,
        "_firm_pct_below_3": pct_below_3,
        "_production_mean": mean_prod,
    }


def compute_all_scores(
    unemployment: np.ndarray,
    inflation: np.ndarray,
    gdp: np.ndarray,
    avg_productivity: np.ndarray,
    avg_employed_wage: np.ndarray,
    avg_price: np.ndarray,
    total_vacancies: np.ndarray,
    n_households: int,
    final_production: np.ndarray,
    burn_in: int = 500,
    destroyed: bool = False,
) -> dict[str, float]:
    """
    Compute all scoring metrics for a simulation run.

    Returns dictionary with individual scores and total score.
    Lower scores are better (0 = perfect match).
    """
    scores: dict[str, float] = {}

    # Collapse penalty
    if destroyed:
        scores["collapse_penalty"] = 1000.0
        scores["total"] = 1000.0
        return scores
    else:
        scores["collapse_penalty"] = 0.0

    # IMPORTANT: Index GDP to period 0 BEFORE applying burn-in
    # This matches the baseline scenario: gdp_indexed = gdp / gdp[0] * 100
    # Then apply burn-in to the indexed series
    gdp_indexed_full = gdp / np.maximum(gdp[0], 1e-10) * 100
    log_gdp_full = np.log(np.maximum(gdp_indexed_full, 1e-10))

    # Apply burn-in to all series
    unemployment_trimmed = unemployment[burn_in:]
    inflation_trimmed = inflation[burn_in:]
    log_gdp = log_gdp_full[burn_in:]  # Log GDP after burn-in (indexed to period 0)
    gdp_trimmed = gdp[burn_in:]  # Raw GDP for growth calculations
    avg_productivity_trimmed = avg_productivity[burn_in:]
    avg_wage_trimmed = avg_employed_wage[burn_in:]
    avg_price_trimmed = avg_price[burn_in:]
    vacancy_trimmed = total_vacancies[burn_in:]

    # Calculate productivity/real wage ratio
    real_wage = np.where(
        avg_price_trimmed > 0, avg_wage_trimmed / avg_price_trimmed, 0.0
    )
    prod_wage_ratio = np.where(real_wage > 0, avg_productivity_trimmed / real_wage, 0.0)

    # Calculate wage inflation
    wage_inflation = np.diff(avg_wage_trimmed) / np.maximum(
        avg_wage_trimmed[:-1], 1e-10
    )

    # Calculate vacancy rate
    vacancy_rate = vacancy_trimmed / n_households

    # Score all 8 metrics
    scores.update(score_log_gdp(log_gdp))
    scores.update(score_unemployment(unemployment_trimmed))
    scores.update(score_inflation(inflation_trimmed))
    scores.update(score_prod_wage_ratio(prod_wage_ratio))
    scores.update(score_phillips_curve(unemployment_trimmed, wage_inflation))
    scores.update(score_okun_curve(unemployment_trimmed, gdp_trimmed))
    scores.update(score_beveridge_curve(unemployment_trimmed, vacancy_rate))
    scores.update(score_firm_size_distribution(final_production))

    # Compute total (exclude debug fields starting with "_")
    scores["total"] = sum(v for k, v in scores.items() if not k.startswith("_"))

    return scores


# ==============================================================================
# CHECKPOINT SYSTEM
# ==============================================================================


def resolve_checkpoint_path(filepath: str) -> str:
    """
    Resolve checkpoint path, searching in OUTPUT_DIR if not found.

    Priority:
    1. If filepath exists as given, use it
    2. If filepath exists in OUTPUT_DIR, use that
    3. Otherwise, return filepath in OUTPUT_DIR (for new checkpoints)
    """
    if os.path.exists(filepath):
        return filepath
    # Try OUTPUT_DIR
    in_output_dir = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
    if os.path.exists(in_output_dir):
        return in_output_dir
    # Return path in OUTPUT_DIR for new checkpoints
    return in_output_dir


class CheckpointManager:
    """Manages checkpointing for calibration runs."""

    def __init__(self, filepath: str, resume: bool = False):
        self.filepath = resolve_checkpoint_path(filepath)
        self.resume = resume
        self.data = self._load_or_create()
        self._configs_evaluated: set[str] = set()
        self._rebuild_evaluated_set()

    def _load_or_create(self) -> dict:
        """Load existing checkpoint or create new one."""
        if self.resume and os.path.exists(self.filepath):
            print(f"Resuming from checkpoint: {self.filepath}")
            with open(self.filepath) as f:
                return json.load(f)
        if self.resume and not os.path.exists(self.filepath):
            print(f"WARNING: Checkpoint not found: {self.filepath}")
            print("Starting fresh calibration instead.")
        elif os.path.exists(self.filepath):
            print("Starting fresh (overwriting existing checkpoint)")
        return {
            "version": "1.0",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_configs_evaluated": 0,
                "stage": "starting",
                "n_seeds": 3,
                "n_periods": 1000,
                "burn_in": 500,
            },
            "fixed_params": {},
            "sensitivity_results": {},
            "grid_results": [],
            "bo_results": {"iterations": [], "best_params": None, "best_score": None},
            "top_configurations": [],
        }

    def _rebuild_evaluated_set(self):
        """Rebuild set of evaluated config hashes for fast lookup."""
        for result in self.data.get("grid_results", []):
            self._configs_evaluated.add(result["config_hash"])

    def config_hash(self, params: dict) -> str:
        """Generate deterministic hash for configuration."""
        sorted_items = sorted((str(k), str(v)) for k, v in params.items())
        return hashlib.md5(str(sorted_items).encode()).hexdigest()[:16]

    def is_evaluated(self, params: dict) -> bool:
        """Check if configuration was already evaluated."""
        return self.config_hash(params) in self._configs_evaluated

    def add_result(
        self,
        params: dict,
        scores: dict,
        seed_scores: list[float],
    ):
        """Add evaluation result."""
        result = {
            "config_hash": self.config_hash(params),
            "params": params,
            "scores": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in scores.items()
            },
            "seed_scores": seed_scores,
            "timestamp": datetime.now().isoformat(),
        }
        self.data["grid_results"].append(result)
        self._configs_evaluated.add(result["config_hash"])
        self.data["metadata"]["total_configs_evaluated"] = len(
            self.data["grid_results"]
        )

    def set_stage(self, stage: str):
        """Update current stage."""
        self.data["metadata"]["stage"] = stage

    def set_sensitivity_results(self, results: dict):
        """Store sensitivity analysis results."""
        self.data["sensitivity_results"] = results

    def save(self):
        """Save checkpoint to disk (atomic write)."""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        temp_path = f"{self.filepath}.tmp"
        with open(temp_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(temp_path, self.filepath)

    def should_checkpoint(self, interval: int = 100) -> bool:
        """Check if it's time to checkpoint."""
        return self.data["metadata"]["total_configs_evaluated"] % interval == 0

    def get_top_configs(self, n: int = 20) -> list[dict]:
        """Get top n configurations by total score."""
        sorted_results = sorted(
            self.data["grid_results"], key=lambda x: x["scores"]["total"]
        )
        return sorted_results[:n]


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================

# Fixed parameters (from book Section 3.9.1)
# Note: Low-impact params at the bottom may be swept in local sensitivity phase
FIXED_PARAMS = {
    "n_firms": 100,
    "n_households": 500,
    "n_banks": 10,
    "h_rho": 0.10,
    "h_xi": 0.05,
    "h_phi": 0.10,
    "h_eta": 0.10,
    "max_M": 4,
    "max_H": 2,
    "max_Z": 2,
    "min_wage_rev_period": 4,
    "theta": 8,
    "beta": 2.50,
    "delta": 0.10,
    "r_bar": 0.02,
    "cap_factor": 2.00,
    "labor_productivity": 0.50,
    "loan_priority_method": "by_leverage",
    "wage_offer_init": 1.0,  # Baseline for offset-based params
    # Low-impact sensitive params fixed to best values (from collapse rate analysis)
    # These had <2% collapse rate spread in grid search data
    "v": 0.10,
    "savings_init": 3,
    "new_firm_scale_factor": 0.8,
    "price_cut_allow_increase": False,
    "loanbook_clear_on_repay": True,
    "firing_method": "random",
    "fragility_cap_method": "none",
}


def apply_config_offsets(config: dict) -> dict:
    """
    Convert offset-based params (price_init_offset, min_wage_ratio) to actual values.

    This centralizes the logic for converting relative params to absolute values:
    - price_init_offset -> price_init = wage_offer_init + offset
    - min_wage_ratio -> min_wage = wage_offer_init * ratio
    """
    config = config.copy()
    wage_base = config.get("wage_offer_init", 1.0)
    if "price_init_offset" in config:
        config["price_init"] = wage_base + config.pop("price_init_offset")
    if "min_wage_ratio" in config:
        config["min_wage"] = wage_base * config.pop("min_wage_ratio")
    return config


def apply_unemployment_smoothing(
    unemployment_raw: np.ndarray, method: str
) -> np.ndarray:
    """
    Apply moving average smoothing if method is 'simple_ma'.

    Parameters
    ----------
    unemployment_raw : np.ndarray
        Raw unemployment rate data
    method : str
        Smoothing method - 'simple_ma' applies 4-period moving average, otherwise no smoothing

    Returns
    -------
    np.ndarray
        Smoothed (or original) unemployment data
    """
    if method == "simple_ma" and len(unemployment_raw) >= 4:
        window = 4
        kernel = np.ones(window) / window
        smoothed = np.convolve(unemployment_raw, kernel, mode="valid")
        return np.concatenate([unemployment_raw[: window - 1], smoothed])
    return unemployment_raw


def run_single_simulation(
    params: dict[str, Any],
    seed: int,
    n_periods: int = 1000,
    burn_in: int = 500,
) -> dict[str, Any]:
    """
    Run a single simulation with given parameters and seed.

    Returns metrics dictionary.
    """
    # Build full config
    config = {**FIXED_PARAMS, **params}
    config["seed"] = seed
    config["n_periods"] = n_periods
    config["logging"] = {"default_level": "ERROR"}
    config = apply_config_offsets(config)

    try:
        sim = bam.Simulation.init(**config)
        results = sim.run(
            collect={
                "Producer": ["production", "labor_productivity"],
                "Worker": ["wage", "employed"],
                "Employer": ["n_vacancies"],
                "Economy": True,
                "aggregate": None,
            }
        )

        # Extract data
        unemployment_raw = np.array(results.economy_data["unemployment_rate"])
        inflation = np.array(results.economy_data["inflation"])
        avg_price = np.array(results.economy_data["avg_price"])

        # Apply smoothing based on config setting
        unemployment = apply_unemployment_smoothing(
            unemployment_raw, sim.config.unemployment_calc_method
        )

        # Production and GDP
        production = np.array(results.role_data["Producer"]["production"])
        gdp = np.sum(production, axis=1)

        # Productivity
        productivity = np.array(results.role_data["Producer"]["labor_productivity"])
        avg_productivity = np.mean(productivity, axis=1)

        # Wages (employed workers only)
        wages = np.array(results.role_data["Worker"]["wage"])
        employed = np.array(results.role_data["Worker"]["employed"])
        employed_wages_sum = np.sum(np.where(employed, wages, 0.0), axis=1)
        employed_count = np.sum(employed, axis=1)
        avg_employed_wage = np.where(
            employed_count > 0, employed_wages_sum / employed_count, 0.0
        )

        # Vacancies
        n_vacancies = np.array(results.role_data["Employer"]["n_vacancies"])
        total_vacancies = np.sum(n_vacancies, axis=1)

        # Final production distribution
        final_production = production[-1]

        # Check for collapse
        destroyed = sim.ec.destroyed if hasattr(sim.ec, "destroyed") else False

        # Compute scores
        scores = compute_all_scores(
            unemployment=unemployment,
            inflation=inflation,
            gdp=gdp,
            avg_productivity=avg_productivity,
            avg_employed_wage=avg_employed_wage,
            avg_price=avg_price,
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

    Returns (averaged_scores, list of total scores per seed).
    """
    all_scores: list[dict] = []

    for seed in range(n_seeds):
        scores = run_single_simulation(params, seed, n_periods, burn_in)
        all_scores.append(scores)

    # Average scores across seeds
    avg_scores: dict[str, float] = {}
    # Collect all keys from all scores (some seeds may crash with fewer keys)
    all_keys = set()
    for s in all_scores:
        all_keys.update(s.keys())

    for key in all_keys:
        # Skip non-numeric keys (e.g., "_error" which is a string)
        if key == "_error":
            # Preserve error message if any seed crashed
            errors = [s.get("_error") for s in all_scores if "_error" in s]
            if errors:
                avg_scores["_error"] = errors[0]  # Keep first error
            continue

        values = [s.get(key, 0) for s in all_scores]
        avg_scores[key] = float(np.mean(values))
        if not key.startswith("_"):
            avg_scores[f"{key}_std"] = float(np.std(values))

    seed_totals = [s["total"] for s in all_scores]

    return avg_scores, seed_totals


# Worker function for parallel execution
def _run_config_worker(args: tuple) -> tuple[dict, dict, list]:
    """Worker function for parallel config evaluation."""
    params, n_seeds, n_periods, burn_in = args
    scores, seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
    return params, scores, seed_totals


# ==============================================================================
# PARAMETER GRIDS
# ==============================================================================

# High-impact parameters (always sweep)
HIGH_IMPACT_GRID = {
    "zero_production_bankrupt": [True, False],
    "contract_poisson_mean": [0, 10],
    "unemployment_calc_method": ["raw", "simple_ma"],
    "unemployment_calc_after": [
        "firms_fire_workers",
        "workers_update_contracts",
        "spawn_replacement_banks",
    ],
    "production_init": [0.5, 1.0, 1.5, 2.0],
}

# Sensitive interdependent parameters (offset-based approach)
INTERDEPENDENT_GRID = {
    "price_init_offset": [0.1, 0.3, 0.5],  # price_init = wage_offer_init + offset
    "min_wage_ratio": [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],  # min_wage = wage_offer_init * ratio
}

# Sensitive parameters - REDUCED to only high-impact based on collapse rate analysis
# Only params with >15% collapse rate spread are kept in the grid
# 6 low-impact params are in LOCAL_SWEEP_PARAMS for later sweep
# 1 param (v) is fixed in FIXED_PARAMS
SENSITIVITY_GRID = {
    "net_worth_init": [1, 5, 10],  # 22% collapse spread - HIGH IMPACT
    "new_firm_price_markup": [1.0, 1.25],  # 16% collapse spread - HIGH IMPACT
    "equity_base_init": [5, 10],  # 2nd highest impact score, test best values
}

# Test mode grids (reduced for quick validation)
TEST_HIGH_IMPACT_GRID = {
    "zero_production_bankrupt": [True],
    "contract_poisson_mean": [0, 10],
    "unemployment_calc_method": ["raw"],
    "unemployment_calc_after": ["spawn_replacement_banks"],
    "production_init": [1.0],
}

TEST_INTERDEPENDENT_GRID = {
    "price_init_offset": [0.3],
    "min_wage_ratio": [0.7],
}

TEST_SENSITIVITY_GRID = {
    "net_worth_init": [5, 10],
    "new_firm_price_markup": [1.0, 1.25],
    "equity_base_init": [5, 10],
}

# Demo mode grids (medium-sized with progress bar demonstration, ~10 minutes)
DEMO_HIGH_IMPACT_GRID = {
    "zero_production_bankrupt": [True, False],
    "contract_poisson_mean": [0, 5, 10],
    "unemployment_calc_method": ["raw", "simple_ma"],
    "unemployment_calc_after": ["spawn_replacement_banks"],
    "production_init": [1.0],
}

DEMO_INTERDEPENDENT_GRID = {
    "price_init_offset": [0.2, 0.4],
    "min_wage_ratio": [0.6, 0.8],
}

DEMO_SENSITIVITY_GRID = {
    "net_worth_init": [1, 5, 10],
    "new_firm_price_markup": [1.0, 1.25],
    "equity_base_init": [5, 10],
}


def generate_combinations(grid: dict[str, list]) -> list[dict]:
    """Generate all parameter combinations from a grid."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=False)))
    return combinations


# ==============================================================================
# CALIBRATION STAGES
# ==============================================================================


def run_sensitivity_analysis(
    checkpoint: CheckpointManager,
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    sensitivity_grid: dict | None = None,
) -> dict[str, dict]:
    """
    Stage 1: One-at-a-time sensitivity analysis.

    Tests each lower-impact parameter individually to identify most impactful.
    """
    if sensitivity_grid is None:
        sensitivity_grid = SENSITIVITY_GRID

    print("\n" + "=" * 60)
    print("STAGE 1: Sensitivity Analysis")
    print("=" * 60)
    checkpoint.set_stage("sensitivity_analysis")
    checkpoint.save()

    # Count total configurations to evaluate
    total_configs = 1 + sum(
        len(values) for values in sensitivity_grid.values()
    )  # +1 for baseline

    # Run baseline first
    print("\nRunning baseline configuration...")
    baseline_scores, _ = run_ensemble({}, n_seeds, n_periods, burn_in)
    baseline_total = baseline_scores["total"]
    print(f"Baseline total score: {baseline_total:.2f}\n")

    results = {}

    with ProgressTracker(
        total=total_configs - 1,  # Exclude baseline (already done)
        stage="Sensitivity",
        update_interval=1,
    ) as progress:
        for param_name, param_values in sensitivity_grid.items():
            impacts = []

            for value in param_values:
                test_params = {param_name: value}
                scores, _ = run_ensemble(test_params, n_seeds, n_periods, burn_in)
                impact = abs(scores["total"] - baseline_total)
                impacts.append(
                    {
                        "value": value,
                        "score": scores["total"],
                        "impact": impact,
                    }
                )
                progress.update(
                    score=scores["total"],
                    params={param_name: value},
                )

            best = min(impacts, key=lambda x: x["score"])
            max_impact = max(i["impact"] for i in impacts)

            results[param_name] = {
                "max_impact": max_impact,
                "best_value": best["value"],
                "best_score": best["score"],
                "all_values": impacts,
            }

    # Rank by impact
    ranked = sorted(results.items(), key=lambda x: x[1]["max_impact"], reverse=True)
    print("\n=== Sensitivity Ranking ===")
    for rank, (param, info) in enumerate(ranked, 1):
        print(
            f"  {rank}. {param}: impact={info['max_impact']:.2f}, best={info['best_value']}"
        )

    checkpoint.set_sensitivity_results(results)
    checkpoint.save()

    return results


def run_grid_search(
    checkpoint: CheckpointManager,
    sensitivity_results: dict[str, dict],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    top_k_sensitive: int = 2,
    high_impact_grid: dict | None = None,
    interdependent_grid: dict | None = None,
) -> list[dict]:
    """
    Stage 2: Full factorial grid search.

    Combines high-impact + interdependent + sensitive parameters from grids.
    """
    if high_impact_grid is None:
        high_impact_grid = HIGH_IMPACT_GRID
    if interdependent_grid is None:
        interdependent_grid = INTERDEPENDENT_GRID

    print("\n" + "=" * 60)
    print("STAGE 2: Grid Search")
    print("=" * 60)
    checkpoint.set_stage("grid_search")
    checkpoint.save()

    # Select sensitive parameters
    # Since SENSITIVITY_GRID has been curated to only high-impact params,
    # we use all of them directly (ignoring old sensitivity_results ranking)
    sensitive_params = list(SENSITIVITY_GRID.keys())
    print(f"Sensitive params from grid: {sensitive_params}")

    # Build combined grid
    combined_grid = {**high_impact_grid, **interdependent_grid}
    for param in sensitive_params:
        combined_grid[param] = SENSITIVITY_GRID[param]

    # Generate combinations
    all_combinations = generate_combinations(combined_grid)
    total_combos = len(all_combinations)
    print(f"\nTotal combinations: {total_combos}")
    print(f"Seeds per combination: {n_seeds}")
    print(f"Total simulations: {total_combos * n_seeds}")

    # Filter already evaluated (checkpoint resume support)
    remaining = [c for c in all_combinations if not checkpoint.is_evaluated(c)]
    already_done = total_combos - len(remaining)
    print(f"Already evaluated: {already_done}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All configurations already evaluated!")
        return checkpoint.get_top_configs(100)

    # Get best score from checkpoint if resuming
    initial_best = float("inf")
    if already_done > 0:
        top_so_far = checkpoint.get_top_configs(1)
        if top_so_far:
            initial_best = top_so_far[0]["scores"]["total"]
            print(f"Best score from checkpoint: {initial_best:.2f}")

    print(f"\nStarting grid search with {n_workers} worker(s)...\n")

    with ProgressTracker(
        total=len(remaining),
        stage="Grid Search",
        update_interval=10 if n_workers > 1 else 5,
    ) as progress:
        # Initialize with checkpoint's best score
        if initial_best < float("inf"):
            progress.stats.best_score = initial_best

        if n_workers > 1:
            # Parallel execution
            args_list = [(c, n_seeds, n_periods, burn_in) for c in remaining]

            with Pool(processes=n_workers) as pool:
                for params, scores, seed_totals in pool.imap_unordered(
                    _run_config_worker, args_list
                ):
                    checkpoint.add_result(params, scores, seed_totals)
                    progress.update(
                        score=scores["total"],
                        params=params,
                    )

                    # Periodic checkpoint save
                    if checkpoint.should_checkpoint(50):
                        checkpoint.save()
        else:
            # Sequential execution
            for params in remaining:
                scores, seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
                checkpoint.add_result(params, scores, seed_totals)
                progress.update(
                    score=scores["total"],
                    params=params,
                )

                # Periodic checkpoint save
                if checkpoint.should_checkpoint(20):
                    checkpoint.save()

    checkpoint.save()

    return checkpoint.get_top_configs(100)


# Parameters to sweep in local sensitivity analysis (excluded from main grid)
LOCAL_SWEEP_PARAMS = {
    "v": [0.06, 0.10],
    "savings_init": [2, 3, 4],
    "new_firm_scale_factor": [0.6, 0.8],
    "price_cut_allow_increase": [True, False],
    "loanbook_clear_on_repay": [True, False],
    "firing_method": ["random", "expensive"],
    "fragility_cap_method": ["credit_demand", "none"],
}


def run_local_sensitivity_sweep(
    checkpoint: CheckpointManager,
    top_configs: list[dict],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    top_k: int = 300,
) -> list[dict]:
    """
    Stage 2.5: Local sensitivity sweep.

    For each top configuration, tests alternative values of low-impact
    parameters to validate fixed values and find potential improvements.
    """
    # Check if already completed (skip on resume)
    if checkpoint.data["metadata"].get("local_sweep_completed", False):
        print("\n" + "=" * 60)
        print("STAGE 2.5: Local Sensitivity Sweep (SKIPPED - already completed)")
        print("=" * 60)
        return checkpoint.get_top_configs(100)

    print("\n" + "=" * 60)
    print("STAGE 2.5: Local Sensitivity Sweep")
    print("=" * 60)
    checkpoint.set_stage("local_sensitivity_sweep")

    # Take top k configurations
    configs_to_sweep = top_configs[:top_k]
    print(f"Sweeping {len(configs_to_sweep)} top configurations")
    print(f"Parameters to sweep: {list(LOCAL_SWEEP_PARAMS.keys())}")

    # Count total variations per config
    total_variations = sum(len(v) for v in LOCAL_SWEEP_PARAMS.values())
    print(f"Variations per config: {total_variations}")

    # Generate all sweep combinations
    sweep_configs = []
    for base_config in configs_to_sweep:
        base_params = base_config["params"].copy()
        # Test each param's alternative values one at a time
        for param, values in LOCAL_SWEEP_PARAMS.items():
            for value in values:
                # Skip if this is already the value in base config
                if base_params.get(param) == value:
                    continue
                # Create variant
                variant = base_params.copy()
                variant[param] = value
                if not checkpoint.is_evaluated(variant):
                    sweep_configs.append(variant)

    total_sweeps = len(sweep_configs)
    print(f"Total sweep configurations: {total_sweeps}")

    if not sweep_configs:
        print("All sweep configurations already evaluated!")
        return checkpoint.get_top_configs(100)

    # Run with workers
    progress = ProgressTracker(
        total=total_sweeps,
        stage="Local Sweep",
        update_interval=10,
    )

    # Get initial best from grid search
    if top_configs:
        progress.stats.best_score = top_configs[0]["scores"]["total"]

    if n_workers > 1:
        # Parallel execution using Pool.imap_unordered (consistent with grid search)
        args_list = [(c, n_seeds, n_periods, burn_in) for c in sweep_configs]

        with Pool(processes=n_workers) as pool:
            for params, scores, seed_totals in pool.imap_unordered(
                _run_config_worker, args_list
            ):
                checkpoint.add_result(params, scores, seed_totals)
                progress.update(score=scores["total"], params=params)

                if checkpoint.should_checkpoint(50):
                    checkpoint.save()
    else:
        for config in sweep_configs:
            scores, seed_totals = run_ensemble(config, n_seeds, n_periods, burn_in)
            checkpoint.add_result(config, scores, seed_totals)
            progress.update(score=scores["total"], params=config)
            if checkpoint.should_checkpoint(20):
                checkpoint.save()

    # Mark local sweep as completed so it's skipped on resume
    checkpoint.data["metadata"]["local_sweep_completed"] = True
    checkpoint.save()

    return checkpoint.get_top_configs(100)


def run_bayesian_optimization(
    checkpoint: CheckpointManager,
    top_configs: list[dict],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_calls: int = 300,
) -> dict:
    """
    Stage 3: Bayesian optimization to refine search.

    Uses scikit-optimize to explore continuous parameter space.
    """
    if not SKOPT_AVAILABLE:
        print("\nSkipping Bayesian Optimization (scikit-optimize not installed)")
        return {}

    print("\n" + "=" * 60)
    print("STAGE 3: Bayesian Optimization")
    print("=" * 60)
    checkpoint.set_stage("bayesian_optimization")

    # Define search space
    space = [
        Categorical([True, False], name="zero_production_bankrupt"),
        Integer(0, 15, name="contract_poisson_mean"),
        Categorical(["raw", "simple_ma"], name="unemployment_calc_method"),
        Categorical(
            [
                "firms_fire_workers",
                "workers_update_contracts",
                "spawn_replacement_banks",
            ],
            name="unemployment_calc_after",
        ),
        Real(0.3, 3.0, name="production_init"),
        Real(0.05, 0.6, name="price_init_offset"),
        Real(0.4, 1.0, name="min_wage_ratio"),
        # Sensitive parameters
        Integer(1, 10, name="net_worth_init"),
        Real(1.0, 1.5, name="new_firm_price_markup"),
        Integer(5, 15, name="equity_base_init"),
    ]

    def decode_params(x):
        return {
            "zero_production_bankrupt": x[0],
            "contract_poisson_mean": x[1],
            "unemployment_calc_method": x[2],
            "unemployment_calc_after": x[3],
            "production_init": x[4],
            "price_init_offset": x[5],
            "min_wage_ratio": x[6],
            "net_worth_init": x[7],
            "new_firm_price_markup": x[8],
            "equity_base_init": x[9],
        }

    # Create progress tracker for BO
    progress = ProgressTracker(
        total=n_calls,
        stage="Bayesian Opt",
        update_interval=5,
    )

    # Get best score from grid search
    if top_configs:
        progress.stats.best_score = top_configs[0]["scores"]["total"]

    def objective(x):
        params = decode_params(x)
        scores, seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
        checkpoint.add_result(params, scores, seed_totals)
        progress.update(score=scores["total"], params=params)
        if checkpoint.should_checkpoint(50):
            checkpoint.save()
        return scores["total"]

    # Initialize with top configs
    x0 = []
    y0 = []
    for cfg in top_configs[: min(20, len(top_configs))]:
        p = cfg["params"]
        try:
            x = [
                p.get("zero_production_bankrupt", True),
                p.get("contract_poisson_mean", 10),
                p.get("unemployment_calc_method", "simple_ma"),
                p.get("unemployment_calc_after", "spawn_replacement_banks"),
                p.get("production_init", 1.0),
                p.get("price_init_offset", 0.3),
                p.get("min_wage_ratio", 0.7),
                p.get("net_worth_init", 5),
                p.get("new_firm_price_markup", 1.0),
                p.get("equity_base_init", 10),
            ]
            x0.append(x)
            y0.append(cfg["scores"]["total"])
        except (KeyError, TypeError):
            continue

    print(f"\nInitializing with {len(x0)} configurations from grid search")
    print(f"Running {n_calls} BO iterations...\n")

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        x0=x0 if x0 else None,
        y0=y0 if y0 else None,
        random_state=42,
        verbose=False,  # Disable skopt's verbose output, we have our own
    )

    progress.close()

    best_params = decode_params(result.x)
    best_score = result.fun

    checkpoint.data["bo_results"] = {
        "best_params": best_params,
        "best_score": best_score,
        "n_iterations": n_calls,
    }
    checkpoint.save()

    print(f"\nBO complete. Best score: {best_score:.2f}")
    print(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_score}


# ==============================================================================
# VISUALIZATION
# ==============================================================================


def visualize_configuration(
    params: dict[str, Any],
    seed: int = 0,
    n_periods: int = 1000,
    burn_in: int = 500,
    title: str = "BAM Calibration Results",
    save_path: str | None = None,
) -> None:
    """
    Generate 8-panel visualization matching baseline scenario format.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return

    print(f"\nRunning visualization for configuration (seed={seed})...")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Build full config
    config = {**FIXED_PARAMS, **params}
    config["seed"] = seed
    config["n_periods"] = n_periods
    config["logging"] = {"default_level": "ERROR"}

    # Handle offset params
    config = apply_config_offsets(config)

    sim = bam.Simulation.init(**config)
    results = sim.run(
        collect={
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
            "Employer": ["n_vacancies"],
            "Economy": True,
            "aggregate": None,
        }
    )

    # Extract data
    unemployment_raw = np.array(results.economy_data["unemployment_rate"])
    inflation = np.array(results.economy_data["inflation"])
    avg_price = np.array(results.economy_data["avg_price"])

    # Apply smoothing based on config setting
    unemployment = apply_unemployment_smoothing(
        unemployment_raw, sim.config.unemployment_calc_method
    )

    # Production and GDP
    production = np.array(results.role_data["Producer"]["production"])
    gdp = np.sum(production, axis=1)

    # Productivity
    productivity = np.array(results.role_data["Producer"]["labor_productivity"])
    avg_productivity = np.mean(productivity, axis=1)

    # Wages (employed workers only)
    wages = np.array(results.role_data["Worker"]["wage"])
    employed = np.array(results.role_data["Worker"]["employed"])
    employed_wages_sum = np.sum(np.where(employed, wages, 0.0), axis=1)
    employed_count = np.sum(employed, axis=1)
    avg_employed_wage = np.where(
        employed_count > 0, employed_wages_sum / employed_count, 0.0
    )

    # Vacancies
    n_vacancies = np.array(results.role_data["Employer"]["n_vacancies"])
    total_vacancies = np.sum(n_vacancies, axis=1)

    # Final production distribution
    final_production = production[-1]

    # Calculate derived metrics
    gdp_indexed = gdp / gdp[0] * 100
    log_gdp = np.log(gdp_indexed)

    real_wage = np.where(avg_price > 0, avg_employed_wage / avg_price, 0.0)
    prod_wage_ratio = np.where(real_wage > 0, avg_productivity / real_wage, 0.0)

    wage_inflation = np.diff(avg_employed_wage) / np.maximum(
        avg_employed_wage[:-1], 1e-10
    )
    gdp_growth = np.diff(gdp) / np.maximum(gdp[:-1], 1e-10)
    vacancy_rate = total_vacancies / config["n_households"]

    # For curves: use RAW unemployment to capture true economic relationships
    unemployment_growth_raw = np.diff(unemployment_raw) / np.maximum(
        unemployment_raw[:-1], 1e-10
    )

    # Apply burn-in
    periods = np.arange(burn_in, len(gdp))
    log_gdp_trimmed = log_gdp[burn_in:]
    unemployment_pct = unemployment[burn_in:] * 100  # SA for time series visualization
    inflation_pct = inflation[burn_in:] * 100
    prod_wage_ratio_trimmed = prod_wage_ratio[burn_in:]

    # Curves use RAW unemployment
    unemployment_phillips = unemployment_raw[burn_in:]
    wage_inflation_trimmed = wage_inflation[burn_in - 1 :]
    gdp_growth_trimmed = gdp_growth[burn_in - 1 :]
    unemployment_growth_trimmed = unemployment_growth_raw[burn_in - 1 :]
    vacancy_rate_trimmed = vacancy_rate[burn_in:]
    unemployment_beveridge = unemployment_raw[burn_in:]

    # Calculate correlations
    phillips_corr = np.corrcoef(unemployment_phillips, wage_inflation_trimmed)[0, 1]
    okun_corr = np.corrcoef(unemployment_growth_trimmed, gdp_growth_trimmed)[0, 1]
    beveridge_corr = np.corrcoef(unemployment_beveridge, vacancy_rate_trimmed)[0, 1]

    # Compute overall score
    scores = compute_all_scores(
        unemployment=unemployment,
        inflation=inflation,
        gdp=gdp,
        avg_productivity=avg_productivity,
        avg_employed_wage=avg_employed_wage,
        avg_price=avg_price,
        total_vacancies=total_vacancies,
        n_households=config["n_households"],
        final_production=final_production,
        burn_in=burn_in,
    )

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(f"{title} (Score: {scores['total']:.2f})", fontsize=16, y=0.995)

    # Colors (matching baseline scenario)
    colors = {
        "gdp": "#2E86AB",
        "unemployment": "#A23B72",
        "inflation": "#F18F01",
        "prod_wage": "#6A994E",
    }

    # Panel (0,0): Log Real GDP
    axes[0, 0].plot(periods, log_gdp_trimmed, linewidth=1.5, color=colors["gdp"])
    axes[0, 0].axhline(
        5.40, color="green", linestyle="--", alpha=0.5, label="Target min"
    )
    axes[0, 0].axhline(
        5.50, color="green", linestyle="--", alpha=0.5, label="Target max"
    )
    axes[0, 0].set_title("Log Real GDP", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Time (periods)")
    axes[0, 0].set_ylabel("Log Output")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=8)

    # Panel (0,1): Unemployment Rate
    axes[0, 1].plot(
        periods, unemployment_pct, linewidth=1.5, color=colors["unemployment"]
    )
    axes[0, 1].axhline(
        5, color="green", linestyle="--", alpha=0.5, label="Target min (5%)"
    )
    axes[0, 1].axhline(
        7, color="green", linestyle="--", alpha=0.5, label="Target max (7%)"
    )
    axes[0, 1].set_title("Unemployment Rate (%)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Time (periods)")
    axes[0, 1].set_ylabel("Unemployment Rate (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    # Panel (1,0): Annual Inflation Rate
    axes[1, 0].plot(periods, inflation_pct, linewidth=1.5, color=colors["inflation"])
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 0].axhline(
        3, color="green", linestyle="--", alpha=0.5, label="Target min (3%)"
    )
    axes[1, 0].axhline(
        7, color="green", linestyle="--", alpha=0.5, label="Target max (7%)"
    )
    axes[1, 0].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Time (periods)")
    axes[1, 0].set_ylabel("Inflation Rate (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(fontsize=8)

    # Panel (1,1): Productivity / Real Wage Ratio
    axes[1, 1].plot(
        periods, prod_wage_ratio_trimmed, linewidth=1.5, color=colors["prod_wage"]
    )
    axes[1, 1].axhline(
        0.35, color="red", linestyle=":", alpha=0.7, label="Book value (~0.35)"
    )
    axes[1, 1].set_title(
        "Productivity / Real Wage Ratio", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Time (periods)")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend(fontsize=8)

    # Panel (2,0): Phillips Curve
    axes[2, 0].scatter(
        unemployment_phillips,
        wage_inflation_trimmed,
        s=10,
        alpha=0.5,
        color=colors["gdp"],
    )
    # Add reference lines: our results and target
    # Slope = r * (std_y / std_x)
    x_mean, y_mean = np.mean(unemployment_phillips), np.mean(wage_inflation_trimmed)
    x_std, y_std = np.std(unemployment_phillips), np.std(wage_inflation_trimmed)
    if x_std > 0:
        x_range = np.array([unemployment_phillips.min(), unemployment_phillips.max()])
        # Our results regression line
        actual_slope = phillips_corr * (y_std / x_std)
        y_actual = y_mean + actual_slope * (x_range - x_mean)
        axes[2, 0].plot(
            x_range,
            y_actual,
            color=colors["gdp"],
            linewidth=2,
            alpha=0.8,
            label=f"Actual (r={phillips_corr:.2f})",
        )
        # Target reference line (r = -0.10)
        target_slope = -0.10 * (y_std / x_std)
        y_target = y_mean + target_slope * (x_range - x_mean)
        axes[2, 0].plot(
            x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.10)"
        )
    axes[2, 0].set_title("Phillips Curve", fontsize=12, fontweight="bold")
    axes[2, 0].set_xlabel("Unemployment Rate")
    axes[2, 0].set_ylabel("Wage Inflation Rate")
    axes[2, 0].grid(True, linestyle="--", alpha=0.6)
    axes[2, 0].legend(fontsize=8, loc="upper right")

    # Panel (2,1): Okun Curve
    axes[2, 1].scatter(
        unemployment_growth_trimmed,
        gdp_growth_trimmed,
        s=10,
        alpha=0.5,
        color=colors["unemployment"],
    )
    # Add reference lines: our results and target
    x_mean, y_mean = np.mean(unemployment_growth_trimmed), np.mean(gdp_growth_trimmed)
    x_std, y_std = np.std(unemployment_growth_trimmed), np.std(gdp_growth_trimmed)
    if x_std > 0:
        x_range = np.array(
            [unemployment_growth_trimmed.min(), unemployment_growth_trimmed.max()]
        )
        # Our results regression line
        actual_slope = okun_corr * (y_std / x_std)
        y_actual = y_mean + actual_slope * (x_range - x_mean)
        axes[2, 1].plot(
            x_range,
            y_actual,
            color=colors["unemployment"],
            linewidth=2,
            alpha=0.8,
            label=f"Actual (r={okun_corr:.2f})",
        )
        # Target reference line (r = -0.70)
        target_slope = -0.70 * (y_std / x_std)
        y_target = y_mean + target_slope * (x_range - x_mean)
        axes[2, 1].plot(
            x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.70)"
        )
    axes[2, 1].set_title("Okun Curve", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Unemployment Growth Rate")
    axes[2, 1].set_ylabel("Output Growth Rate")
    axes[2, 1].grid(True, linestyle="--", alpha=0.6)
    axes[2, 1].legend(fontsize=8, loc="upper right")

    # Panel (3,0): Beveridge Curve
    axes[3, 0].scatter(
        unemployment_beveridge,
        vacancy_rate_trimmed,
        s=10,
        alpha=0.5,
        color=colors["inflation"],
    )
    # Add reference lines: our results and target
    x_mean, y_mean = np.mean(unemployment_beveridge), np.mean(vacancy_rate_trimmed)
    x_std, y_std = np.std(unemployment_beveridge), np.std(vacancy_rate_trimmed)
    if x_std > 0:
        x_range = np.array([unemployment_beveridge.min(), unemployment_beveridge.max()])
        # Our results regression line
        actual_slope = beveridge_corr * (y_std / x_std)
        y_actual = y_mean + actual_slope * (x_range - x_mean)
        axes[3, 0].plot(
            x_range,
            y_actual,
            color=colors["inflation"],
            linewidth=2,
            alpha=0.8,
            label=f"Actual (r={beveridge_corr:.2f})",
        )
        # Target reference line (r = -0.27)
        target_slope = -0.27 * (y_std / x_std)
        y_target = y_mean + target_slope * (x_range - x_mean)
        axes[3, 0].plot(
            x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.27)"
        )
    axes[3, 0].set_title("Beveridge Curve", fontsize=12, fontweight="bold")
    axes[3, 0].set_xlabel("Unemployment Rate")
    axes[3, 0].set_ylabel("Vacancy Rate")
    axes[3, 0].grid(True, linestyle="--", alpha=0.6)
    axes[3, 0].legend(fontsize=8, loc="upper right")

    # Panel (3,1): Firm Size Distribution
    pct_below_3 = np.sum(final_production < 3) / len(final_production) * 100
    axes[3, 1].hist(
        final_production,
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color=colors["prod_wage"],
    )
    # Add vertical line at production=3 (target threshold)
    axes[3, 1].axvline(
        x=3,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Target threshold (90% below)",
    )
    # Add ideal exponential-like distribution overlay
    x_ideal = np.linspace(0, final_production.max(), 100)
    # Exponential decay: most firms produce very little
    ideal_dist = len(final_production) * 0.8 * np.exp(-x_ideal / 1.0)
    axes[3, 1].plot(
        x_ideal, ideal_dist, "g-", linewidth=2, alpha=0.7, label="Ideal (right-skewed)"
    )
    axes[3, 1].set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
    axes[3, 1].set_xlabel("Production")
    axes[3, 1].set_ylabel("Frequency")
    axes[3, 1].grid(True, linestyle="--", alpha=0.6)
    axes[3, 1].legend(fontsize=8, loc="upper right")
    axes[3, 1].text(
        0.98,
        0.60,
        f"{pct_below_3:.0f}% below prod=3\n(Book target: 90%)",
        transform=axes[3, 1].transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (Post Burn-In)")
    print("=" * 60)
    print(
        f"\nLog GDP: Mean={np.mean(log_gdp_trimmed):.2f}, Std={np.std(log_gdp_trimmed):.3f}"
    )
    print(
        f"Unemployment: Mean={np.mean(unemployment_pct):.1f}%, "
        f"Std={np.std(unemployment_pct):.1f}%, "
        f"Range=[{np.min(unemployment_pct):.1f}%, {np.max(unemployment_pct):.1f}%]"
    )
    print(
        f"Inflation: Mean={np.mean(inflation_pct):.1f}%, "
        f"Positive={np.mean(inflation_pct > 0) * 100:.0f}%"
    )
    print("\nCurve Correlations vs Book Targets:")
    print(f"  Phillips:  r={phillips_corr:+.2f} (book target: -0.10)")
    print(f"  Okun:      r={okun_corr:+.2f} (book target: < -0.70)")
    print(f"  Beveridge: r={beveridge_corr:+.2f} (book target: -0.27)")
    print(f"\nFirm Size: {pct_below_3:.0f}% below production 3 (book target: 90%)")
    print(f"\nTotal Score: {scores['total']:.2f}")
    print("=" * 60)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def run_calibration(
    checkpoint_path: str = "calibration_checkpoint.json",
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    test_mode: bool = False,
    skip_bo: bool = False,
    resume: bool = False,
):
    """
    Run full calibration pipeline.
    """
    print("=" * 60)
    print("BAM Engine Parameter Calibration")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Seeds per config: {n_seeds}")
    print(f"  Periods: {n_periods}")
    print(f"  Burn-in: {burn_in}")
    print(f"  Workers: {n_workers}")
    print(f"  Test mode: {test_mode}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Select grids based on mode
    if test_mode:
        high_impact_grid = TEST_HIGH_IMPACT_GRID
        interdependent_grid = TEST_INTERDEPENDENT_GRID
        sensitivity_grid = TEST_SENSITIVITY_GRID
        skip_bo = True
    else:
        high_impact_grid = HIGH_IMPACT_GRID
        interdependent_grid = INTERDEPENDENT_GRID
        sensitivity_grid = SENSITIVITY_GRID

    # Initialize checkpoint
    checkpoint = CheckpointManager(checkpoint_path, resume=resume)
    checkpoint.data["metadata"]["n_seeds"] = n_seeds
    checkpoint.data["metadata"]["n_periods"] = n_periods
    checkpoint.data["metadata"]["burn_in"] = burn_in
    checkpoint.data["fixed_params"] = FIXED_PARAMS

    start_time = time.time()

    # Stage 1: Sensitivity Analysis
    if not checkpoint.data["sensitivity_results"]:
        sensitivity_results = run_sensitivity_analysis(
            checkpoint=checkpoint,
            n_seeds=n_seeds,
            n_periods=n_periods,
            burn_in=burn_in,
            n_workers=n_workers,
            sensitivity_grid=sensitivity_grid,
        )
    else:
        print("\nUsing cached sensitivity results from checkpoint")
        sensitivity_results = checkpoint.data["sensitivity_results"]

    # Stage 2: Grid Search
    top_configs = run_grid_search(
        checkpoint=checkpoint,
        sensitivity_results=sensitivity_results,
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        high_impact_grid=high_impact_grid,
        interdependent_grid=interdependent_grid,
    )

    # Stage 2.5: Local Sensitivity Sweep
    top_configs = run_local_sensitivity_sweep(
        checkpoint=checkpoint,
        top_configs=top_configs,
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        top_k=300 if not test_mode else 10,
    )

    # Stage 3: Bayesian Optimization
    if not skip_bo and SKOPT_AVAILABLE:
        run_bayesian_optimization(
            checkpoint=checkpoint,
            top_configs=top_configs,
            n_seeds=n_seeds,
            n_periods=n_periods,
            burn_in=burn_in,
            n_calls=300 if not test_mode else 10,
        )
    else:
        if skip_bo:
            print("\nSkipping Bayesian Optimization (--skip-bo flag)")
        elif not SKOPT_AVAILABLE:
            print("\nSkipping Bayesian Optimization (scikit-optimize not installed)")
            print("Install with: pip install scikit-optimize")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(
        f"Total configs evaluated: {checkpoint.data['metadata']['total_configs_evaluated']}"
    )

    # Show top configurations
    top_5 = checkpoint.get_top_configs(5)
    print("\n=== Top 5 Configurations ===")
    for rank, cfg in enumerate(top_5, 1):
        print(f"\nRank {rank} (Score: {cfg['scores']['total']:.2f}):")
        for key, value in cfg["params"].items():
            print(f"  {key}: {value}")
        print(f"  Okun corr: {cfg['scores'].get('_okun_corr', 'N/A')}")
        print(
            f"  Unemployment mean: {cfg['scores'].get('_unemployment_mean', 0) * 100:.1f}%"
        )

    # Save final checkpoint
    checkpoint.data["top_configurations"] = top_5
    checkpoint.save()
    print(f"\nResults saved to {checkpoint_path}")

    # Visualize best config
    if top_5 and MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualization for best configuration...")
        visualize_configuration(
            params=top_5[0]["params"],
            seed=0,
            n_periods=n_periods,
            burn_in=burn_in,
            title="BAM Calibration - Best Configuration",
            save_path=os.path.join(OUTPUT_DIR, "best_config_visualization.png"),
        )

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="BAM Engine Parameter Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (quick validation, ~1-2 minutes)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint instead of starting fresh",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Number of parallel workers (default: {max(1, cpu_count() - 1)})",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        type=int,
        const=1,
        metavar="RANK",
        help="Visualize configuration from checkpoint (default: rank 1 = best)",
    )
    parser.add_argument(
        "--skip-bo",
        action="store_true",
        help="Skip Bayesian optimization stage",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per configuration (default: 3)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Number of simulation periods (default: 1000)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run single simulation with default parameters for comparison",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (~10 minutes) to demonstrate progress monitoring",
    )

    args = parser.parse_args()

    # Visualization mode
    if args.visualize:
        checkpoint_path = os.path.join(OUTPUT_DIR, "calibration_checkpoint.json")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

        with open(checkpoint_path) as f:
            data = json.load(f)

        top_configs = sorted(data["grid_results"], key=lambda x: x["scores"]["total"])
        rank = args.visualize
        if rank > len(top_configs):
            print(f"Error: Only {len(top_configs)} configurations available")
            sys.exit(1)

        cfg = top_configs[rank - 1]
        print(f"Visualizing rank {rank} configuration")
        visualize_configuration(
            params=cfg["params"],
            seed=0,
            n_periods=args.periods,
            burn_in=args.periods // 2,
            title=f"BAM Calibration - Rank {rank}",
        )
        return

    # Baseline mode: single run with default parameters
    if args.baseline:
        print("=" * 60)
        print("BASELINE MODE: Running with defaults.yml parameters")
        print("=" * 60)
        print("\nThis runs a single simulation with default parameters")
        print("to compare against the baseline scenario script.\n")
        n_periods = args.periods
        burn_in = n_periods // 2
        visualize_configuration(
            params={},  # Empty params = use defaults from FIXED_PARAMS only
            seed=0,
            n_periods=n_periods,
            burn_in=burn_in,
            title="BAM Calibration - Baseline (defaults.yml)",
            save_path=os.path.join(
                OUTPUT_DIR, "baseline_calibration_visualization.png"
            ),
        )
        return

    # Demo mode: medium-sized run to demonstrate progress monitoring
    if args.demo:
        print("=" * 60)
        print("DEMO MODE: Progress Monitoring Demonstration")
        print("=" * 60)
        print("\nThis mode runs a medium-sized calibration (~10 minutes)")
        print("to demonstrate the progress monitoring features.\n")

        # Demo settings: enough configs to see progress bars in action
        n_seeds = 2
        n_periods = 300
        burn_in = 150
        n_workers = 2  # Parallel with 2 workers

        # Initialize checkpoint (use temporary file for demo)
        checkpoint = CheckpointManager(os.path.join(OUTPUT_DIR, "demo_checkpoint.json"))
        checkpoint.data["metadata"]["n_seeds"] = n_seeds
        checkpoint.data["metadata"]["n_periods"] = n_periods
        checkpoint.data["metadata"]["burn_in"] = burn_in
        checkpoint.data["fixed_params"] = FIXED_PARAMS

        # Run sensitivity analysis with demo grid
        sensitivity_results = run_sensitivity_analysis(
            checkpoint=checkpoint,
            n_seeds=n_seeds,
            n_periods=n_periods,
            burn_in=burn_in,
            n_workers=n_workers,
            sensitivity_grid=DEMO_SENSITIVITY_GRID,
        )

        # Run grid search with demo grids
        top_configs = run_grid_search(
            checkpoint=checkpoint,
            sensitivity_results=sensitivity_results,
            n_seeds=n_seeds,
            n_periods=n_periods,
            burn_in=burn_in,
            n_workers=n_workers,
            top_k_sensitive=2,  # Only top 2 for demo
            high_impact_grid=DEMO_HIGH_IMPACT_GRID,
            interdependent_grid=DEMO_INTERDEPENDENT_GRID,
        )

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print(
            f"Total configs evaluated: {checkpoint.data['metadata']['total_configs_evaluated']}"
        )
        if top_configs:
            print(f"Best score: {top_configs[0]['scores']['total']:.2f}")

        # Demo checkpoint is kept for inspection/resume
        print(
            f"\nDemo checkpoint saved to: {os.path.join(OUTPUT_DIR, 'demo_checkpoint.json')}"
        )
        return

    # Test mode settings
    if args.test:
        n_seeds = 1
        n_periods = 100
        burn_in = 50
        n_workers = 1
    else:
        n_seeds = args.seeds
        n_periods = args.periods
        burn_in = n_periods // 2
        n_workers = args.workers

    # Run calibration
    run_calibration(
        checkpoint_path=os.path.join(OUTPUT_DIR, "calibration_checkpoint.json"),
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        n_workers=n_workers,
        test_mode=args.test,
        skip_bo=args.skip_bo,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
