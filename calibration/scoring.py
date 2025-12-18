"""
Scoring System
==============

Scoring functions for evaluating BAM simulations against target macroeconomic
patterns from Delli Gatti et al. (2011), Section 3.9.1.

The scoring system uses a lower-is-better approach where:
- Score of 0 = perfect match to target
- Higher scores = worse calibration

Scoring priorities (highest to lowest):
1. Real wage magnitude - most important for income distribution
2. Curve correlations - key macroeconomic relationships
3. Deflation periods - characteristic model dynamics
4. Log real GDP - output level and stability
5. Unemployment stability - labor market dynamics
6. Inflation stability - price dynamics
7. Firm size distribution - heterogeneity patterns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ==============================================================================
# SCORE CONFIGURATION
# ==============================================================================
# Centralized configuration for all scoring targets and weights.
# Avoid referring to specific values in comments - use descriptive language.


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for a single scoring metric."""

    weight: float
    description: str


# Score weights - higher weight = more important
SCORE_WEIGHTS = {
    # Highest priority: Real wage magnitude
    "real_wage_level": 200.0,
    "real_wage_stability": 100.0,
    # High priority: Curve correlations
    "okun_shape": 25.0,  # Reduced weight (was 100)
    "beveridge_shape": 40.0,
    "beveridge_vacancy_range": 40.0,
    "phillips_shape": 40.0,
    "phillips_positive": 25.0,
    # Medium priority: Deflation and GDP
    "deflation_penalty": 50.0,
    "log_gdp_level": 50.0,
    "log_gdp_stability": 50.0,
    # Lower priority: Stability metrics
    "unemployment_range": 10.0,
    "unemployment_mean": 30.0,
    "unemployment_stability": 30.0,
    "inflation_mean": 20.0,
    "inflation_stability": 20.0,
    # Lowest priority: Firm size
    "firm_size_dist": 20.0,
    "production_level": 10.0,
    # Collapse penalty
    "collapse_penalty": 1000.0,
}

# Score targets - values the model should achieve
SCORE_TARGETS = {
    # Real wage targets
    "real_wage_min": 0.30,
    "real_wage_max": 0.40,
    "real_wage_std_max": 0.03,
    # Log GDP targets
    "log_gdp_min": 5.40,
    "log_gdp_max": 5.50,
    "log_gdp_std_max": 0.03,
    # Unemployment targets
    "unemployment_range_min": 0.02,
    "unemployment_range_max": 0.12,
    "unemployment_mean_min": 0.05,
    "unemployment_mean_max": 0.07,
    "unemployment_std_max": 0.05,
    # Inflation targets
    "inflation_mean_min": -0.05,
    "inflation_mean_max": 0.11,
    # Deflation targets (percentage of deflation episodes)
    "deflation_pct_min": 5.0,
    "deflation_pct_max": 10.0,
    "deflation_pct_excessive": 15.0,
    # Curve correlation targets
    "okun_threshold": -0.70,
    "beveridge_target": -0.27,
    "beveridge_tolerance": 0.10,
    "beveridge_vacancy_min": 0.08,
    "beveridge_vacancy_max": 0.20,
    "phillips_target": -0.10,
    "phillips_tolerance": 0.10,
    "phillips_wage_infl_positive_pct": 0.85,
    # Firm size targets
    "firm_pct_below_threshold": 0.90,
    "firm_mean_max": 3.0,
}


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================


def score_real_wage(real_wage: np.ndarray) -> dict[str, float]:
    """
    Score absolute real wage level against target range.

    This is the highest priority metric. The real wage represents the
    purchasing power of workers and is key to understanding income
    distribution dynamics in the model. The productivity line is stable
    at the labor_productivity parameter, so only the real wage needs
    scoring to ensure the productivity-wage relationship is correct.

    The target range represents empirically observed real wage levels
    from the reference model implementation where the ratio between
    productivity and real wage is stable.

    Parameters
    ----------
    real_wage : np.ndarray
        Real wage time series (nominal_wage / price_level).

    Returns
    -------
    dict
        Score components and diagnostic values (prefixed with '_').
    """
    # Filter valid values
    valid = real_wage[real_wage > 0]
    if len(valid) < 100:
        return {
            "real_wage_level": 10.0,
            "real_wage_stability": 5.0,
            "_real_wage_mean": 0.0,
            "_real_wage_std": 1.0,
        }

    mean_val = float(np.nanmean(valid))
    std_val = float(np.nanstd(valid))

    # Level score: penalty for being outside target range
    target_min = SCORE_TARGETS["real_wage_min"]
    target_max = SCORE_TARGETS["real_wage_max"]

    if target_min <= mean_val <= target_max:
        level_score = 0.0
    else:
        dist = min(abs(mean_val - target_min), abs(mean_val - target_max))
        level_score = dist * SCORE_WEIGHTS["real_wage_level"]

    # Stability score: penalty for excessive volatility
    std_max = SCORE_TARGETS["real_wage_std_max"]
    if std_val <= std_max:
        stability_score = 0.0
    else:
        stability_score = (std_val - std_max) * SCORE_WEIGHTS["real_wage_stability"]

    return {
        "real_wage_level": level_score,
        "real_wage_stability": stability_score,
        "_real_wage_mean": mean_val,
        "_real_wage_std": std_val,
    }


def score_log_gdp(log_gdp: np.ndarray) -> dict[str, float]:
    """
    Score log GDP level and stability against target range.

    Log GDP measures the overall output level of the economy. The target
    range ensures the model produces economically meaningful output levels.
    Stability scoring ensures the model exhibits realistic business cycle
    volatility without excessive fluctuations.

    Parameters
    ----------
    log_gdp : np.ndarray
        Log of indexed real GDP time series.

    Returns
    -------
    dict
        Score components and diagnostic values.
    """
    mean_val = float(np.nanmean(log_gdp))
    std_val = float(np.nanstd(log_gdp))

    # Level score
    target_min = SCORE_TARGETS["log_gdp_min"]
    target_max = SCORE_TARGETS["log_gdp_max"]

    if target_min <= mean_val <= target_max:
        level_score = 0.0
    else:
        dist = min(abs(mean_val - target_min), abs(mean_val - target_max))
        level_score = dist * SCORE_WEIGHTS["log_gdp_level"]

    # Stability score
    std_max = SCORE_TARGETS["log_gdp_std_max"]
    if std_val <= std_max:
        stability_score = 0.0
    else:
        stability_score = (std_val - std_max) * SCORE_WEIGHTS["log_gdp_stability"]

    return {
        "log_gdp_level": level_score,
        "log_gdp_stability": stability_score,
        "_log_gdp_mean": mean_val,
        "_log_gdp_std": std_val,
    }


def score_unemployment(unemployment: np.ndarray) -> dict[str, float]:
    """
    Score unemployment rate against target range and stability.

    Unemployment is a key labor market indicator. The scoring evaluates:
    - Range: whether unemployment stays within economically realistic bounds
    - Mean: whether average unemployment is in the target range
    - Stability: whether unemployment volatility is reasonable

    Lower volatility is preferred as it indicates more stable labor markets.

    Parameters
    ----------
    unemployment : np.ndarray
        Unemployment rate time series (smoothed).

    Returns
    -------
    dict
        Score components and diagnostic values.
    """
    mean_val = float(np.nanmean(unemployment))
    std_val = float(np.nanstd(unemployment))
    min_val = float(np.nanmin(unemployment))
    max_val = float(np.nanmax(unemployment))

    # Range violations
    range_min = SCORE_TARGETS["unemployment_range_min"]
    range_max = SCORE_TARGETS["unemployment_range_max"]
    violations = np.sum(unemployment < range_min) + np.sum(unemployment > range_max)
    range_score = (violations / len(unemployment)) * SCORE_WEIGHTS["unemployment_range"]

    # Mean score
    mean_min = SCORE_TARGETS["unemployment_mean_min"]
    mean_max = SCORE_TARGETS["unemployment_mean_max"]
    if mean_min <= mean_val <= mean_max:
        mean_score = 0.0
    else:
        dist = min(abs(mean_val - mean_min), abs(mean_val - mean_max))
        mean_score = dist * SCORE_WEIGHTS["unemployment_mean"]

    # Stability score
    std_max = SCORE_TARGETS["unemployment_std_max"]
    if std_val <= std_max:
        stability_score = 0.0
    else:
        stability_score = (std_val - std_max) * SCORE_WEIGHTS["unemployment_stability"]

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
    Score inflation rate against target range and stability.

    Inflation measures price dynamics in the economy. The scoring evaluates:
    - Mean: whether average inflation is in a realistic range
    - Stability: lower volatility preferred, but occasional spikes allowed

    The model should exhibit mostly positive inflation with occasional
    deflationary episodes during economic downturns.

    Parameters
    ----------
    inflation : np.ndarray
        Inflation rate time series.

    Returns
    -------
    dict
        Score components and diagnostic values.
    """
    mean_val = float(np.nanmean(inflation))
    std_val = float(np.nanstd(inflation))

    # Mean score
    mean_min = SCORE_TARGETS["inflation_mean_min"]
    mean_max = SCORE_TARGETS["inflation_mean_max"]
    if mean_min <= mean_val <= mean_max:
        mean_score = 0.0
    else:
        dist = min(abs(mean_val - mean_min), abs(mean_val - mean_max))
        mean_score = dist * SCORE_WEIGHTS["inflation_mean"]

    # Stability score (mild penalty for high volatility)
    # Allow for spikes, so use a gentler penalty
    stability_score = std_val * SCORE_WEIGHTS["inflation_stability"] * 0.5

    return {
        "inflation_mean": mean_score,
        "inflation_stability": stability_score,
        "_inflation_mean": mean_val,
        "_inflation_std": std_val,
    }


def score_deflation(inflation: np.ndarray) -> dict[str, float]:
    """
    Score deflation episodes against target range.

    The BAM model should exhibit occasional deflationary periods as a
    characteristic feature of endogenous business cycle dynamics. Too
    few deflation episodes suggests the model lacks realistic downturns,
    while too many suggests excessive instability.

    The target is for a moderate percentage of periods to show deflation,
    with penalties for being outside this range.

    Parameters
    ----------
    inflation : np.ndarray
        Inflation rate time series.

    Returns
    -------
    dict
        Deflation penalty score and diagnostic values.
    """
    positive_pct = float(np.mean(inflation > 0))
    deflation_pct = (1 - positive_pct) * 100  # As percentage

    target_min = SCORE_TARGETS["deflation_pct_min"]
    target_max = SCORE_TARGETS["deflation_pct_max"]
    excessive = SCORE_TARGETS["deflation_pct_excessive"]
    weight = SCORE_WEIGHTS["deflation_penalty"]

    if target_min <= deflation_pct <= target_max:
        # Sweet spot - no penalty
        penalty = 0.0
    elif deflation_pct < target_min:
        # Not enough deflation
        penalty = (target_min - deflation_pct) * weight * 0.1
    elif deflation_pct > excessive:
        # Too much deflation
        penalty = (deflation_pct - excessive) * weight * 0.1
    else:
        # Between target_max and excessive - small penalty
        penalty = (deflation_pct - target_max) * weight * 0.05

    return {
        "deflation_penalty": penalty,
        "_deflation_pct": deflation_pct,
        "_inflation_positive_pct": positive_pct,
    }


def score_okun_curve(
    unemployment_growth: np.ndarray, gdp_growth: np.ndarray
) -> dict[str, float]:
    """
    Score Okun curve correlation against target threshold.

    Okun's Law describes the negative relationship between changes in
    unemployment and GDP growth. A strong negative correlation indicates
    the model correctly captures this fundamental macroeconomic relationship
    where economic expansions reduce unemployment and contractions increase it.

    The weight is reduced compared to other correlations since Okun's Law
    can be difficult to achieve perfectly in agent-based models.

    Parameters
    ----------
    unemployment_growth : np.ndarray
        Unemployment growth rate time series.
    gdp_growth : np.ndarray
        GDP growth rate time series.

    Returns
    -------
    dict
        Okun correlation score and diagnostic values.
    """
    if len(unemployment_growth) < 10 or len(gdp_growth) < 10:
        return {"okun_shape": 100.0, "_okun_corr": 0.0}

    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = float(np.corrcoef(unemployment_growth, gdp_growth)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    except (ValueError, RuntimeWarning):
        corr = 0.0

    threshold = SCORE_TARGETS["okun_threshold"]
    weight = SCORE_WEIGHTS["okun_shape"]

    if corr < threshold:
        shape_score = 0.0
    else:
        shape_score = (corr - threshold) * weight

    # Extra penalty for positive correlation (wrong sign)
    if corr > 0:
        shape_score += corr * weight * 0.5

    return {
        "okun_shape": shape_score,
        "_okun_corr": corr,
    }


def score_beveridge_curve(
    unemployment: np.ndarray, vacancy_rate: np.ndarray
) -> dict[str, float]:
    """
    Score Beveridge curve correlation and vacancy rate range.

    The Beveridge curve describes the negative relationship between
    unemployment and job vacancy rates. When the labor market is tight
    (low unemployment), vacancies are high; when slack (high unemployment),
    vacancies are low. This captures labor market matching efficiency.

    Parameters
    ----------
    unemployment : np.ndarray
        Unemployment rate time series (raw, not smoothed).
    vacancy_rate : np.ndarray
        Vacancy rate time series.

    Returns
    -------
    dict
        Beveridge correlation and vacancy range scores with diagnostics.
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

    # Correlation score
    target = SCORE_TARGETS["beveridge_target"]
    tolerance = SCORE_TARGETS["beveridge_tolerance"]
    weight = SCORE_WEIGHTS["beveridge_shape"]

    corr_distance = abs(corr - target)
    if corr_distance <= tolerance:
        shape_score = 0.0
    else:
        shape_score = (corr_distance - tolerance) * weight

    # Vacancy rate range score
    mean_vr = float(np.mean(vacancy_rate))
    vr_min = SCORE_TARGETS["beveridge_vacancy_min"]
    vr_max = SCORE_TARGETS["beveridge_vacancy_max"]
    vr_weight = SCORE_WEIGHTS["beveridge_vacancy_range"]

    if vr_min <= mean_vr <= vr_max:
        vr_score = 0.0
    else:
        dist = min(abs(mean_vr - vr_min), abs(mean_vr - vr_max))
        vr_score = dist * vr_weight

    return {
        "beveridge_shape": shape_score,
        "beveridge_vacancy_range": vr_score,
        "_beveridge_corr": corr,
        "_vacancy_rate_mean": mean_vr,
    }


def score_phillips_curve(
    unemployment: np.ndarray, wage_inflation: np.ndarray
) -> dict[str, float]:
    """
    Score Phillips curve correlation and wage inflation positivity.

    The Phillips curve describes the relationship between unemployment
    and wage inflation. In the model, this should show a weak negative
    correlation - when unemployment is high, wage growth is lower.

    Additionally, most wage inflation periods should be positive,
    indicating workers generally receive wage increases.

    Parameters
    ----------
    unemployment : np.ndarray
        Unemployment rate time series (raw, not smoothed).
    wage_inflation : np.ndarray
        Wage inflation rate time series.

    Returns
    -------
    dict
        Phillips correlation and wage positivity scores with diagnostics.
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

    # Correlation score
    target = SCORE_TARGETS["phillips_target"]
    tolerance = SCORE_TARGETS["phillips_tolerance"]
    weight = SCORE_WEIGHTS["phillips_shape"]

    corr_distance = abs(corr - target)
    if corr_distance <= tolerance:
        shape_score = 0.0
    else:
        shape_score = (corr_distance - tolerance) * weight

    # Wage inflation positive percentage
    positive_pct = float(np.mean(wage_inflation > 0))
    target_pct = SCORE_TARGETS["phillips_wage_infl_positive_pct"]
    pos_weight = SCORE_WEIGHTS["phillips_positive"]

    if positive_pct >= target_pct:
        positive_score = 0.0
    else:
        positive_score = (target_pct - positive_pct) * pos_weight

    return {
        "phillips_shape": shape_score,
        "phillips_positive": positive_score,
        "_phillips_corr": corr,
        "_wage_inflation_positive_pct": positive_pct,
    }


def score_firm_size_distribution(final_production: np.ndarray) -> dict[str, float]:
    """
    Score firm size distribution against target pattern.

    The BAM model should exhibit a highly right-skewed firm size
    distribution where most firms are small and few are large. This
    matches empirical observations of firm heterogeneity and is a
    characteristic emergent property of the model.

    Parameters
    ----------
    final_production : np.ndarray
        Production levels of all firms in the final period.

    Returns
    -------
    dict
        Firm size distribution scores and diagnostics.
    """
    if len(final_production) == 0:
        return {
            "firm_size_dist": 10.0,
            "production_level": 10.0,
            "_firm_pct_below_3": 0.0,
            "_production_mean": 0.0,
        }

    threshold = SCORE_TARGETS["firm_mean_max"]
    pct_below = float(np.mean(final_production < threshold))
    mean_prod = float(np.mean(final_production))

    # Percentage below threshold score
    target_pct = SCORE_TARGETS["firm_pct_below_threshold"]
    dist_weight = SCORE_WEIGHTS["firm_size_dist"]

    if pct_below >= target_pct:
        pct_score = 0.0
    else:
        pct_score = (target_pct - pct_below) * dist_weight

    # Mean production score
    prod_weight = SCORE_WEIGHTS["production_level"]
    if mean_prod < threshold:
        mean_score = 0.0
    else:
        mean_score = (mean_prod - threshold) * prod_weight

    return {
        "firm_size_dist": pct_score,
        "production_level": mean_score,
        "_firm_pct_below_3": pct_below,
        "_production_mean": mean_prod,
    }


# ==============================================================================
# AGGREGATE SCORING
# ==============================================================================


def compute_all_scores(
    unemployment: np.ndarray,
    unemployment_raw: np.ndarray,
    inflation: np.ndarray,
    gdp: np.ndarray,
    avg_productivity: np.ndarray,
    avg_employed_wage: np.ndarray,
    avg_price: np.ndarray,
    real_wage: np.ndarray,
    total_vacancies: np.ndarray,
    n_households: int,
    final_production: np.ndarray,
    burn_in: int = 500,
    destroyed: bool = False,
) -> dict[str, float]:
    """
    Compute all scoring metrics for a simulation run.

    This aggregates scores from all individual metrics into a single
    dictionary with a total score. Lower scores are better (0 = perfect).

    Parameters
    ----------
    unemployment : np.ndarray
        Smoothed unemployment rate time series.
    unemployment_raw : np.ndarray
        Raw unemployment rate (for curve correlations).
    inflation : np.ndarray
        Inflation rate time series.
    gdp : np.ndarray
        Real GDP time series.
    avg_productivity : np.ndarray
        Average labor productivity time series.
    avg_employed_wage : np.ndarray
        Average wage of employed workers time series.
    avg_price : np.ndarray
        Average price level time series.
    real_wage : np.ndarray
        Real wage time series (nominal_wage / price).
    total_vacancies : np.ndarray
        Total vacancies time series.
    n_households : int
        Number of households (for vacancy rate calculation).
    final_production : np.ndarray
        Production levels in final period.
    burn_in : int
        Number of initial periods to exclude.
    destroyed : bool
        Whether the economy collapsed during simulation.

    Returns
    -------
    dict
        All score components and total score.
    """
    scores: dict[str, float] = {}

    # Collapse penalty
    if destroyed:
        scores["collapse_penalty"] = SCORE_WEIGHTS["collapse_penalty"]
        scores["total"] = scores["collapse_penalty"]
        return scores
    else:
        scores["collapse_penalty"] = 0.0

    # Index GDP to period 0 before applying burn-in
    gdp_indexed = gdp / np.maximum(gdp[0], 1e-10) * 100
    log_gdp_full = np.log(np.maximum(gdp_indexed, 1e-10))

    # Apply burn-in to all series
    unemployment_trimmed = unemployment[burn_in:]
    unemployment_raw_trimmed = unemployment_raw[burn_in:]
    inflation_trimmed = inflation[burn_in:]
    log_gdp = log_gdp_full[burn_in:]
    gdp_trimmed = gdp[burn_in:]
    real_wage_trimmed = real_wage[burn_in:]
    avg_wage_trimmed = avg_employed_wage[burn_in:]
    vacancy_trimmed = total_vacancies[burn_in:]

    # Calculate derived metrics
    vacancy_rate = vacancy_trimmed / n_households

    # Growth rates (for Okun curve)
    gdp_growth = np.diff(gdp_trimmed) / np.maximum(gdp_trimmed[:-1], 1e-10)
    unemp_growth = np.diff(unemployment_raw_trimmed) / np.maximum(
        unemployment_raw_trimmed[:-1], 1e-10
    )

    # Wage inflation (for Phillips curve)
    wage_inflation = np.diff(avg_wage_trimmed) / np.maximum(
        avg_wage_trimmed[:-1], 1e-10
    )

    # Score all metrics
    scores.update(score_real_wage(real_wage_trimmed))
    scores.update(score_log_gdp(log_gdp))
    scores.update(score_unemployment(unemployment_trimmed))
    scores.update(score_inflation(inflation_trimmed))
    scores.update(score_deflation(inflation_trimmed))
    scores.update(score_okun_curve(unemp_growth, gdp_growth))
    scores.update(score_beveridge_curve(unemployment_raw_trimmed, vacancy_rate))
    scores.update(score_phillips_curve(unemployment_raw_trimmed, wage_inflation))
    scores.update(score_firm_size_distribution(final_production))

    # Compute total (exclude diagnostic fields starting with "_")
    scores["total"] = sum(v for k, v in scores.items() if not k.startswith("_"))

    return scores
