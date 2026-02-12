"""Growth+ scenario definition (Section 3.9.2).

This module defines the Growth+ validation scenario from Delli Gatti et al. (2011).
It contains the metrics dataclass, computation function, and scenario configuration.

The Growth+ scenario extends the baseline with endogenous productivity growth
through R&D investment.

For visualization, see viz.py (in this package).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats
from scipy.signal import find_peaks

import bamengine as bam
from bamengine import SimulationResults, ops
from bamengine.utils import EPS
from validation.scenarios._utils import adjust_burn_in, filter_outliers_iqr
from validation.types import CheckType, MetricFormat, MetricGroup, MetricSpec, Scenario

# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class GrowthPlusMetrics:
    """All computed metrics from a Growth+ simulation run.

    Includes time series, curve correlations, distribution metrics,
    growth-specific metrics, and financial dynamics.
    """

    # Time series (full, for visualization)
    unemployment: NDArray[np.floating]
    inflation: NDArray[np.floating]
    log_gdp: NDArray[np.floating]
    real_wage: NDArray[np.floating]
    avg_productivity: NDArray[np.floating]
    vacancy_rate: NDArray[np.floating]

    # Curve data
    wage_inflation: NDArray[np.floating]
    gdp_growth: NDArray[np.floating]
    unemployment_growth: NDArray[np.floating]

    # Distribution data
    final_production: NDArray[np.floating]

    # Summary statistics
    unemployment_mean: float
    unemployment_std: float
    unemployment_max: float
    unemployment_pct_in_bounds: float
    inflation_mean: float
    inflation_std: float
    inflation_max: float
    inflation_min: float
    inflation_pct_in_bounds: float
    log_gdp_mean: float
    log_gdp_std: float
    real_wage_mean: float
    real_wage_std: float
    vacancy_rate_mean: float
    vacancy_rate_pct_in_bounds: float  # Percentage of periods in [0.08, 0.20]

    # Correlations
    phillips_corr: float
    okun_corr: float
    beveridge_corr: float

    # Distribution metrics
    firm_size_skewness: float
    firm_size_pct_below_threshold: float
    firm_size_tail_ratio: float  # max(production) / median(production)
    firm_size_pct_below_medium: float  # pct below medium threshold (100)

    # Growth-specific metrics
    productivity_growth_rate: NDArray[np.floating]
    productivity_trend_coefficient: float
    initial_productivity: float
    final_productivity: float
    total_productivity_growth: float
    real_wage_initial: float
    real_wage_final: float
    total_real_wage_growth: float
    log_gdp_trend_coefficient: float
    log_gdp_total_growth: float

    # Productivity-wage co-movement
    productivity_wage_correlation: float  # Detrended correlation
    wage_productivity_ratio_mean: float  # Mean of wage/productivity ratio
    wage_productivity_ratio_std: float  # Std of wage/productivity ratio

    # Financial dynamics
    n_firm_bankruptcies: NDArray[np.int_]
    n_bank_bankruptcies: NDArray[np.int_]
    real_interest_rate: NDArray[np.floating]
    avg_financial_fragility: NDArray[np.floating]
    price_ratio: NDArray[np.floating]
    price_dispersion: NDArray[np.floating]
    equity_dispersion: NDArray[np.floating]
    sales_dispersion: NDArray[np.floating]

    # Growth rate distributions
    output_growth_rates: NDArray[np.floating]
    networth_growth_rates: NDArray[np.floating]

    # Tiered distribution metrics
    output_growth_pct_within_tight: float
    output_growth_pct_within_normal: float
    output_growth_pct_outliers: float
    output_growth_tent_r2: float
    output_growth_positive_frac: float
    networth_growth_pct_within_tight: float
    networth_growth_pct_within_normal: float
    networth_growth_pct_outliers: float
    networth_growth_tent_r2: float

    # Recession detection
    recession_mask: NDArray[np.bool_]
    n_recessions: int
    avg_recession_length: float

    # Minsky classification
    minsky_hedge_pct: float
    minsky_speculative_pct: float
    minsky_ponzi_pct: float

    # Summary statistics
    bankruptcies_mean: float
    real_interest_rate_mean: float
    real_interest_rate_std: float
    real_interest_rate_pct_in_bounds: float
    avg_fragility_mean: float
    financial_fragility_cv: float  # Coefficient of variation (std/mean)
    fragility_gdp_correlation: float  # Detrended pro-cyclical correlation with GDP
    price_ratio_mean: float
    price_ratio_cv: float  # Coefficient of variation (std/mean)
    price_ratio_min: float  # Minimum value (floor check)
    price_ratio_p5: float  # 5th percentile (low-end behavior)
    price_ratio_gdp_correlation: (
        float  # Detrended counter-cyclical correlation with GDP
    )
    price_dispersion_mean: float
    price_dispersion_cv: float  # Coefficient of variation (std/mean)
    price_dispersion_gdp_correlation: (
        float  # Detrended pro-cyclical correlation with GDP
    )
    equity_dispersion_mean: float
    equity_dispersion_cv: float  # Coefficient of variation (std/mean)
    equity_dispersion_gdp_correlation: (
        float  # Detrended pro-cyclical correlation with GDP
    )
    sales_dispersion_mean: float
    sales_dispersion_cv: float  # Coefficient of variation (std/mean)
    sales_dispersion_gdp_correlation: (
        float  # Detrended pro-cyclical correlation with GDP
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _smooth_series(x: NDArray[np.floating], window: int = 5) -> NDArray[np.floating]:
    """Apply centered moving average smoothing."""
    if window < 1:
        return x
    if window == 1:
        return x.copy()

    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode="same")

    half = window // 2
    if half > 0:
        smoothed[:half] = x[:half]
        smoothed[-half:] = x[-half:]

    return smoothed


def _detect_recessions(
    log_gdp: NDArray[np.floating],
    smoothing_window: int = 5,
    peak_prominence: float = 0.03,
    peak_distance: int = 20,
    min_gap: int = 10,
    extension_after_trough: int = 10,
) -> NDArray[np.bool_]:
    """Detect recession episodes using peak-to-trough algorithm."""
    if len(log_gdp) < smoothing_window * 2:
        return np.zeros(len(log_gdp), dtype=bool)

    smoothed = _smooth_series(log_gdp, window=smoothing_window)

    peaks, _ = find_peaks(smoothed, prominence=peak_prominence, distance=peak_distance)
    troughs, _ = find_peaks(
        -smoothed, prominence=peak_prominence, distance=peak_distance
    )

    n_periods = len(log_gdp)
    recession_mask = np.zeros(n_periods, dtype=bool)

    for peak in peaks:
        future_troughs = troughs[troughs > peak]
        if len(future_troughs) > 0:
            trough = future_troughs[0]
            end = min(trough + extension_after_trough, n_periods)
            recession_mask[peak:end] = True

    # Bridge short gaps
    if np.any(recession_mask):
        padded = np.concatenate([[False], recession_mask, [False]])
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]

        for i in range(len(ends) - 1):
            gap = starts[i + 1] - ends[i]
            if gap < min_gap:
                recession_mask[ends[i] : starts[i + 1]] = True

    return recession_mask


def _count_recession_episodes(recession_mask: NDArray[np.bool_]) -> tuple[int, float]:
    """Count recession episodes and compute average length."""
    if not np.any(recession_mask):
        return 0, 0.0

    padded = np.concatenate([[False], recession_mask, [False]])
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]

    n_episodes = len(starts)
    if n_episodes == 0:
        return 0, 0.0

    lengths = ends - starts
    avg_length = float(np.mean(lengths))

    return n_episodes, avg_length


def _compute_pct_within_range(
    values: NDArray[np.floating], range_min: float, range_max: float
) -> float:
    """Compute percentage of values within a range."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return 0.0
    within = np.sum((valid >= range_min) & (valid <= range_max))
    return float(within / len(valid))


def _compute_detrended_correlation(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> float:
    """Compute correlation of linearly detrended series.

    Removes linear trend from each series before computing correlation,
    which measures cyclical co-movement rather than trend co-movement.
    """
    if len(x) < 10 or len(y) < 10:
        return 0.0
    t = np.arange(len(x))
    # Remove linear trend from each series
    try:
        x_trend = np.polyval(np.polyfit(t, x, 1), t)
        y_trend = np.polyval(np.polyfit(t, y, 1), t)
    except np.linalg.LinAlgError:
        return 0.0
    x_detrended = x - x_trend
    y_detrended = y - y_trend
    corr = np.corrcoef(x_detrended, y_detrended)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _compute_cv(series: NDArray[np.floating]) -> float:
    """Compute coefficient of variation (std / mean)."""
    mean_val = float(np.mean(series))
    return float(np.std(series) / max(mean_val, EPS))


def _compute_gdp_cyclicality(
    series: NDArray[np.floating], gdp: NDArray[np.floating]
) -> float:
    """Compute detrended correlation of smoothed series with smoothed GDP.

    Both ``series`` and ``gdp`` are smoothed internally so callers can pass
    raw (unsmoothed) arrays.
    """
    smoothed = _smooth_series(series)
    smoothed_gdp = _smooth_series(gdp)
    return _compute_detrended_correlation(smoothed, smoothed_gdp)


def _compute_tent_shape_r2(
    growth_rates: NDArray[np.floating], trim_pct: float = 0.10
) -> float:
    """Compute R-squared measuring Laplace tent-shape fit of growth rate distribution.

    A Laplace distribution appears as a straight line (tent shape) on a
    log-rank plot. This function measures how well the empirical distribution
    matches that linear pattern by fitting log(rank) ~ growth_rate on each
    side and averaging the R-squared values.

    The outermost ``trim_pct`` of points on each side are excluded because
    the book notes that "both tails happen to be sensibly fatter than
    predicted by the Laplace model" (Delli Gatti et al., 2011).

    Parameters
    ----------
    growth_rates : NDArray
        Array of growth rate values.
    trim_pct : float
        Fraction of outermost points to trim on each side (default 0.10).

    Returns
    -------
    float
        Average R-squared across negative and positive sides, or 0.0 if
        insufficient data.
    """
    valid = growth_rates[np.isfinite(growth_rates)]
    if len(valid) < 20:
        return 0.0

    negative = valid[valid < 0]
    positive = valid[valid >= 0]

    if len(negative) < 10 or len(positive) < 10:
        return 0.0

    r2_values: list[float] = []

    for side_data, reverse in [(negative, False), (positive, True)]:
        sorted_data = np.sort(side_data)
        if reverse:
            sorted_data = sorted_data[::-1]
        n = len(sorted_data)
        ranks = np.arange(1, n + 1)
        log_ranks = np.log(ranks)

        # Trim outermost points (fat-tail region)
        # Note: remaining points keep their original rank positions
        # (not re-ranked after trimming), standard for log-rank analysis
        n_trim = max(1, int(n * trim_pct))
        trimmed_data = sorted_data[n_trim:]
        trimmed_log_ranks = log_ranks[n_trim:]

        if len(trimmed_data) < 5:
            continue

        # Linear regression: log(rank) = a * growth_rate + b
        try:
            coeffs = np.polyfit(trimmed_data, trimmed_log_ranks, 1)
        except np.linalg.LinAlgError:
            # SVD can fail on degenerate data (e.g., constant values)
            continue

        predicted = np.polyval(coeffs, trimmed_data)

        ss_res = np.sum((trimmed_log_ranks - predicted) ** 2)
        ss_tot = np.sum((trimmed_log_ranks - np.mean(trimmed_log_ranks)) ** 2)

        if ss_tot > 0:
            r2_values.append(1.0 - ss_res / ss_tot)

    if len(r2_values) == 0:
        return 0.0

    return float(np.mean(r2_values))


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_growth_plus_metrics(
    sim: bam.Simulation,
    results: SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 150.0,
    firm_size_threshold_medium: float = 100.0,
) -> GrowthPlusMetrics:
    """Compute all validation metrics from Growth+ simulation results."""
    # Extract raw data
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    inventory = results.role_data["Producer"]["inventory"]
    labor_productivity = results.role_data["Producer"]["labor_productivity"]
    prices = results.role_data["Producer"]["price"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]
    net_worth = results.role_data["Borrower"]["net_worth"]
    total_funds = results.role_data["Borrower"]["total_funds"]
    consumer_budget = results.role_data["Consumer"]["income_to_spend"]
    n_firm_bankruptcies = np.array(
        results.economy_data["n_firm_bankruptcies"], dtype=np.int_
    )
    n_bank_bankruptcies = np.array(
        results.economy_data["n_bank_bankruptcies"], dtype=np.int_
    )

    loan_principals = results.relationship_data["LoanBook"]["principal"]
    loan_rates = results.relationship_data["LoanBook"]["rate"]
    loan_source_ids = results.relationship_data["LoanBook"]["source_ids"]

    # Compute time series
    unemployment = 1 - ops.mean(employed.astype(float), axis=1)
    gdp = ops.sum(production, axis=1)
    log_gdp = ops.log(gdp)

    weighted_productivity = ops.sum(
        ops.multiply(labor_productivity, production), axis=1
    )
    avg_productivity = ops.divide(weighted_productivity, gdp)

    employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
    employed_count = ops.sum(employed, axis=1)
    avg_employed_wage = ops.where(
        ops.greater(employed_count, 0),
        ops.divide(employed_wages_sum, employed_count),
        0.0,
    )
    real_wage = ops.divide(avg_employed_wage, avg_price)

    total_vacancies = ops.sum(n_vacancies, axis=1)
    vacancy_rate = ops.divide(total_vacancies, sim.n_households)

    # Curve data
    wage_inflation = ops.divide(
        avg_employed_wage[1:] - avg_employed_wage[:-1],
        ops.where(ops.greater(avg_employed_wage[:-1], 0), avg_employed_wage[:-1], 1.0),
    )
    gdp_growth = ops.divide(gdp[1:] - gdp[:-1], gdp[:-1])
    unemployment_growth = ops.divide(
        unemployment[1:] - unemployment[:-1],
        ops.where(ops.greater(unemployment[:-1], 0), unemployment[:-1], 1.0),
    )
    productivity_growth_rate = ops.divide(
        avg_productivity[1:] - avg_productivity[:-1],
        ops.where(ops.greater(avg_productivity[:-1], 0), avg_productivity[:-1], 1.0),
    )

    # Apply burn-in
    unemployment_ss = unemployment[burn_in:]
    log_gdp_ss = log_gdp[burn_in:]
    real_wage_ss = real_wage[burn_in:]
    vacancy_rate_ss = vacancy_rate[burn_in:]
    avg_productivity_ss = avg_productivity[burn_in:]
    inflation_ss = inflation[burn_in:]

    wage_inflation_ss = wage_inflation[burn_in - 1 :]
    gdp_growth_ss = gdp_growth[burn_in - 1 :]
    unemployment_growth_ss = unemployment_growth[burn_in - 1 :]

    # Correlations
    phillips_corr = float(np.corrcoef(unemployment_ss, wage_inflation_ss)[0, 1])
    unemp_filtered, gdp_filtered = filter_outliers_iqr(
        unemployment_growth_ss, gdp_growth_ss
    )
    okun_corr = float(np.corrcoef(unemp_filtered, gdp_filtered)[0, 1])
    beveridge_corr = float(np.corrcoef(unemployment_ss, vacancy_rate_ss)[0, 1])

    # Distribution metrics
    final_production_arr = production[-1]
    firm_size_skewness = float(stats.skew(final_production_arr))
    firm_size_pct_below = float(
        np.sum(final_production_arr < firm_size_threshold) / len(final_production_arr)
    )
    median_production = float(np.median(final_production_arr))
    firm_size_tail_ratio = (
        float(np.max(final_production_arr) / median_production)
        if median_production > 0
        else 0.0
    )
    firm_size_pct_below_medium = float(
        np.sum(final_production_arr < firm_size_threshold_medium)
        / len(final_production_arr)
    )

    # Growth-specific metrics
    initial_productivity = float(avg_productivity_ss[0])
    final_productivity_val = float(avg_productivity_ss[-1])
    total_productivity_growth = (
        (final_productivity_val - initial_productivity) / initial_productivity
        if initial_productivity > 0
        else 0.0
    )

    real_wage_initial = float(real_wage_ss[0])
    real_wage_final = float(real_wage_ss[-1])
    total_real_wage_growth = (
        (real_wage_final - real_wage_initial) / real_wage_initial
        if real_wage_initial > 0
        else 0.0
    )

    time_axis = np.arange(len(avg_productivity_ss))
    log_productivity = np.log(
        np.where(avg_productivity_ss > 0, avg_productivity_ss, EPS)
    )
    try:
        trend_coef, _ = np.polyfit(time_axis, log_productivity, 1)
        productivity_trend_coefficient = float(trend_coef)
    except np.linalg.LinAlgError:
        productivity_trend_coefficient = 0.0

    try:
        log_gdp_trend_coef, _ = np.polyfit(time_axis, log_gdp_ss, 1)
        log_gdp_trend_coefficient = float(log_gdp_trend_coef)
    except np.linalg.LinAlgError:
        log_gdp_trend_coefficient = 0.0
    log_gdp_total_growth = float(log_gdp_ss[-1] - log_gdp_ss[0])

    # Productivity-wage co-movement metrics
    productivity_wage_correlation = _compute_detrended_correlation(
        avg_productivity_ss, real_wage_ss
    )

    # Wage-to-productivity ratio
    wage_prod_ratio = ops.divide(real_wage_ss, avg_productivity_ss)
    wage_productivity_ratio_mean = float(np.mean(wage_prod_ratio))
    wage_productivity_ratio_std = float(np.std(wage_prod_ratio))

    # Financial dynamics
    n_periods = len(inflation)
    real_interest_rate = np.zeros(n_periods)
    for t in range(n_periods):
        principals_t = loan_principals[t]
        rates_t = loan_rates[t]
        if len(principals_t) > 0 and np.sum(principals_t) > 0:
            weighted_nominal = float(
                np.sum(rates_t * principals_t) / np.sum(principals_t)
            )
        else:
            weighted_nominal = sim.r_bar
        real_interest_rate[t] = weighted_nominal - inflation[t]

    total_wage_bill = ops.sum(wages * employed.astype(float), axis=1)
    total_net_worth = ops.sum(net_worth, axis=1)
    safe_total_nw = ops.where(ops.greater(total_net_worth, EPS), total_net_worth, EPS)
    avg_financial_fragility = ops.divide(total_wage_bill, safe_total_nw)

    safe_gdp = ops.where(ops.greater(gdp, EPS), gdp, EPS)
    total_demand = ops.sum(consumer_budget, axis=1)
    market_clearing_price = ops.divide(total_demand, safe_gdp)
    price_ratio = ops.divide(avg_price, market_clearing_price)

    # Dispersions
    safe_prod = np.where(production > 0, production, 0.0)
    prod_sum = np.sum(safe_prod, axis=1, keepdims=True)
    prod_weights = safe_prod / np.where(prod_sum > EPS, prod_sum, EPS)
    weighted_price_mean = np.sum(prod_weights * prices, axis=1, keepdims=True)
    weighted_price_var = np.sum(
        prod_weights * (prices - weighted_price_mean) ** 2, axis=1
    )
    weighted_price_std = np.sqrt(np.maximum(weighted_price_var, 0.0))
    w_mean_flat = weighted_price_mean.squeeze()
    safe_w_mean = np.where(w_mean_flat > EPS, w_mean_flat, EPS)
    price_dispersion = weighted_price_std / safe_w_mean

    nw_mean = ops.mean(net_worth, axis=1)
    equity_dispersion = ops.divide(
        ops.std(net_worth, axis=1),
        ops.where(ops.greater(np.abs(nw_mean), EPS), np.abs(nw_mean), EPS),
    )

    qty_sold = np.subtract(production, inventory)
    sales = ops.multiply(prices, qty_sold)
    sales_mean = ops.mean(sales, axis=1)
    sales_dispersion = ops.divide(
        ops.std(sales, axis=1),
        ops.where(ops.greater(np.abs(sales_mean), EPS), np.abs(sales_mean), EPS),
    )

    # Growth rate distributions
    gdp_after_burnin = gdp[burn_in:]
    output_growth_rates = np.diff(gdp_after_burnin) / gdp_after_burnin[:-1]

    nw_prev = net_worth[-2]
    nw_final = net_worth[-1]
    valid_firms = (nw_prev > 0) & (nw_final > 0)
    nw_prev_valid = nw_prev[valid_firms]
    nw_final_valid = nw_final[valid_firms]
    networth_growth_rates = (nw_final_valid - nw_prev_valid) / nw_prev_valid

    # Tiered distribution metrics
    output_growth_pct_within_tight = _compute_pct_within_range(
        output_growth_rates, -0.05, 0.05
    )
    output_growth_pct_within_normal = _compute_pct_within_range(
        output_growth_rates, -0.10, 0.10
    )
    output_growth_pct_outliers = 1.0 - _compute_pct_within_range(
        output_growth_rates, -0.15, 0.15
    )

    output_growth_tent_r2 = _compute_tent_shape_r2(output_growth_rates, trim_pct=0.10)

    valid_output_growth = output_growth_rates[np.isfinite(output_growth_rates)]
    output_growth_positive_frac = (
        float(np.sum(valid_output_growth >= 0) / len(valid_output_growth))
        if len(valid_output_growth) > 0
        else 0.0
    )

    networth_growth_pct_within_tight = _compute_pct_within_range(
        networth_growth_rates, -0.05, 0.05
    )
    networth_growth_pct_within_normal = _compute_pct_within_range(
        networth_growth_rates, -0.10, 0.10
    )
    networth_growth_pct_outliers = 1.0 - _compute_pct_within_range(
        networth_growth_rates, -0.50, 0.20
    )

    networth_growth_tent_r2 = _compute_tent_shape_r2(
        networth_growth_rates, trim_pct=0.10
    )

    # Recession detection
    recession_mask = _detect_recessions(log_gdp)
    n_recessions, avg_recession_length = _count_recession_episodes(
        recession_mask[burn_in:]
    )

    # Minsky classification
    n_active_firms = sim.n_firms
    n_periods_total = total_funds.shape[0]

    hedge_pcts: list[float] = []
    speculative_pcts: list[float] = []
    ponzi_pcts: list[float] = []

    for t in range(burn_in, n_periods_total):
        tf = total_funds[t]
        p = loan_principals[t]
        r = loan_rates[t]
        src = loan_source_ids[t]

        debt_per_firm = np.zeros(n_active_firms)
        interest_per_firm = np.zeros(n_active_firms)
        if len(p) > 0:
            np.add.at(debt_per_firm, src, p * (1.0 + r))
            np.add.at(interest_per_firm, src, p * r)

        hedge = tf >= debt_per_firm
        ponzi = tf < interest_per_firm
        speculative = (~hedge) & (~ponzi)

        hedge_pcts.append(float(np.sum(hedge)) / n_active_firms)
        speculative_pcts.append(float(np.sum(speculative)) / n_active_firms)
        ponzi_pcts.append(float(np.sum(ponzi)) / n_active_firms)

    minsky_hedge_pct = float(np.mean(hedge_pcts))
    minsky_speculative_pct = float(np.mean(speculative_pcts))
    minsky_ponzi_pct = float(np.mean(ponzi_pcts))

    # Summary statistics
    bankruptcies_ss = n_firm_bankruptcies[burn_in:]
    real_ir_ss = real_interest_rate[burn_in:]
    fragility_ss = avg_financial_fragility[burn_in:]

    # Financial fragility CV and pro-cyclicality (Minsky hypothesis)
    financial_fragility_cv = _compute_cv(fragility_ss)
    fragility_gdp_correlation = _compute_gdp_cyclicality(fragility_ss, log_gdp_ss)

    price_ratio_ss = price_ratio[burn_in:]

    # Price ratio derived metrics
    price_ratio_mean_val = float(np.mean(price_ratio_ss))
    price_ratio_cv = _compute_cv(price_ratio_ss)
    price_ratio_min_val = float(np.min(price_ratio_ss))
    price_ratio_p5_val = float(np.percentile(price_ratio_ss, 5))
    price_ratio_gdp_corr = _compute_gdp_cyclicality(price_ratio_ss, log_gdp_ss)

    price_disp_ss = price_dispersion[burn_in:]

    # Price dispersion CV and pro-cyclicality
    price_dispersion_cv = _compute_cv(price_disp_ss)
    price_dispersion_gdp_correlation = _compute_gdp_cyclicality(
        price_disp_ss, log_gdp_ss
    )

    equity_disp_ss = equity_dispersion[burn_in:]
    sales_disp_ss = sales_dispersion[burn_in:]

    # Equity and sales dispersion CV and pro-cyclicality
    equity_dispersion_cv = _compute_cv(equity_disp_ss)
    equity_dispersion_gdp_corr = _compute_gdp_cyclicality(equity_disp_ss, log_gdp_ss)
    sales_dispersion_cv = _compute_cv(sales_disp_ss)
    sales_dispersion_gdp_corr = _compute_gdp_cyclicality(sales_disp_ss, log_gdp_ss)

    return GrowthPlusMetrics(
        unemployment=unemployment,
        inflation=inflation,
        log_gdp=log_gdp,
        real_wage=real_wage,
        avg_productivity=avg_productivity,
        vacancy_rate=vacancy_rate,
        wage_inflation=wage_inflation,
        gdp_growth=gdp_growth,
        unemployment_growth=unemployment_growth,
        final_production=final_production_arr,
        unemployment_mean=float(np.mean(unemployment_ss)),
        unemployment_std=float(np.std(unemployment_ss)),
        unemployment_max=float(np.max(unemployment_ss)),
        unemployment_pct_in_bounds=float(
            np.sum((unemployment_ss >= 0.02) & (unemployment_ss <= 0.15))
            / len(unemployment_ss)
        ),
        inflation_mean=float(np.mean(inflation_ss)),
        inflation_std=float(np.std(inflation_ss)),
        inflation_max=float(np.max(inflation_ss)),
        inflation_min=float(np.min(inflation_ss)),
        inflation_pct_in_bounds=float(
            np.mean((inflation_ss >= -0.02) & (inflation_ss <= 0.10))
        ),
        log_gdp_mean=float(np.mean(log_gdp_ss)),
        log_gdp_std=float(np.std(log_gdp_ss)),
        real_wage_mean=float(np.mean(real_wage_ss)),
        real_wage_std=float(np.std(real_wage_ss)),
        vacancy_rate_mean=float(np.mean(vacancy_rate_ss)),
        vacancy_rate_pct_in_bounds=float(
            np.mean((vacancy_rate_ss >= 0.08) & (vacancy_rate_ss <= 0.20))
        ),
        phillips_corr=phillips_corr,
        okun_corr=okun_corr,
        beveridge_corr=beveridge_corr,
        firm_size_skewness=firm_size_skewness,
        firm_size_pct_below_threshold=firm_size_pct_below,
        firm_size_tail_ratio=firm_size_tail_ratio,
        firm_size_pct_below_medium=firm_size_pct_below_medium,
        productivity_growth_rate=productivity_growth_rate,
        productivity_trend_coefficient=productivity_trend_coefficient,
        initial_productivity=initial_productivity,
        final_productivity=final_productivity_val,
        total_productivity_growth=total_productivity_growth,
        real_wage_initial=real_wage_initial,
        real_wage_final=real_wage_final,
        total_real_wage_growth=total_real_wage_growth,
        log_gdp_trend_coefficient=log_gdp_trend_coefficient,
        log_gdp_total_growth=log_gdp_total_growth,
        productivity_wage_correlation=productivity_wage_correlation,
        wage_productivity_ratio_mean=wage_productivity_ratio_mean,
        wage_productivity_ratio_std=wage_productivity_ratio_std,
        n_firm_bankruptcies=n_firm_bankruptcies,
        n_bank_bankruptcies=n_bank_bankruptcies,
        real_interest_rate=real_interest_rate,
        avg_financial_fragility=avg_financial_fragility,
        price_ratio=price_ratio,
        price_dispersion=price_dispersion,
        equity_dispersion=equity_dispersion,
        sales_dispersion=sales_dispersion,
        output_growth_rates=output_growth_rates,
        networth_growth_rates=networth_growth_rates,
        output_growth_pct_within_tight=output_growth_pct_within_tight,
        output_growth_pct_within_normal=output_growth_pct_within_normal,
        output_growth_pct_outliers=output_growth_pct_outliers,
        output_growth_tent_r2=output_growth_tent_r2,
        output_growth_positive_frac=output_growth_positive_frac,
        networth_growth_pct_within_tight=networth_growth_pct_within_tight,
        networth_growth_pct_within_normal=networth_growth_pct_within_normal,
        networth_growth_pct_outliers=networth_growth_pct_outliers,
        networth_growth_tent_r2=networth_growth_tent_r2,
        recession_mask=recession_mask,
        n_recessions=n_recessions,
        avg_recession_length=avg_recession_length,
        minsky_hedge_pct=minsky_hedge_pct,
        minsky_speculative_pct=minsky_speculative_pct,
        minsky_ponzi_pct=minsky_ponzi_pct,
        bankruptcies_mean=float(np.mean(bankruptcies_ss)),
        real_interest_rate_mean=float(np.mean(real_ir_ss)),
        real_interest_rate_std=float(np.std(real_ir_ss)),
        real_interest_rate_pct_in_bounds=float(
            np.mean((real_ir_ss >= -0.02) & (real_ir_ss <= 0.12))
        ),
        avg_fragility_mean=float(np.mean(fragility_ss)),
        financial_fragility_cv=financial_fragility_cv,
        fragility_gdp_correlation=fragility_gdp_correlation,
        price_ratio_mean=price_ratio_mean_val,
        price_ratio_cv=price_ratio_cv,
        price_ratio_min=price_ratio_min_val,
        price_ratio_p5=price_ratio_p5_val,
        price_ratio_gdp_correlation=price_ratio_gdp_corr,
        price_dispersion_mean=float(np.mean(price_disp_ss)),
        price_dispersion_cv=price_dispersion_cv,
        price_dispersion_gdp_correlation=price_dispersion_gdp_correlation,
        equity_dispersion_mean=float(np.mean(equity_disp_ss)),
        equity_dispersion_cv=equity_dispersion_cv,
        equity_dispersion_gdp_correlation=equity_dispersion_gdp_corr,
        sales_dispersion_mean=float(np.mean(sales_disp_ss)),
        sales_dispersion_cv=sales_dispersion_cv,
        sales_dispersion_gdp_correlation=sales_dispersion_gdp_corr,
    )


# =============================================================================
# Collection Configuration
# =============================================================================

COLLECT_CONFIG = {
    "Producer": ["production", "labor_productivity", "price", "inventory"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth", "gross_profit", "total_funds"],
    "Consumer": ["income_to_spend"],
    "LoanBook": ["principal", "rate", "source_ids"],
    "Economy": True,
    "capture_timing": {
        "Worker.wage": "firms_run_production",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Producer.labor_productivity": "firms_apply_productivity_growth",
        "Producer.price": "firms_adjust_price",
        "Producer.inventory": "consumers_finalize_purchases",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "Borrower.net_worth": "firms_run_production",
        "Borrower.gross_profit": "firms_collect_revenue",
        "Borrower.total_funds": "firms_collect_revenue",
        "Consumer.income_to_spend": "consumers_decide_income_to_spend",
        "LoanBook.principal": "banks_provide_loans",
        "LoanBook.rate": "banks_provide_loans",
        "LoanBook.source_ids": "banks_provide_loans",
        "Economy.n_firm_bankruptcies": "mark_bankrupt_firms",
        "Economy.n_bank_bankruptcies": "mark_bankrupt_banks",
    },
}

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG: dict[str, Any] = {}

# =============================================================================
# Metric Specifications
# =============================================================================

METRIC_SPECS = [
    # Time series metrics
    MetricSpec(
        name="unemployment_rate_mean",
        field="unemployment_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.unemployment_rate_mean",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="unemployment_hard_ceiling",
        field="unemployment_max",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.unemployment_hard_ceiling",
        weight=3.0,
        group=MetricGroup.TIME_SERIES,
        threshold=0.30,
        invert=True,
        target_desc="< 30% (model stability)",
    ),
    MetricSpec(
        name="unemployment_std",
        field="unemployment_std",
        check_type=CheckType.RANGE,
        target_path="metrics.unemployment_std",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="unemployment_pct_in_bounds",
        field="unemployment_pct_in_bounds",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.unemployment_pct_in_bounds",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="inflation_rate_mean",
        field="inflation_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.inflation_rate_mean",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="inflation_hard_ceiling_upper",
        field="inflation_max",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.inflation_hard_ceiling_upper",
        weight=3.0,
        group=MetricGroup.TIME_SERIES,
        threshold=0.25,
        invert=True,  # max must be < 25%
        target_desc="< 25% (model stability)",
    ),
    MetricSpec(
        name="inflation_hard_floor",
        field="inflation_min",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.inflation_hard_floor",
        weight=3.0,
        group=MetricGroup.TIME_SERIES,
        threshold=-0.15,
        invert=False,  # min must be > -15%
        target_desc="> -15% (model stability)",
    ),
    MetricSpec(
        name="inflation_non_degenerate",
        field="inflation_mean",
        check_type=CheckType.RANGE,
        target_path="metrics.inflation_non_degenerate",
        weight=2.0,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="inflation_pct_in_bounds",
        field="inflation_pct_in_bounds",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.inflation_pct_in_bounds",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="log_gdp_mean",
        field="log_gdp_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.log_gdp_mean",
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="log_gdp_trend",
        field="log_gdp_trend_coefficient",
        check_type=CheckType.RANGE,
        target_path="metrics.log_gdp_trend",
        weight=2.0,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.TREND,
    ),
    MetricSpec(
        name="log_gdp_trend_positive",
        field="log_gdp_trend_coefficient",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.log_gdp_trend_positive",
        weight=2.0,
        group=MetricGroup.TIME_SERIES,
        threshold=0.0,
        target_desc="> 0 (must be positive)",
    ),
    MetricSpec(
        name="log_gdp_total_growth",
        field="log_gdp_total_growth",
        check_type=CheckType.RANGE,
        target_path="metrics.log_gdp_total_growth",
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.TREND,
    ),
    MetricSpec(
        name="vacancy_rate_mean",
        field="vacancy_rate_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.vacancy_rate_mean",
        weight=0.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="vacancy_rate_pct_in_bounds",
        field="vacancy_rate_pct_in_bounds",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.vacancy_rate_pct_in_bounds",
        weight=0.5,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    # Curve correlations
    MetricSpec(
        name="phillips_correlation",
        field="phillips_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.phillips_correlation",
        weight=1.5,
        group=MetricGroup.CURVES,
    ),
    MetricSpec(
        name="phillips_negative_sign",
        field="phillips_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.phillips_negative_sign",
        weight=3.0,  # CRITICAL - structural validity
        group=MetricGroup.CURVES,
        target_desc="correlation must be < 0",
    ),
    MetricSpec(
        name="okun_correlation",
        field="okun_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.okun_correlation",
        weight=3.0,  # CRITICAL: Okun's Law is fundamental macroeconomic relationship
        group=MetricGroup.CURVES,
    ),
    MetricSpec(
        name="okun_weak_fail",
        field="okun_corr",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.okun_weak_fail",
        weight=3.0,
        group=MetricGroup.CURVES,
        threshold=-0.50,
        invert=True,  # PASS if r < -0.50, FAIL if r >= -0.50
        target_desc="< -0.50 (minimum acceptable)",
    ),
    MetricSpec(
        name="beveridge_correlation",
        field="beveridge_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.beveridge_correlation",
        weight=2.0,
        group=MetricGroup.CURVES,
    ),
    MetricSpec(
        name="beveridge_negative_sign",
        field="beveridge_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.beveridge_negative_sign",
        weight=3.0,  # CRITICAL - structural validity
        group=MetricGroup.CURVES,
        target_desc="correlation must be < 0",
    ),
    # Distribution metrics
    MetricSpec(
        name="firm_size_skewness",
        field="firm_size_skewness",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.firm_size_skewness",
        weight=1.5,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="firm_size_pct_below",
        field="firm_size_pct_below_threshold",
        check_type=CheckType.RANGE,
        target_path="metrics.firm_size_pct_below",
        weight=0.5,
        group=MetricGroup.DISTRIBUTION,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="firm_size_tail_ratio",
        field="firm_size_tail_ratio",
        check_type=CheckType.RANGE,
        target_path="metrics.firm_size_tail_ratio",
        weight=0.75,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="firm_size_pct_below_medium",
        field="firm_size_pct_below_medium",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.firm_size_pct_below_medium",
        weight=0.5,
        group=MetricGroup.DISTRIBUTION,
        format=MetricFormat.PERCENT,
    ),
    # Growth metrics
    MetricSpec(
        name="productivity_growth",
        field="total_productivity_growth",
        check_type=CheckType.RANGE,
        target_path="metrics.productivity_growth",
        weight=1.5,
        group=MetricGroup.GROWTH,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="real_wage_growth",
        field="total_real_wage_growth",
        check_type=CheckType.RANGE,
        target_path="metrics.real_wage_growth",
        weight=1.0,
        group=MetricGroup.GROWTH,
        format=MetricFormat.PERCENT,
    ),
    # Productivity-Wage Co-movement
    MetricSpec(
        name="productivity_wage_correlation",
        field="productivity_wage_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.productivity_wage_correlation",
        weight=2.0,  # CRITICAL - key validation from Figure 3.4(d)
        group=MetricGroup.GROWTH,
    ),
    MetricSpec(
        name="wage_productivity_ratio_mean",
        field="wage_productivity_ratio_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.wage_productivity_ratio_mean",
        weight=1.5,  # IMPORTANT
        group=MetricGroup.GROWTH,
    ),
    MetricSpec(
        name="wage_productivity_ratio_std",
        field="wage_productivity_ratio_std",
        check_type=CheckType.RANGE,
        target_path="metrics.wage_productivity_ratio_std",
        weight=1.5,  # IMPORTANT - detects divergence
        group=MetricGroup.GROWTH,
    ),
    MetricSpec(
        name="productivity_trend",
        field="productivity_trend_coefficient",
        check_type=CheckType.RANGE,
        target_path="metrics.productivity_trend",
        weight=1.0,
        group=MetricGroup.GROWTH,
        format=MetricFormat.TREND,
    ),
    MetricSpec(
        name="n_recessions",
        field="n_recessions",
        check_type=CheckType.RANGE,
        target_path="metrics.n_recessions",
        weight=1.5,
        group=MetricGroup.GROWTH,
        format=MetricFormat.INTEGER,
    ),
    MetricSpec(
        name="avg_recession_length",
        field="avg_recession_length",
        check_type=CheckType.RANGE,
        target_path="metrics.avg_recession_length",
        weight=0.5,
        group=MetricGroup.GROWTH,
    ),
    # Financial dynamics
    MetricSpec(
        name="bankruptcies_mean",
        field="bankruptcies_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.bankruptcies_mean",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="real_interest_rate_mean",
        field="real_interest_rate_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.real_interest_rate_mean",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="real_interest_rate_std",
        field="real_interest_rate_std",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.real_interest_rate_std",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="real_interest_rate_pct_in_bounds",
        field="real_interest_rate_pct_in_bounds",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.real_interest_rate_pct_in_bounds",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="financial_fragility_mean",
        field="avg_fragility_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.financial_fragility_mean",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="financial_fragility_cv",
        field="financial_fragility_cv",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.financial_fragility_cv",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="fragility_gdp_correlation",
        field="fragility_gdp_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.fragility_gdp_correlation",
        weight=2.0,  # VERY IMPORTANT - key Minsky validation
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_ratio_mean",
        field="price_ratio_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.price_ratio_mean",
        weight=0.75,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_ratio_cv",
        field="price_ratio_cv",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.price_ratio_cv",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_ratio_floor",
        field="price_ratio_min",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.price_ratio_floor",
        weight=3.0,  # CRITICAL - structural validity
        group=MetricGroup.FINANCIAL,
        threshold=1.0,
        invert=False,  # PASS if min > 1.0
        target_desc="> 1.0 (price ratio floor)",
    ),
    MetricSpec(
        name="price_ratio_gdp_correlation",
        field="price_ratio_gdp_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.price_ratio_gdp_correlation",
        weight=1.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_ratio_low_end",
        field="price_ratio_p5",
        check_type=CheckType.RANGE,
        target_path="metrics.price_ratio_low_end",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_dispersion_mean",
        field="price_dispersion_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.price_dispersion_mean",
        weight=0.75,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_dispersion_cv",
        field="price_dispersion_cv",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.price_dispersion_cv",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="price_dispersion_gdp_correlation",
        field="price_dispersion_gdp_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.price_dispersion_gdp_correlation",
        weight=1.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="equity_dispersion_mean",
        field="equity_dispersion_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.equity_dispersion_mean",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="equity_dispersion_cv",
        field="equity_dispersion_cv",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.equity_dispersion_cv",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="equity_dispersion_gdp_correlation",
        field="equity_dispersion_gdp_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.equity_dispersion_gdp_correlation",
        weight=1.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="sales_dispersion_mean",
        field="sales_dispersion_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.sales_dispersion_mean",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="sales_dispersion_cv",
        field="sales_dispersion_cv",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.sales_dispersion_cv",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="sales_dispersion_gdp_correlation",
        field="sales_dispersion_gdp_correlation",
        check_type=CheckType.RANGE,
        target_path="metrics.sales_dispersion_gdp_correlation",
        weight=1.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="minsky_hedge_pct",
        field="minsky_hedge_pct",
        check_type=CheckType.RANGE,
        target_path="metrics.minsky_hedge_pct",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="minsky_ponzi_pct",
        field="minsky_ponzi_pct",
        check_type=CheckType.RANGE,
        target_path="metrics.minsky_ponzi_pct",
        weight=0.5,
        group=MetricGroup.FINANCIAL,
        format=MetricFormat.PERCENT,
    ),
    # Growth rate distributions
    MetricSpec(
        name="output_growth_pct_tight",
        field="output_growth_pct_within_tight",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.output_growth_pct_tight",
        weight=1.0,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="output_growth_pct_normal",
        field="output_growth_pct_within_normal",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.output_growth_pct_normal",
        weight=0.5,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="output_growth_outliers",
        field="output_growth_pct_outliers",
        check_type=CheckType.OUTLIER,
        target_path="metrics.output_growth_outliers",
        weight=1.5,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="output_growth_tent_r2",
        field="output_growth_tent_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.output_growth_tent_r2",
        weight=2.0,
        group=MetricGroup.GROWTH_RATE_DIST,
    ),
    MetricSpec(
        name="output_growth_positive_frac",
        field="output_growth_positive_frac",
        check_type=CheckType.RANGE,
        target_path="metrics.output_growth_positive_frac",
        weight=0.5,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="networth_growth_pct_tight",
        field="networth_growth_pct_within_tight",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.networth_growth_pct_tight",
        weight=1.0,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="networth_growth_pct_normal",
        field="networth_growth_pct_within_normal",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.networth_growth_pct_normal",
        weight=0.5,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="networth_growth_outliers",
        field="networth_growth_pct_outliers",
        check_type=CheckType.OUTLIER,
        target_path="metrics.networth_growth_outliers",
        weight=1.5,
        group=MetricGroup.GROWTH_RATE_DIST,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="networth_growth_tent_r2",
        field="networth_growth_tent_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.networth_growth_tent_r2",
        weight=1.5,
        group=MetricGroup.GROWTH_RATE_DIST,
    ),
]


# =============================================================================
# Setup Hook for RnD Extension
# =============================================================================


def _setup_rnd(sim: bam.Simulation | None) -> None:
    """Setup hook to import and attach RnD extension."""
    if sim is None:
        # Pre-import call - just import to register event classes
        from extensions.rnd import RnD
    else:
        # Attach RnD role and events to simulation
        from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

        sim.use_role(RnD)
        sim.use_events(*RND_EVENTS)
        sim.use_config(RND_CONFIG)


# =============================================================================
# Compute Metrics Wrapper
# =============================================================================


def _compute_metrics_wrapper(
    sim: bam.Simulation, results: SimulationResults, burn_in: int
) -> GrowthPlusMetrics:
    """Wrapper for compute_growth_plus_metrics that loads params from YAML."""
    with open(Path(__file__).parent / "targets.yaml") as f:
        targets = yaml.safe_load(f)

    params = targets["metadata"]["params"]

    return compute_growth_plus_metrics(
        sim,
        results,
        burn_in=burn_in,
        firm_size_threshold=params["firm_size_threshold"],
        firm_size_threshold_medium=params["firm_size_threshold_medium"],
    )


# =============================================================================
# Scenario Definition
# =============================================================================

SCENARIO = Scenario(
    name="growth_plus",
    metric_specs=METRIC_SPECS,
    collect_config=COLLECT_CONFIG,
    targets_path=Path(__file__).parent / "targets.yaml",
    default_config=DEFAULT_CONFIG,
    compute_metrics=_compute_metrics_wrapper,
    setup_hook=_setup_rnd,
    title="GROWTH+ SCENARIO VALIDATION",
    stability_title="GROWTH+ SEED STABILITY TEST",
)


# =============================================================================
# Public API
# =============================================================================


def load_growth_plus_targets() -> dict[str, Any]:
    """Load Growth+ validation targets from YAML for visualization."""
    with open(Path(__file__).parent / "targets.yaml") as f:
        data = yaml.safe_load(f)

    viz = data["metadata"]["visualization"]
    ts = viz["time_series"]
    curves = viz["curves"]
    dist = viz["distributions"]
    fin = viz.get("financial_dynamics", {})

    def _transform_curve_targets(raw: dict[str, Any]) -> dict[str, Any]:
        """Transform curve targets to expected keys for visualization."""
        return {
            "target": raw.get("correlation_target"),
            "min": raw.get("correlation_min"),
            "max": raw.get("correlation_max"),
        }

    def _transform_firm_size_targets(raw: dict[str, Any]) -> dict[str, Any]:
        """Transform firm size targets to expected keys for visualization."""
        skew_target = raw.get("skewness_target")
        skew_tol = raw.get("skewness_tolerance")
        return {
            "threshold": raw.get("threshold_small"),
            "threshold_medium": raw.get("threshold_medium"),
            "pct_below_target": raw.get("pct_below_small_target"),
            "pct_below_min": raw.get("pct_below_small_min"),
            "pct_below_max": raw.get("pct_below_small_max"),
            "pct_below_medium_target": raw.get("pct_below_medium_target"),
            "pct_below_medium_min": raw.get("pct_below_medium_min"),
            "skewness_target": skew_target,
            "skewness_tolerance": skew_tol,
            "skewness_min": skew_target - skew_tol,
            "skewness_max": skew_target + skew_tol,
            "skewness_hard_min": raw.get("skewness_hard_min"),
            "skewness_hard_max": raw.get("skewness_hard_max"),
            "tail_ratio_min": raw.get("tail_ratio_min"),
            "tail_ratio_max": raw.get("tail_ratio_max"),
        }

    return {
        "log_gdp": ts["log_gdp"]["targets"],
        "unemployment": ts["unemployment_rate"]["targets"],
        "inflation": ts["inflation_rate"]["targets"],
        "productivity": ts["productivity"]["targets"],
        "real_wage": ts["real_wage"]["targets"],
        "phillips_corr": _transform_curve_targets(curves["phillips"]["targets"]),
        "okun_corr": _transform_curve_targets(curves["okun"]["targets"]),
        "beveridge_corr": _transform_curve_targets(curves["beveridge"]["targets"]),
        "firm_size": _transform_firm_size_targets(dist["firm_size"]["targets"]),
        "financial_dynamics": fin,
    }


# =============================================================================
# Scenario Runner
# =============================================================================


def run_scenario(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    burn_in: int = 500,
    show_plot: bool = True,
) -> GrowthPlusMetrics:
    """Run Growth+ scenario simulation with optional visualization.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    burn_in : int
        Number of burn-in periods to exclude from analysis.
    show_plot : bool
        Whether to display the visualization.

    Returns
    -------
    GrowthPlusMetrics
        Computed metrics from the simulation.
    """
    from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

    # Initialize simulation with Growth+ default parameters
    sim = bam.Simulation.init(
        n_periods=n_periods, seed=seed, logging={"default_level": "ERROR"}
    )

    # Attach custom RnD role, events, and config
    rnd = sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    sim.use_config(RND_CONFIG)

    print("Growth+ simulation initialized:")
    print(f"  - {sim.n_firms} firms")
    print(f"  - {sim.n_households} households")
    print(f"  - {sim.n_banks} banks")
    print(f"  - Custom RnD role attached: {rnd is not None}")
    print(
        f"  - Extension params: sigma_min={sim.sigma_min}, sigma_max={sim.sigma_max}, "
        f"sigma_decay={sim.sigma_decay}"
    )

    # Run simulation
    results = sim.run(collect=COLLECT_CONFIG)

    print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
    print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

    burn_in = adjust_burn_in(burn_in, n_periods, verbose=True)

    # Compute metrics
    metrics = _compute_metrics_wrapper(sim, results, burn_in)

    print(
        f"\nComputed metrics for {len(metrics.unemployment) - burn_in} periods (after burn-in)"
    )
    print(f"  Initial productivity: {metrics.initial_productivity:.4f}")
    print(f"  Final productivity: {metrics.final_productivity:.4f}")
    print(
        f"  Total productivity growth: {metrics.total_productivity_growth * 100:.1f}%"
    )

    # Visualize if requested (lazy import to avoid circular dependency)
    if show_plot:
        from validation.scenarios.growth_plus.viz import (
            visualize_financial_dynamics,
            visualize_growth_plus_results,
        )

        bounds = load_growth_plus_targets()
        visualize_growth_plus_results(metrics, bounds, burn_in=burn_in)
        visualize_financial_dynamics(metrics, bounds, burn_in=burn_in)

    return metrics
