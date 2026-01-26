"""Growth+ scenario metrics (Section 3.9.2).

This module provides the GrowthPlusMetrics dataclass and compute function
for the Growth+ extension scenario from Delli Gatti et al. (2011).

The Growth+ scenario extends the baseline with endogenous productivity
growth through R&D investment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

import bamengine as bam
from bamengine import ops
from bamengine.utils import EPS
from validation.metrics._utils import (
    detect_recessions,
    filter_outliers_iqr,
    get_targets_dir,
)


@dataclass
class GrowthPlusMetrics:
    """All computed metrics from a Growth+ simulation run.

    The Growth+ scenario extends the baseline with endogenous productivity
    growth through R&D investment. Key differences from baseline:
    - Productivity and real wage grow over time (non-stationary)
    - Phillips correlation is stronger (-0.19 vs -0.10)
    - Firm size distribution has larger values due to productivity growth

    Attributes:
        Time series (full, before burn-in applied):
            unemployment: Unemployment rate per period
            inflation: Inflation rate per period
            log_gdp: Log of total production per period (GROWING)
            real_wage: Average real wage of employed workers (GROWING)
            avg_productivity: Production-weighted average labor productivity (GROWING)
            vacancy_rate: Total vacancies / total households per period

        Curve data (for scatter plots):
            wage_inflation: Period-over-period wage change rate
            gdp_growth: Period-over-period GDP change rate
            unemployment_growth: Period-over-period unemployment change rate

        Distribution data:
            final_production: Production per firm at final period

        Summary statistics (computed after burn-in):
            unemployment_mean, unemployment_std
            inflation_mean, inflation_std
            log_gdp_mean, log_gdp_std
            real_wage_mean, real_wage_std
            vacancy_rate_mean

        Curve correlations:
            phillips_corr: Correlation(unemployment, wage_inflation)
            okun_corr: Correlation(unemployment_growth, gdp_growth)
            beveridge_corr: Correlation(unemployment, vacancy_rate)

        Distribution metrics:
            firm_size_skewness: Skewness of firm size distribution
            firm_size_pct_below_threshold: Fraction of firms below threshold

        Growth-specific metrics:
            productivity_growth_rate: Period-over-period productivity growth rates
            productivity_trend_coefficient: Linear trend slope of log(productivity)
            initial_productivity: Productivity at burn-in period
            final_productivity: Productivity at final period
            total_productivity_growth: (final - initial) / initial
            real_wage_initial: Real wage at burn-in period
            real_wage_final: Real wage at final period
            total_real_wage_growth: (final - initial) / initial

        Financial dynamics metrics:
            n_firm_bankruptcies: Number of firm bankruptcies per period
            n_bank_bankruptcies: Number of bank bankruptcies per period
            real_interest_rate: Weighted average real interest rate per period
            avg_financial_fragility: Cross-section mean of wage_bill/net_worth
            price_ratio: Ratio of market price to market-clearing price
            price_dispersion: Coefficient of variation of firm prices
            equity_dispersion: Coefficient of variation of firm net worth
            sales_dispersion: Coefficient of variation of firm production

        Growth rate distributions (final period, per firm):
            output_growth_rates: Firm-level output growth rates
            networth_growth_rates: Firm-level net worth growth rates

        Recession detection:
            recession_mask: Boolean array marking recession periods
            n_recessions: Number of recession episodes (after burn-in)
            avg_recession_length: Average length of recession episodes

        Minsky classification (averaged over post-burn-in periods):
            minsky_hedge_pct: Fraction of Hedge firms (total_funds >= debt)
            minsky_speculative_pct: Fraction of Speculative firms (interest <= total_funds < debt)
            minsky_ponzi_pct: Fraction of Ponzi firms (total_funds < interest)

        Additional summary statistics:
            bankruptcies_mean: Mean bankruptcies per period (after burn-in)
            real_interest_rate_mean: Mean real interest rate (after burn-in)
            avg_fragility_mean: Mean financial fragility (after burn-in)
            price_ratio_mean: Mean price ratio (after burn-in)
    """

    # Time series (full)
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

    # Distribution
    final_production: NDArray[np.floating]

    # Summary statistics (after burn-in)
    unemployment_mean: float
    unemployment_std: float
    inflation_mean: float
    inflation_std: float
    log_gdp_mean: float
    log_gdp_std: float
    real_wage_mean: float
    real_wage_std: float
    vacancy_rate_mean: float

    # Correlations
    phillips_corr: float
    okun_corr: float
    beveridge_corr: float

    # Distribution metrics
    firm_size_skewness: float
    firm_size_pct_below_threshold: float

    # Growth-specific metrics
    productivity_growth_rate: NDArray[np.floating]
    productivity_trend_coefficient: float
    initial_productivity: float
    final_productivity: float
    total_productivity_growth: float
    real_wage_initial: float
    real_wage_final: float
    total_real_wage_growth: float

    # Financial dynamics metrics
    n_firm_bankruptcies: NDArray[np.int_]
    n_bank_bankruptcies: NDArray[np.int_]
    real_interest_rate: NDArray[np.floating]
    avg_financial_fragility: NDArray[np.floating]
    price_ratio: NDArray[np.floating]
    price_dispersion: NDArray[np.floating]
    equity_dispersion: NDArray[np.floating]
    sales_dispersion: NDArray[np.floating]

    # Growth rate distributions (final period)
    output_growth_rates: NDArray[np.floating]
    networth_growth_rates: NDArray[np.floating]

    # Tiered distribution metrics (computed from growth rates)
    output_growth_pct_within_tight: float
    output_growth_pct_within_normal: float
    output_growth_pct_outliers: float
    networth_growth_pct_within_tight: float
    networth_growth_pct_within_normal: float
    networth_growth_pct_outliers: float

    # Recession detection
    recession_mask: NDArray[np.bool_]
    n_recessions: int
    avg_recession_length: float

    # Minsky classification
    minsky_hedge_pct: float
    minsky_speculative_pct: float
    minsky_ponzi_pct: float

    # Additional summary statistics
    bankruptcies_mean: float
    real_interest_rate_mean: float
    real_interest_rate_std: float
    avg_fragility_mean: float
    avg_fragility_std: float
    price_ratio_mean: float
    price_ratio_std: float
    price_dispersion_mean: float
    price_dispersion_std: float
    equity_dispersion_mean: float
    equity_dispersion_std: float
    sales_dispersion_mean: float
    sales_dispersion_std: float


def _count_recession_episodes(recession_mask: NDArray[np.bool_]) -> tuple[int, float]:
    """Count recession episodes and compute average length.

    A recession episode is a sequence of consecutive True values.

    Args:
        recession_mask: Boolean array where True indicates a recession period.

    Returns:
        Tuple of (number of episodes, average episode length).
    """
    if not np.any(recession_mask):
        return 0, 0.0

    # Find start/end of each episode by detecting transitions
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
    """Compute percentage of values within a range.

    Args:
        values: Array of values to check.
        range_min: Minimum of the range (inclusive).
        range_max: Maximum of the range (inclusive).

    Returns:
        Fraction of valid (finite) values within [range_min, range_max].
    """
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return 0.0
    within = np.sum((valid >= range_min) & (valid <= range_max))
    return float(within / len(valid))


def compute_growth_plus_metrics(
    sim: bam.Simulation,
    results: bam.SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 150.0,
) -> GrowthPlusMetrics:
    """Compute all validation metrics from Growth+ simulation results.

    This function computes metrics corresponding to the 8 figures in
    Delli Gatti et al. (2011), Section 3.9.2, with additional growth-specific
    metrics for productivity and real wage trends.

    Args:
        sim: The simulation instance (needed for n_households)
        results: SimulationResults from sim.run() with the required data collected
        burn_in: Number of initial periods to exclude for steady-state analysis
        firm_size_threshold: Threshold for firm size distribution percentile
            (default 150.0 for Growth+ vs 5.0 for baseline due to productivity growth)

    Returns:
        GrowthPlusMetrics dataclass containing all computed metrics
    """
    # =========================================================================
    # Extract raw data from results
    # =========================================================================
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

    # LoanBook data is stored as lists of arrays (one per period)
    loan_principals = results.relationship_data["LoanBook"]["principal"]
    loan_rates = results.relationship_data["LoanBook"]["rate"]
    loan_source_ids = results.relationship_data["LoanBook"]["source_ids"]

    # =========================================================================
    # Compute time series metrics
    # =========================================================================

    # Unemployment rate: fraction of households not employed
    unemployment = 1 - ops.mean(employed.astype(float), axis=1)

    # GDP and log GDP
    gdp = ops.sum(production, axis=1)
    log_gdp = ops.log(gdp)

    # Production-weighted average labor productivity
    weighted_productivity = ops.sum(
        ops.multiply(labor_productivity, production), axis=1
    )
    avg_productivity = ops.divide(weighted_productivity, gdp)

    # Average wage for employed workers only
    # (unemployed workers have wage=0, which would skew the average)
    employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
    employed_count = ops.sum(employed, axis=1)
    avg_employed_wage = ops.where(
        ops.greater(employed_count, 0),
        ops.divide(employed_wages_sum, employed_count),
        0.0,
    )

    # Real wage: nominal wage / price level
    real_wage = ops.divide(avg_employed_wage, avg_price)

    # Vacancy rate: total vacancies / total households
    total_vacancies = ops.sum(n_vacancies, axis=1)
    vacancy_rate = ops.divide(total_vacancies, sim.n_households)

    # =========================================================================
    # Compute curve data (growth rates for scatter plots)
    # =========================================================================

    # Wage inflation: period-over-period wage change
    wage_inflation = ops.divide(
        avg_employed_wage[1:] - avg_employed_wage[:-1],
        ops.where(ops.greater(avg_employed_wage[:-1], 0), avg_employed_wage[:-1], 1.0),
    )

    # GDP growth rate
    gdp_growth = ops.divide(gdp[1:] - gdp[:-1], gdp[:-1])

    # Unemployment growth rate
    unemployment_growth = ops.divide(
        unemployment[1:] - unemployment[:-1],
        ops.where(ops.greater(unemployment[:-1], 0), unemployment[:-1], 1.0),
    )

    # Productivity growth rate (Growth+ specific)
    productivity_growth_rate = ops.divide(
        avg_productivity[1:] - avg_productivity[:-1],
        ops.where(ops.greater(avg_productivity[:-1], 0), avg_productivity[:-1], 1.0),
    )

    # =========================================================================
    # Apply burn-in and compute summary statistics
    # =========================================================================
    unemployment_ss = unemployment[burn_in:]
    inflation_ss = inflation[burn_in:]
    log_gdp_ss = log_gdp[burn_in:]
    real_wage_ss = real_wage[burn_in:]
    vacancy_rate_ss = vacancy_rate[burn_in:]
    avg_productivity_ss = avg_productivity[burn_in:]

    # Curve data aligned with burn-in
    # (growth rates have length n-1, so burn_in-1 aligns with period burn_in)
    wage_inflation_ss = wage_inflation[burn_in - 1 :]
    gdp_growth_ss = gdp_growth[burn_in - 1 :]
    unemployment_growth_ss = unemployment_growth[burn_in - 1 :]

    # =========================================================================
    # Compute correlations for macroeconomic curves
    # =========================================================================

    # Phillips curve: wage inflation vs unemployment
    phillips_corr = float(np.corrcoef(unemployment_ss, wage_inflation_ss)[0, 1])

    # Okun curve: output growth vs unemployment growth (with IQR outlier filtering)
    unemp_filtered, gdp_filtered = filter_outliers_iqr(
        unemployment_growth_ss, gdp_growth_ss
    )
    okun_corr = float(np.corrcoef(unemp_filtered, gdp_filtered)[0, 1])

    # Beveridge curve: vacancy rate vs unemployment
    beveridge_corr = float(np.corrcoef(unemployment_ss, vacancy_rate_ss)[0, 1])

    # =========================================================================
    # Compute firm size distribution metrics
    # =========================================================================
    final_production_arr = production[-1]
    firm_size_skewness = float(stats.skew(final_production_arr))
    firm_size_pct_below = float(
        np.sum(final_production_arr < firm_size_threshold) / len(final_production_arr)
    )

    # =========================================================================
    # Compute Growth+ specific metrics
    # =========================================================================

    # Initial and final productivity (after burn-in)
    initial_productivity = float(avg_productivity_ss[0])
    final_productivity_val = float(avg_productivity_ss[-1])
    total_productivity_growth = (
        (final_productivity_val - initial_productivity) / initial_productivity
        if initial_productivity > 0
        else 0.0
    )

    # Initial and final real wage (after burn-in)
    real_wage_initial = float(real_wage_ss[0])
    real_wage_final = float(real_wage_ss[-1])
    total_real_wage_growth = (
        (real_wage_final - real_wage_initial) / real_wage_initial
        if real_wage_initial > 0
        else 0.0
    )

    # Productivity trend coefficient via linear regression on log(productivity)
    # y = a + b*t, where b is the trend coefficient
    time_axis = np.arange(len(avg_productivity_ss))
    log_productivity = np.log(
        np.where(avg_productivity_ss > 0, avg_productivity_ss, EPS)
    )
    # Use numpy polyfit for linear regression
    trend_coef, _ = np.polyfit(time_axis, log_productivity, 1)
    productivity_trend_coefficient = float(trend_coef)

    # =========================================================================
    # Compute financial dynamics metrics
    # =========================================================================

    # Real interest rate: weighted average nominal rate minus inflation
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
            weighted_nominal = sim.r_bar  # Fallback to baseline rate
        real_interest_rate[t] = weighted_nominal - inflation[t]

    # Average financial fragility = total_wage_bill / total_net_worth
    # This is the aggregate wage-bill to equity ratio for the economy
    total_wage_bill = ops.sum(wages * employed.astype(float), axis=1)
    total_net_worth = ops.sum(net_worth, axis=1)
    # Safe division avoiding zero net worth
    safe_total_nw = ops.where(ops.greater(total_net_worth, EPS), total_net_worth, EPS)
    avg_financial_fragility = ops.divide(total_wage_bill, safe_total_nw)

    # Price ratio: market price / market-clearing price
    # Market price P = production-weighted average price (what consumers actually pay)
    # Market-clearing price P* = total demand / total supply
    safe_gdp = ops.where(ops.greater(gdp, EPS), gdp, EPS)
    total_demand = ops.sum(consumer_budget, axis=1)
    market_clearing_price = ops.divide(total_demand, safe_gdp)
    price_ratio = ops.divide(avg_price, market_clearing_price)

    # Dispersions (coefficient of variation = weighted_std / weighted_mean)
    # Production-weighted std: sqrt(sum(w * (x - w_mean)^2) / sum(w))
    # This weights dispersion by economic activity, reducing noise from small firms
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

    # =========================================================================
    # Compute growth rate distributions (GDP and net worth, after burn-in)
    # =========================================================================
    # Figure 3.6a shows GDP growth rate distribution across periods (not firm-level)
    gdp_after_burnin = gdp[burn_in:]
    output_growth_rates = np.diff(gdp_after_burnin) / gdp_after_burnin[:-1]

    # Figure 3.6b: firm-level net worth growth rate (last period, cross-sectional)
    # Exclude firms that went bankrupt (nw_final <= 0) or were just replaced (nw_prev <= 0)
    nw_prev = net_worth[-2]  # All firms at period T-1
    nw_final = net_worth[-1]  # All firms at period T
    valid_firms = (nw_prev > 0) & (nw_final > 0)
    nw_prev_valid = nw_prev[valid_firms]
    nw_final_valid = nw_final[valid_firms]
    networth_growth_rates = (nw_final_valid - nw_prev_valid) / nw_prev_valid

    # =========================================================================
    # Compute tiered distribution metrics
    # =========================================================================
    # Output growth rate distribution metrics
    output_growth_pct_within_tight = _compute_pct_within_range(
        output_growth_rates, -0.05, 0.05
    )
    output_growth_pct_within_normal = _compute_pct_within_range(
        output_growth_rates, -0.10, 0.10
    )
    output_growth_pct_outliers = 1.0 - _compute_pct_within_range(
        output_growth_rates, -0.15, 0.15
    )

    # Net worth growth rate distribution metrics
    networth_growth_pct_within_tight = _compute_pct_within_range(
        networth_growth_rates, -0.05, 0.05
    )
    networth_growth_pct_within_normal = _compute_pct_within_range(
        networth_growth_rates, -0.10, 0.10
    )
    networth_growth_pct_outliers = 1.0 - _compute_pct_within_range(
        networth_growth_rates, -0.50, 0.20
    )

    # =========================================================================
    # Recession detection (peak-to-trough algorithm)
    # =========================================================================
    # Use peak-to-trough detection for broader recession episodes that match
    # the book's recession shading pattern (includes slowdowns, partial recovery)
    recession_mask = detect_recessions(log_gdp)
    n_recessions, avg_recession_length = _count_recession_episodes(
        recession_mask[burn_in:]
    )

    # =========================================================================
    # Minsky classification (averaged over post-burn-in periods)
    # =========================================================================
    # Hedge: total_funds >= debt (can pay principal + interest)
    # Speculative: interest <= total_funds < debt (can pay interest only)
    # Ponzi: total_funds < interest (cannot even pay interest)
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

    # Additional summary statistics (after burn-in)
    bankruptcies_ss = n_firm_bankruptcies[burn_in:]
    real_ir_ss = real_interest_rate[burn_in:]
    fragility_ss = avg_financial_fragility[burn_in:]
    price_ratio_ss = price_ratio[burn_in:]
    price_disp_ss = price_dispersion[burn_in:]
    equity_disp_ss = equity_dispersion[burn_in:]
    sales_disp_ss = sales_dispersion[burn_in:]

    # =========================================================================
    # Return all metrics
    # =========================================================================
    return GrowthPlusMetrics(
        # Time series (full)
        unemployment=unemployment,
        inflation=inflation,
        log_gdp=log_gdp,
        real_wage=real_wage,
        avg_productivity=avg_productivity,
        vacancy_rate=vacancy_rate,
        # Curve data
        wage_inflation=wage_inflation,
        gdp_growth=gdp_growth,
        unemployment_growth=unemployment_growth,
        # Distribution
        final_production=final_production_arr,
        # Summary statistics
        unemployment_mean=float(np.mean(unemployment_ss)),
        unemployment_std=float(np.std(unemployment_ss)),
        inflation_mean=float(np.mean(inflation_ss)),
        inflation_std=float(np.std(inflation_ss)),
        log_gdp_mean=float(np.mean(log_gdp_ss)),
        log_gdp_std=float(np.std(log_gdp_ss)),
        real_wage_mean=float(np.mean(real_wage_ss)),
        real_wage_std=float(np.std(real_wage_ss)),
        vacancy_rate_mean=float(np.mean(vacancy_rate_ss)),
        # Correlations
        phillips_corr=phillips_corr,
        okun_corr=okun_corr,
        beveridge_corr=beveridge_corr,
        # Distribution metrics
        firm_size_skewness=firm_size_skewness,
        firm_size_pct_below_threshold=firm_size_pct_below,
        # Growth-specific metrics
        productivity_growth_rate=productivity_growth_rate,
        productivity_trend_coefficient=productivity_trend_coefficient,
        initial_productivity=initial_productivity,
        final_productivity=final_productivity_val,
        total_productivity_growth=total_productivity_growth,
        real_wage_initial=real_wage_initial,
        real_wage_final=real_wage_final,
        total_real_wage_growth=total_real_wage_growth,
        # Financial dynamics metrics
        n_firm_bankruptcies=n_firm_bankruptcies,
        n_bank_bankruptcies=n_bank_bankruptcies,
        real_interest_rate=real_interest_rate,
        avg_financial_fragility=avg_financial_fragility,
        price_ratio=price_ratio,
        price_dispersion=price_dispersion,
        equity_dispersion=equity_dispersion,
        sales_dispersion=sales_dispersion,
        # Growth rate distributions
        output_growth_rates=output_growth_rates,
        networth_growth_rates=networth_growth_rates,
        # Tiered distribution metrics
        output_growth_pct_within_tight=output_growth_pct_within_tight,
        output_growth_pct_within_normal=output_growth_pct_within_normal,
        output_growth_pct_outliers=output_growth_pct_outliers,
        networth_growth_pct_within_tight=networth_growth_pct_within_tight,
        networth_growth_pct_within_normal=networth_growth_pct_within_normal,
        networth_growth_pct_outliers=networth_growth_pct_outliers,
        # Recession detection
        recession_mask=recession_mask,
        n_recessions=n_recessions,
        avg_recession_length=avg_recession_length,
        # Minsky classification
        minsky_hedge_pct=minsky_hedge_pct,
        minsky_speculative_pct=minsky_speculative_pct,
        minsky_ponzi_pct=minsky_ponzi_pct,
        # Additional summary statistics
        bankruptcies_mean=float(np.mean(bankruptcies_ss)),
        real_interest_rate_mean=float(np.mean(real_ir_ss)),
        real_interest_rate_std=float(np.std(real_ir_ss)),
        avg_fragility_mean=float(np.mean(fragility_ss)),
        avg_fragility_std=float(np.std(fragility_ss)),
        price_ratio_mean=float(np.mean(price_ratio_ss)),
        price_ratio_std=float(np.std(price_ratio_ss)),
        price_dispersion_mean=float(np.mean(price_disp_ss)),
        price_dispersion_std=float(np.std(price_disp_ss)),
        equity_dispersion_mean=float(np.mean(equity_disp_ss)),
        equity_dispersion_std=float(np.std(equity_disp_ss)),
        sales_dispersion_mean=float(np.mean(sales_disp_ss)),
        sales_dispersion_std=float(np.std(sales_disp_ss)),
    )


# Standard collection config for Growth+ scenario
GROWTH_PLUS_COLLECT_CONFIG = {
    "Producer": ["production", "labor_productivity", "price", "inventory"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth", "gross_profit", "total_funds"],
    "Consumer": ["income_to_spend"],
    "LoanBook": ["principal", "rate", "source_ids"],
    "Economy": True,
    "aggregate": None,
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


def load_growth_plus_targets() -> dict[str, dict[str, float]]:
    """Load Growth+ validation targets from YAML.

    Returns a dictionary in the BOUNDS format used by scenario visualizations,
    with keys: log_gdp, unemployment, inflation, productivity, real_wage,
    phillips_corr, okun_corr, beveridge_corr, firm_size.

    Returns
    -------
    dict[str, dict[str, float]]
        Targets dictionary compatible with scenario visualization code.
    """
    import os

    import yaml

    yaml_path = os.path.join(get_targets_dir(), "growth_plus.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    ts = data["time_series"]
    curves = data["curves"]
    dist = data["distributions"]
    fin = data.get("financial_dynamics", {})

    return {
        "log_gdp": {
            "normal_min": ts["log_gdp"]["targets"]["normal_min"],
            "normal_max": ts["log_gdp"]["targets"]["normal_max"],
            "extreme_min": ts["log_gdp"]["targets"]["extreme_min"],
            "extreme_max": ts["log_gdp"]["targets"]["extreme_max"],
            "mean_target": ts["log_gdp"]["targets"]["mean_target"],
        },
        "unemployment": {
            "normal_min": ts["unemployment_rate"]["targets"]["normal_min"],
            "normal_max": ts["unemployment_rate"]["targets"]["normal_max"],
            "extreme_min": ts["unemployment_rate"]["targets"]["extreme_min"],
            "extreme_max": ts["unemployment_rate"]["targets"]["extreme_max"],
            "mean_target": ts["unemployment_rate"]["targets"]["mean_target"],
        },
        "inflation": {
            "normal_min": ts["inflation_rate"]["targets"]["normal_min"],
            "normal_max": ts["inflation_rate"]["targets"]["normal_max"],
            "extreme_min": ts["inflation_rate"]["targets"]["extreme_min"],
            "extreme_max": ts["inflation_rate"]["targets"]["extreme_max"],
            "mean_target": ts["inflation_rate"]["targets"]["mean_target"],
        },
        "productivity": {
            "initial_min": ts["productivity"]["targets"]["initial_min"],
            "initial_max": ts["productivity"]["targets"]["initial_max"],
            "final_min": ts["productivity"]["targets"]["final_min"],
            "final_max": ts["productivity"]["targets"]["final_max"],
            "total_growth_min": ts["productivity"]["targets"]["total_growth_min"],
            "total_growth_max": ts["productivity"]["targets"]["total_growth_max"],
        },
        "real_wage": {
            "initial_min": ts["real_wage"]["targets"]["initial_min"],
            "initial_max": ts["real_wage"]["targets"]["initial_max"],
            "final_min": ts["real_wage"]["targets"]["final_min"],
            "final_max": ts["real_wage"]["targets"]["final_max"],
            "total_growth_min": ts["real_wage"]["targets"]["total_growth_min"],
            "total_growth_max": ts["real_wage"]["targets"]["total_growth_max"],
        },
        "phillips_corr": {
            "target": curves["phillips"]["targets"]["correlation_target"],
            "min": curves["phillips"]["targets"]["correlation_min"],
            "max": curves["phillips"]["targets"]["correlation_max"],
        },
        "okun_corr": {
            "target": curves["okun"]["targets"]["correlation_target"],
            "min": curves["okun"]["targets"]["correlation_min"],
            "max": curves["okun"]["targets"]["correlation_max"],
        },
        "beveridge_corr": {
            "target": curves["beveridge"]["targets"]["correlation_target"],
            "min": curves["beveridge"]["targets"]["correlation_min"],
            "max": curves["beveridge"]["targets"]["correlation_max"],
        },
        "firm_size": {
            "threshold": dist["firm_size"]["targets"]["threshold_small"],
            "pct_below_target": dist["firm_size"]["targets"]["pct_below_small_target"],
            "skewness_min": dist["firm_size"]["targets"]["skewness_min"],
            "skewness_max": dist["firm_size"]["targets"]["skewness_max"],
        },
        "bankruptcies": {
            "mean_target": fin["bankruptcies"]["targets"]["mean_target"],
            "mean_tolerance": fin["bankruptcies"]["targets"]["mean_tolerance"],
        },
        "real_interest_rate": {
            "mean_target": fin["real_interest_rate"]["targets"]["mean_target"],
            "mean_tolerance": fin["real_interest_rate"]["targets"]["mean_tolerance"],
            "std_target": fin["real_interest_rate"]["targets"]["std_target"],
            "std_tolerance": fin["real_interest_rate"]["targets"]["std_tolerance"],
            "normal_min": fin["real_interest_rate"]["targets"]["normal_min"],
            "normal_max": fin["real_interest_rate"]["targets"]["normal_max"],
            "extreme_min": fin["real_interest_rate"]["targets"]["extreme_min"],
            "extreme_max": fin["real_interest_rate"]["targets"]["extreme_max"],
        },
        "financial_fragility": {
            "mean_target": fin["financial_fragility"]["targets"]["mean_target"],
            "mean_tolerance": fin["financial_fragility"]["targets"]["mean_tolerance"],
            "std_target": fin["financial_fragility"]["targets"]["std_target"],
            "std_tolerance": fin["financial_fragility"]["targets"]["std_tolerance"],
            "normal_min": fin["financial_fragility"]["targets"]["normal_min"],
            "normal_max": fin["financial_fragility"]["targets"]["normal_max"],
            "extreme_min": fin["financial_fragility"]["targets"]["extreme_min"],
            "extreme_max": fin["financial_fragility"]["targets"]["extreme_max"],
        },
        "price_ratio": {
            "mean_target": fin["price_ratio"]["targets"]["mean_target"],
            "mean_tolerance": fin["price_ratio"]["targets"]["mean_tolerance"],
            "std_target": fin["price_ratio"]["targets"]["std_target"],
            "std_tolerance": fin["price_ratio"]["targets"]["std_tolerance"],
            "normal_min": fin["price_ratio"]["targets"]["normal_min"],
            "normal_max": fin["price_ratio"]["targets"]["normal_max"],
            "extreme_min": fin["price_ratio"]["targets"]["extreme_min"],
            "extreme_max": fin["price_ratio"]["targets"]["extreme_max"],
        },
        "price_dispersion": {
            "mean_target": fin["price_dispersion"]["targets"]["mean_target"],
            "mean_tolerance": fin["price_dispersion"]["targets"]["mean_tolerance"],
            "std_target": fin["price_dispersion"]["targets"]["std_target"],
            "std_tolerance": fin["price_dispersion"]["targets"]["std_tolerance"],
            "normal_min": fin["price_dispersion"]["targets"]["normal_min"],
            "normal_max": fin["price_dispersion"]["targets"]["normal_max"],
            "extreme_min": fin["price_dispersion"]["targets"]["extreme_min"],
            "extreme_max": fin["price_dispersion"]["targets"]["extreme_max"],
        },
        "equity_dispersion": {
            "mean_target": fin["equity_dispersion"]["targets"]["mean_target"],
            "mean_tolerance": fin["equity_dispersion"]["targets"]["mean_tolerance"],
            "std_target": fin["equity_dispersion"]["targets"]["std_target"],
            "std_tolerance": fin["equity_dispersion"]["targets"]["std_tolerance"],
            "normal_min": fin["equity_dispersion"]["targets"]["normal_min"],
            "normal_max": fin["equity_dispersion"]["targets"]["normal_max"],
            "extreme_min": fin["equity_dispersion"]["targets"]["extreme_min"],
            "extreme_max": fin["equity_dispersion"]["targets"]["extreme_max"],
        },
        "sales_dispersion": {
            "mean_target": fin["sales_dispersion"]["targets"]["mean_target"],
            "mean_tolerance": fin["sales_dispersion"]["targets"]["mean_tolerance"],
            "std_target": fin["sales_dispersion"]["targets"]["std_target"],
            "std_tolerance": fin["sales_dispersion"]["targets"]["std_tolerance"],
            "normal_min": fin["sales_dispersion"]["targets"]["normal_min"],
            "normal_max": fin["sales_dispersion"]["targets"]["normal_max"],
            "extreme_min": fin["sales_dispersion"]["targets"]["extreme_min"],
            "extreme_max": fin["sales_dispersion"]["targets"]["extreme_max"],
        },
        "minsky": {
            "hedge_pct_target": fin["minsky_classification"]["targets"][
                "hedge_pct_target"
            ],
            "hedge_pct_min": fin["minsky_classification"]["targets"]["hedge_pct_min"],
            "hedge_pct_max": fin["minsky_classification"]["targets"]["hedge_pct_max"],
            "ponzi_pct_target": fin["minsky_classification"]["targets"][
                "ponzi_pct_target"
            ],
            "ponzi_pct_min": fin["minsky_classification"]["targets"]["ponzi_pct_min"],
            "ponzi_pct_max": fin["minsky_classification"]["targets"]["ponzi_pct_max"],
        },
        # Raw financial_dynamics section for visualization code
        "financial_dynamics": fin,
    }
