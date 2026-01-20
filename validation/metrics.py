"""Shared metric computation for BAM validation.

This module provides functions to compute validation metrics from simulation
results. These metrics are used both by example scripts for visualization
and by validation tests for comparing against book targets.

The metrics correspond to the 8 figures in Delli Gatti et al. (2011),
Section 3.9.1:
    1. Log GDP (real output)
    2. Unemployment rate
    3. Inflation rate
    4. Productivity / Real wage
    5. Phillips curve (wage inflation vs unemployment)
    6. Okun curve (output growth vs unemployment growth)
    7. Beveridge curve (vacancy rate vs unemployment)
    8. Firm size distribution
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

import bamengine as bam
from bamengine import ops


def _filter_outliers_iqr(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Filter outliers using IQR method for both x and y variables.

    Keeps points where both x and y are within 1.5×IQR of their quartiles.
    This is the standard statistical method for outlier detection.

    Parameters
    ----------
    x : NDArray
        First variable (e.g., unemployment growth rate).
    y : NDArray
        Second variable (e.g., GDP growth rate).

    Returns
    -------
    tuple[NDArray, NDArray]
        Filtered x and y arrays with outliers removed.
    """
    # Handle empty or too-small arrays
    if len(x) < 4 or len(y) < 4:
        # Need at least 4 points for meaningful quartiles
        return x, y

    # Compute IQR bounds for x
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    x_lower, x_upper = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x

    # Compute IQR bounds for y
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    y_lower, y_upper = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y

    # Keep points within bounds for BOTH variables
    mask = (x >= x_lower) & (x <= x_upper) & (y >= y_lower) & (y <= y_upper)
    return x[mask], y[mask]


@dataclass
class BaselineMetrics:
    """All computed metrics from a baseline simulation run.

    Attributes:
        Time series (full, before burn-in applied):
            unemployment: Unemployment rate per period
            inflation: Inflation rate per period
            log_gdp: Log of total production per period
            real_wage: Average real wage of employed workers per period
            avg_productivity: Production-weighted average labor productivity
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


def compute_baseline_metrics(
    sim: bam.Simulation,
    results: bam.SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 5.0,
) -> BaselineMetrics:
    """Compute all validation metrics from baseline simulation results.

    This function computes metrics corresponding to the 8 figures in
    Delli Gatti et al. (2011), Section 3.9.1.

    Args:
        sim: The simulation instance (needed for n_households)
        results: SimulationResults from sim.run() with the required data collected
        burn_in: Number of initial periods to exclude for steady-state analysis
        firm_size_threshold: Threshold for firm size distribution percentile

    Returns:
        BaselineMetrics dataclass containing all computed metrics

    Required collection config for results:
        collect={
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
            "Employer": ["n_vacancies"],
            "Economy": True,
            "aggregate": None,
            "capture_timing": {
                "Worker.wage": "workers_receive_wage",
                "Worker.employed": "firms_run_production",
                "Producer.production": "firms_run_production",
                "Employer.n_vacancies": "firms_decide_vacancies",
            },
        }
    """
    # =========================================================================
    # Extract raw data from results
    # =========================================================================
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    labor_productivity = results.role_data["Producer"]["labor_productivity"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]

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

    # =========================================================================
    # Apply burn-in and compute summary statistics
    # =========================================================================
    unemployment_ss = unemployment[burn_in:]
    inflation_ss = inflation[burn_in:]
    log_gdp_ss = log_gdp[burn_in:]
    real_wage_ss = real_wage[burn_in:]
    vacancy_rate_ss = vacancy_rate[burn_in:]

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
    unemp_filtered, gdp_filtered = _filter_outliers_iqr(
        unemployment_growth_ss, gdp_growth_ss
    )
    okun_corr = float(np.corrcoef(unemp_filtered, gdp_filtered)[0, 1])

    # Beveridge curve: vacancy rate vs unemployment
    beveridge_corr = float(np.corrcoef(unemployment_ss, vacancy_rate_ss)[0, 1])

    # =========================================================================
    # Compute firm size distribution metrics
    # =========================================================================
    final_production = production[-1]
    firm_size_skewness = float(stats.skew(final_production))
    firm_size_pct_below = float(
        np.sum(final_production < firm_size_threshold) / len(final_production)
    )

    # =========================================================================
    # Return all metrics
    # =========================================================================
    return BaselineMetrics(
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
        final_production=final_production,
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
    )


# Standard collection config for baseline scenario
BASELINE_COLLECT_CONFIG = {
    "Producer": ["production", "labor_productivity"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Economy": True,
    "aggregate": None,
    "capture_timing": {
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
    },
}


# =============================================================================
# Growth+ Scenario Metrics (Section 3.8)
# =============================================================================


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


def compute_growth_plus_metrics(
    sim: bam.Simulation,
    results: bam.SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 150.0,
) -> GrowthPlusMetrics:
    """Compute all validation metrics from Growth+ simulation results.

    This function computes metrics corresponding to the 8 figures in
    Delli Gatti et al. (2011), Section 3.8, with additional growth-specific
    metrics for productivity and real wage trends.

    Args:
        sim: The simulation instance (needed for n_households)
        results: SimulationResults from sim.run() with the required data collected
        burn_in: Number of initial periods to exclude for steady-state analysis
        firm_size_threshold: Threshold for firm size distribution percentile
            (default 150.0 for Growth+ vs 5.0 for baseline due to productivity growth)

    Returns:
        GrowthPlusMetrics dataclass containing all computed metrics

    Required collection config for results:
        collect={
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
            "Employer": ["n_vacancies"],
            "Economy": True,
            "aggregate": None,
            "capture_timing": {
                "Worker.wage": "workers_receive_wage",
                "Worker.employed": "firms_run_production",
                "Producer.production": "firms_run_production",
                "Producer.labor_productivity": "firms_apply_productivity_growth",
                "Employer.n_vacancies": "firms_decide_vacancies",
            },
        }
    """
    # =========================================================================
    # Extract raw data from results
    # =========================================================================
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    labor_productivity = results.role_data["Producer"]["labor_productivity"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]

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
    unemp_filtered, gdp_filtered = _filter_outliers_iqr(
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
        np.where(avg_productivity_ss > 0, avg_productivity_ss, 1e-10)
    )
    # Use numpy polyfit for linear regression
    trend_coef, _ = np.polyfit(time_axis, log_productivity, 1)
    productivity_trend_coefficient = float(trend_coef)

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
    )


# Standard collection config for Growth+ scenario
GROWTH_PLUS_COLLECT_CONFIG = {
    "Producer": ["production", "labor_productivity"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Economy": True,
    "aggregate": None,
    "capture_timing": {
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Producer.labor_productivity": "firms_apply_productivity_growth",
        "Employer.n_vacancies": "firms_decide_vacancies",
    },
}


# =============================================================================
# Target Loading Utilities
# =============================================================================


def _get_targets_dir() -> str:
    """Get the path to the validation/targets directory."""
    import os

    return os.path.join(os.path.dirname(__file__), "targets")


def load_baseline_targets() -> dict[str, dict[str, float]]:
    """Load baseline validation targets from YAML.

    Returns a dictionary in the BOUNDS format used by example visualizations,
    with keys: log_gdp, unemployment, inflation, real_wage, phillips_corr,
    okun_corr, beveridge_corr, firm_size.

    Returns
    -------
    dict[str, dict[str, float]]
        Targets dictionary compatible with example visualization code.
    """
    import os

    import yaml

    yaml_path = os.path.join(_get_targets_dir(), "baseline.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    ts = data["time_series"]
    curves = data["curves"]
    dist = data["distributions"]

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
        "real_wage": {
            "normal_min": ts["real_wage"]["targets"]["normal_min"],
            "normal_max": ts["real_wage"]["targets"]["normal_max"],
            "extreme_min": ts["real_wage"]["targets"]["extreme_min"],
            "extreme_max": ts["real_wage"]["targets"]["extreme_max"],
            "mean_target": ts["real_wage"]["targets"]["mean_target"],
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
    }


def load_growth_plus_targets() -> dict[str, dict[str, float]]:
    """Load Growth+ validation targets from YAML.

    Returns a dictionary in the BOUNDS format used by example visualizations,
    with keys: log_gdp, unemployment, inflation, productivity, real_wage,
    phillips_corr, okun_corr, beveridge_corr, firm_size.

    Returns
    -------
    dict[str, dict[str, float]]
        Targets dictionary compatible with example visualization code.
    """
    import os

    import yaml

    yaml_path = os.path.join(_get_targets_dir(), "growth_plus.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    ts = data["time_series"]
    curves = data["curves"]
    dist = data["distributions"]

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
    }
