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

    # Okun curve: output growth vs unemployment growth
    okun_corr = float(np.corrcoef(unemployment_growth_ss, gdp_growth_ss)[0, 1])

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
