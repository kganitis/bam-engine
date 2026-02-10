"""Buffer-stock consumption scenario definition (Section 3.9.3).

This module defines the buffer-stock validation scenario from Delli Gatti et al.
(2011). It contains the metrics dataclass, computation function, and scenario
configuration.

The buffer-stock scenario replaces the baseline mean-field MPC with an
individual adaptive rule based on buffer-stock saving theory. Validation
focuses on reproducing the wealth distribution (Figure 3.8) fitted with
Singh-Maddala, Dagum, and GB2 distributions.

For visualization, see buffer_stock_viz.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats

import bamengine as bam
from bamengine import SimulationResults, ops
from validation.scenarios._utils import adjust_burn_in, filter_outliers_iqr
from validation.types import CheckType, MetricFormat, MetricGroup, MetricSpec, Scenario

# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class BufferStockMetrics:
    """All computed metrics from a buffer-stock simulation run.

    Includes baseline macro metrics, distribution fitting results,
    and buffer-stock specific metrics.
    """

    # Time series (full, for visualization)
    unemployment: NDArray[np.floating]
    inflation: NDArray[np.floating]
    log_gdp: NDArray[np.floating]
    real_wage: NDArray[np.floating]
    vacancy_rate: NDArray[np.floating]

    # Curve data
    wage_inflation: NDArray[np.floating]
    gdp_growth: NDArray[np.floating]
    unemployment_growth: NDArray[np.floating]

    # Distribution data
    final_production: NDArray[np.floating]
    final_savings: NDArray[np.floating]
    final_propensity: NDArray[np.floating]

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
    vacancy_rate_pct_in_bounds: float

    # Correlations
    phillips_corr: float
    okun_corr: float
    beveridge_corr: float

    # Distribution metrics (firm size)
    firm_size_skewness: float
    firm_size_pct_below_threshold: float
    firm_size_tail_ratio: float
    firm_size_pct_below_medium: float

    # Distribution fitting (Singh-Maddala)
    sm_params: tuple
    sm_ks_stat: float
    sm_ccdf_r2: float

    # Distribution fitting (Dagum)
    dagum_params: tuple
    dagum_ks_stat: float
    dagum_ccdf_r2: float

    # Distribution fitting (GB2 via betaprime)
    gb2_params: tuple
    gb2_ks_stat: float
    gb2_ccdf_r2: float

    # Best distribution fit
    best_fit: str
    best_r2: float

    # Buffer-stock specific metrics
    wealth_gini: float
    wealth_skewness: float
    mean_mpc: float
    std_mpc: float
    pct_dissaving: float

    # Financial dynamics (subset)
    n_firm_bankruptcies: NDArray[np.int_]
    bankruptcies_mean: float
    real_interest_rate: NDArray[np.floating]
    real_interest_rate_mean: float


# =============================================================================
# Helper Functions
# =============================================================================


def _ccdf_r2_loglog(
    data: NDArray[np.floating],
    dist: stats.rv_continuous,
    params: tuple,
) -> float:
    """Compute R-squared between empirical and theoretical CCDF on log-log scale."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    empirical_ccdf = 1.0 - np.arange(1, n + 1) / n
    theoretical_ccdf = 1.0 - dist.cdf(sorted_data, *params)

    # Filter for log-log (both must be positive and finite)
    mask = (
        (empirical_ccdf > 0)
        & (theoretical_ccdf > 0)
        & (sorted_data > 0)
        & np.isfinite(theoretical_ccdf)
    )
    if np.sum(mask) < 10:
        return 0.0

    log_emp = np.log10(empirical_ccdf[mask])
    log_theo = np.log10(theoretical_ccdf[mask])

    ss_res = np.sum((log_emp - log_theo) ** 2)
    ss_tot = np.sum((log_emp - np.mean(log_emp)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _compute_gini(data: NDArray[np.floating]) -> float:
    """Compute Gini coefficient for a distribution."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n == 0 or np.sum(sorted_data) == 0:
        return 0.0
    cumulative = np.cumsum(sorted_data)
    return float(
        (2.0 * np.sum((np.arange(1, n + 1) * sorted_data)) / (n * cumulative[-1]))
        - (n + 1) / n
    )


def _fit_distributions(savings_data: NDArray[np.floating]) -> dict[str, Any]:
    """Fit GB2, Singh-Maddala, and Dagum to cross-section wealth data.

    Returns dict with keys 'SM', 'D', 'GB2', each containing
    (params, ks_stat, ccdf_r2).

    Singh-Maddala (burr12) and Dagum (mielke) both have ``x^k`` terms in
    their PDFs.  Unconstrained MLE sometimes converges to extreme shape
    parameters (k > 100), causing ``x^k`` overflow for wealth values in
    the tens or hundreds.  When overflow is detected (NaN in the CDF),
    we fall back to profile likelihood: fix k at values in [5..50],
    optimize the remaining parameters, and select the k that maximizes
    the log-likelihood.  GB2 (betaprime) does not exhibit this issue.
    """
    data = savings_data[savings_data > 0]
    if len(data) < 50:
        empty = ((), 1.0, 0.0)
        return {"SM": empty, "D": empty, "GB2": empty}

    results: dict[str, tuple[tuple, float, float]] = {}

    # Singh-Maddala = Burr Type XII (burr12 in scipy) — same x^c overflow
    # risk as Dagum; fall back to profile likelihood over c if needed.
    try:
        sm_params = None
        with np.errstate(over="ignore", invalid="ignore"):
            params = stats.burr12.fit(data, floc=0)
            test_cdf = stats.burr12.cdf(data[:5], *params)
            if np.all(np.isfinite(test_cdf)):
                sm_params = params
            else:
                best_ll = -np.inf
                for c in [2, 5, 10, 15, 20, 30, 50]:
                    try:
                        p = stats.burr12.fit(data, fc=c, floc=0)
                        ll = float(np.sum(stats.burr12.logpdf(data, *p)))
                        if np.isfinite(ll) and ll > best_ll:
                            best_ll = ll
                            sm_params = p
                    except Exception:
                        continue
        if sm_params is None:
            raise ValueError("Singh-Maddala fit failed")
        sm_ks = stats.kstest(data, "burr12", args=sm_params)
        sm_r2 = _ccdf_r2_loglog(data, stats.burr12, sm_params)
        results["SM"] = (sm_params, float(sm_ks.statistic), sm_r2)
    except Exception:
        results["SM"] = ((), 1.0, 0.0)

    # Dagum (Mielke/Burr Type III) — unconstrained MLE sometimes finds
    # extreme k (>100), causing x^k overflow.  Fall back to profile
    # likelihood over k when that happens.
    try:
        dagum_params = None
        with np.errstate(over="ignore", invalid="ignore"):
            params = stats.mielke.fit(data, floc=0)
            test_cdf = stats.mielke.cdf(data[:5], *params)
            if np.all(np.isfinite(test_cdf)):
                dagum_params = params
            else:
                # Profile likelihood: fix k, optimize s and scale
                best_ll = -np.inf
                for k in [5, 10, 15, 20, 30, 50]:
                    try:
                        p = stats.mielke.fit(data, fk=k, floc=0)
                        ll = float(np.sum(stats.mielke.logpdf(data, *p)))
                        if np.isfinite(ll) and ll > best_ll:
                            best_ll = ll
                            dagum_params = p
                    except Exception:
                        continue
        if dagum_params is None:
            raise ValueError("Dagum fit failed")
        dagum_ks = stats.kstest(data, "mielke", args=dagum_params)
        dagum_r2 = _ccdf_r2_loglog(data, stats.mielke, dagum_params)
        results["D"] = (dagum_params, float(dagum_ks.statistic), dagum_r2)
    except Exception:
        results["D"] = ((), 1.0, 0.0)

    # GB2 approximated via betaprime (generalized form)
    try:
        gb2_params = stats.betaprime.fit(data, floc=0)
        gb2_ks = stats.kstest(data, "betaprime", args=gb2_params)
        gb2_r2 = _ccdf_r2_loglog(data, stats.betaprime, gb2_params)
        results["GB2"] = (gb2_params, float(gb2_ks.statistic), gb2_r2)
    except Exception:
        results["GB2"] = ((), 1.0, 0.0)

    return results


def _compute_detrended_correlation(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> float:
    """Compute correlation of linearly detrended series."""
    if len(x) < 10 or len(y) < 10:
        return 0.0
    t = np.arange(len(x))
    try:
        x_trend = np.polyval(np.polyfit(t, x, 1), t)
        y_trend = np.polyval(np.polyfit(t, y, 1), t)
    except np.linalg.LinAlgError:
        return 0.0
    x_detrended = x - x_trend
    y_detrended = y - y_trend
    corr = np.corrcoef(x_detrended, y_detrended)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_buffer_stock_metrics(
    sim: bam.Simulation,
    results: SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 150.0,
    firm_size_threshold_medium: float = 100.0,
) -> BufferStockMetrics:
    """Compute all validation metrics from buffer-stock simulation results."""
    # Extract raw data
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]
    consumer_savings = results.role_data["Consumer"]["savings"]
    buf_propensity = results.role_data["BufferStock"]["propensity"]
    n_firm_bankruptcies = np.array(
        results.economy_data["n_firm_bankruptcies"], dtype=np.int_
    )

    loan_principals = results.relationship_data["LoanBook"]["principal"]
    loan_rates = results.relationship_data["LoanBook"]["rate"]

    # Compute time series
    unemployment = 1 - ops.mean(employed.astype(float), axis=1)
    gdp = ops.sum(production, axis=1)
    log_gdp = ops.log(gdp)

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

    # Apply burn-in
    unemployment_ss = unemployment[burn_in:]
    log_gdp_ss = log_gdp[burn_in:]
    real_wage_ss = real_wage[burn_in:]
    vacancy_rate_ss = vacancy_rate[burn_in:]
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

    # Firm size distribution
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

    # Wealth distribution fitting — exclude unemployed households.
    # Unemployed households have c=1/h (gradual savings drawdown) and converge to a
    # dividend-floor equilibrium, creating a spurious point mass that distorts
    # the heavy-tail shape the fitted distributions are meant to capture.
    final_employed = employed[-1].astype(bool)
    final_savings_arr = consumer_savings[-1][final_employed]
    dist_fits = _fit_distributions(final_savings_arr)

    sm_params, sm_ks_stat, sm_r2 = dist_fits["SM"]
    dagum_params, dagum_ks_stat, dagum_r2 = dist_fits["D"]
    gb2_params, gb2_ks_stat, gb2_r2 = dist_fits["GB2"]

    # Determine best fit
    fits = {"SM": sm_r2, "D": dagum_r2, "GB2": gb2_r2}
    best_fit = max(fits, key=fits.get)
    best_r2 = fits[best_fit]

    # Buffer-stock specific metrics (employed only)
    wealth_gini = _compute_gini(final_savings_arr)
    wealth_skewness = float(stats.skew(final_savings_arr))

    # MPC statistics from final period (employed only — unemployed have
    # a fixed c=1/h that doesn't reflect the buffer-stock formula)
    final_propensity = buf_propensity[-1][final_employed]

    # Adjust MPC for the dividend artifact: dividends inflate savings above
    # the labor-income buffer target S* = h*W, causing the formula to produce
    # c > 1 to drain the surplus. Subtracting D/W isolates the labor-income
    # component of the MPC. See plan context for detailed derivation.
    shareholder_dividends = results.role_data["Shareholder"]["dividends"]
    final_dividends = shareholder_dividends[-1][final_employed]
    final_wages = wages[-1][final_employed]
    adjustment = np.where(final_wages > 0, final_dividends / final_wages, 0.0)
    adjusted_propensity = final_propensity - adjustment

    mean_mpc = float(np.mean(adjusted_propensity))
    std_mpc = float(np.std(adjusted_propensity))
    pct_dissaving = float(np.sum(adjusted_propensity > 1.0) / len(adjusted_propensity))

    # Financial dynamics (subset)
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

    bankruptcies_ss = n_firm_bankruptcies[burn_in:]

    return BufferStockMetrics(
        unemployment=unemployment,
        inflation=inflation,
        log_gdp=log_gdp,
        real_wage=real_wage,
        vacancy_rate=vacancy_rate,
        wage_inflation=wage_inflation,
        gdp_growth=gdp_growth,
        unemployment_growth=unemployment_growth,
        final_production=final_production_arr,
        final_savings=final_savings_arr,
        final_propensity=final_propensity,
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
        sm_params=sm_params,
        sm_ks_stat=sm_ks_stat,
        sm_ccdf_r2=sm_r2,
        dagum_params=dagum_params,
        dagum_ks_stat=dagum_ks_stat,
        dagum_ccdf_r2=dagum_r2,
        gb2_params=gb2_params,
        gb2_ks_stat=gb2_ks_stat,
        gb2_ccdf_r2=gb2_r2,
        best_fit=best_fit,
        best_r2=best_r2,
        wealth_gini=wealth_gini,
        wealth_skewness=wealth_skewness,
        mean_mpc=mean_mpc,
        std_mpc=std_mpc,
        pct_dissaving=pct_dissaving,
        n_firm_bankruptcies=n_firm_bankruptcies,
        bankruptcies_mean=float(np.mean(bankruptcies_ss)),
        real_interest_rate=real_interest_rate,
        real_interest_rate_mean=float(np.mean(real_interest_rate[burn_in:])),
    )


# =============================================================================
# Collection Configuration
# =============================================================================

COLLECT_CONFIG = {
    "Producer": ["production"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth"],
    "Consumer": ["savings"],
    "BufferStock": ["propensity"],
    "Shareholder": ["dividends"],
    "LoanBook": ["principal", "rate"],
    "Economy": True,
    "aggregate": None,
    "capture_timing": {
        "Worker.wage": "firms_run_production",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "Borrower.net_worth": "firms_run_production",
        "Consumer.savings": None,  # end of period
        "BufferStock.propensity": "consumers_calc_buffer_stock_propensity",
        "Shareholder.dividends": "consumers_calc_buffer_stock_propensity",
        "LoanBook.principal": "banks_provide_loans",
        "LoanBook.rate": "banks_provide_loans",
        "Economy.n_firm_bankruptcies": "mark_bankrupt_firms",
    },
}

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "n_firms": 100,
    "n_households": 500,
    "n_banks": 10,
    # Buffer-stock extension parameter
    "buffer_stock_h": 2.0,
    # Growth+ R&D parameters (Section 3.9.3 builds on Growth+)
    "new_firm_size_factor": 0.5,
    "new_firm_production_factor": 0.5,
    "new_firm_wage_factor": 0.5,
    "new_firm_price_markup": 1.5,
    "max_loan_to_net_worth": 5,
    "job_search_method": "all_firms",
    "sigma_min": 0.0,
    "sigma_max": 0.1,
    "sigma_decay": -1.0,
}

# =============================================================================
# Metric Specifications
# =============================================================================

METRIC_SPECS = [
    # === Time series metrics ===
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
        invert=True,
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
        invert=False,
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
    # === Curve correlations ===
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
        weight=3.0,
        group=MetricGroup.CURVES,
        target_desc="correlation must be < 0",
    ),
    MetricSpec(
        name="okun_correlation",
        field="okun_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.okun_correlation",
        weight=3.0,
        group=MetricGroup.CURVES,
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
        weight=3.0,
        group=MetricGroup.CURVES,
        target_desc="correlation must be < 0",
    ),
    # === Distribution metrics (firm size) ===
    MetricSpec(
        name="firm_size_skewness",
        field="firm_size_skewness",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.firm_size_skewness",
        weight=1.0,
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
        weight=0.5,
        group=MetricGroup.DISTRIBUTION,
    ),
    # === Wealth distribution fitting ===
    MetricSpec(
        name="sm_ccdf_r2",
        field="sm_ccdf_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.sm_ccdf_r2",
        weight=3.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="dagum_ccdf_r2",
        field="dagum_ccdf_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.dagum_ccdf_r2",
        weight=3.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="gb2_ccdf_r2",
        field="gb2_ccdf_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.gb2_ccdf_r2",
        weight=2.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="best_r2",
        field="best_r2",
        check_type=CheckType.RANGE,
        target_path="metrics.best_r2",
        weight=3.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="wealth_gini",
        field="wealth_gini",
        check_type=CheckType.RANGE,
        target_path="metrics.wealth_gini",
        weight=2.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="wealth_skewness",
        field="wealth_skewness",
        check_type=CheckType.RANGE,
        target_path="metrics.wealth_skewness",
        weight=1.0,
        group=MetricGroup.DISTRIBUTION,
    ),
    # === Buffer-stock specific ===
    MetricSpec(
        name="mean_mpc",
        field="mean_mpc",
        check_type=CheckType.RANGE,
        target_path="metrics.mean_mpc",
        weight=1.5,
        group=MetricGroup.FINANCIAL,
    ),
    MetricSpec(
        name="pct_dissaving",
        field="pct_dissaving",
        check_type=CheckType.RANGE,
        target_path="metrics.pct_dissaving",
        weight=1.0,
        group=MetricGroup.FINANCIAL,
        format=MetricFormat.PERCENT,
    ),
    # === Financial dynamics ===
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
]


# =============================================================================
# Setup Hook for Buffer-Stock Extension
# =============================================================================


def _setup_buffer_stock(sim: bam.Simulation | None) -> None:
    """Setup hook to import and attach buffer-stock + RnD extensions."""
    if sim is None:
        # Pre-import call - just import to register event classes
        from extensions.buffer_stock import BufferStock
        from extensions.rnd import RnD
    else:
        # Attach both extensions to simulation
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock
        from extensions.rnd import RND_EVENTS, RnD

        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_role(RnD)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)


# =============================================================================
# Compute Metrics Wrapper
# =============================================================================


def _compute_metrics_wrapper(
    sim: bam.Simulation, results: SimulationResults, burn_in: int
) -> BufferStockMetrics:
    """Wrapper for compute_buffer_stock_metrics that loads params from YAML."""
    targets_path = Path(__file__).parent.parent / "targets" / "buffer_stock.yaml"
    with open(targets_path) as f:
        targets = yaml.safe_load(f)

    params = targets["metadata"]["params"]

    return compute_buffer_stock_metrics(
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
    name="buffer_stock",
    metric_specs=METRIC_SPECS,
    collect_config=COLLECT_CONFIG,
    targets_file="buffer_stock.yaml",
    default_config=DEFAULT_CONFIG,
    compute_metrics=_compute_metrics_wrapper,
    setup_hook=_setup_buffer_stock,
)


# =============================================================================
# Public API
# =============================================================================


def load_buffer_stock_targets() -> dict[str, Any]:
    """Load buffer-stock validation targets from YAML for visualization."""
    targets_path = Path(__file__).parent.parent / "targets" / "buffer_stock.yaml"
    with open(targets_path) as f:
        data = yaml.safe_load(f)

    viz = data["metadata"]["visualization"]
    ts = viz["time_series"]
    curves = viz["curves"]
    dist = viz["distributions"]

    def _transform_curve_targets(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "target": raw.get("correlation_target"),
            "min": raw.get("correlation_min"),
            "max": raw.get("correlation_max"),
        }

    return {
        "log_gdp": ts["log_gdp"]["targets"],
        "unemployment": ts["unemployment_rate"]["targets"],
        "inflation": ts["inflation_rate"]["targets"],
        "real_wage": ts["real_wage"]["targets"],
        "phillips_corr": _transform_curve_targets(curves["phillips"]["targets"]),
        "okun_corr": _transform_curve_targets(curves["okun"]["targets"]),
        "beveridge_corr": _transform_curve_targets(curves["beveridge"]["targets"]),
        "firm_size": dist["firm_size"]["targets"],
        "wealth_distribution": dist.get("wealth", {}).get("targets", {}),
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
) -> BufferStockMetrics:
    """Run buffer-stock scenario simulation with optional visualization.

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
    BufferStockMetrics
        Computed metrics from the simulation.
    """
    from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock
    from extensions.rnd import RND_EVENTS, RnD

    config = {
        **DEFAULT_CONFIG,
        "n_periods": n_periods,
        "seed": seed,
        "logging": {"default_level": "ERROR"},
    }
    sim = bam.Simulation.init(**config)
    sim.use_role(BufferStock, n_agents=sim.n_households)
    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)

    print("Buffer-stock simulation initialized:")
    print(f"  - {sim.n_firms} firms")
    print(f"  - {sim.n_households} households")
    print(f"  - {sim.n_banks} banks")
    print(f"  - buffer_stock_h={sim.buffer_stock_h}")
    print(f"  - R&D enabled (sigma=[{sim.sigma_min}, {sim.sigma_max}])")

    results = sim.run(collect=COLLECT_CONFIG)

    print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
    print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

    burn_in = adjust_burn_in(burn_in, n_periods, verbose=True)

    metrics = _compute_metrics_wrapper(sim, results, burn_in)

    print(
        f"\nComputed metrics for {len(metrics.unemployment) - burn_in} "
        "periods (after burn-in)"
    )
    print(f"  Best distribution fit: {metrics.best_fit} (R²={metrics.best_r2:.4f})")
    print(f"  Wealth Gini: {metrics.wealth_gini:.4f}")
    print(f"  Mean MPC: {metrics.mean_mpc:.4f}")
    print(f"  % dissaving: {metrics.pct_dissaving * 100:.1f}%")

    if show_plot:
        from validation.scenarios.buffer_stock_viz import visualize_buffer_stock_results

        bounds = load_buffer_stock_targets()
        visualize_buffer_stock_results(metrics, bounds, burn_in=burn_in)

    return metrics


if __name__ == "__main__":
    run_scenario(seed=0, n_periods=1000, burn_in=500, show_plot=True)
