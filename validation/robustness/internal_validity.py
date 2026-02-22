"""Internal validity analysis (Section 3.10.1, Part 1).

Runs multiple simulations with different random seeds using default
parameters, then computes cross-simulation statistics to verify that
the model's qualitative results are robust to stochastic variation.

Key outputs:
- Cross-simulation variance of macroeconomic variables
- Co-movement analysis at leads and lags (Figure 3.9)
- AR model fitting and impulse-response functions
- Firm size distribution invariance
- Empirical curve persistence (Phillips, Okun, Beveridge)
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

import bamengine as bam
from bamengine import SimulationResults, ops
from validation.robustness.stats import (
    cross_correlation,
    fit_ar,
    hp_filter,
    impulse_response,
)
from validation.scenarios._utils import adjust_burn_in, filter_outliers_iqr

# ─── Collection Configuration ───────────────────────────────────────────────

ROBUSTNESS_COLLECT_CONFIG: dict[str, Any] = {
    "Producer": ["production", "labor_productivity"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth"],
    "LoanBook": ["principal", "rate"],
    "Economy": True,
    "capture_timing": {
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "LoanBook.principal": "banks_provide_loans",
        "LoanBook.rate": "banks_provide_loans",
    },
}

# ─── Growth+ Collection Config ─────────────────────────────────────────────

GROWTH_PLUS_COLLECT_CONFIG: dict[str, Any] = {
    **ROBUSTNESS_COLLECT_CONFIG,
    "capture_timing": {
        **ROBUSTNESS_COLLECT_CONFIG["capture_timing"],
        "Producer.labor_productivity": "firms_apply_productivity_growth",
    },
}


# ─── Growth+ Setup Hook ───────────────────────────────────────────────────


def setup_growth_plus(sim: bam.Simulation) -> None:
    """Attach R&D extension to a simulation (for Growth+ robustness).

    Module-level function so it can be pickled by ``ProcessPoolExecutor``.
    """
    from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    sim.use_config(RND_CONFIG)


# ─── Variables for co-movement analysis ─────────────────────────────────────

COMOVEMENT_VARIABLES = [
    "unemployment",
    "productivity",
    "price_index",
    "interest_rate",
    "real_wage",
]


# ─── Result Types ───────────────────────────────────────────────────────────


@dataclass
class SeedTimeSeries:
    """Extracted time series from a single simulation run.

    All arrays have length ``n_periods`` (full series, before burn-in).
    """

    seed: int
    collapsed: bool
    n_periods_actual: int

    # Macro time series
    gdp: NDArray[np.floating]
    log_gdp: NDArray[np.floating]
    unemployment: NDArray[np.floating]
    avg_productivity: NDArray[np.floating]
    avg_price: NDArray[np.floating]
    real_interest_rate: NDArray[np.floating]
    real_wage: NDArray[np.floating]
    inflation: NDArray[np.floating]
    vacancy_rate: NDArray[np.floating]
    n_firm_bankruptcies: NDArray[np.floating]

    # Curve data
    wage_inflation: NDArray[np.floating]
    gdp_growth: NDArray[np.floating]
    unemployment_growth: NDArray[np.floating]

    # Distribution data (final period)
    final_production: NDArray[np.floating]
    final_net_worth: NDArray[np.floating]


@dataclass
class SeedAnalysis:
    """Analysis results for a single seed."""

    seed: int
    collapsed: bool

    # Co-movement cross-correlations (one per variable)
    comovements: dict[str, NDArray[np.floating]]

    # AR fit for GDP cyclical component
    ar_coeffs: NDArray[np.floating]
    ar_order: int
    ar_r_squared: float
    irf: NDArray[np.floating]

    # Summary statistics (post burn-in)
    unemployment_mean: float
    unemployment_std: float
    inflation_mean: float
    inflation_std: float
    gdp_growth_mean: float
    gdp_growth_std: float
    real_wage_mean: float
    productivity_mean: float

    # Curve correlations
    phillips_corr: float
    okun_corr: float
    beveridge_corr: float

    # Distribution metrics
    firm_size_skewness_sales: float
    firm_size_skewness_net_worth: float
    normality_pvalue_sales: float
    normality_pvalue_net_worth: float

    # Distribution shape metrics
    firm_size_kurtosis_sales: float
    firm_size_kurtosis_net_worth: float
    firm_size_tail_index: float  # log-log rank-size slope (negative = heavier tail)

    # Peak timing for co-movement classification
    peak_lags: dict[str, int]  # lag of max |correlation| per variable

    # Wage-productivity ratio
    wage_productivity_ratio: float

    # HP-filtered GDP cycle (for cross-seed averaging)
    hp_gdp_cycle: NDArray[np.floating]

    # Degenerate dynamics detection
    degenerate: bool = False
    degenerate_reasons: list[str] = field(default_factory=list)
    firm_size_shape: str = ""  # "pareto-like", "exponential", "uniform-like"


@dataclass
class InternalValidityResult:
    """Full internal validity analysis result.

    Contains per-seed results and cross-simulation aggregates.
    """

    n_seeds: int
    n_periods: int
    burn_in: int
    seed_analyses: list[SeedAnalysis]

    # Cross-simulation co-movement averages (Figure 3.9)
    baseline_comovements: dict[str, NDArray[np.floating]]
    mean_comovements: dict[str, NDArray[np.floating]]
    std_comovements: dict[str, NDArray[np.floating]]

    # AR fit on cross-simulation average GDP cycle
    mean_ar_coeffs: NDArray[np.floating]
    mean_ar_order: int
    mean_ar_r_squared: float
    mean_irf: NDArray[np.floating]

    # Cross-simulation variance of key statistics
    cross_sim_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Collapse / degenerate counts
    n_collapsed: int = 0
    n_degenerate: int = 0

    @property
    def collapse_rate(self) -> float:
        """Fraction of seeds that led to economic collapse."""
        return self.n_collapsed / self.n_seeds if self.n_seeds > 0 else 0.0

    @property
    def degenerate_rate(self) -> float:
        """Fraction of seeds with degenerate dynamics."""
        return self.n_degenerate / self.n_seeds if self.n_seeds > 0 else 0.0


# ─── Time Series Extraction ────────────────────────────────────────────────


def _extract_time_series(
    seed: int,
    sim: bam.Simulation,
    results: SimulationResults,
) -> SeedTimeSeries:
    """Extract macro time series from simulation results."""
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    labor_productivity = results.role_data["Producer"]["labor_productivity"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]
    loan_principals = results.relationship_data["LoanBook"]["principal"]
    loan_rates = results.relationship_data["LoanBook"]["rate"]
    net_worth = results.role_data["Borrower"]["net_worth"]

    # GDP and related
    gdp = ops.sum(production, axis=1)
    log_gdp = ops.log(gdp)

    # Unemployment
    unemployment = 1 - ops.mean(employed.astype(float), axis=1)

    # Weighted average productivity
    weighted_prod = ops.sum(ops.multiply(labor_productivity, production), axis=1)
    avg_productivity = ops.divide(weighted_prod, gdp)

    # Real wage
    employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
    employed_count = ops.sum(employed, axis=1)
    avg_employed_wage = ops.where(
        ops.greater(employed_count, 0),
        ops.divide(employed_wages_sum, employed_count),
        0.0,
    )
    real_wage = ops.divide(avg_employed_wage, avg_price)

    # Weighted average loan interest rate (weighted by actual loan principals)
    n_periods_total = len(inflation)
    real_interest_rate = np.zeros(n_periods_total)
    for t in range(n_periods_total):
        principals_t = loan_principals[t]
        rates_t = loan_rates[t]
        if len(principals_t) > 0 and np.sum(principals_t) > 0:
            weighted_nominal = float(
                np.sum(rates_t * principals_t) / np.sum(principals_t)
            )
        else:
            weighted_nominal = sim.config.r_bar
        real_interest_rate[t] = weighted_nominal - inflation[t]

    # Vacancy rate
    total_vacancies = ops.sum(n_vacancies, axis=1)
    vacancy_rate = ops.divide(total_vacancies, sim.n_households)

    # Bankruptcy count
    n_bankruptcies = results.economy_data.get(
        "n_firm_bankruptcies",
        np.zeros(len(gdp)),
    )

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

    return SeedTimeSeries(
        seed=seed,
        collapsed=sim.ec.collapsed,
        n_periods_actual=sim.t,
        gdp=gdp,
        log_gdp=log_gdp,
        unemployment=unemployment,
        avg_productivity=avg_productivity,
        avg_price=avg_price,
        real_interest_rate=real_interest_rate,
        real_wage=real_wage,
        inflation=inflation,
        vacancy_rate=vacancy_rate,
        n_firm_bankruptcies=n_bankruptcies,
        wage_inflation=wage_inflation,
        gdp_growth=gdp_growth,
        unemployment_growth=unemployment_growth,
        final_production=production[-1],
        final_net_worth=net_worth[-1],
    )


# ─── Single Seed Analysis ──────────────────────────────────────────────────


def _analyze_seed(
    ts: SeedTimeSeries,
    burn_in: int,
    max_lag: int = 4,
    ar_order: int = 2,
    irf_periods: int = 20,
) -> SeedAnalysis:
    """Compute co-movements, AR fit, and summary stats for one seed."""
    if ts.collapsed:
        # Return a minimal result for collapsed simulations
        n_lags = 2 * max_lag + 1
        return SeedAnalysis(
            seed=ts.seed,
            collapsed=True,
            comovements={v: np.full(n_lags, np.nan) for v in COMOVEMENT_VARIABLES},
            ar_coeffs=np.zeros(ar_order + 1),
            ar_order=ar_order,
            ar_r_squared=0.0,
            irf=np.zeros(irf_periods),
            unemployment_mean=np.nan,
            unemployment_std=np.nan,
            inflation_mean=np.nan,
            inflation_std=np.nan,
            gdp_growth_mean=np.nan,
            gdp_growth_std=np.nan,
            real_wage_mean=np.nan,
            productivity_mean=np.nan,
            phillips_corr=np.nan,
            okun_corr=np.nan,
            beveridge_corr=np.nan,
            firm_size_skewness_sales=np.nan,
            firm_size_skewness_net_worth=np.nan,
            normality_pvalue_sales=np.nan,
            normality_pvalue_net_worth=np.nan,
            firm_size_kurtosis_sales=np.nan,
            firm_size_kurtosis_net_worth=np.nan,
            firm_size_tail_index=np.nan,
            peak_lags={v: 0 for v in COMOVEMENT_VARIABLES},
            wage_productivity_ratio=np.nan,
            hp_gdp_cycle=np.array([]),
            degenerate=True,
            degenerate_reasons=["collapsed"],
        )

    bi = burn_in

    # HP-filter GDP and extract cyclical components
    _, gdp_cycle = hp_filter(ts.log_gdp[bi:])

    # Co-movement variable mapping
    series_map = {
        "unemployment": ts.unemployment[bi:],
        "productivity": ts.avg_productivity[bi:],
        "price_index": ts.avg_price[bi:],
        "interest_rate": ts.real_interest_rate[bi:],
        "real_wage": ts.real_wage[bi:],
    }

    # HP-filter each variable and compute cross-correlations with GDP cycle
    comovements: dict[str, NDArray[np.floating]] = {}
    for var_name, series in series_map.items():
        _, var_cycle = hp_filter(series)
        comovements[var_name] = cross_correlation(gdp_cycle, var_cycle, max_lag)

    # AR model fit on GDP cyclical component
    try:
        ar_coeffs, ar_r2 = fit_ar(gdp_cycle, order=ar_order)
        irf_vals = impulse_response(ar_coeffs, n_periods=irf_periods)
    except (ValueError, np.linalg.LinAlgError):
        ar_coeffs = np.zeros(ar_order + 1)
        ar_r2 = 0.0
        irf_vals = np.zeros(irf_periods)

    # Summary statistics (post burn-in)
    unemp_ss = ts.unemployment[bi:]
    gdp_growth_ss = ts.gdp_growth[bi - 1 :]

    # Curve correlations
    wage_inflation_ss = ts.wage_inflation[bi - 1 :]
    phillips_corr = float(np.corrcoef(unemp_ss, wage_inflation_ss)[0, 1])

    unemp_growth_ss = ts.unemployment_growth[bi - 1 :]
    unemp_filt, gdp_filt = filter_outliers_iqr(unemp_growth_ss, gdp_growth_ss)
    okun_corr = (
        float(np.corrcoef(unemp_filt, gdp_filt)[0, 1])
        if len(unemp_filt) > 2
        else np.nan
    )

    vacancy_ss = ts.vacancy_rate[bi:]
    beveridge_corr = float(np.corrcoef(unemp_ss, vacancy_ss)[0, 1])

    # Distribution metrics
    prod = ts.final_production
    nw = ts.final_net_worth
    prod_positive = prod[prod > 0]
    nw_positive = nw[nw > 0]

    firm_skew_sales = (
        float(stats.skew(prod_positive)) if len(prod_positive) > 2 else np.nan
    )
    firm_skew_nw = float(stats.skew(nw_positive)) if len(nw_positive) > 2 else np.nan

    # Shapiro-Wilk test for normality (p < 0.05 → reject normality)
    norm_p_sales = (
        float(stats.shapiro(prod_positive[:500])[1])
        if len(prod_positive) > 3
        else np.nan
    )
    norm_p_nw = (
        float(stats.shapiro(nw_positive[:500])[1]) if len(nw_positive) > 3 else np.nan
    )

    # Kurtosis (excess kurtosis: 0 = normal)
    kurt_sales = (
        float(stats.kurtosis(prod_positive)) if len(prod_positive) > 3 else np.nan
    )
    kurt_nw = float(stats.kurtosis(nw_positive)) if len(nw_positive) > 3 else np.nan

    # Tail index: log-log rank-size slope on positive net worth
    if len(nw_positive) > 10:
        sorted_nw = np.sort(nw_positive)[::-1]
        ranks = np.arange(1, len(sorted_nw) + 1, dtype=float)
        log_ranks = np.log(ranks)
        log_values = np.log(sorted_nw)
        valid_mask = np.isfinite(log_ranks) & np.isfinite(log_values)
        if np.sum(valid_mask) > 2:
            slope, _, _, _, _ = stats.linregress(
                log_ranks[valid_mask], log_values[valid_mask]
            )
            tail_index = float(slope)
        else:
            tail_index = np.nan
    else:
        tail_index = np.nan

    # Distribution shape classification (kurtosis-based)
    if not np.isnan(kurt_nw):
        if kurt_nw > 6 and (not np.isnan(tail_index) and tail_index < -1.5):
            firm_size_shape = "pareto-like"
        elif kurt_nw < 0:
            firm_size_shape = "uniform-like"
        else:
            firm_size_shape = "exponential"
    else:
        firm_size_shape = ""

    # Peak lag detection: lag of max |correlation| per variable
    peak_lags: dict[str, int] = {}
    for var_name in COMOVEMENT_VARIABLES:
        corr_arr = comovements[var_name]
        if np.any(np.isfinite(corr_arr)):
            abs_corr = np.abs(corr_arr)
            peak_idx = int(np.nanargmax(abs_corr))
            peak_lags[var_name] = peak_idx - max_lag
        else:
            peak_lags[var_name] = 0

    # Wage-productivity ratio
    wage_productivity_ratio = (
        float(np.mean(ts.real_wage[bi:]) / np.mean(ts.avg_productivity[bi:]))
        if np.mean(ts.avg_productivity[bi:]) > 1e-10
        else np.nan
    )

    # ── Degenerate dynamics detection ─────────────────────────────
    degenerate_reasons: list[str] = []

    if float(np.mean(unemp_ss)) < 0.005:
        degenerate_reasons.append("near-zero unemployment")

    if float(np.mean(unemp_ss)) > 0.50:
        degenerate_reasons.append("extreme unemployment")

    if float(np.std(gdp_growth_ss)) < 1e-6:
        degenerate_reasons.append("zero GDP variance")

    if any(np.isnan(comovements[v]).any() for v in COMOVEMENT_VARIABLES):
        degenerate_reasons.append("NaN in co-movements")

    return SeedAnalysis(
        seed=ts.seed,
        collapsed=False,
        comovements=comovements,
        ar_coeffs=ar_coeffs,
        ar_order=ar_order,
        ar_r_squared=ar_r2,
        irf=irf_vals,
        unemployment_mean=float(np.mean(unemp_ss)),
        unemployment_std=float(np.std(unemp_ss)),
        inflation_mean=float(np.mean(ts.inflation[bi:])),
        inflation_std=float(np.std(ts.inflation[bi:])),
        gdp_growth_mean=float(np.nanmean(gdp_growth_ss)),
        gdp_growth_std=float(np.nanstd(gdp_growth_ss)),
        real_wage_mean=float(np.mean(ts.real_wage[bi:])),
        productivity_mean=float(np.mean(ts.avg_productivity[bi:])),
        phillips_corr=phillips_corr,
        okun_corr=okun_corr,
        beveridge_corr=beveridge_corr,
        firm_size_skewness_sales=firm_skew_sales,
        firm_size_skewness_net_worth=firm_skew_nw,
        normality_pvalue_sales=norm_p_sales,
        normality_pvalue_net_worth=norm_p_nw,
        firm_size_kurtosis_sales=kurt_sales,
        firm_size_kurtosis_net_worth=kurt_nw,
        firm_size_tail_index=tail_index,
        peak_lags=peak_lags,
        wage_productivity_ratio=wage_productivity_ratio,
        hp_gdp_cycle=gdp_cycle,
        degenerate=len(degenerate_reasons) > 0,
        degenerate_reasons=degenerate_reasons,
        firm_size_shape=firm_size_shape,
    )


# ─── Worker Function ────────────────────────────────────────────────────────


def _run_seed(
    seed: int,
    n_periods: int,
    burn_in: int,
    config: dict[str, Any],
    max_lag: int,
    ar_order: int,
    irf_periods: int,
    setup_hook: Callable[[bam.Simulation], None] | None = None,
    collect_config: dict[str, Any] | None = None,
    exp_setup_fn: Callable[[bam.Simulation], None] | None = None,
) -> SeedAnalysis:
    """Run a single simulation and return analysis.

    Designed for ``ProcessPoolExecutor`` — all arguments must be picklable.
    In particular, *setup_hook* and *exp_setup_fn* must be **module-level
    functions** (not lambdas or closures).

    Parameters
    ----------
    setup_hook : callable or None
        Global setup (e.g. attach R&D extension for Growth+).
    exp_setup_fn : callable or None
        Per-experiment setup (e.g. attach taxation extension).
    """
    sim = bam.Simulation.init(
        seed=seed,
        n_periods=n_periods,
        logging={"default_level": "ERROR"},
        **config,
    )
    if setup_hook is not None:
        setup_hook(sim)
    if exp_setup_fn is not None:
        exp_setup_fn(sim)
    results = sim.run(collect=collect_config or ROBUSTNESS_COLLECT_CONFIG)

    ts = _extract_time_series(seed, sim, results)
    return _analyze_seed(ts, burn_in, max_lag, ar_order, irf_periods)


# ─── Main Entry Point ──────────────────────────────────────────────────────


def run_internal_validity(
    n_seeds: int = 20,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 10,
    max_lag: int = 4,
    ar_order_single: int = 2,
    ar_order_mean: int = 1,
    irf_periods: int = 20,
    baseline_seed: int = 0,
    verbose: bool = True,
    setup_hook: Callable[[bam.Simulation], None] | None = None,
    collect_config: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> InternalValidityResult:
    """Run internal validity analysis (Section 3.10.1, Part 1).

    Runs ``n_seeds`` simulations with different random seeds using
    default parameters, collecting time series for co-movement analysis,
    AR model fitting, and distribution checks.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds to test.
    n_periods : int
        Simulation periods per seed.
    burn_in : int
        Burn-in periods to discard from analysis.
    n_workers : int
        Parallel workers for simulation execution.
    max_lag : int
        Maximum lead/lag for cross-correlation computation.
    ar_order_single : int
        AR order for single-seed GDP cycle (book uses AR(2)).
    ar_order_mean : int
        AR order for cross-seed average GDP cycle (book uses AR(1)).
    irf_periods : int
        Number of periods for impulse-response function.
    baseline_seed : int
        Seed to use as the 'baseline' reference in co-movement plots.
    verbose : bool
        Print progress messages.
    setup_hook : callable or None
        Optional function ``(sim) -> None`` called after ``Simulation.init()``
        to attach extension roles, events, and config (e.g. R&D for Growth+).
        **Must be a module-level function** for ``ProcessPoolExecutor``
        pickling — lambdas and closures will fail.
    collect_config : dict or None
        Custom collection configuration. When *None*, uses the default
        ``ROBUSTNESS_COLLECT_CONFIG``.
    **config_overrides
        Additional simulation config overrides.

    Returns
    -------
    InternalValidityResult
        Complete analysis with per-seed results and aggregates.
    """
    burn_in = adjust_burn_in(burn_in, n_periods)
    seeds = list(range(n_seeds))

    if verbose:
        print(
            f"Running internal validity analysis: {n_seeds} seeds, "
            f"{n_periods} periods, {burn_in} burn-in"
        )

    # Run all seeds (parallel or sequential)
    seed_analyses: list[SeedAnalysis] = []

    def _report(seed: int, analysis: SeedAnalysis) -> None:
        if verbose:
            if analysis.collapsed:
                status = "COLLAPSED"
            elif analysis.degenerate:
                status = f"DEGENERATE ({', '.join(analysis.degenerate_reasons)})"
            else:
                status = "OK"
            print(f"  Seed {seed:>3d}: {status}")

    if n_workers == 1:
        for seed in seeds:
            analysis = _run_seed(
                seed,
                n_periods,
                burn_in,
                config_overrides,
                max_lag,
                ar_order_single,
                irf_periods,
                setup_hook,
                collect_config,
            )
            seed_analyses.append(analysis)
            _report(seed, analysis)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _run_seed,
                    seed,
                    n_periods,
                    burn_in,
                    config_overrides,
                    max_lag,
                    ar_order_single,
                    irf_periods,
                    setup_hook,
                    collect_config,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                analysis = future.result()
                seed_analyses.append(analysis)
                _report(seed, analysis)

    # Sort by seed for consistent ordering
    seed_analyses.sort(key=lambda a: a.seed)

    # Separate valid and collapsed/degenerate results
    # Degenerate is a superset of collapsed — exclude both from averages
    valid = [a for a in seed_analyses if not a.degenerate]
    n_collapsed = sum(1 for a in seed_analyses if a.collapsed)
    n_degenerate = sum(1 for a in seed_analyses if a.degenerate)

    if verbose:
        print(
            f"\nCompleted: {len(valid)} valid, "
            f"{n_collapsed} collapsed, {n_degenerate} degenerate"
        )

    # ── Aggregate co-movements ──────────────────────────────────────────

    n_lags = 2 * max_lag + 1

    # Baseline co-movements (from the baseline_seed, or first valid seed)
    baseline_analysis = next(
        (a for a in valid if a.seed == baseline_seed),
        valid[0] if valid else None,
    )
    baseline_comovements = (
        baseline_analysis.comovements
        if baseline_analysis
        else {v: np.full(n_lags, np.nan) for v in COMOVEMENT_VARIABLES}
    )

    # Mean and std co-movements across valid seeds
    mean_comovements: dict[str, NDArray[np.floating]] = {}
    std_comovements: dict[str, NDArray[np.floating]] = {}

    for var in COMOVEMENT_VARIABLES:
        if valid:
            all_corrs = np.array([a.comovements[var] for a in valid])
            mean_comovements[var] = np.nanmean(all_corrs, axis=0)
            std_comovements[var] = np.nanstd(all_corrs, axis=0)
        else:
            mean_comovements[var] = np.full(n_lags, np.nan)
            std_comovements[var] = np.full(n_lags, np.nan)

    # ── AR fit on cross-simulation average GDP cycle ────────────────────

    # Collect HP-filtered GDP cycles from valid seeds, compute the
    # pointwise mean cycle, and fit AR(1) on it.  This matches the book's
    # methodology: the averaged cyclical component is best described by an
    # AR(1) structure (individual seeds are AR(2), but the second-order
    # dynamics cancel out when averaging).

    if valid:
        try:
            cycles = [a.hp_gdp_cycle for a in valid if len(a.hp_gdp_cycle) > 0]
            if cycles:
                min_len = min(len(c) for c in cycles)
                stacked = np.array([c[:min_len] for c in cycles])
                mean_cycle = np.mean(stacked, axis=0)
                mean_ar_coeffs, mean_ar_r2 = fit_ar(mean_cycle, order=ar_order_mean)
                mean_irf = impulse_response(mean_ar_coeffs, n_periods=irf_periods)
            else:
                mean_ar_coeffs = np.zeros(ar_order_mean + 1)
                mean_ar_r2 = 0.0
                mean_irf = np.zeros(irf_periods)
        except (IndexError, ValueError, np.linalg.LinAlgError):
            mean_ar_coeffs = np.zeros(ar_order_mean + 1)
            mean_ar_r2 = 0.0
            mean_irf = np.zeros(irf_periods)
    else:
        mean_ar_coeffs = np.zeros(ar_order_mean + 1)
        mean_ar_r2 = 0.0
        mean_irf = np.zeros(irf_periods)

    # ── Cross-simulation statistics ─────────────────────────────────────

    stat_fields = [
        ("unemployment_mean", "unemployment_mean"),
        ("unemployment_std", "unemployment_std"),
        ("inflation_mean", "inflation_mean"),
        ("inflation_std", "inflation_std"),
        ("gdp_growth_mean", "gdp_growth_mean"),
        ("gdp_growth_std", "gdp_growth_std"),
        ("real_wage_mean", "real_wage_mean"),
        ("productivity_mean", "productivity_mean"),
        ("phillips_corr", "phillips_corr"),
        ("okun_corr", "okun_corr"),
        ("beveridge_corr", "beveridge_corr"),
        ("firm_size_skewness_sales", "firm_size_skewness_sales"),
        ("firm_size_skewness_net_worth", "firm_size_skewness_net_worth"),
        ("firm_size_kurtosis_sales", "firm_size_kurtosis_sales"),
        ("firm_size_kurtosis_net_worth", "firm_size_kurtosis_net_worth"),
        ("firm_size_tail_index", "firm_size_tail_index"),
        ("wage_productivity_ratio", "wage_productivity_ratio"),
    ]

    cross_sim_stats: dict[str, dict[str, float]] = {}
    for display_name, attr_name in stat_fields:
        values = [
            getattr(a, attr_name) for a in valid if not np.isnan(getattr(a, attr_name))
        ]
        if values:
            cross_sim_stats[display_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv": float(np.std(values) / abs(np.mean(values)))
                if abs(np.mean(values)) > 1e-10
                else 0.0,
            }

    return InternalValidityResult(
        n_seeds=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
        seed_analyses=seed_analyses,
        baseline_comovements=baseline_comovements,
        mean_comovements=mean_comovements,
        std_comovements=std_comovements,
        mean_ar_coeffs=mean_ar_coeffs,
        mean_ar_order=ar_order_mean,
        mean_ar_r_squared=mean_ar_r2,
        mean_irf=mean_irf,
        cross_sim_stats=cross_sim_stats,
        n_collapsed=n_collapsed,
        n_degenerate=n_degenerate,
    )
