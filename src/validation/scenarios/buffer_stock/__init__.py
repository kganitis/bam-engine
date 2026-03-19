"""Buffer-stock consumption scenario definition (Section 3.9.4).

This module defines the buffer-stock validation scenario from Delli Gatti et al.
(2011). It validates the buffer-stock extension by measuring:

1. **Unique metrics** (Figure 3.8): Wealth distribution fitted with
   Singh-Maddala, Dagum, and GB2 distributions, plus MPC behavioral metrics.
2. **Improvement over Growth+**: Per-metric comparison showing how the
   buffer-stock extension changes Growth+ macro dynamics (per-seed pairing).

The validation runs Growth+ first as a baseline (same seed), then runs the
buffer-stock simulation and computes improvement deltas for each Growth+ metric.
The total score is a weighted blend of unique metric scores and improvement scores.

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

import bamengine as bam
from bamengine import SimulationResults
from validation.scenarios._utils import adjust_burn_in
from validation.types import (
    BufferStockValidationScore,
    CheckType,
    MetricFormat,
    MetricGroup,
    MetricResult,
    MetricSpec,
    Scenario,
    ValidationScore,
)

# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class BufferStockMetrics:
    """Composed metrics: Growth+ metrics + buffer-stock unique fields.

    The Growth+ fields are computed on the buffer-stock simulation (not
    copied from the baseline). This allows comparing how the same metrics
    behave with vs without buffer-stock consumption.
    """

    # Growth+ metrics computed on the buffer-stock simulation
    growth_plus: Any  # GrowthPlusMetrics (avoid circular import)

    # Buffer-stock unique: distribution data
    final_savings: NDArray[np.floating]
    final_propensity: NDArray[np.floating]

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


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_buffer_stock_metrics(
    sim: bam.Simulation,
    results: SimulationResults,
    burn_in: int = 500,
) -> BufferStockMetrics:
    """Compute buffer-stock validation metrics.

    Delegates Growth+ macro metric computation to
    :func:`~validation.scenarios.growth_plus.compute_growth_plus_metrics`,
    then computes buffer-stock-unique metrics (wealth distribution fitting,
    MPC, Gini).

    Parameters
    ----------
    sim : Simulation
        The simulation instance (with buffer-stock + R&D extensions active).
    results : SimulationResults
        Collected simulation results.
    burn_in : int
        Number of burn-in periods to exclude.

    Returns
    -------
    BufferStockMetrics
        Composed metrics with Growth+ fields + buffer-stock unique fields.
    """
    from validation.scenarios.growth_plus import (
        GrowthPlusMetrics,
        compute_growth_plus_metrics,
    )

    # Load Growth+ targets for parameter extraction
    gp_targets_path = Path(__file__).parents[1] / "growth_plus" / "targets.yaml"
    with open(gp_targets_path) as f:
        gp_targets_yaml = yaml.safe_load(f)

    gp_metrics_yaml = gp_targets_yaml["metrics"]
    gp_infl = gp_metrics_yaml["inflation_pct_in_bounds"]
    gp_vac = gp_metrics_yaml["vacancy_rate_pct_in_bounds"]
    gp_rir = gp_metrics_yaml["real_interest_rate_pct_in_bounds"]

    # Step 1: Compute Growth+ metrics on the buffer-stock simulation
    gp_metrics: GrowthPlusMetrics = compute_growth_plus_metrics(
        sim,
        results,
        burn_in=burn_in,
        firm_size_threshold=gp_metrics_yaml["firm_size_pct_below"]["threshold"],
        firm_size_threshold_medium=gp_metrics_yaml["firm_size_pct_below_medium"][
            "threshold"
        ],
        unemployment_floor=gp_metrics_yaml["unemployment_pct_above_floor"]["floor"],
        unemployment_ceiling=gp_metrics_yaml["unemployment_pct_below_ceiling"][
            "ceiling"
        ],
        inflation_bounds=(gp_infl["bounds_min"], gp_infl["bounds_max"]),
        vacancy_rate_bounds=(gp_vac["bounds_min"], gp_vac["bounds_max"]),
        real_interest_rate_bounds=(gp_rir["bounds_min"], gp_rir["bounds_max"]),
    )

    # Step 2: Compute buffer-stock unique metrics
    employed = results.role_data["Worker"]["employed"]
    consumer_savings = results.role_data["Consumer"]["savings"]
    buf_propensity = results.role_data["BufferStock"]["propensity"]
    wages = results.role_data["Worker"]["wage"]
    shareholder_dividends = results.role_data["Shareholder"]["dividends"]

    # Wealth distribution fitting — exclude unemployed households.
    final_employed = employed[-1].astype(bool)
    final_savings_arr = consumer_savings[-1][final_employed]
    dist_fits = _fit_distributions(final_savings_arr)

    sm_params, sm_ks_stat, sm_r2 = dist_fits["SM"]
    dagum_params, dagum_ks_stat, dagum_r2 = dist_fits["D"]
    gb2_params, gb2_ks_stat, gb2_r2 = dist_fits["GB2"]

    # Determine best fit
    fits = {"SM": sm_r2, "D": dagum_r2, "GB2": gb2_r2}
    best_fit = max(fits, key=lambda k: fits[k])
    best_r2 = fits[best_fit]

    # Wealth distribution stats
    wealth_gini = _compute_gini(final_savings_arr)
    wealth_skewness = float(stats.skew(final_savings_arr))

    # MPC statistics (employed only)
    final_propensity = buf_propensity[-1][final_employed]

    # Adjust MPC for the dividend artifact
    final_dividends = shareholder_dividends[-1][final_employed]
    final_wages = wages[-1][final_employed]
    adjustment = np.where(final_wages > 0, final_dividends / final_wages, 0.0)
    adjusted_propensity = final_propensity - adjustment

    mean_mpc = float(np.mean(adjusted_propensity))
    std_mpc = float(np.std(adjusted_propensity))
    pct_dissaving = float(np.sum(adjusted_propensity > 1.0) / len(adjusted_propensity))

    return BufferStockMetrics(
        growth_plus=gp_metrics,
        final_savings=final_savings_arr,
        final_propensity=final_propensity,
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
    )


# =============================================================================
# Collection Configuration (superset of Growth+ + buffer-stock fields)
# =============================================================================

COLLECT_CONFIG = {
    # From Growth+ (needed for compute_growth_plus_metrics)
    "Producer": ["production", "labor_productivity", "price", "inventory"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Borrower": ["net_worth", "gross_profit", "total_funds"],
    "Consumer": ["savings", "income_to_spend"],
    # Buffer-stock unique
    "BufferStock": ["propensity"],
    "Shareholder": ["dividends"],
    # Shared
    "LoanBook": ["principal", "rate", "source_ids"],
    "capture_timing": {
        # Growth+ timings
        "Producer.labor_productivity": "firms_apply_productivity_growth",
        "Producer.price": "firms_adjust_price",
        "Producer.production": "firms_run_production",
        "Producer.inventory": "consumers_finalize_purchases",
        "Borrower.gross_profit": "firms_collect_revenue",
        "Borrower.total_funds": "firms_collect_revenue",
        "Consumer.income_to_spend": "consumers_decide_buffer_stock_spending",
        # Buffer-stock timings
        "BufferStock.propensity": "consumers_calc_buffer_stock_propensity",
        "Shareholder.dividends": "consumers_calc_buffer_stock_propensity",
        # Shared timings
        "Worker.wage": "firms_run_production",
        "Worker.employed": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
        "Borrower.net_worth": "firms_run_production",
        "Consumer.savings": None,  # end of period
        "LoanBook.principal": "credit_market_round",
        "LoanBook.rate": "credit_market_round",
        "LoanBook.source_ids": "credit_market_round",
    },
}

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG: dict[str, Any] = {}

# =============================================================================
# Unique Metric Specifications (8 buffer-stock-specific metrics)
# =============================================================================

UNIQUE_METRIC_SPECS = [
    # === Wealth distribution fitting (Figure 3.8) ===
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
    # === Buffer-stock behavioral ===
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
        from extensions.buffer_stock import (
            BUFFER_STOCK_CONFIG,
            BUFFER_STOCK_EVENTS,
            BufferStock,
        )
        from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_role(RnD)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)
        sim.use_config(RND_CONFIG)
        sim.use_config(BUFFER_STOCK_CONFIG)


# =============================================================================
# Compute Metrics Wrapper (for Scenario dataclass compatibility)
# =============================================================================


def _compute_metrics_wrapper(
    sim: bam.Simulation, results: SimulationResults, burn_in: int
) -> BufferStockMetrics:
    """Wrapper for compute_buffer_stock_metrics."""
    return compute_buffer_stock_metrics(sim, results, burn_in=burn_in)


# =============================================================================
# Core Validation Flow
# =============================================================================


def validate_buffer_stock(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    growth_plus_result: ValidationScore | None = None,
    **config_overrides: Any,
) -> BufferStockValidationScore:
    """Run buffer-stock validation for a single seed.

    Per-seed PASS/FAIL is determined by the 8 unique buffer-stock metrics
    only (wealth distribution fits, MPC, dissaving). Improvement deltas
    vs Growth+ are computed and stored for informational/aggregate use
    but do not affect ``passed`` or ``total_score``.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    growth_plus_result : ValidationScore or None
        Optional pre-computed Growth+ validation result for the same seed.
        If ``None``, Growth+ validation is run internally.
    **config_overrides
        Simulation config overrides.

    Returns
    -------
    BufferStockValidationScore
        Result with 8 unique metrics (PASS/FAIL) and improvement deltas
        (informational).
    """
    from validation.engine import evaluate_metric, load_targets, validate
    from validation.scenarios import get_scenario

    # Step 1: Get Growth+ baseline
    if growth_plus_result is None:
        gp_scenario = get_scenario("growth_plus")
        growth_plus_result = validate(
            gp_scenario, seed=seed, n_periods=n_periods, **config_overrides
        )

    # Handle collapsed baseline: if GP failed catastrophically, propagate
    if _is_collapsed(growth_plus_result):
        return _make_collapsed_result(
            growth_plus_result, seed, n_periods, config_overrides
        )

    # Step 2: Load buffer-stock targets
    with open(Path(__file__).parent / "targets.yaml") as f:
        bs_targets = yaml.safe_load(f)

    improvement_config = bs_targets.get("improvement", {})
    blend_alpha = improvement_config.get("blend_alpha", 0.6)  # informational only

    # Step 3: Run buffer-stock simulation
    config = {
        **DEFAULT_CONFIG,
        "n_periods": n_periods,
        "seed": seed,
        "log_level": "ERROR",
        **config_overrides,
    }

    _setup_buffer_stock(None)  # Pre-import
    sim = bam.Simulation.init(**config)
    _setup_buffer_stock(sim)  # Attach extensions

    results = sim.run(collect=COLLECT_CONFIG)

    # Step 4: Compute composed metrics
    burn_in = bs_targets["metadata"]["validation"]["burn_in_periods"]
    burn_in = adjust_burn_in(burn_in, n_periods)

    bs_metrics = compute_buffer_stock_metrics(sim, results, burn_in=burn_in)

    # Step 5: Evaluate unique metrics (these determine per-seed PASS/FAIL)
    unique_results: list[MetricResult] = []
    for spec in UNIQUE_METRIC_SPECS:
        mr = evaluate_metric(spec, bs_metrics, bs_targets)
        unique_results.append(mr)

    # Step 6: Compute improvement deltas (informational — not used for PASS/FAIL)
    # Improvement is assessed at the aggregate level in stability testing,
    # not per seed, because per-seed variance is too high.
    gp_scenario = get_scenario("growth_plus")
    gp_targets = load_targets(gp_scenario)

    gp_on_bs_results: list[MetricResult] = []
    for spec in gp_scenario.metric_specs:
        mr = evaluate_metric(spec, bs_metrics.growth_plus, gp_targets)
        gp_on_bs_results.append(mr)

    deltas: dict[str, float] = {}
    assert [r.name for r in gp_on_bs_results] == [
        r.name for r in growth_plus_result.metric_results
    ], "Growth+ metric ordering mismatch — result may be from a different code version"
    for bs_mr, gp_mr in zip(
        gp_on_bs_results, growth_plus_result.metric_results, strict=True
    ):
        deltas[gp_mr.name] = bs_mr.score - gp_mr.score

    # Step 7: Score from unique metrics only
    total_weight = sum(r.weight for r in unique_results)
    total_score = (
        sum(r.score * r.weight for r in unique_results) / total_weight
        if total_weight > 0
        else 0.0
    )

    n_pass = sum(1 for r in unique_results if r.status == "PASS")
    n_warn = sum(1 for r in unique_results if r.status == "WARN")
    n_fail = sum(1 for r in unique_results if r.status == "FAIL")

    return BufferStockValidationScore(
        metric_results=unique_results,
        total_score=total_score,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        config=config,
        baseline_score=growth_plus_result,
        improvement_deltas=deltas,
        degraded_metrics=[],  # Populated at aggregate level in stability test
        blend_alpha=blend_alpha,
    )


def _is_collapsed(result: ValidationScore) -> bool:
    """Check if a validation result indicates a collapsed simulation.

    A collapsed simulation has very low total score and many failures,
    typically from the economy dying (unemployment > 80%).
    """
    return result.total_score < 0.1 and result.n_fail > result.n_pass


def _make_collapsed_result(
    baseline: ValidationScore,
    seed: int,
    n_periods: int,
    config_overrides: dict[str, Any],
) -> BufferStockValidationScore:
    """Create a non-evaluable result when the Growth+ baseline collapsed."""
    return BufferStockValidationScore(
        metric_results=[],
        total_score=float("nan"),
        n_pass=0,
        n_warn=0,
        n_fail=1,
        config={"seed": seed, "n_periods": n_periods, **config_overrides},
        baseline_score=baseline,
        improvement_deltas={},
        degraded_metrics=[],
        blend_alpha=0.6,
    )


# =============================================================================
# Scenario Definition (for registry compatibility)
# =============================================================================

# The Scenario object is used by the generic engine and registry.
# Buffer-stock validation should use validate_buffer_stock() directly,
# but the Scenario is needed for get_scenario("buffer_stock") lookups
# and for BUFFER_STOCK_WEIGHTS derivation.
SCENARIO = Scenario(
    name="buffer_stock",
    metric_specs=UNIQUE_METRIC_SPECS,
    collect_config=COLLECT_CONFIG,
    targets_path=Path(__file__).parent / "targets.yaml",
    default_config=DEFAULT_CONFIG,
    compute_metrics=_compute_metrics_wrapper,
    setup_hook=_setup_buffer_stock,
    title="BUFFER-STOCK SCENARIO VALIDATION",
    stability_title="BUFFER-STOCK SEED STABILITY TEST",
)


# =============================================================================
# Public API
# =============================================================================


def load_buffer_stock_targets() -> dict[str, Any]:
    """Load buffer-stock validation targets from YAML for visualization."""
    with open(Path(__file__).parent / "targets.yaml") as f:
        data = yaml.safe_load(f)

    viz = data.get("metadata", {}).get("visualization", {})
    dist = viz.get("distributions", {})

    return {
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
) -> BufferStockValidationScore:
    """Run buffer-stock scenario with improvement comparison and visualization.

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
    BufferStockValidationScore
        Validation result with unique metrics and improvement deltas.
    """
    from validation import print_buffer_stock_report, run_growth_plus_validation

    # Section 1: Growth+ baseline
    print("=" * 70)
    print("SECTION 1: GROWTH+ BASELINE")
    print("=" * 70)
    gp_result = run_growth_plus_validation(seed=seed, n_periods=n_periods)
    print(
        f"Growth+ score: {gp_result.total_score:.3f}, "
        f"passed: {gp_result.passed} "
        f"(pass={gp_result.n_pass}, warn={gp_result.n_warn}, fail={gp_result.n_fail})"
    )

    # Section 2: Buffer-stock with improvement comparison
    print(f"\n{'=' * 70}")
    print("SECTION 2: BUFFER-STOCK IMPROVEMENT")
    print("=" * 70)
    bs_result = validate_buffer_stock(
        seed=seed, n_periods=n_periods, growth_plus_result=gp_result
    )

    print_buffer_stock_report(bs_result)

    if show_plot and not np.isnan(bs_result.total_score):
        from validation.scenarios.buffer_stock.viz import visualize_buffer_stock_results

        visualize_buffer_stock_results(bs_result)

    return bs_result
