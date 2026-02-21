"""Baseline scenario definition (Section 3.9.1).

This module defines the baseline validation scenario from Delli Gatti et al. (2011).
It contains the metrics dataclass, computation function, and scenario configuration.

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
from bamengine import SimulationResults, ops
from validation.scenarios._utils import adjust_burn_in, filter_outliers_iqr
from validation.types import CheckType, MetricFormat, MetricGroup, MetricSpec, Scenario

# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class BaselineMetrics:
    """All computed metrics from a baseline simulation run.

    Includes time series, curve correlations, and distribution metrics
    needed for validation against book targets.
    """

    # Time series (full, for visualization)
    unemployment: NDArray[np.floating]
    inflation: NDArray[np.floating]
    log_gdp: NDArray[np.floating]
    real_wage: NDArray[np.floating]
    avg_productivity: NDArray[np.floating]
    vacancy_rate: NDArray[np.floating]

    # Curve data (for scatter plots)
    wage_inflation: NDArray[np.floating]
    gdp_growth: NDArray[np.floating]
    unemployment_growth: NDArray[np.floating]

    # Distribution data
    final_production: NDArray[np.floating]

    # Summary statistics (after burn-in)
    unemployment_mean: float
    unemployment_std: float
    unemployment_max: float
    unemployment_pct_above_floor: float
    unemployment_pct_below_ceiling: float
    inflation_mean: float
    inflation_std: float
    inflation_pct_in_bounds: float
    log_gdp_mean: float
    log_gdp_std: float
    real_wage_mean: float
    real_wage_std: float
    avg_productivity_mean: float
    avg_productivity_std: float
    vacancy_rate_mean: float
    vacancy_rate_pct_in_bounds: float

    # Derived metrics
    wage_to_productivity_ratio: float

    # Correlations
    phillips_corr: float
    okun_corr: float
    okun_r_squared: float
    beveridge_corr: float

    # Distribution metrics
    firm_size_skewness: float
    firm_size_pct_below_threshold: float
    firm_size_tail_ratio: float
    firm_size_pct_below_medium: float


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_baseline_metrics(
    sim: bam.Simulation,
    results: SimulationResults,
    burn_in: int = 500,
    firm_size_threshold: float = 5.0,
    firm_size_threshold_medium: float = 10.0,
    unemployment_floor: float = 0.02,
    unemployment_ceiling: float = 0.12,
    inflation_extreme_bounds: tuple[float, float] = (-0.05, 0.15),
    vacancy_rate_bounds: tuple[float, float] = (0.08, 0.20),
) -> BaselineMetrics:
    """Compute all validation metrics from baseline simulation results."""
    # Extract raw data
    inflation = results.economy_data["inflation"]
    avg_price = results.economy_data["avg_price"]
    production = results.role_data["Producer"]["production"]
    labor_productivity = results.role_data["Producer"]["labor_productivity"]
    wages = results.role_data["Worker"]["wage"]
    employed = results.role_data["Worker"]["employed"]
    n_vacancies = results.role_data["Employer"]["n_vacancies"]

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

    # Compute curve data
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
    avg_productivity_ss = avg_productivity[burn_in:]
    vacancy_rate_ss = vacancy_rate[burn_in:]

    wage_inflation_ss = wage_inflation[burn_in - 1 :]
    gdp_growth_ss = gdp_growth[burn_in - 1 :]
    unemployment_growth_ss = unemployment_growth[burn_in - 1 :]

    # Unemployment dynamics
    unemployment_max = float(np.max(unemployment_ss))
    unemployment_pct_above_floor = float(np.mean(unemployment_ss >= unemployment_floor))
    unemployment_pct_below_ceiling = float(
        np.mean(unemployment_ss <= unemployment_ceiling)
    )

    # Inflation dynamics (computed on full series, not post-burn-in)
    # This matches the book's Figure 3.2(c) which shows all 1000 periods
    inflation_pct_in_bounds = float(
        np.mean(
            (inflation >= inflation_extreme_bounds[0])
            & (inflation <= inflation_extreme_bounds[1])
        )
    )

    # Correlations
    phillips_corr = float(np.corrcoef(unemployment_ss, wage_inflation_ss)[0, 1])

    unemp_filtered, gdp_filtered = filter_outliers_iqr(
        unemployment_growth_ss, gdp_growth_ss
    )
    okun_corr = float(np.corrcoef(unemp_filtered, gdp_filtered)[0, 1])
    okun_r_squared = okun_corr**2

    beveridge_corr = float(np.corrcoef(unemployment_ss, vacancy_rate_ss)[0, 1])

    # Distribution metrics
    final_production = production[-1]
    firm_size_skewness = float(stats.skew(final_production))
    firm_size_pct_below = float(
        np.sum(final_production < firm_size_threshold) / len(final_production)
    )
    median_production = float(np.median(final_production))
    firm_size_tail_ratio = (
        float(np.max(final_production) / median_production)
        if median_production > 0
        else 0.0
    )
    firm_size_pct_below_medium = float(
        np.sum(final_production < firm_size_threshold_medium) / len(final_production)
    )

    # Derived metrics
    avg_productivity_mean = float(np.mean(avg_productivity_ss))
    real_wage_mean = float(np.mean(real_wage_ss))
    wage_to_productivity_ratio = real_wage_mean / avg_productivity_mean

    return BaselineMetrics(
        unemployment=unemployment,
        inflation=inflation,
        log_gdp=log_gdp,
        real_wage=real_wage,
        avg_productivity=avg_productivity,
        vacancy_rate=vacancy_rate,
        wage_inflation=wage_inflation,
        gdp_growth=gdp_growth,
        unemployment_growth=unemployment_growth,
        final_production=final_production,
        unemployment_mean=float(np.mean(unemployment_ss)),
        unemployment_std=float(np.std(unemployment_ss)),
        unemployment_max=unemployment_max,
        unemployment_pct_above_floor=unemployment_pct_above_floor,
        unemployment_pct_below_ceiling=unemployment_pct_below_ceiling,
        inflation_mean=float(np.mean(inflation)),
        inflation_std=float(np.std(inflation)),
        inflation_pct_in_bounds=inflation_pct_in_bounds,
        log_gdp_mean=float(np.mean(log_gdp_ss)),
        log_gdp_std=float(np.std(log_gdp_ss)),
        real_wage_mean=real_wage_mean,
        real_wage_std=float(np.std(real_wage_ss)),
        avg_productivity_mean=avg_productivity_mean,
        avg_productivity_std=float(np.std(avg_productivity_ss)),
        vacancy_rate_mean=float(np.mean(vacancy_rate_ss)),
        vacancy_rate_pct_in_bounds=float(
            np.mean(
                (vacancy_rate_ss >= vacancy_rate_bounds[0])
                & (vacancy_rate_ss <= vacancy_rate_bounds[1])
            )
        ),
        wage_to_productivity_ratio=wage_to_productivity_ratio,
        phillips_corr=phillips_corr,
        okun_corr=okun_corr,
        okun_r_squared=okun_r_squared,
        beveridge_corr=beveridge_corr,
        firm_size_skewness=firm_size_skewness,
        firm_size_pct_below_threshold=firm_size_pct_below,
        firm_size_tail_ratio=firm_size_tail_ratio,
        firm_size_pct_below_medium=firm_size_pct_below_medium,
    )


# =============================================================================
# Collection Configuration
# =============================================================================

COLLECT_CONFIG = {
    "Producer": ["production", "labor_productivity"],
    "Worker": ["wage", "employed"],
    "Employer": ["n_vacancies"],
    "Economy": True,
    "capture_timing": {
        "Worker.wage": "workers_receive_wage",
        "Worker.employed": "firms_run_production",
        "Producer.production": "firms_run_production",
        "Employer.n_vacancies": "firms_decide_vacancies",
    },
}

# =============================================================================
# Metric Specifications
# =============================================================================

METRIC_SPECS = [
    # Unemployment metrics
    MetricSpec(
        name="unemployment_rate_mean",
        field="unemployment_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.unemployment_rate_mean",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="unemployment_absolute_ceiling",
        field="unemployment_max",
        check_type=CheckType.BOOLEAN,
        target_path="metrics.unemployment_absolute_ceiling",
        weight=3.0,
        group=MetricGroup.TIME_SERIES,
        threshold=0.20,
        invert=True,
        target_desc="< 20% (model collapse gate)",
    ),
    MetricSpec(
        name="unemployment_std",
        field="unemployment_std",
        check_type=CheckType.RANGE,
        target_path="metrics.unemployment_std",
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="unemployment_pct_above_floor",
        field="unemployment_pct_above_floor",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.unemployment_pct_above_floor",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="unemployment_pct_below_ceiling",
        field="unemployment_pct_below_ceiling",
        check_type=CheckType.PCT_WITHIN,
        target_path="metrics.unemployment_pct_below_ceiling",
        weight=0.5,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    # Inflation metrics
    MetricSpec(
        name="inflation_rate_mean",
        field="inflation_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.inflation_rate_mean",
        weight=1.5,
        group=MetricGroup.TIME_SERIES,
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
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
        format=MetricFormat.PERCENT,
    ),
    MetricSpec(
        name="inflation_std",
        field="inflation_std",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.inflation_std",
        weight=0.5,
        group=MetricGroup.TIME_SERIES,
    ),
    # Wage and productivity metrics
    MetricSpec(
        name="real_wage_mean",
        field="real_wage_mean",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.real_wage_mean",
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="real_wage_std",
        field="real_wage_std",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.real_wage_std",
        weight=0.5,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="productivity_flatness",
        field="avg_productivity_std",
        check_type=CheckType.MEAN_TOLERANCE,
        target_path="metrics.productivity_flatness",
        weight=2.0,
        group=MetricGroup.TIME_SERIES,
    ),
    MetricSpec(
        name="wage_to_productivity_ratio",
        field="wage_to_productivity_ratio",
        check_type=CheckType.RANGE,
        target_path="metrics.wage_to_productivity_ratio",
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
    # Curve correlations
    MetricSpec(
        name="phillips_correlation",
        field="phillips_corr",
        check_type=CheckType.RANGE,
        target_path="metrics.phillips_correlation",
        weight=1.0,
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
        weight=1.5,
        group=MetricGroup.CURVES,
    ),
    MetricSpec(
        name="okun_r_squared",
        field="okun_r_squared",
        check_type=CheckType.RANGE,
        target_path="metrics.okun_r_squared",
        weight=0.5,
        group=MetricGroup.CURVES,
        target_desc="R\u00b2 >= 0.50",
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
        check_type=CheckType.RANGE,
        target_path="metrics.firm_size_skewness",
        weight=1.5,
        group=MetricGroup.DISTRIBUTION,
    ),
    MetricSpec(
        name="firm_size_pct_below",
        field="firm_size_pct_below_threshold",
        check_type=CheckType.RANGE,
        target_path="metrics.firm_size_pct_below",
        weight=1.0,
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
]


# =============================================================================
# Compute Metrics Wrapper (for Scenario)
# =============================================================================


def _compute_metrics_wrapper(
    sim: bam.Simulation, results: SimulationResults, burn_in: int
) -> BaselineMetrics:
    """Wrapper for compute_baseline_metrics that loads params from YAML."""
    with open(Path(__file__).parent / "targets.yaml") as f:
        targets = yaml.safe_load(f)

    metrics = targets["metrics"]

    infl_bounds = metrics["inflation_pct_in_bounds"]
    vac_bounds = metrics["vacancy_rate_pct_in_bounds"]

    return compute_baseline_metrics(
        sim,
        results,
        burn_in=burn_in,
        firm_size_threshold=metrics["firm_size_pct_below"]["threshold"],
        firm_size_threshold_medium=metrics["firm_size_pct_below_medium"]["threshold"],
        unemployment_floor=metrics["unemployment_pct_above_floor"]["floor"],
        unemployment_ceiling=metrics["unemployment_pct_below_ceiling"]["ceiling"],
        inflation_extreme_bounds=(
            infl_bounds["bounds_min"],
            infl_bounds["bounds_max"],
        ),
        vacancy_rate_bounds=(vac_bounds["bounds_min"], vac_bounds["bounds_max"]),
    )


# =============================================================================
# Scenario Definition
# =============================================================================

SCENARIO = Scenario(
    name="baseline",
    metric_specs=METRIC_SPECS,
    collect_config=COLLECT_CONFIG,
    targets_path=Path(__file__).parent / "targets.yaml",
    default_config={},  # no scenario-specific parameters to override defaults
    compute_metrics=_compute_metrics_wrapper,
    setup_hook=None,
    title="BASELINE SCENARIO VALIDATION",
    stability_title="SEED STABILITY TEST",
)


# =============================================================================
# Public API
# =============================================================================


# For backwards compatibility with visualization module
def load_baseline_targets() -> dict[str, Any]:
    """Load baseline validation targets from YAML for visualization."""
    with open(Path(__file__).parent / "targets.yaml") as f:
        data = yaml.safe_load(f)

    viz = data["metadata"]["visualization"]
    ts = viz["time_series"]
    curves = viz["curves"]
    dist = viz["distributions"]

    def _transform_curve_targets(raw: dict[str, Any]) -> dict[str, Any]:
        """Transform curve targets to expected keys for visualization."""
        return {
            "target": raw.get("correlation_target"),
            "min": raw.get("correlation_min"),
            "max": raw.get("correlation_max"),
        }

    def _transform_firm_size_targets(raw: dict[str, Any]) -> dict[str, Any]:
        """Transform firm size targets to expected keys for visualization."""
        return {
            "threshold": raw.get("threshold_small"),
            "pct_below_target": raw.get("pct_below_small_target"),
            "pct_below_min": raw.get("pct_below_small_min"),
            "pct_below_max": raw.get("pct_below_small_max"),
            "skewness_min": raw.get("skewness_min"),
            "skewness_max": raw.get("skewness_max"),
        }

    return {
        "log_gdp": ts["log_gdp"]["targets"],
        "unemployment": ts["unemployment_rate"]["targets"],
        "inflation": ts["inflation_rate"]["targets"],
        "real_wage": ts["real_wage"]["targets"],
        "phillips_corr": _transform_curve_targets(curves["phillips"]["targets"]),
        "okun_corr": _transform_curve_targets(curves["okun"]["targets"]),
        "beveridge_corr": _transform_curve_targets(curves["beveridge"]["targets"]),
        "firm_size": _transform_firm_size_targets(dist["firm_size"]["targets"]),
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
) -> BaselineMetrics:
    """Run baseline scenario simulation with optional visualization.

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
    BaselineMetrics
        Computed metrics from the simulation.
    """
    # Initialize simulation
    sim = bam.Simulation.init(
        n_periods=n_periods,
        seed=seed,
        logging={"default_level": "ERROR"},
    )

    print("Initialized baseline scenario with:")
    print(f"  - {sim.n_firms} firms")
    print(f"  - {sim.n_households} households")
    print(f"  - {sim.n_banks} banks")

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

    # Visualize if requested (lazy import to avoid circular dependency)
    if show_plot:
        from validation.scenarios.baseline.viz import visualize_baseline_results

        bounds = load_baseline_targets()
        visualize_baseline_results(metrics, bounds, burn_in=burn_in)

    return metrics
