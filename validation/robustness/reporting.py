"""Text reporting for robustness analysis.

Generates formatted console output summarising internal validity
and sensitivity analysis results.  Reports are designed to mirror the
qualitative findings described in Section 3.10.1 of the book.
"""

from __future__ import annotations

import numpy as np

from validation.robustness.internal_validity import (
    COMOVEMENT_VARIABLES,
    InternalValidityResult,
)
from validation.robustness.sensitivity import ExperimentResult, SensitivityResult

# ─── Formatting Helpers ──────────────────────────────────────────────────────


def _header(title: str, width: int = 60) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def _subheader(title: str, width: int = 60) -> str:
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def _fmt(value: float, fmt: str = ".4f") -> str:
    """Format a float, returning 'N/A' for NaN values."""
    if np.isnan(value):
        return "N/A"
    return f"{value:{fmt}}"


def _pct(value: float) -> str:
    """Format a float as a percentage."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.1%}"


# ─── Internal Validity Report ────────────────────────────────────────────────


def format_internal_validity_report(result: InternalValidityResult) -> str:
    """Format complete internal validity analysis as a text report.

    Parameters
    ----------
    result : InternalValidityResult
        Result from :func:`run_internal_validity`.

    Returns
    -------
    str
        Formatted multi-section report.
    """
    lines: list[str] = []

    lines.append(_header("Internal Validity Analysis (Section 3.10.1, Part 1)"))
    lines.append(
        f"\nConfiguration: {result.n_seeds} seeds, "
        f"{result.n_periods} periods, {result.burn_in} burn-in"
    )
    n_valid = result.n_seeds - result.n_degenerate
    lines.append(
        f"Valid: {n_valid}, "
        f"Collapsed: {result.n_collapsed}, "
        f"Degenerate: {result.n_degenerate} "
        f"({_pct(result.degenerate_rate)})"
    )

    # ── Cross-simulation variance ────────────────────────────────────
    lines.append(_subheader("1. Cross-Simulation Variance"))
    lines.append(
        f"  {'Statistic':<30s} {'Mean':>10s} {'Std':>10s} "
        f"{'CV':>8s} {'Min':>10s} {'Max':>10s}"
    )

    display_names = {
        "unemployment_mean": "Unemployment rate",
        "unemployment_std": "Unemployment volatility",
        "inflation_mean": "Inflation rate",
        "inflation_std": "Inflation volatility",
        "gdp_growth_mean": "GDP growth rate",
        "gdp_growth_std": "GDP growth volatility",
        "real_wage_mean": "Real wage",
        "productivity_mean": "Avg productivity",
    }

    for key, label in display_names.items():
        if key in result.cross_sim_stats:
            s = result.cross_sim_stats[key]
            lines.append(
                f"  {label:<30s} {_fmt(s['mean'], '.4f'):>10s} "
                f"{_fmt(s['std'], '.4f'):>10s} "
                f"{_fmt(s['cv'], '.3f'):>8s} "
                f"{_fmt(s['min'], '.4f'):>10s} "
                f"{_fmt(s['max'], '.4f'):>10s}"
            )

    # ── Co-movement summary ──────────────────────────────────────────
    lines.append(_subheader("2. Co-Movement Structure (contemporaneous, lag=0)"))

    var_labels = {
        "unemployment": "Unemployment",
        "productivity": "Productivity",
        "price_index": "Price index",
        "interest_rate": "Real interest rate",
        "real_wage": "Real wage",
    }
    max_lag = (len(next(iter(result.mean_comovements.values()))) - 1) // 2

    lines.append(
        f"  {'Variable':<25s} {'Baseline':>10s} {'Mean':>10s} "
        f"{'Std':>8s} {'Peak lag':>9s}  Classification"
    )
    # Compute mean peak lags from valid seeds
    valid = [a for a in result.seed_analyses if not a.collapsed and not a.degenerate]
    mean_peak_lags: dict[str, int] = {}
    if valid:
        for var in COMOVEMENT_VARIABLES:
            lag_vals = [a.peak_lags[var] for a in valid if var in a.peak_lags]
            if lag_vals:
                values, counts = np.unique(lag_vals, return_counts=True)
                mean_peak_lags[var] = int(values[np.argmax(counts)])

    for var in COMOVEMENT_VARIABLES:
        baseline_val = result.baseline_comovements[var][max_lag]
        mean_val = result.mean_comovements[var][max_lag]
        std_val = result.std_comovements[var][max_lag]
        peak_lag = mean_peak_lags.get(var, 0)

        # Classify: procyclical (>0.2), countercyclical (<-0.2), acyclical
        if abs(mean_val) <= 0.2:
            classification = "acyclical"
        elif mean_val > 0:
            classification = "procyclical"
        else:
            classification = "countercyclical"

        lines.append(
            f"  {var_labels[var]:<25s} {baseline_val:>10.3f} "
            f"{mean_val:>10.3f} {std_val:>8.3f} {peak_lag:>+9d}  {classification}"
        )

    # ── AR model fit ─────────────────────────────────────────────────
    lines.append(_subheader("3. GDP Cyclical Component: AR Structure"))
    lines.append("  (Mean AR fitted on pointwise-averaged GDP cycle across seeds)")

    if valid:
        baseline = next((a for a in valid if a.seed == 0), valid[0])
        lines.append(
            f"  Baseline (seed {baseline.seed}):"
            f" AR({baseline.ar_order}),"
            f" phi_1={baseline.ar_coeffs[1]:.3f},"
        )
        if baseline.ar_order >= 2 and len(baseline.ar_coeffs) > 2:
            lines[-1] += f" phi_2={baseline.ar_coeffs[2]:.3f},"
        lines[-1] += f" R²={baseline.ar_r_squared:.3f}"

        lines.append(
            f"  Cross-sim mean:"
            f" AR({result.mean_ar_order}),"
            f" phi_1={result.mean_ar_coeffs[1]:.3f},"
            f" R²={result.mean_ar_r_squared:.3f}"
        )

        # Book finding: averaging individual AR(2) processes yields AR(1)-like
        ar_coeffs_1 = [a.ar_coeffs[1] for a in valid]
        lines.append(
            f"  Individual phi_1 range:"
            f" [{min(ar_coeffs_1):.3f}, {max(ar_coeffs_1):.3f}],"
            f" mean={np.mean(ar_coeffs_1):.3f}"
        )

    # ── Empirical curves ─────────────────────────────────────────────
    lines.append(_subheader("4. Empirical Curves Persistence"))

    curve_fields = {
        "phillips_corr": ("Phillips curve", "unemp vs wage inflation", "negative"),
        "okun_corr": ("Okun's law", "unemp growth vs GDP growth", "negative"),
        "beveridge_corr": ("Beveridge curve", "unemp vs vacancies", "negative"),
    }

    lines.append(
        f"  {'Curve':<20s} {'Mean corr':>10s} {'Std':>8s} {'Expected':>10s}  Confirmed?"
    )
    for key, (name, _desc, expected_sign) in curve_fields.items():
        if key in result.cross_sim_stats:
            s = result.cross_sim_stats[key]
            mean_val = s["mean"]
            confirmed = (
                "YES" if (expected_sign == "negative" and mean_val < 0) else "NO"
            )
            lines.append(
                f"  {name:<20s} {_fmt(mean_val, '.3f'):>10s} "
                f"{_fmt(s['std'], '.3f'):>8s} {expected_sign:>10s}  {confirmed}"
            )

    # ── Distribution invariance ──────────────────────────────────────
    lines.append(_subheader("5. Firm Size Distribution Invariance"))

    dist_metrics = [
        (
            "Sales (production)",
            "firm_size_skewness_sales",
            "firm_size_kurtosis_sales",
            "normality_pvalue_sales",
        ),
        (
            "Net worth",
            "firm_size_skewness_net_worth",
            "firm_size_kurtosis_net_worth",
            "normality_pvalue_net_worth",
        ),
    ]
    for dist_type, skew_key, kurt_key, norm_key in dist_metrics:
        lines.append(f"  {dist_type}:")
        if skew_key in result.cross_sim_stats:
            s = result.cross_sim_stats[skew_key]
            lines.append(
                f"    Skewness: mean={_fmt(s['mean'], '.3f')},"
                f" std={_fmt(s['std'], '.3f')}"
                f" (positive = right-skewed)"
            )

        if kurt_key in result.cross_sim_stats:
            s = result.cross_sim_stats[kurt_key]
            lines.append(
                f"    Kurtosis: mean={_fmt(s['mean'], '.3f')},"
                f" std={_fmt(s['std'], '.3f')}"
                f" (excess; 0 = normal)"
            )

        # Normality test results from individual seeds
        norm_values = [
            getattr(a, norm_key) for a in valid if not np.isnan(getattr(a, norm_key))
        ]
        if norm_values:
            reject_count = sum(1 for p in norm_values if p < 0.05)
            lines.append(
                f"    Normality rejected (p<0.05): {reject_count}/{len(norm_values)}"
                f" seeds ({_pct(reject_count / len(norm_values))})"
            )

    # Tail index and shape classification (net worth)
    if "firm_size_tail_index" in result.cross_sim_stats:
        s = result.cross_sim_stats["firm_size_tail_index"]
        lines.append(
            f"  Tail index (net worth): mean={_fmt(s['mean'], '.3f')},"
            f" std={_fmt(s['std'], '.3f')}"
            f" (log-log slope; more negative = heavier tail)"
        )

    # Shape classification mode across seeds
    shapes = [a.firm_size_shape for a in valid if a.firm_size_shape]
    if shapes:
        from collections import Counter

        shape_counts = Counter(shapes)
        mode_shape = shape_counts.most_common(1)[0]
        lines.append(
            f"  Shape classification: {mode_shape[0]}"
            f" ({mode_shape[1]}/{len(shapes)} seeds)"
        )

    lines.append("")
    return "\n".join(lines)


def print_internal_validity_report(result: InternalValidityResult) -> None:
    """Print internal validity report to stdout."""
    print(format_internal_validity_report(result))


# ─── Sensitivity Report ──────────────────────────────────────────────────────


def _format_experiment_report(exp_result: ExperimentResult) -> str:
    """Format report for a single sensitivity experiment."""
    lines: list[str] = []
    exp = exp_result.experiment

    lines.append(_subheader(f"Experiment: {exp.description}"))

    # Summary table
    lines.append(
        f"\n  {'Value':<30s} {'Unemp':>8s} {'Infl':>8s} "
        f"{'GDP gr':>8s} {'GDP vol':>8s} {'W/P':>6s} "
        f"{'Collapses':>10s} {'Degen':>10s}"
    )
    lines.append(f"  {'-' * 94}")

    for i, vr in enumerate(exp_result.value_results):
        marker = " *" if i == exp_result.baseline_idx else "  "
        u = _pct(vr.stats.get("unemployment_mean", {}).get("mean", float("nan")))
        inf = _fmt(vr.stats.get("inflation_mean", {}).get("mean", float("nan")), ".4f")
        g = _pct(vr.stats.get("gdp_growth_mean", {}).get("mean", float("nan")))
        vol = _fmt(vr.stats.get("gdp_growth_std", {}).get("mean", float("nan")), ".4f")
        wp = _fmt(
            vr.stats.get("wage_productivity_ratio", {}).get("mean", float("nan")),
            ".3f",
        )
        collapse = f"{vr.n_collapsed}/{vr.n_seeds}"
        degen = f"{vr.n_degenerate}/{vr.n_seeds}"

        lines.append(
            f"{marker}{vr.label:<30s} {u:>8s} {inf:>8s} "
            f"{g:>8s} {vol:>8s} {wp:>6s} "
            f"{collapse:>10s} {degen:>10s}"
        )

    # Co-movement changes (contemporaneous correlation)
    lines.append("\n  Contemporaneous co-movements (lag=0):")
    var_labels = {
        "unemployment": "Unemp",
        "productivity": "Prod",
        "price_index": "Price",
        "interest_rate": "Rate",
        "real_wage": "Wage",
    }

    header_parts = [f"{'Value':<20s}"]
    for var in COMOVEMENT_VARIABLES:
        header_parts.append(f"{var_labels[var]:>8s}")
    lines.append("  " + " ".join(header_parts))

    for i, vr in enumerate(exp_result.value_results):
        if not vr.mean_comovements:
            continue
        first_arr = next(iter(vr.mean_comovements.values()))
        if np.ndim(first_arr) == 0:
            # All seeds degenerate — scalar NaN, skip this row
            marker = " *" if i == exp_result.baseline_idx else "  "
            lines.append(
                f"{marker}{vr.label:<20s}" + "     N/A" * len(COMOVEMENT_VARIABLES)
            )
            continue
        max_lag = (len(first_arr) - 1) // 2
        parts = [f"{vr.label:<20s}"]
        for var in COMOVEMENT_VARIABLES:
            val = vr.mean_comovements[var][max_lag]
            parts.append(f"{_fmt(val, '.3f'):>8s}")
        marker = " *" if i == exp_result.baseline_idx else "  "
        lines.append(marker + " ".join(parts))

    lines.append("\n  (* = baseline value)")
    return "\n".join(lines)


def format_sensitivity_report(result: SensitivityResult) -> str:
    """Format complete sensitivity analysis as a text report.

    Parameters
    ----------
    result : SensitivityResult
        Result from :func:`run_sensitivity_analysis`.

    Returns
    -------
    str
        Formatted multi-section report.
    """
    lines: list[str] = []

    lines.append(_header("Sensitivity Analysis (Section 3.10.1, Part 2)"))
    lines.append(
        f"\nConfiguration: {result.n_seeds_per_value} seeds per value, "
        f"{result.n_periods} periods, {result.burn_in} burn-in"
    )
    lines.append(f"Experiments: {len(result.experiments)}")

    for _exp_name, exp_result in result.experiments.items():
        lines.append(_format_experiment_report(exp_result))

    lines.append("")
    return "\n".join(lines)


def print_sensitivity_report(result: SensitivityResult) -> None:
    """Print sensitivity analysis report to stdout."""
    print(format_sensitivity_report(result))
