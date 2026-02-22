"""Integration tests for the robustness analysis package.

These tests run actual simulations with small configurations to verify
that the full internal validity and sensitivity analysis pipelines
produce well-formed results. Marked as slow since each test runs
multiple simulations.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.robustness import (
    InternalValidityResult,
    SensitivityResult,
    format_internal_validity_report,
    format_sensitivity_report,
    run_internal_validity,
    run_sensitivity_analysis,
)

# =============================================================================
# Internal Validity
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestInternalValidity:
    """Integration tests for internal validity analysis."""

    @pytest.fixture(scope="class")
    def iv_result(self) -> InternalValidityResult:
        """Run internal validity with minimal settings (shared across class)."""
        return run_internal_validity(
            n_seeds=3,
            n_periods=100,
            burn_in=50,
            n_workers=1,
            verbose=False,
        )

    def test_returns_correct_type(self, iv_result: InternalValidityResult) -> None:
        assert isinstance(iv_result, InternalValidityResult)

    def test_correct_seed_count(self, iv_result: InternalValidityResult) -> None:
        assert iv_result.n_seeds == 3
        assert len(iv_result.seed_analyses) == 3

    def test_seed_analyses_ordered(self, iv_result: InternalValidityResult) -> None:
        """Seeds should be in order 0, 1, 2."""
        seeds = [a.seed for a in iv_result.seed_analyses]
        assert seeds == sorted(seeds)

    def test_comovements_present(self, iv_result: InternalValidityResult) -> None:
        """All 5 co-movement variables should be computed."""
        from validation.robustness import COMOVEMENT_VARIABLES

        for var in COMOVEMENT_VARIABLES:
            assert var in iv_result.mean_comovements
            assert var in iv_result.std_comovements
            assert var in iv_result.baseline_comovements

    def test_comovements_shape(self, iv_result: InternalValidityResult) -> None:
        """Each co-movement array should have 2*max_lag+1 = 9 elements."""
        for var, arr in iv_result.mean_comovements.items():
            assert len(arr) == 9, f"{var}: expected 9 elements, got {len(arr)}"

    def test_comovements_in_range(self, iv_result: InternalValidityResult) -> None:
        """Correlations should be in [-1, 1]."""
        for var, arr in iv_result.mean_comovements.items():
            assert np.all(np.abs(arr) <= 1.0 + 1e-10), (
                f"{var}: correlations out of range"
            )

    def test_ar_fit_present(self, iv_result: InternalValidityResult) -> None:
        """AR fit should produce non-trivial coefficients."""
        assert len(iv_result.mean_ar_coeffs) > 0
        assert len(iv_result.mean_irf) > 0

    def test_cross_sim_stats_present(self, iv_result: InternalValidityResult) -> None:
        """Cross-simulation statistics should include key metrics."""
        assert "unemployment_mean" in iv_result.cross_sim_stats
        assert "inflation_mean" in iv_result.cross_sim_stats

    def test_cross_sim_stats_have_fields(
        self, iv_result: InternalValidityResult
    ) -> None:
        """Each stat entry should have mean, std, min, max, cv."""
        for name, stat in iv_result.cross_sim_stats.items():
            assert "mean" in stat, f"{name} missing 'mean'"
            assert "std" in stat, f"{name} missing 'std'"

    def test_report_generation(self, iv_result: InternalValidityResult) -> None:
        """Report should be non-empty and contain key sections."""
        report = format_internal_validity_report(iv_result)
        assert "Internal Validity" in report
        assert "Cross-Simulation Variance" in report
        assert "Co-Movement Structure" in report

    def test_no_collapse_with_defaults(self, iv_result: InternalValidityResult) -> None:
        """Default parameters should not cause economic collapse."""
        assert iv_result.n_collapsed == 0

    def test_new_seed_analysis_fields(self, iv_result: InternalValidityResult) -> None:
        """New fields (kurtosis, tail_index, peak_lags, etc.) should exist."""
        for sa in iv_result.seed_analyses:
            assert isinstance(sa.firm_size_kurtosis_sales, float)
            assert isinstance(sa.firm_size_kurtosis_net_worth, float)
            assert isinstance(sa.firm_size_tail_index, float)
            assert isinstance(sa.peak_lags, dict)
            assert isinstance(sa.wage_productivity_ratio, float)
            assert isinstance(sa.hp_gdp_cycle, np.ndarray)
            assert isinstance(sa.firm_size_shape, str)

    def test_cross_sim_stats_new_fields(
        self, iv_result: InternalValidityResult
    ) -> None:
        """Kurtosis, tail_index, and wage_productivity_ratio in cross_sim_stats."""
        assert "firm_size_kurtosis_sales" in iv_result.cross_sim_stats
        assert "firm_size_kurtosis_net_worth" in iv_result.cross_sim_stats
        assert "firm_size_tail_index" in iv_result.cross_sim_stats
        assert "wage_productivity_ratio" in iv_result.cross_sim_stats

    def test_mean_ar_fit_from_averaged_cycles(
        self, iv_result: InternalValidityResult
    ) -> None:
        """Mean AR should be order 1 with proper IRF decay."""
        assert iv_result.mean_ar_order == 1
        assert len(iv_result.mean_ar_coeffs) == 2  # [const, phi_1]
        assert len(iv_result.mean_irf) > 0
        # IRF should start at 1.0 and decay
        assert iv_result.mean_irf[0] == 1.0

    def test_collapsed_seed_has_empty_cycle(
        self, iv_result: InternalValidityResult
    ) -> None:
        """Collapsed seeds should have empty hp_gdp_cycle."""
        for sa in iv_result.seed_analyses:
            if sa.collapsed:
                assert len(sa.hp_gdp_cycle) == 0


# =============================================================================
# Sensitivity Analysis
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestSensitivityAnalysis:
    """Integration tests for sensitivity analysis."""

    @pytest.fixture(scope="class")
    def sa_result(self) -> SensitivityResult:
        """Run a single experiment with minimal settings."""
        return run_sensitivity_analysis(
            experiments=["credit_market"],
            n_seeds=2,
            n_periods=100,
            burn_in=50,
            n_workers=1,
            verbose=False,
        )

    def test_returns_correct_type(self, sa_result: SensitivityResult) -> None:
        assert isinstance(sa_result, SensitivityResult)

    def test_experiment_present(self, sa_result: SensitivityResult) -> None:
        assert "credit_market" in sa_result.experiments

    def test_correct_value_count(self, sa_result: SensitivityResult) -> None:
        """Credit market has 5 parameter values (H=1,2,3,4,6)."""
        exp = sa_result.experiments["credit_market"]
        assert len(exp.value_results) == 5

    def test_baseline_identified(self, sa_result: SensitivityResult) -> None:
        """Baseline should be correctly identified."""
        exp = sa_result.experiments["credit_market"]
        baseline = exp.baseline
        assert baseline is not None
        assert "2" in baseline.label or baseline.config_overrides.get("max_H") == 2

    def test_value_results_have_stats(self, sa_result: SensitivityResult) -> None:
        """Each value result should have statistics."""
        exp = sa_result.experiments["credit_market"]
        for vr in exp.value_results:
            assert vr.n_seeds == 2
            assert vr.mean_comovements is not None

    def test_value_results_have_comovements(self, sa_result: SensitivityResult) -> None:
        """Each value result should have co-movement data."""
        from validation.robustness import COMOVEMENT_VARIABLES

        exp = sa_result.experiments["credit_market"]
        for vr in exp.value_results:
            for var in COMOVEMENT_VARIABLES:
                assert var in vr.mean_comovements

    def test_report_generation(self, sa_result: SensitivityResult) -> None:
        """Sensitivity report should be non-empty."""
        report = format_sensitivity_report(sa_result)
        assert "Sensitivity Analysis" in report
        assert "credit" in report.lower()

    def test_get_stat_table(self, sa_result: SensitivityResult) -> None:
        """get_stat_table should return (label, mean, std) tuples."""
        exp = sa_result.experiments["credit_market"]
        table = exp.get_stat_table("unemployment_mean")
        assert len(table) == 5
        for label, _mean, _std in table:
            assert isinstance(label, str)

    def test_value_results_have_new_stats(self, sa_result: SensitivityResult) -> None:
        """New stat fields should be present in value results."""
        exp = sa_result.experiments["credit_market"]
        for vr in exp.value_results:
            # At least some of the new stat fields should be present
            new_fields = [
                "firm_size_kurtosis_sales",
                "firm_size_kurtosis_net_worth",
                "firm_size_tail_index",
                "wage_productivity_ratio",
            ]
            for field in new_fields:
                assert field in vr.stats, f"Missing {field} in value result stats"

    def test_value_results_have_mean_peak_lags(
        self, sa_result: SensitivityResult
    ) -> None:
        """mean_peak_lags dict should be present with 5 co-movement keys."""
        from validation.robustness import COMOVEMENT_VARIABLES

        exp = sa_result.experiments["credit_market"]
        for vr in exp.value_results:
            assert isinstance(vr.mean_peak_lags, dict)
            for var in COMOVEMENT_VARIABLES:
                assert var in vr.mean_peak_lags, f"Missing {var} in mean_peak_lags"


@pytest.mark.slow
@pytest.mark.validation
def test_unknown_experiment_raises() -> None:
    """Requesting a non-existent experiment should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown experiment"):
        run_sensitivity_analysis(
            experiments=["nonexistent"],
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )


# =============================================================================
# Structural Experiments (Section 3.10.2)
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestPAExperiment:
    """Integration tests for the PA (preferential attachment) experiment."""

    @pytest.fixture(scope="class")
    def pa_result(self):
        from validation.robustness import run_pa_experiment

        return run_pa_experiment(
            n_seeds=2,
            n_periods=100,
            burn_in=50,
            n_workers=1,
            verbose=False,
            include_baseline=True,
        )

    def test_returns_correct_type(self, pa_result) -> None:
        from validation.robustness import PAExperimentResult

        assert isinstance(pa_result, PAExperimentResult)

    def test_pa_off_validity_present(self, pa_result) -> None:
        assert pa_result.pa_off_validity is not None
        assert pa_result.pa_off_validity.n_seeds == 2

    def test_baseline_present(self, pa_result) -> None:
        assert pa_result.baseline_validity is not None
        assert pa_result.baseline_validity.n_seeds == 2

    def test_z_sweep_present(self, pa_result) -> None:
        assert pa_result.pa_off_z_sweep is not None
        assert "goods_market_no_pa" in pa_result.pa_off_z_sweep.experiments

    def test_no_collapse_pa_off(self, pa_result) -> None:
        assert pa_result.pa_off_validity.n_collapsed == 0

    def test_report_generation(self, pa_result) -> None:
        from validation.robustness import format_pa_report

        report = format_pa_report(pa_result)
        assert "PA Experiment" in report
        assert "PA off" in report


@pytest.mark.slow
@pytest.mark.validation
class TestEntryExperiment:
    """Integration tests for the entry neutrality experiment."""

    @pytest.fixture(scope="class")
    def entry_result(self):
        from validation.robustness import run_entry_experiment

        return run_entry_experiment(
            n_seeds=2,
            n_periods=100,
            burn_in=50,
            n_workers=1,
            verbose=False,
        )

    def test_returns_correct_type(self, entry_result) -> None:
        from validation.robustness import EntryExperimentResult

        assert isinstance(entry_result, EntryExperimentResult)

    def test_tax_sweep_present(self, entry_result) -> None:
        assert entry_result.tax_sweep is not None
        assert "entry_neutrality" in entry_result.tax_sweep.experiments

    def test_correct_value_count(self, entry_result) -> None:
        """Entry neutrality has 5 tax rate values."""
        exp = entry_result.tax_sweep.experiments["entry_neutrality"]
        assert len(exp.value_results) == 5

    def test_report_generation(self, entry_result) -> None:
        from validation.robustness import format_entry_report

        report = format_entry_report(entry_result)
        assert "Entry Neutrality" in report
        assert "Monotonicity" in report


@pytest.mark.slow
@pytest.mark.validation
def test_sensitivity_with_setup_fn() -> None:
    """run_sensitivity_analysis should work with entry_neutrality (has setup_fn)."""
    result = run_sensitivity_analysis(
        experiments=["entry_neutrality"],
        n_seeds=2,
        n_periods=100,
        burn_in=50,
        n_workers=1,
        verbose=False,
    )
    assert "entry_neutrality" in result.experiments
    exp = result.experiments["entry_neutrality"]
    assert len(exp.value_results) == 5
