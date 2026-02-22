"""Unit tests for validation.robustness.stats module.

Tests the pure statistical functions: HP filter, cross-correlation,
AR model fitting, and impulse-response function computation.
"""

import numpy as np
import pytest

from validation.robustness.stats import (
    cross_correlation,
    fit_ar,
    hp_filter,
    impulse_response,
)

# =============================================================================
# HP Filter
# =============================================================================


class TestHPFilter:
    """Test Hodrick-Prescott filter decomposition."""

    def test_constant_series_zero_cycle(self) -> None:
        """A constant series has zero cyclical component."""
        y = np.full(100, 5.0)
        trend, cycle = hp_filter(y)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-10)
        np.testing.assert_allclose(trend, 5.0, atol=1e-10)

    def test_linear_series_zero_cycle(self) -> None:
        """A linear series has zero cyclical component.

        The HP filter's penalty is on the second difference of the trend,
        so a linear trend has zero penalty and perfectly fits.
        """
        y = np.linspace(1, 10, 200)
        _trend, cycle = hp_filter(y)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-6)

    def test_cycle_plus_trend_recovery(self) -> None:
        """A known sinusoidal cycle should be largely recovered."""
        t = np.arange(200)
        trend_true = 100 + 0.5 * t
        cycle_true = 5.0 * np.sin(2 * np.pi * t / 20)  # Period=20
        y = trend_true + cycle_true

        _trend_est, cycle_est = hp_filter(y, lamb=1600.0)

        # Cycle should be correlated with true cycle
        corr = np.corrcoef(cycle_true, cycle_est)[0, 1]
        assert corr > 0.9, f"Cycle correlation too low: {corr:.3f}"

    def test_lambda_zero_returns_identity(self) -> None:
        """With lambda=0, trend equals the series (no smoothing)."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(50)
        trend, cycle = hp_filter(y, lamb=0.0)
        np.testing.assert_allclose(trend, y, atol=1e-10)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-10)

    def test_high_lambda_approaches_linear(self) -> None:
        """Very high lambda forces trend toward linear (OLS line)."""
        rng = np.random.default_rng(42)
        t = np.arange(100, dtype=float)
        y = 2.0 * t + 1.0 + rng.standard_normal(100) * 5

        trend, _ = hp_filter(y, lamb=1e8)

        # Trend should be nearly linear
        second_diffs = np.diff(trend, n=2)
        np.testing.assert_allclose(second_diffs, 0.0, atol=0.1)

    def test_short_series_fallback(self) -> None:
        """Series shorter than 3 returns identity."""
        y = np.array([1.0, 2.0])
        trend, cycle = hp_filter(y)
        np.testing.assert_array_equal(trend, y)
        np.testing.assert_array_equal(cycle, np.zeros(2))

    def test_output_shapes(self) -> None:
        """Trend and cycle have same length as input."""
        y = np.random.default_rng(0).standard_normal(150)
        trend, cycle = hp_filter(y)
        assert trend.shape == y.shape
        assert cycle.shape == y.shape

    def test_decomposition_sums_to_original(self) -> None:
        """Trend + cycle = original series."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100)) + 50
        trend, cycle = hp_filter(y)
        np.testing.assert_allclose(trend + cycle, y, atol=1e-10)


# =============================================================================
# Cross-Correlation
# =============================================================================


class TestCrossCorrelation:
    """Test cross-correlation at leads and lags."""

    def test_identical_series_peak_at_zero(self) -> None:
        """Auto-correlation peaks at lag 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        result = cross_correlation(x, x, max_lag=4)

        # Peak should be at lag 0 (index max_lag)
        assert result[4] == pytest.approx(1.0, abs=1e-10)
        assert np.argmax(result) == 4

    def test_lagged_series_peak_at_correct_lag(self) -> None:
        """A shifted copy should peak at the correct lag.

        We construct y such that y_{t} = x_{t-2}, i.e. y lags x by 2.
        Then corr(x_t, y_{t+k}) peaks at k=-2 because y_{t-2} = x_{t-4}
        but more directly: y_{t+k} = x_{t+k-2}, so corr(x_t, x_{t+k-2})
        is maximised at k=2 where it becomes corr(x_t, x_t).
        """
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        # y_t = x_{t-2}: y lags x by 2
        y = np.zeros(n)
        y[2:] = x[:-2]

        result = cross_correlation(x, y, max_lag=4)

        # corr(x_t, y_{t+k}) = corr(x_t, x_{t+k-2}) peaks at k=+2
        # Index for k=+2 is max_lag + 2 = 6
        peak_idx = np.argmax(result)
        assert peak_idx == 6, f"Expected peak at index 6 (lag=+2), got {peak_idx}"
        assert result[6] > 0.9

    def test_output_length(self) -> None:
        """Output should have 2*max_lag + 1 elements."""
        x = np.random.default_rng(0).standard_normal(100)
        for max_lag in [1, 2, 4, 8]:
            result = cross_correlation(x, x, max_lag=max_lag)
            assert len(result) == 2 * max_lag + 1

    def test_correlation_range(self) -> None:
        """All values should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        result = cross_correlation(x, y, max_lag=4)
        assert np.all(np.abs(result) <= 1.0 + 1e-10)

    def test_unequal_length_raises(self) -> None:
        """Should raise ValueError for unequal-length inputs."""
        with pytest.raises(ValueError, match="equal length"):
            cross_correlation(np.zeros(10), np.zeros(15))

    def test_uncorrelated_series_near_zero(self) -> None:
        """Independent random series should have near-zero correlation."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        result = cross_correlation(x, y, max_lag=4)
        assert np.all(np.abs(result) < 0.15)

    def test_negative_correlation(self) -> None:
        """Perfectly anti-correlated series should give -1 at lag 0."""
        x = np.sin(np.linspace(0, 10 * np.pi, 200))
        y = -x
        result = cross_correlation(x, y, max_lag=2)
        assert result[2] == pytest.approx(-1.0, abs=1e-10)


# =============================================================================
# AR Model Fitting
# =============================================================================


class TestFitAR:
    """Test autoregressive model fitting via OLS."""

    def test_known_ar1_recovery(self) -> None:
        """Should recover known AR(1) parameter from generated data.

        Uses cumulative AR(1) to produce a persistent series with high
        signal-to-noise ratio, ensuring R² is well above zero.
        """
        rng = np.random.default_rng(42)
        phi = 0.7
        n = 2000
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.standard_normal()

        coeffs, r2 = fit_ar(y, order=1)

        # coeffs = [constant, phi_1]
        assert coeffs[1] == pytest.approx(phi, abs=0.05)
        assert r2 > 0.4

    def test_known_ar2_recovery(self) -> None:
        """Should recover known AR(2) parameters."""
        rng = np.random.default_rng(42)
        phi1, phi2 = 0.6, -0.3
        n = 2000
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.standard_normal()

        coeffs, r2 = fit_ar(y, order=2)

        assert coeffs[1] == pytest.approx(phi1, abs=0.05)
        assert coeffs[2] == pytest.approx(phi2, abs=0.05)
        assert r2 > 0.3

    def test_white_noise_low_r_squared(self) -> None:
        """White noise should yield near-zero R² and small coefficients."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        coeffs, r2 = fit_ar(y, order=2)

        assert r2 < 0.1
        assert abs(coeffs[1]) < 0.15
        assert abs(coeffs[2]) < 0.15

    def test_too_short_series_raises(self) -> None:
        """Series shorter than order+2 should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            fit_ar(np.array([1.0, 2.0]), order=2)

    def test_coefficients_shape(self) -> None:
        """Should return order+1 coefficients (constant + AR params)."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        for order in [1, 2, 3]:
            coeffs, _ = fit_ar(y, order=order)
            assert len(coeffs) == order + 1

    def test_r_squared_range(self) -> None:
        """R² should be in [0, 1] for well-behaved data."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200))
        _, r2 = fit_ar(y, order=1)
        assert 0.0 <= r2 <= 1.0


# =============================================================================
# Impulse-Response Function
# =============================================================================


class TestImpulseResponse:
    """Test impulse-response function computation."""

    def test_ar1_exponential_decay(self) -> None:
        """AR(1) IRF should decay exponentially: phi^t."""
        phi = 0.8
        coeffs = np.array([0.0, phi])  # [constant, phi_1]
        irf = impulse_response(coeffs, n_periods=20)

        # IRF should be phi^t
        expected = np.array([phi**t for t in range(20)])
        expected[0] = 1.0  # Unit shock at t=0
        np.testing.assert_allclose(irf, expected, atol=1e-10)

    def test_ar2_hump_shaped(self) -> None:
        """AR(2) with positive phi1 and negative phi2 gives hump-shaped IRF."""
        coeffs = np.array([0.0, 1.2, -0.4])
        irf = impulse_response(coeffs, n_periods=20)

        # Should start at 1.0, increase, then decrease
        assert irf[0] == 1.0
        assert irf[1] > irf[0]  # Hump
        # Should eventually converge toward zero (stable system)
        assert abs(irf[-1]) < abs(irf[1])

    def test_unit_shock_at_zero(self) -> None:
        """IRF should always start with 1.0 at t=0."""
        coeffs = np.array([0.0, 0.5])
        irf = impulse_response(coeffs, n_periods=10)
        assert irf[0] == 1.0

    def test_zero_coefficients_impulse_only(self) -> None:
        """With zero AR coefficients, IRF should be [1, 0, 0, ...]."""
        coeffs = np.array([0.0, 0.0])
        irf = impulse_response(coeffs, n_periods=5)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(irf, expected)

    def test_output_length(self) -> None:
        """IRF should have exactly n_periods elements."""
        coeffs = np.array([0.0, 0.8])
        for n in [5, 10, 20, 50]:
            irf = impulse_response(coeffs, n_periods=n)
            assert len(irf) == n

    def test_stable_ar1_converges(self) -> None:
        """Stable AR(1) (|phi|<1) should converge to zero."""
        coeffs = np.array([0.0, 0.5])
        irf = impulse_response(coeffs, n_periods=50)
        assert abs(irf[-1]) < 0.01


# =============================================================================
# Experiments
# =============================================================================


class TestExperiments:
    """Test experiment definitions."""

    def test_all_experiments_registered(self) -> None:
        """All experiments should be in the registry."""
        from validation.robustness.experiments import ALL_EXPERIMENT_NAMES, EXPERIMENTS

        assert len(EXPERIMENTS) == 7  # 5 parameter + 2 structural
        assert set(ALL_EXPERIMENT_NAMES) == {
            "credit_market",
            "goods_market",
            "labor_applications",
            "contract_length",
            "economy_size",
            "goods_market_no_pa",
            "entry_neutrality",
        }

    def test_experiment_labels_match_values(self) -> None:
        """Each experiment should have labels matching its values count."""
        from validation.robustness.experiments import EXPERIMENTS

        for name, exp in EXPERIMENTS.items():
            labels = exp.get_labels()
            assert len(labels) == len(exp.values), (
                f"{name}: {len(labels)} labels vs {len(exp.values)} values"
            )

    def test_baseline_value_in_values(self) -> None:
        """Each experiment's baseline_value should be in its values list."""
        from validation.robustness.experiments import EXPERIMENTS

        for name, exp in EXPERIMENTS.items():
            assert exp.baseline_value in exp.values, (
                f"{name}: baseline {exp.baseline_value} not in {exp.values}"
            )

    def test_get_config_single_param(self) -> None:
        """Single-param experiments return {param: value}."""
        from validation.robustness.experiments import CREDIT_MARKET

        config = CREDIT_MARKET.get_config(0)
        assert config == {"max_H": 1}

    def test_get_config_multi_param(self) -> None:
        """Multi-param experiments return full override dict."""
        from validation.robustness.experiments import ECONOMY_SIZE

        config = ECONOMY_SIZE.get_config(0)
        assert "n_firms" in config
        assert "n_households" in config
        assert "n_banks" in config

    def test_credit_market_values(self) -> None:
        """Credit market experiment has correct parameter values."""
        from validation.robustness.experiments import CREDIT_MARKET

        assert CREDIT_MARKET.param == "max_H"
        assert CREDIT_MARKET.values == [1, 2, 3, 4, 6]
        assert CREDIT_MARKET.baseline_value == 2

    def test_contract_length_values(self) -> None:
        """Contract length covers 1 to 14 quarters."""
        from validation.robustness.experiments import CONTRACT_LENGTH

        assert CONTRACT_LENGTH.param == "theta"
        assert 1 in CONTRACT_LENGTH.values
        assert 14 in CONTRACT_LENGTH.values
        assert CONTRACT_LENGTH.baseline_value == 8

    def test_economy_size_has_seven_configs(self) -> None:
        """Economy size has 4 proportional + 3 class-specific = 7 configs."""
        from validation.robustness.experiments import ECONOMY_SIZE

        assert len(ECONOMY_SIZE.values) == 7

    def test_structural_experiments_registered(self) -> None:
        """Structural experiments should be in both registries."""
        from validation.robustness.experiments import (
            STRUCTURAL_EXPERIMENT_NAMES,
            STRUCTURAL_EXPERIMENTS,
        )

        assert set(STRUCTURAL_EXPERIMENT_NAMES) == {
            "goods_market_no_pa",
            "entry_neutrality",
        }
        assert len(STRUCTURAL_EXPERIMENTS) == 2

    def test_entry_neutrality_has_setup_fn(self) -> None:
        """Entry neutrality should have a setup_fn for taxation extension."""
        from validation.robustness.experiments import ENTRY_NEUTRALITY

        assert ENTRY_NEUTRALITY.setup_fn is not None
        assert callable(ENTRY_NEUTRALITY.setup_fn)

    def test_goods_market_no_pa_has_no_setup_fn(self) -> None:
        """Goods market no-PA should not need a setup_fn."""
        from validation.robustness.experiments import GOODS_MARKET_NO_PA

        assert GOODS_MARKET_NO_PA.setup_fn is None

    def test_entry_neutrality_values(self) -> None:
        """Entry neutrality should sweep tax rates from 0 to 90%."""
        from validation.robustness.experiments import ENTRY_NEUTRALITY

        assert ENTRY_NEUTRALITY.param == "profit_tax_rate"
        assert ENTRY_NEUTRALITY.values == [0.0, 0.3, 0.5, 0.7, 0.9]
        assert ENTRY_NEUTRALITY.baseline_value == 0.0

    def test_goods_market_no_pa_values(self) -> None:
        """Goods market no-PA should combine random matching with Z sweep."""
        from validation.robustness.experiments import GOODS_MARKET_NO_PA

        assert GOODS_MARKET_NO_PA.param is None  # multi-param
        assert len(GOODS_MARKET_NO_PA.values) == 5
        for v in GOODS_MARKET_NO_PA.values:
            assert v["consumer_matching"] == "random"
            assert "max_Z" in v

    def test_setup_taxation_is_picklable(self) -> None:
        """setup_taxation must be picklable for ProcessPoolExecutor."""
        import pickle

        from validation.robustness.experiments import setup_taxation

        # Module-level functions are picklable; lambdas/closures are not
        pickled = pickle.dumps(setup_taxation)
        restored = pickle.loads(pickled)
        assert callable(restored)
