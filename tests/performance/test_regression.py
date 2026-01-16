"""Performance regression tests.

These tests ensure that performance hasn't regressed beyond acceptable
thresholds. Baselines are established from benchmarking runs and updated
after confirmed improvements.

**Important**: These tests are designed for local development on consistent
hardware. They are SKIPPED in CI because:
- CI runners are virtualized and performance varies significantly (1.5-3x)
- Different OS runners have different baseline performance
- pytest-based regression tests need machine-specific baselines

**Future Work**: Consider using ASV (Airspeed Velocity) for proper performance
tracking with machine-specific baselines and historical commit tracking.

To run regression tests locally:
    pytest -m regression
    pytest  # runs all tests including regression
"""

import logging
import time

import pytest

from bamengine import Simulation

# Disable logging for performance tests
logging.getLogger("bamengine").setLevel(logging.ERROR)


# Performance baselines (seconds) - updated January 16, 2026
# Note: These include pytest framework overhead (~20-25% slower than pure benchmark)
# Update these after confirming performance improvements
# Major optimization (Jan 2026): logging guards + vectorized firm selection (~75-80% faster)
BASELINE_SMALL = 3.0  # 100 firms, 500 households, 1000 periods
BASELINE_MEDIUM = 6.0  # 200 firms, 1000 households, 1000 periods
BASELINE_LARGE = 18.0  # 500 firms, 2500 households, 1000 periods

# Allowed regression threshold (15% slower than baseline)
# More relaxed than pure benchmarks due to test framework overhead variability
REGRESSION_THRESHOLD = 1.15


def benchmark_configuration(n_firms, n_households, n_periods, seed=42):
    """Benchmark a single simulation configuration.

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_households : int
        Number of households
    n_periods : int
        Number of periods to simulate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    float
        Elapsed time in seconds
    """
    sim = Simulation.init(
        n_firms=n_firms,
        n_households=n_households,
        seed=seed,
        logging={"default_level": "ERROR"},  # Disable logging for performance tests
    )

    start = time.perf_counter()
    sim.run(n_periods)
    elapsed = time.perf_counter() - start

    return elapsed


@pytest.mark.slow
@pytest.mark.regression
def test_no_performance_regression_small():
    """Ensure performance hasn't regressed for small configuration."""
    elapsed = benchmark_configuration(100, 500, 1000)

    threshold = BASELINE_SMALL * REGRESSION_THRESHOLD

    assert elapsed < threshold, (
        f"Performance regression detected (Small config): "
        f"{elapsed:.3f}s > {threshold:.3f}s threshold "
        f"(baseline: {BASELINE_SMALL:.3f}s, +{(elapsed / BASELINE_SMALL - 1) * 100:.1f}%)"
    )

    # Print performance info (even on success)
    change_pct = (elapsed / BASELINE_SMALL - 1) * 100
    print(
        f"\nSmall config: {elapsed:.3f}s"
        f" (baseline: {BASELINE_SMALL:.3f}s, {change_pct:+.1f}%)"
    )


@pytest.mark.slow
@pytest.mark.regression
def test_no_performance_regression_medium():
    """Ensure performance hasn't regressed for medium configuration."""
    elapsed = benchmark_configuration(200, 1000, 1000)

    threshold = BASELINE_MEDIUM * REGRESSION_THRESHOLD

    assert elapsed < threshold, (
        f"Performance regression detected (Medium config): "
        f"{elapsed:.3f}s > {threshold:.3f}s threshold "
        f"(baseline: {BASELINE_MEDIUM:.3f}s, +{(elapsed / BASELINE_MEDIUM - 1) * 100:.1f}%)"
    )

    # Print performance info (even on success)
    change_pct = (elapsed / BASELINE_MEDIUM - 1) * 100
    print(
        f"\nMedium config: {elapsed:.3f}s"
        f" (baseline: {BASELINE_MEDIUM:.3f}s, {change_pct:+.1f}%)"
    )


@pytest.mark.slow
@pytest.mark.regression
def test_no_performance_regression_large():
    """Ensure performance hasn't regressed for large configuration."""
    elapsed = benchmark_configuration(500, 2500, 1000)

    threshold = BASELINE_LARGE * REGRESSION_THRESHOLD

    assert elapsed < threshold, (
        f"Performance regression detected (Large config): "
        f"{elapsed:.3f}s > {threshold:.3f}s threshold "
        f"(baseline: {BASELINE_LARGE:.3f}s, +{(elapsed / BASELINE_LARGE - 1) * 100:.1f}%)"
    )

    # Print performance info (even on success)
    change_pct = (elapsed / BASELINE_LARGE - 1) * 100
    print(
        f"\nLarge config: {elapsed:.3f}s"
        f" (baseline: {BASELINE_LARGE:.3f}s, {change_pct:+.1f}%)"
    )
