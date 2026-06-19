import numpy as np

from comparison.equivalence.metrics import METRIC_KEYS, compute_metrics, iqr_filter


def _series(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    u = 0.07 + 0.01 * rng.standard_normal(n)
    # construct wage_inflation strongly negatively correlated with u (Phillips)
    wage_inf = 0.05 - 2.0 * (u - 0.07) + 0.001 * rng.standard_normal(n)
    log_gdp = np.cumsum(0.001 * rng.standard_normal(n)) + 5.4
    vac = 0.13 - 0.5 * (u - 0.07) + 0.002 * rng.standard_normal(n)  # Beveridge negative
    return {
        "unemployment": u.tolist(),
        "price_inflation": (0.05 + 0.01 * rng.standard_normal(n)).tolist(),
        "wage_inflation": wage_inf.tolist(),
        "log_gdp": log_gdp.tolist(),
        "vacancy_rate": vac.tolist(),
        "production_final": np.abs(rng.standard_gamma(2.0, size=120)).tolist(),
    }


def test_returns_all_metric_keys():
    m = compute_metrics(_series(), burn_in=500)
    assert set(METRIC_KEYS) <= set(m)


def test_phillips_and_beveridge_negative():
    m = compute_metrics(_series(), burn_in=500)
    assert m["phillips_corr"] < 0
    assert m["beveridge_corr"] < 0


def test_mean_unemployment_close_to_input():
    m = compute_metrics(_series(seed=1), burn_in=500)
    assert 0.05 < m["mean_unemployment"] < 0.09


def test_iqr_filter_drops_outlier_pair():
    x = np.array([1.0, 2, 3, 4, 100])
    y = np.array([1.0, 2, 3, 4, 5])
    fx, fy = iqr_filter(x, y)
    assert 100 not in fx
    assert len(fx) == len(fy) == 4
