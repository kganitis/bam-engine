from __future__ import annotations

import numpy as np
from scipy.stats import skew

METRIC_KEYS = (
    "mean_unemployment",
    "mean_inflation",
    "firm_size_skew",
    "phillips_corr",
    "okun_corr",
    "beveridge_corr",
)


def iqr_filter(x, y, k: float = 1.5):
    """Filter outlier pairs from two series using IQR rule.

    Parameters
    ----------
    x, y : array-like
        Input series of equal length.
    k : float
        IQR multiplier (default 1.5).

    Returns
    -------
    x_filtered, y_filtered : np.ndarray
        Paired filtered arrays with identical length.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.ones(len(x), bool)
    for s in (x, y):
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        keep &= (s >= q1 - k * iqr) & (s <= q3 + k * iqr)
    return x[keep], y[keep]


def _corr(a, b) -> float:
    """Compute Pearson correlation, returning NaN for degenerate cases.

    Parameters
    ----------
    a, b : array-like
        Input series.

    Returns
    -------
    float
        Correlation coefficient or NaN.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(series: dict, burn_in: int = 500) -> dict:
    """Compute behavioral equivalence metrics from raw series.

    Parameters
    ----------
    series : dict
        Keys: unemployment, price_inflation, wage_inflation, log_gdp,
        vacancy_rate (each a list/array of length n_periods),
        production_final (per-firm array).
    burn_in : int
        Number of periods to exclude from the start (default 500).

    Returns
    -------
    dict
        Keys: mean_unemployment, mean_inflation, firm_size_skew,
        phillips_corr, okun_corr, beveridge_corr.
    """
    bi = burn_in
    unemp = np.asarray(series["unemployment"], float)
    infl = np.asarray(series["price_inflation"], float)
    wage_inf = np.asarray(series["wage_inflation"], float)
    log_gdp = np.asarray(series["log_gdp"], float)
    vac = np.asarray(series["vacancy_rate"], float)
    prod = np.asarray(series["production_final"], float)

    gdp_growth = np.diff(log_gdp)  # length n-1
    unemp_growth = np.diff(unemp)  # length n-1

    unemp_ss = unemp[bi:]
    phillips = _corr(unemp_ss, wage_inf[bi:])
    beveridge = _corr(unemp_ss, vac[bi:])
    ug_ss, gg_ss = unemp_growth[bi - 1 :], gdp_growth[bi - 1 :]
    fug, fgg = iqr_filter(ug_ss, gg_ss)
    okun = _corr(fug, fgg)

    pos = prod[prod > 0]
    fs_skew = float(skew(pos)) if len(pos) > 2 else float("nan")

    return {
        "mean_unemployment": float(np.mean(unemp_ss)),
        "mean_inflation": float(np.mean(infl[bi:])),
        "firm_size_skew": fs_skew,
        "phillips_corr": phillips,
        "okun_corr": okun,
        "beveridge_corr": beveridge,
    }
