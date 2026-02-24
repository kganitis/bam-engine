"""Shared utility functions for validation scenarios."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_detrended_correlation(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> float:
    """Compute correlation of linearly detrended series.

    Removes linear trend from each series before computing correlation,
    which measures cyclical co-movement rather than trend co-movement.

    Parameters
    ----------
    x, y : NDArray
        Time series of equal length.

    Returns
    -------
    float
        Pearson correlation of detrended series, or 0.0 if series
        are too short or detrending fails.
    """
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


def filter_outliers_iqr(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Filter outliers using IQR method for both x and y variables.

    Points are removed if either coordinate falls outside 1.5 * IQR
    of its respective variable's quartile range.

    Parameters
    ----------
    x, y : NDArray
        Paired data arrays of equal length.

    Returns
    -------
    tuple of NDArray
        Filtered (x, y) with outliers removed.
    """
    if len(x) < 4 or len(y) < 4:
        return x, y

    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    x_lower, x_upper = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x

    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    y_lower, y_upper = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y

    mask = (x >= x_lower) & (x <= x_upper) & (y >= y_lower) & (y <= y_upper)
    return x[mask], y[mask]


def compute_real_interest_rate(
    loan_principals: list[NDArray[np.floating]],
    loan_rates: list[NDArray[np.floating]],
    inflation: NDArray[np.floating],
    fallback_rate: float,
) -> NDArray[np.floating]:
    """Compute per-period real interest rate from loan data.

    For each period, computes the principal-weighted average nominal
    interest rate across all active loans, then subtracts inflation.

    Parameters
    ----------
    loan_principals : list of NDArray
        Per-period arrays of loan principal amounts.
    loan_rates : list of NDArray
        Per-period arrays of loan interest rates.
    inflation : NDArray
        Inflation rate per period.
    fallback_rate : float
        Nominal rate to use when no active loans exist (typically r_bar).

    Returns
    -------
    NDArray
        Real interest rate per period.
    """
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
            weighted_nominal = fallback_rate
        real_interest_rate[t] = weighted_nominal - inflation[t]
    return real_interest_rate


def adjust_burn_in(burn_in: int, n_periods: int, *, verbose: bool = False) -> int:
    """Adjust burn-in period if it exceeds available simulation periods.

    When ``burn_in >= n_periods`` (e.g., for quick tests with few periods),
    the burn-in is reduced to half of ``n_periods`` to ensure some data
    remains for analysis.

    Parameters
    ----------
    burn_in : int
        Requested burn-in periods.
    n_periods : int
        Total simulation periods.
    verbose : bool
        If True, print a message when burn-in is adjusted.

    Returns
    -------
    int
        Adjusted burn-in value.
    """
    if burn_in >= n_periods:
        burn_in = max(0, n_periods // 2)
        if verbose:
            print(f"  (burn_in adjusted to {burn_in} for short simulation)")
    return burn_in
