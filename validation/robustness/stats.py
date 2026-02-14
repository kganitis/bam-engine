"""Statistical tools for robustness analysis.

Pure functions for time series analysis: Hodrick-Prescott filtering,
lead-lag cross-correlations, autoregressive model fitting, and
impulse-response function computation. No simulation dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve


def hp_filter(
    y: NDArray[np.floating],
    lamb: float = 1600.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Hodrick-Prescott filter decomposition into trend and cycle.

    Solves the penalized least-squares problem:

        min_τ  Σ(y_t - τ_t)² + λ Σ(τ_{t+1} - 2τ_t + τ_{t-1})²

    via the sparse linear system ``(I + λ K'K) τ = y``.

    Parameters
    ----------
    y : NDArray
        Time series of length T.
    lamb : float
        Smoothing parameter. Standard values: 1600 for quarterly data,
        6.25 for annual, 129600 for monthly.

    Returns
    -------
    trend : NDArray
        Trend component (τ).
    cycle : NDArray
        Cyclical component (y - τ).
    """
    t = len(y)
    if t < 3:
        return y.copy(), np.zeros_like(y)

    # Build second-difference matrix K of shape (T-2, T)
    # K[i, :] = [0, ..., 0, 1, -2, 1, 0, ..., 0] with 1 at position i
    diags = np.array([np.ones(t - 2), -2 * np.ones(t - 2), np.ones(t - 2)])
    offsets = [0, 1, 2]
    k_mat = sparse.diags(diags, offsets, shape=(t - 2, t), format="csc")

    # Solve (I + λ K'K) τ = y
    identity = sparse.eye(t, format="csc")
    a_mat = identity + lamb * (k_mat.T @ k_mat)
    trend = spsolve(a_mat, y)

    cycle = y - trend
    return trend, cycle


def cross_correlation(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    max_lag: int = 4,
) -> NDArray[np.floating]:
    """Cross-correlation of x and y at integer leads and lags.

    Computes ``corr(x_t, y_{t+k})`` for ``k = -max_lag, ..., 0, ..., +max_lag``.

    At lag k=0 this is the contemporaneous correlation. Positive k means
    y *leads* x (y is shifted forward relative to x).

    Parameters
    ----------
    x, y : NDArray
        Time series of equal length T.
    max_lag : int
        Maximum lead/lag to compute.

    Returns
    -------
    NDArray
        Array of shape ``(2 * max_lag + 1,)`` with correlations.
        Index ``max_lag + k`` holds ``corr(x_t, y_{t+k})``.
    """
    n = len(x)
    if n != len(y):
        raise ValueError(f"Series must have equal length, got {n} and {len(y)}")

    result = np.zeros(2 * max_lag + 1)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x_seg = x[: n - k]
            y_seg = y[k:]
        else:
            x_seg = x[-k:]
            y_seg = y[: n + k]

        if len(x_seg) < 3:
            result[max_lag + k] = np.nan
        else:
            result[max_lag + k] = np.corrcoef(x_seg, y_seg)[0, 1]

    return result


def fit_ar(
    y: NDArray[np.floating],
    order: int = 2,
) -> tuple[NDArray[np.floating], float]:
    """Fit an AR(p) model via ordinary least squares.

    Estimates the model ``y_t = c + φ_1 y_{t-1} + ... + φ_p y_{t-p} + ε_t``.

    Parameters
    ----------
    y : NDArray
        Time series of length T.
    order : int
        AR order (p). Must be >= 1.

    Returns
    -------
    coeffs : NDArray
        Estimated coefficients ``[c, φ_1, ..., φ_p]`` of length ``order + 1``.
    r_squared : float
        Coefficient of determination (R²).
    """
    t = len(y)
    if t <= order + 1:
        raise ValueError(
            f"Series length {t} too short for AR({order}), need > {order + 1}"
        )

    # Build design matrix: [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    x_mat = np.ones((t - order, order + 1))
    for lag in range(1, order + 1):
        x_mat[:, lag] = y[order - lag : t - lag]

    y_dep = y[order:]

    # Solve via least squares
    coeffs, _residuals, _, _ = np.linalg.lstsq(x_mat, y_dep, rcond=None)

    # Compute R²
    y_pred = x_mat @ coeffs
    ss_res = np.sum((y_dep - y_pred) ** 2)
    ss_tot = np.sum((y_dep - np.mean(y_dep)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, float(r_squared)


def impulse_response(
    ar_coeffs: NDArray[np.floating],
    n_periods: int = 20,
) -> NDArray[np.floating]:
    """Compute impulse-response function from AR coefficients.

    Simulates the response to a unit shock at t=0 through the AR recursion.

    Parameters
    ----------
    ar_coeffs : NDArray
        AR coefficients ``[c, φ_1, ..., φ_p]`` from :func:`fit_ar`.
    n_periods : int
        Number of periods to simulate.

    Returns
    -------
    NDArray
        Impulse-response values of length ``n_periods``.
    """
    order = len(ar_coeffs) - 1  # Exclude constant
    phi = ar_coeffs[1:]  # AR coefficients only (no constant)

    irf = np.zeros(n_periods)
    irf[0] = 1.0  # Unit shock

    for t in range(1, n_periods):
        for lag in range(min(t, order)):
            irf[t] += phi[lag] * irf[t - lag - 1]

    return irf
