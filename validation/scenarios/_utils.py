"""Shared utility functions for validation scenarios."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
