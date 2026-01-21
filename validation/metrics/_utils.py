"""Internal utilities for metrics computation.

This module provides shared helper functions used by both baseline and
growth_plus metric modules.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray


def filter_outliers_iqr(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Filter outliers using IQR method for both x and y variables.

    Keeps points where both x and y are within 1.5×IQR of their quartiles.
    This is the standard statistical method for outlier detection.

    Parameters
    ----------
    x : NDArray
        First variable (e.g., unemployment growth rate).
    y : NDArray
        Second variable (e.g., GDP growth rate).

    Returns
    -------
    tuple[NDArray, NDArray]
        Filtered x and y arrays with outliers removed.
    """
    # Handle empty or too-small arrays
    if len(x) < 4 or len(y) < 4:
        # Need at least 4 points for meaningful quartiles
        return x, y

    # Compute IQR bounds for x
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    x_lower, x_upper = q1_x - 1.5 * iqr_x, q3_x + 1.5 * iqr_x

    # Compute IQR bounds for y
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    y_lower, y_upper = q1_y - 1.5 * iqr_y, q3_y + 1.5 * iqr_y

    # Keep points within bounds for BOTH variables
    mask = (x >= x_lower) & (x <= x_upper) & (y >= y_lower) & (y <= y_upper)
    return x[mask], y[mask]


def get_targets_dir() -> str:
    """Get the path to the validation/targets directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "targets")
