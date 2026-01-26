"""Internal utilities for metrics computation.

This module provides shared helper functions used by both baseline and
growth_plus metric modules.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


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


# =============================================================================
# Recession Detection Utilities
# =============================================================================


def smooth_series(x: NDArray[np.floating], window: int = 5) -> NDArray[np.floating]:
    """Apply centered moving average smoothing.

    Parameters
    ----------
    x : NDArray
        Time series to smooth.
    window : int, default=5
        Window size for moving average (odd number preferred).

    Returns
    -------
    NDArray
        Smoothed time series (same length as input).
    """
    if window < 1:
        return x
    if window == 1:
        return x.copy()

    kernel = np.ones(window) / window
    # Use 'same' mode to preserve length
    smoothed = np.convolve(x, kernel, mode="same")

    # Fix edge effects by using original values at boundaries
    half = window // 2
    if half > 0:
        smoothed[:half] = x[:half]
        smoothed[-half:] = x[-half:]

    return smoothed


def find_turning_points(
    series: NDArray[np.floating],
    prominence: float = 0.03,
    distance: int = 20,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Find peaks (local maxima) and troughs (local minima) in a time series.

    Uses scipy.signal.find_peaks with prominence filtering to identify
    significant turning points in the business cycle.

    Parameters
    ----------
    series : NDArray
        Time series (e.g., smoothed log GDP).
    prominence : float, default=0.03
        Minimum prominence for a peak/trough to be detected.
        Higher values filter out minor fluctuations.
    distance : int, default=20
        Minimum number of periods between peaks/troughs.

    Returns
    -------
    peaks : NDArray
        Indices of local maxima (business cycle peaks).
    troughs : NDArray
        Indices of local minima (business cycle troughs).
    """
    # Find peaks (local maxima)
    peaks, _ = find_peaks(series, prominence=prominence, distance=distance)

    # Find troughs (local minima) by inverting the signal
    troughs, _ = find_peaks(-series, prominence=prominence, distance=distance)

    return peaks, troughs


def create_recession_mask(
    n_periods: int,
    peaks: NDArray[np.intp],
    troughs: NDArray[np.intp],
    extension_after_trough: int = 10,
) -> NDArray[np.bool_]:
    """Create recession mask from peak to trough + extension.

    Marks the period from each business cycle peak to the subsequent
    trough (plus a small extension) as a recession episode.

    Parameters
    ----------
    n_periods : int
        Total number of periods in the time series.
    peaks : NDArray
        Indices of business cycle peaks.
    troughs : NDArray
        Indices of business cycle troughs.
    extension_after_trough : int, default=10
        Number of periods to extend recession after trough.
        This captures the early recovery phase.

    Returns
    -------
    recession_mask : NDArray[np.bool_]
        Boolean array where True indicates a recession period.
    """
    recession_mask = np.zeros(n_periods, dtype=bool)

    for peak in peaks:
        # Find the next trough after this peak
        future_troughs = troughs[troughs > peak]
        if len(future_troughs) > 0:
            trough = future_troughs[0]
            # Mark from peak to trough + extension
            end = min(trough + extension_after_trough, n_periods)
            recession_mask[peak:end] = True

    return recession_mask


def bridge_short_gaps(
    recession_mask: NDArray[np.bool_],
    min_gap: int = 10,
) -> NDArray[np.bool_]:
    """Bridge short gaps between recession episodes.

    If two recession episodes are separated by fewer than min_gap periods,
    treat them as a single continuous episode. This handles brief positive
    blips that occur during ongoing recessions.

    Parameters
    ----------
    recession_mask : NDArray[np.bool_]
        Initial recession mask.
    min_gap : int, default=10
        Minimum gap between separate recession episodes.
        Shorter gaps are bridged.

    Returns
    -------
    NDArray[np.bool_]
        Recession mask with short gaps bridged.
    """
    if not np.any(recession_mask):
        return recession_mask

    result = recession_mask.copy()

    # Find episode boundaries
    padded = np.concatenate([[False], recession_mask, [False]])
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]

    # Bridge gaps that are too short
    for i in range(len(ends) - 1):
        gap = starts[i + 1] - ends[i]
        if gap < min_gap:
            result[ends[i] : starts[i + 1]] = True

    return result


def detect_recessions(
    log_gdp: NDArray[np.floating],
    smoothing_window: int = 5,
    peak_prominence: float = 0.03,
    peak_distance: int = 20,
    min_gap: int = 10,
    extension_after_trough: int = 10,
) -> NDArray[np.bool_]:
    """Detect recession episodes using peak-to-trough algorithm.

    This algorithm identifies business cycle peaks and troughs in the
    log GDP series, then marks the period from each peak to the
    subsequent trough (plus a small extension) as a recession.

    The approach is inspired by the NBER business cycle dating methodology
    and adapted for the BAM model's characteristics.

    Parameters
    ----------
    log_gdp : NDArray
        Log of GDP time series.
    smoothing_window : int, default=5
        Window size for moving average smoothing (odd number preferred).
        Reduces noise before peak detection.
    peak_prominence : float, default=0.03
        Minimum prominence for peak/trough detection.
        Higher values filter out minor fluctuations.
        A value of 0.03 means ~3% change in log GDP is significant.
    peak_distance : int, default=20
        Minimum periods between detected peaks/troughs.
        Prevents detecting multiple peaks within the same cycle.
    min_gap : int, default=10
        Minimum gap between separate recession episodes.
        Shorter gaps are bridged to form single episodes.
    extension_after_trough : int, default=10
        Periods to extend recession after trough detection.
        Captures the early recovery phase.

    Returns
    -------
    recession_mask : NDArray[np.bool_]
        Boolean array where True indicates a recession period.

    Notes
    -----
    Default parameters are tuned for the Growth+ scenario with
    ~500 analysis periods (after burn-in). Expected results:
    - 2-4 recession episodes
    - Average episode length: 40-70 periods

    Examples
    --------
    >>> import numpy as np
    >>> log_gdp = np.cumsum(np.random.randn(500) * 0.01) + 8.0
    >>> mask = detect_recessions(log_gdp)
    >>> n_recession_periods = np.sum(mask)
    """
    # Handle edge cases
    if len(log_gdp) < smoothing_window * 2:
        return np.zeros(len(log_gdp), dtype=bool)

    # Step 1: Smooth the series
    smoothed = smooth_series(log_gdp, window=smoothing_window)

    # Step 2: Find turning points
    peaks, troughs = find_turning_points(
        smoothed, prominence=peak_prominence, distance=peak_distance
    )

    # Step 3: Create recession episodes
    recession_mask = create_recession_mask(
        n_periods=len(log_gdp),
        peaks=peaks,
        troughs=troughs,
        extension_after_trough=extension_after_trough,
    )

    # Step 4: Bridge short gaps
    recession_mask = bridge_short_gaps(recession_mask, min_gap=min_gap)

    return recession_mask
