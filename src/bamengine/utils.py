"""
Utility functions for BAM Engine.

This module provides common utility functions used throughout BAM Engine,
including statistical operations (trimmed means), random sampling utilities,
and efficient array operations.

Constants
---------
EPS : float
    Machine epsilon for numerical comparisons (1.0e-9). Used to avoid
    division by zero and detect near-zero values.

Functions
---------
trim_mean
    Calculate two-sided trimmed mean (SciPy-style).
trimmed_weighted_mean
    Calculate weighted trimmed mean with optional weight filtering.
sample_beta_with_mean
    Draw samples from Beta distribution with specified mean.
select_top_k_indices_sorted
    Efficiently select and sort top-k elements using argpartition.

See Also
--------
bamengine.typing : Type aliases used in this module
scipy.stats.trim_mean : SciPy implementation of trimmed mean

Notes
-----
- Trimmed means are robust statistics that exclude extreme values
- Beta sampling is used for initialization with controlled variance
- Top-k selection uses argpartition for O(n) performance vs O(n log n) for sort
"""

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.typing import Float1D, Idx1D

EPS = 1.0e-9


def trim_mean(values: Float1D, trim_pct: float = 0.05) -> float:
    """
    Calculate two-sided trimmed mean (robust statistic).

    Computes the mean after removing a percentage of the smallest and largest
    values. This provides a robust estimate of central tendency that is less
    sensitive to outliers than the arithmetic mean.

    Parameters
    ----------
    values : Float1D
        1D array of values to average.
    trim_pct : float, optional
        Proportion of values to trim from each tail (default: 0.05 = 5%).
        For example, 0.05 removes the bottom 5% and top 5% of values.

    Returns
    -------
    float
        Trimmed mean of the values. Returns 0.0 if input array is empty.

    Examples
    --------
    Calculate trimmed mean removing 10% from each tail:

    >>> import numpy as np
    >>> from bamengine.utils import trim_mean
    >>> values = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
    >>> trim_mean(values, trim_pct=0.10)
    3.5

    Default 5% trimming:

    >>> values = np.arange(1, 101)  # 1 to 100
    >>> mean = trim_mean(values)  # Removes bottom/top 5% (5 values each)

    Notes
    -----
    - Uses `np.argpartition` for O(n) selection instead of O(n log n) sorting
    - If trim_pct results in k=0, returns regular mean
    - Compatible with scipy.stats.trim_mean behavior
    - Widely used in BAM Engine for initializing new firms/banks from survivors

    See Also
    --------
    trimmed_weighted_mean : Weighted version with optional weight filtering
    scipy.stats.trim_mean : SciPy implementation
    """
    if values.size == 0:
        return 0.0
    k = int(round(trim_pct * values.size))
    if k == 0:
        return float(values.mean())
    idx = np.argpartition(values, (k, values.size - k - 1))
    core = values[idx[k : values.size - k]]
    return float(core.mean())


def trimmed_weighted_mean(
    values: Float1D,
    weights: Float1D | None = None,
    trim_pct: float = 0.05,
    min_weight: float = 1e-3,
) -> float:
    """
    Calculate trimmed weighted mean with optional weight filtering.

    Computes a weighted mean after (1) filtering out entries with negligible
    weights, and (2) trimming extreme values. If no weights are provided,
    falls back to unweighted trimmed mean.

    Parameters
    ----------
    values : Float1D
        1D array of values to average.
    weights : Float1D, optional
        1D array of weights (same length as values). If None, computes
        unweighted trimmed mean (ignores min_weight parameter).
    trim_pct : float, optional
        Proportion of values to trim from each tail (default: 0.05 = 5%).
        Trimming is applied after weight filtering.
    min_weight : float, optional
        Minimum weight threshold (default: 1e-3). Entries with weights below
        this are excluded before computing mean. Ignored if weights is None.

    Returns
    -------
    float
        Trimmed weighted mean. Returns 0.0 if no valid entries remain after
        filtering and trimming.

    Examples
    --------
    Weighted mean with weight filtering:

    >>> import numpy as np
    >>> from bamengine.utils import trimmed_weighted_mean
    >>> values = np.array([10, 20, 30, 40])
    >>> weights = np.array([0.5, 1.0, 1.5, 0.0001])  # Last weight too small
    >>> trimmed_weighted_mean(values, weights, trim_pct=0.0, min_weight=0.01)
    22.0  # Only first 3 values included

    Trimmed weighted mean:

    >>> values = np.array([1, 2, 3, 100])  # 100 is outlier
    >>> weights = np.array([1, 1, 1, 1])
    >>> trimmed_weighted_mean(values, weights, trim_pct=0.25)  # Trims 1 from each end
    2.5  # Mean of [2, 3]

    Unweighted mode (weights=None):

    >>> values = np.array([1, 2, 3, 4, 5])
    >>> trimmed_weighted_mean(values, weights=None, trim_pct=0.20)
    3.0  # Regular trimmed mean

    Notes
    -----
    - Weight filtering occurs before trimming
    - Trimming is based on value ordering (not weight ordering)
    - Falls back to unweighted mean if all weights are zero
    - If weights is None, min_weight parameter is ignored

    See Also
    --------
    trim_mean : Unweighted trimmed mean (faster if no weights needed)
    numpy.average : NumPy weighted average (no trimming)
    """
    values = np.asarray(values)

    # If weights is None, use unweighted logic
    if weights is None:
        if trim_pct == 0.0:
            return float(values.mean())
        # Trimming only, unweighted
        k = int(round(trim_pct * values.size))
        if k == 0 or values.size == 0:  # TODO Branch uncovered by tests
            return float(values.mean())
        idx = np.argsort(values)
        trimmed = values[idx][k : values.size - k]
        if trimmed.size == 0:  # TODO Branch uncovered by tests
            return 0.0
        return float(trimmed.mean())

    # Weighted logic
    weights = np.asarray(weights)
    mask = weights >= min_weight
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return 0.0

    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    k = int(round(trim_pct * values.size))
    if k == 0:
        # Standard weighted mean
        return float(np.average(values, weights=weights))
    # Apply trim
    values_trimmed = values[k : values.size - k]
    weights_trimmed = weights[k : weights.size - k]
    if weights_trimmed.sum() == 0:
        return (
            float(values_trimmed.mean()) if values_trimmed.size else 0.0
        )  # fallback: unweighted mean
    return float(np.average(values_trimmed, weights=weights_trimmed))


def sample_beta_with_mean(
    mean: float,
    n: int = 1,
    low: float | None = None,
    high: float | None = None,
    concentration: float = 12.0,
    *,
    relative_margin: float = 0.50,
    rng: Generator | None = None,
) -> float | Float1D:
    """
    Draw *n* samples from a Beta distribution scaled to [low, high),
    such that the **scaled mean is approximately ``mean``**.

    Parameters
    ----------
    mean : float
        Desired mean of the returned samples. Can be any positive value.
    n : int, default 1
        Number of samples to draw.
    low, high : float or None, optional
        Bounds of the target interval.  If either is None, it is derived as::

            low = mean * (1 - relative_margin)
            high = mean * (1 + relative_margin)

        making the mean the midpoint of the interval.  A tiny eps is added
        when needed to guarantee ``low < mean < high``.
    concentration : float, default 12
        Total pseudo–sample size of the Beta (a+b).  Larger values
        concentrate the draws more tightly around *mean*.
    relative_margin : float, default 0.50
        Half-width of the automatically generated interval as a fraction of
        *mean*.  Set 0.25 for ±25 %, 1.0 for ±100 %, etc.
    rng : np.random.Generator, optional
        Random number generator (falls back to ``default_rng()``).

    Returns
    -------
    float or ndarray
        *n* samples if ``n > 1``; otherwise a scalar.

    Notes
    -----
    • If *mean* is very close to zero, the automatically chosen ``low`` may be
      negative; it is then clipped to zero and an eps-wide gap is kept so that
      ``mean`` stays strictly inside the interval.

    • The function raises ``ValueError`` for invalid arguments.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if concentration <= 0:
        raise ValueError("concentration must be > 0")
    if relative_margin <= 0:
        raise ValueError("relative_margin must be > 0")

    # Automatic bounds
    if low is None or high is None:
        half_span = abs(mean) * relative_margin
        low = mean - half_span if low is None else low
        high = mean + half_span if high is None else high

    # Ensure ordering and strict inequality
    eps = max(abs(mean), 1.0) * 1e-12  # machine-safe gap
    if not (low < mean < high):
        # Shift the offending bound just enough to satisfy the inequality
        if low >= mean:
            low = mean - eps
        if high <= mean:
            high = mean + eps
        if not (low < mean < high):  # pragma: no cover
            raise ValueError(
                f"Could not place mean ({mean}) strictly inside (low, high): "
                f"low={low}, high={high}"
            )

    # Map mean to (0,1) for Beta parameterization
    m = (mean - low) / (high - low)
    a = m * concentration
    b = (1.0 - m) * concentration

    rng = default_rng() if rng is None else rng
    samples = rng.beta(a, b, size=n)
    scaled = low + (high - low) * samples
    return scaled.item() if n == 1 else scaled


def select_top_k_indices_sorted(
    values: Float1D, k: int, descending: bool = True
) -> Idx1D:
    """
    Returns indices of k smallest/largest elements, sorted along the last axis.

    Identifies the k elements (smallest or largest based on the `descending`
    flag) along the last axis of the input N-dimensional array `values`.
    It then returns the original indices of these k elements, sorted such
    that `np.take_along_axis(values, returned_indices, axis=-1)` yields
    values in the specified order (ascending or descending).

    Parameters
    ----------
    values : numpy.ndarray
        N-dimensional array of numerical values. Selection occurs along the last axis.
    k : int
        The number of indices to select.
    descending : bool, optional
        If True (default), selects k largest elements (sorted largest to smallest).
        If False, selects k smallest elements (sorted smallest to largest).

    Returns
    -------
    numpy.ndarray
        N-dimensional array of integer indices, shaped `values.shape[:-1] +
        (selected_k,)`. `selected_k` is `k` (or 0 if `k<=0`, or `values.shape[-1]`
        if `k` exceeds the last axis dimension). Indices refer to positions along the
        last axis of `values`.

    Notes
    -----
    - Operates on the last axis of N-dimensional arrays.
    - Uses `np.argpartition` for efficient selection when `k` is less than the
      size of the last dimension.
    - Scalar inputs are treated as 1-element arrays.
    - Handles empty input arrays and `k <= 0` by returning appropriately
      shaped empty index arrays.
    """
    # Ensure input is a NumPy array (defensive: accepts list input at runtime).
    if not isinstance(values, np.ndarray):
        values = np.array(values, dtype=float)  # type: ignore[unreachable]

    # Ensure values is at least 1D for consistent axis=-1 operations.
    if values.ndim == 0:  # TODO Branch uncovered by tests
        values = np.atleast_1d(values)

    # If k is non-positive, return an empty array with appropriate shape.
    if k <= 0:
        return np.empty(values.shape[:-1] + (0,), dtype=np.intp)

    # If array is empty (and k > 0), also return an empty array.
    if values.size == 0:
        return np.empty(values.shape[:-1] + (0,), dtype=np.intp)

    n = values.shape[-1]  # Size of the last dimension.

    # If k >= n, all elements are selected; sort all indices along the last axis.
    if k >= n:
        # Sort all elements if k is large enough.
        # Negate values for descending sort with np.argsort.
        return np.argsort(-values if descending else values, axis=-1)

    # For k < n:
    # Determine values to partition (negate for descending to find largest).
    values_to_partition = -values if descending else values

    # Efficiently find indices of the k smallest/largest elements
    # (unsorted among themselves).
    # `kth=k-1` because argpartition is 0-indexed and we want the first k elements.
    partitioned_indices = np.argpartition(values_to_partition, kth=k - 1, axis=-1)

    # Take the indices of the first k elements from the partitioned result.
    k_indices_unsorted = partitioned_indices[..., :k]

    # Get the actual values corresponding to these k selected indices.
    k_values = np.take_along_axis(values, k_indices_unsorted, axis=-1)

    # Determine values to sort within the k-selection.
    k_values_to_sort = -k_values if descending else k_values

    # Get the order to sort these k values.
    order_within_k_selection = np.argsort(k_values_to_sort, axis=-1)

    # Apply this order to `k_indices_unsorted` to get the final sorted indices.
    k_indices_sorted_final = np.take_along_axis(
        k_indices_unsorted, order_within_k_selection, axis=-1
    )

    return k_indices_sorted_final
