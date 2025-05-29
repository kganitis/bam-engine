# src/bamengine/helpers.py
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from bamengine.typing import Float1D, Idx1D


def trim_mean(x: Float1D, p: float = 0.05) -> float:
    """Return the ``p`` % two-sided trimmed mean ( SciPy-style )."""
    if x.size == 0:
        return 0.0
    k = int(round(p * x.size))
    if k == 0:
        return float(x.mean())
    idx = np.argpartition(x, (k, x.size - k - 1))
    core = x[idx[k : x.size - k]]
    return float(core.mean())


def sample_beta_with_mean(
    mean: float,
    n: int = 1,
    low: float = 0.1,
    high: float = 1.0,
    concentration: float = 12,
    rng: Generator = default_rng(),
) -> float | Float1D:
    """
    Sample n values from a Beta distribution scaled to [low, high),
    with the scaled mean approximately equal to `mean`.

    Parameters
    ----------
    mean : float
        Desired mean value in [low, high).
    n : int
        Number of samples to draw.
    low : float, optional
        Lower bound (inclusive) of the output interval.
    high : float, optional
        Upper bound (exclusive) of the output interval.
    concentration : float, optional
        Higher values concentrate the samples more tightly around the mean.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    samples : ndarray or float
        Array of n samples (or a scalar if n == 1).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not (low < mean < high):
        raise ValueError(f"Mean must be strictly between low ({low}) and high ({high})")
    if concentration <= 0:
        raise ValueError("concentration must be > 0")

    # Map mean to [0, 1] for Beta parameterization
    m = (mean - low) / (high - low)
    a = m * concentration
    b = (1 - m) * concentration

    samples = rng.beta(a, b, size=n)
    scaled = low + (high - low) * samples
    if n == 1:
        return scaled.item()  # return a scalar, not a 0d array
    return scaled


def select_top_k_indices(values: Float1D, k: int, descending: bool = True) -> Idx1D:
    """
    Return indices of the *k* elements based on the 'descending' flag.
    If descending is True, returns indices of the k largest elements.
    If descending is False, returns indices of the k smallest elements.
    The returned k indices are themselves not sorted.

    Args:
        values: 1-D NumPy array of float values.
        k: The number of top/bottom indices to return.
        descending: If True, find k largest. If False, find k smallest.

    Returns:
        1-D NumPy array of indices.
    """
    if k <= 0:
        return np.array([], dtype=np.intp)

    # Degenerate case: k is too large, return all indices
    if k >= values.shape[-1]:
        vals_for_partition = -values if descending else values
        return np.argpartition(vals_for_partition, kth=0, axis=-1)

    values_to_partition = -values if descending else values
    part = np.argpartition(values_to_partition, kth=k - 1, axis=-1)

    return part[..., :k]
