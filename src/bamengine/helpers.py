# src/bamengine/helpers.py
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


def sample_beta_with_mean(
    mean: float,
    n: int = 1,
    low: float = 0.1,
    high: float = 1.0,
    concentration: float = 12,
    rng: Generator = default_rng(),
) -> float | NDArray[np.float64]:
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
