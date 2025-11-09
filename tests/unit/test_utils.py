# tests/unit/test_utils.py
from typing import Callable

import numpy as np
import pytest
from bamengine import make_rng

from bamengine.utils import (
    sample_beta_with_mean,
    select_top_k_indices_sorted,
    trim_mean,
    trimmed_weighted_mean,
)


def test_select_top_k_indices_sorted_descending_2d() -> None:
    """
    Test selecting k largest elements from a 2D array,
    ensuring the result is sorted descending.
    """
    vals = np.array([[5.0, 1.0, 3.0, 4.0, 8.0], [10.0, 50.0, 5.0, 80.0, 20.0]])
    k = 3

    # Test descending
    idx_desc = select_top_k_indices_sorted(vals, k=k, descending=True)

    # Expected shape (num_rows, k)
    assert idx_desc.shape == (vals.shape[0], k)

    # Check first row
    selected_vals_row0_desc = vals[0, idx_desc[0]]
    expected_vals_row0_desc = np.array([8.0, 5.0, 4.0])
    assert set(selected_vals_row0_desc) == set(
        expected_vals_row0_desc
    ), "Selected elements are not the k largest for row 0"
    assert np.array_equal(
        selected_vals_row0_desc, expected_vals_row0_desc
    ), "Selected elements are not sorted descending for row 0"

    # Check second row
    selected_vals_row1_desc = vals[1, idx_desc[1]]
    expected_vals_row1_desc = np.array([80.0, 50.0, 20.0])
    assert set(selected_vals_row1_desc) == set(
        expected_vals_row1_desc
    ), "Selected elements are not the k largest for row 1"
    assert np.array_equal(
        selected_vals_row1_desc, expected_vals_row1_desc
    ), "Selected elements are not sorted descending for row 1"


def test_select_top_k_indices_sorted_ascending_1d() -> None:
    """
    Test selecting k smallest elements from a 1D array,
    ensuring the result is sorted ascending.
    """
    vals = np.array([10, 50, 5, 80, 20, 90, 35])
    k = 3

    # Test ascending
    idx_asc = select_top_k_indices_sorted(vals, k=k, descending=False)

    # Expected shape (k,) for 1D input
    assert idx_asc.shape == (k,)

    selected_vals_asc = vals[idx_asc]
    expected_vals_asc = np.array([5, 10, 20])
    assert set(selected_vals_asc) == set(
        expected_vals_asc
    ), "Selected elements are not the k smallest"
    assert np.array_equal(
        selected_vals_asc, expected_vals_asc
    ), "Selected elements are not sorted ascending"


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize(
    "k_ratio, expected_k_func",
    [
        (1.0, lambda n: n),  # k = n
        (1.5, lambda n: n),  # k > n
        (0.0, lambda n: 0),  # k = 0
        (0.5, lambda n: n // 2),  # k < n
    ],
)
def test_select_top_k_indices_sorted_edge_cases(
    descending: bool,
    k_ratio: float,
    expected_k_func: Callable[[int], int],
) -> None:
    """
    Test edge cases for k: k=0, k=n, k>n, and a typical k<n.
    """
    vals = np.array([10, 50, 5, 80, 20, 90, 35, 5, 90])  # Includes duplicates
    n = vals.shape[0]

    if k_ratio == 0.0:  # k=0
        k = 0
    elif k_ratio == 0.5:  # k<n
        k = n // 2
    elif k_ratio == 1.0:  # k=n
        k = n
    else:  # k > n
        k = int(n * k_ratio)

    expected_k = expected_k_func(n)

    idx = select_top_k_indices_sorted(vals, k=k, descending=descending)

    assert idx.shape == (expected_k,)

    selected_vals = vals[idx]

    # Determine expected sorted values
    if expected_k == 0:
        expected_sorted_vals = np.array([])
    else:
        full_sort_indices = np.argsort(vals)
        if descending:
            full_sort_indices = full_sort_indices[::-1]  # Reverse for descending
        expected_sorted_vals = vals[full_sort_indices][:expected_k]

    assert np.array_equal(selected_vals, expected_sorted_vals), (
        f"Failed for k={k}, descending={descending}. "
        f"Got {selected_vals}, expected {expected_sorted_vals}"
    )


def test_select_top_k_indices_sorted_empty_input() -> None:
    """Test with empty input array."""
    vals = np.array([])
    k = 3
    idx = select_top_k_indices_sorted(vals, k=k, descending=True)
    assert idx.shape == (0,)
    idx = select_top_k_indices_sorted(vals, k=k, descending=False)
    assert idx.shape == (0,)
    idx = select_top_k_indices_sorted(vals, k=0, descending=False)  # k=0 with empty
    assert idx.shape == (0,)


def test_select_top_k_indices_sorted_2d_k_equals_n_cols() -> None:
    """Test for 2D array where k equals the number of columns."""
    vals = np.array([[5.0, 1.0, 8.0], [10.0, 50.0, 5.0]])
    k = 3  # k equals number of columns

    idx_desc = select_top_k_indices_sorted(vals, k=k, descending=True)
    assert idx_desc.shape == (2, 3)
    assert np.array_equal(vals[0, idx_desc[0]], np.array([8.0, 5.0, 1.0]))
    assert np.array_equal(vals[1, idx_desc[1]], np.array([50.0, 10.0, 5.0]))

    idx_asc = select_top_k_indices_sorted(vals, k=k, descending=False)
    assert idx_asc.shape == (2, 3)
    assert np.array_equal(vals[0, idx_asc[0]], np.array([1.0, 5.0, 8.0]))
    assert np.array_equal(vals[1, idx_asc[1]], np.array([5.0, 10.0, 50.0]))


# --- NEW TESTS FOR trim_mean ----------------------------------------------------


def test_trim_mean_empty_returns_zero() -> None:
    assert trim_mean(np.array([], dtype=np.float64)) == 0.0


@pytest.mark.parametrize("trim_pct", [0.0, 0.0001])
def test_trim_mean_no_trim_equals_mean(trim_pct: float) -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert trim_mean(vals, trim_pct=trim_pct) == pytest.approx(vals.mean())


def test_trim_mean_symmetric_known_case() -> None:
    # 0..9, trim 20% two-sided => drop 2 lowest and 2 highest => keep 2..7 => mean=4.5
    vals = np.arange(10, dtype=np.float64)
    assert trim_mean(vals, trim_pct=0.2) == pytest.approx(4.5)


# --- NEW TESTS FOR trimmed_weighted_mean ---------------------------------------


def test_twm_unweighted_no_trim_equals_mean() -> None:
    vals = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    assert trimmed_weighted_mean(vals, weights=None, trim_pct=0.0) == pytest.approx(
        vals.mean()
    )


def test_twm_unweighted_with_trim() -> None:
    vals = np.array([0.0, 1.0, 2.0, 100.0], dtype=np.float64)
    # trim 25% => k=1 each side -> keep [1,2] -> mean=1.5
    assert trimmed_weighted_mean(vals, weights=None, trim_pct=0.25) == pytest.approx(
        1.5
    )


def test_twm_weighted_no_trim_matches_np_average() -> None:
    vals = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float64)
    wts = np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float64)
    expected = np.average(vals, weights=wts)
    assert trimmed_weighted_mean(vals, weights=wts, trim_pct=0.0) == pytest.approx(
        expected
    )


def test_twm_weighted_with_trim() -> None:
    vals = np.array([1.0, 2.0, 10.0, 20.0, 100.0], dtype=np.float64)
    wts = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    # sort by value => [1,2,10,20,100];
    # trim 20% => drop first and last => keep [2,10,20]
    expected = np.average(
        np.array([2.0, 10.0, 20.0]), weights=np.array([2.0, 3.0, 4.0])
    )
    assert trimmed_weighted_mean(vals, weights=wts, trim_pct=0.20) == pytest.approx(
        expected
    )


def test_twm_min_weight_filters_all_returns_zero() -> None:
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wts = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    # Set min_weight above all weights -> empty after mask -> 0.0
    assert trimmed_weighted_mean(vals, weights=wts, trim_pct=0.1, min_weight=1.0) == 0.0


def test_twm_zero_weight_sum_after_trim_falls_back_to_unweighted() -> None:
    vals = np.array([1.0, 100.0, 1000.0], dtype=np.float64)
    wts = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Allow zero weights through, then trim 1 each side -> keep [100] with weight 0
    # sum(weights_trimmed)==0 -> fallback to unweighted mean of [100] -> 100
    assert (
        trimmed_weighted_mean(vals, weights=wts, trim_pct=0.25, min_weight=0.0) == 100.0
    )


# --- NEW TESTS FOR sample_beta_with_mean ---------------------------------------


def test_sample_beta_with_mean_invalid_args() -> None:
    with pytest.raises(ValueError):
        sample_beta_with_mean(mean=1.0, n=0)
    with pytest.raises(ValueError):
        sample_beta_with_mean(mean=1.0, concentration=0.0)
    with pytest.raises(ValueError):
        sample_beta_with_mean(mean=1.0, relative_margin=0.0)


def test_sample_beta_with_mean_scalar_vs_array() -> None:
    x = sample_beta_with_mean(mean=10.0, n=1, rng=make_rng(0))
    assert isinstance(x, float)
    xs = sample_beta_with_mean(mean=10.0, n=5, rng=make_rng(0))
    assert isinstance(xs, np.ndarray) and xs.shape == (5,)


def test_sample_beta_with_mean_auto_bounds_contain_samples() -> None:
    rng = make_rng(123)
    # mean near zero triggers tiny symmetric bounds around mean after eps-fix
    xs = sample_beta_with_mean(mean=0.0, n=1000, rng=rng)
    assert np.all(np.isfinite(xs))
    assert xs.min() <= 0.0 <= xs.max()  # centered tightly around mean


def test_sample_beta_with_mean_explicit_bounds_respected() -> None:
    rng = make_rng(42)
    low, high = 8.0, 12.0
    xs = sample_beta_with_mean(mean=10.0, n=1000, low=low, high=high, rng=rng)
    assert np.all(xs >= low) and np.all(xs <= high)


def test_sample_beta_with_mean_mean_match_approximately() -> None:
    rng = make_rng(7)
    mean = 25.0
    xs = sample_beta_with_mean(mean=mean, n=200_000, concentration=200.0, rng=rng)
    # With high concentration and many samples, sample mean ~ desired mean
    assert abs(xs.mean() - mean) < 0.2  # tight tolerance


# ------------------ trimmed_weighted_mean (unweighted branch) ------------------


def test_twm_unweighted_k_zero_branch_returns_mean() -> None:
    """
    When weights is None and k == 0 due to tiny trim_pct, the function should
    return the plain mean (hit: `if k == 0 ... return values.mean()`).
    """
    vals = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    # k = round(0.001 * 4) = 0
    res = trimmed_weighted_mean(vals, weights=None, trim_pct=0.001)
    assert res == pytest.approx(vals.mean())


def test_twm_unweighted_empty_values_hits_size_zero_branch_returns_nan() -> None:
    """
    When weights is None and values.size == 0, the code returns values.mean(),
    which is NaN. Assert behavior (and the expected RuntimeWarning).
    """
    vals = np.array([], dtype=np.float64)
    with pytest.warns(RuntimeWarning):
        res = trimmed_weighted_mean(vals, weights=None, trim_pct=0.25)
    assert np.isnan(res)


def test_twm_unweighted_trimmed_empty_returns_zero() -> None:
    """
    When trimming removes all elements (trimmed.size == 0), the function returns 0.0.
    Example: len=2, trim_pct=0.6 -> k=1, slice [1:1] empty.
    """
    vals = np.array([10.0, 20.0], dtype=np.float64)
    res = trimmed_weighted_mean(vals, weights=None, trim_pct=0.6)
    assert res == 0.0


# ------------------ select_top_k_indices_sorted special inputs ------------------


# noinspection PyTypeChecker
def test_select_top_k_indices_sorted_accepts_python_list() -> None:
    """
    Hit the `if not isinstance(values, np.ndarray)` path by passing a list.
    Ensure it selects correctly in descending order.
    """
    vals_list = [3.0, 1.0, 4.0, 2.0]
    idx = select_top_k_indices_sorted(vals_list, k=2, descending=True)
    sel = np.array(vals_list)[idx]
    np.testing.assert_array_equal(sel, np.array([4.0, 3.0]))


def test_select_top_k_indices_sorted_scalar_ndarray() -> None:
    """
    Hit the `if values.ndim == 0` path by passing a 0-D ndarray.
    Expect shape (1,) with the only index 0, and selected value equal to the scalar.
    """
    scalar = np.array(5.0)  # 0-D
    idx = select_top_k_indices_sorted(scalar, k=1, descending=True)
    assert idx.shape == (1,)
    sel = np.atleast_1d(scalar)[idx]
    np.testing.assert_array_equal(sel, np.array([5.0]))
