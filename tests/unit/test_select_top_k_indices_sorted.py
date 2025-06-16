from typing import Callable

import numpy as np
import pytest

from bamengine.utils import select_top_k_indices_sorted


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
