"""
Comprehensive unit tests for LoanBook relationship class.

Tests cover:
- Basic initialization and properties
- Capacity management and resizing
- Loan append operations
- Aggregation methods (debt, interest per borrower)
- Purge operations (borrowers and lenders)
- Drop operations with masks
- Backward compatibility aliases
- Edge cases and boundary conditions
"""

from __future__ import annotations

import numpy as np

from bamengine.relationships import LoanBook
from tests.helpers.factories import mock_loanbook

# ============================================================================
# Initialization and Properties
# ============================================================================


def test_loanbook_initialization_empty() -> None:
    """Test empty LoanBook initialization."""
    lb = LoanBook()

    assert lb.size == 0
    assert lb.capacity == 128  # default capacity
    assert lb.source_ids.size == 0
    assert lb.target_ids.size == 0
    assert lb.principal.size == 0
    assert lb.rate.size == 0
    assert lb.interest.size == 0
    assert lb.debt.size == 0


def test_loanbook_backward_compatibility_aliases() -> None:
    """Test borrower/lender aliases for source_ids/target_ids."""
    lb = mock_loanbook(n=10)

    # Initially same object
    assert lb.borrower is lb.source_ids
    assert lb.lender is lb.target_ids

    # Test setter
    new_borrowers = np.array([1, 2, 3], dtype=np.int64)
    new_lenders = np.array([4, 5, 6], dtype=np.int64)

    lb.borrower = new_borrowers
    lb.lender = new_lenders

    np.testing.assert_array_equal(lb.source_ids, new_borrowers)
    np.testing.assert_array_equal(lb.target_ids, new_lenders)


# ============================================================================
# Capacity Management
# ============================================================================


def test_ensure_capacity_no_resize_when_sufficient() -> None:
    """Test that _ensure_capacity doesn't resize when capacity is sufficient."""
    lb = mock_loanbook(n=100)
    initial_cap = lb.capacity

    lb._ensure_capacity(50)  # well below capacity

    assert lb.capacity == initial_cap


def test_ensure_capacity_doubles_when_needed() -> None:
    """Test that _ensure_capacity doubles capacity when needed."""
    lb = mock_loanbook(n=10)
    lb.size = 8
    initial_cap = lb.capacity

    lb._ensure_capacity(5)  # size + 5 = 13 > 10

    assert lb.capacity >= initial_cap * 2
    assert lb.capacity >= 13


def test_ensure_capacity_handles_large_requests() -> None:
    """Test _ensure_capacity with very large extra requests."""
    lb = mock_loanbook(n=10)
    lb.size = 5

    lb._ensure_capacity(1000)  # much larger than double

    assert lb.capacity >= 1005  # size + extra


def test_ensure_capacity_resizes_all_arrays() -> None:
    """Test that all arrays are resized consistently."""
    lb = mock_loanbook(n=10)
    lb.size = 8

    lb._ensure_capacity(100)

    # All arrays should have same size = capacity
    assert lb.source_ids.size == lb.capacity
    assert lb.target_ids.size == lb.capacity
    assert lb.principal.size == lb.capacity
    assert lb.rate.size == lb.capacity
    assert lb.interest.size == lb.capacity
    assert lb.debt.size == lb.capacity


# ============================================================================
# Append Operations
# ============================================================================


def test_append_loans_single_borrower() -> None:
    """Test appending a single loan."""
    lb = LoanBook()

    lender_idx = 0
    borrower_indices = np.array([5], dtype=np.int64)
    amounts = np.array([100.0])
    rates = np.array([0.05])

    lb.append_loans_for_lender(lender_idx, borrower_indices, amounts, rates)

    assert lb.size == 1
    assert lb.source_ids[0] == 5
    assert lb.target_ids[0] == 0
    assert lb.principal[0] == 100.0
    assert lb.rate[0] == 0.05
    assert lb.interest[0] == 5.0  # 100 * 0.05
    assert lb.debt[0] == 105.0  # 100 * (1 + 0.05)


def test_append_loans_multiple_borrowers() -> None:
    """Test appending multiple loans from one lender."""
    lb = LoanBook()

    lender_idx = 2
    borrower_indices = np.array([1, 3, 5], dtype=np.int64)
    amounts = np.array([100.0, 200.0, 150.0])
    rates = np.array([0.05, 0.08, 0.06])

    lb.append_loans_for_lender(lender_idx, borrower_indices, amounts, rates)

    assert lb.size == 3

    # Check all values
    np.testing.assert_array_equal(lb.source_ids[:3], borrower_indices)
    np.testing.assert_array_equal(lb.target_ids[:3], [2, 2, 2])
    np.testing.assert_array_equal(lb.principal[:3], amounts)
    np.testing.assert_array_equal(lb.rate[:3], rates)
    np.testing.assert_allclose(lb.interest[:3], amounts * rates)
    np.testing.assert_allclose(lb.debt[:3], amounts * (1.0 + rates))


def test_append_loans_triggers_resize() -> None:
    """Test that appending triggers resize when capacity exceeded."""
    lb = mock_loanbook(n=5)  # small capacity
    lb.capacity = 5
    lb.size = 0

    # Append 6 loans (exceeds capacity of 5)
    lender_idx = 0
    borrower_indices = np.arange(6, dtype=np.int64)
    amounts = np.full(6, 100.0)
    rates = np.full(6, 0.05)

    lb.append_loans_for_lender(lender_idx, borrower_indices, amounts, rates)

    assert lb.size == 6
    assert lb.capacity >= 6  # should have resized


def test_append_loans_multiple_times() -> None:
    """Test appending loans incrementally."""
    lb = LoanBook()

    # First append
    lb.append_loans_for_lender(
        0,
        np.array([1, 2], dtype=np.int64),
        np.array([100.0, 200.0]),
        np.array([0.05, 0.06]),
    )

    assert lb.size == 2

    # Second append
    lb.append_loans_for_lender(
        1, np.array([3], dtype=np.int64), np.array([150.0]), np.array([0.07])
    )

    assert lb.size == 3

    # Check all values preserved
    np.testing.assert_array_equal(lb.source_ids[:3], [1, 2, 3])
    np.testing.assert_array_equal(lb.target_ids[:3], [0, 0, 1])
    np.testing.assert_allclose(lb.principal[:3], [100.0, 200.0, 150.0])


# ============================================================================
# Aggregation Methods
# ============================================================================


def test_debt_per_borrower_empty() -> None:
    """Test debt_per_borrower with no loans."""
    lb = LoanBook()
    result = lb.debt_per_borrower(n_borrowers=5)

    assert result.size == 5
    np.testing.assert_array_equal(result, np.zeros(5))


def test_debt_per_borrower_single_loan() -> None:
    """Test debt_per_borrower with single loan."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0, np.array([2], dtype=np.int64), np.array([100.0]), np.array([0.05])
    )

    result = lb.debt_per_borrower(n_borrowers=5)

    expected = np.array([0.0, 0.0, 105.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_debt_per_borrower_multiple_loans_same_borrower() -> None:
    """Test debt_per_borrower when borrower has multiple loans."""
    lb = LoanBook()

    # Borrower 1 has two loans from different lenders
    lb.append_loans_for_lender(
        0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([0.05])
    )
    lb.append_loans_for_lender(
        1, np.array([1], dtype=np.int64), np.array([200.0]), np.array([0.06])
    )

    result = lb.debt_per_borrower(n_borrowers=3)

    # Borrower 1 debt = 100*1.05 + 200*1.06 = 105 + 212 = 317
    expected = np.array([0.0, 317.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_interest_per_borrower_empty() -> None:
    """Test interest_per_borrower with no loans."""
    lb = LoanBook()
    result = lb.interest_per_borrower(n_borrowers=5)

    assert result.size == 5
    np.testing.assert_array_equal(result, np.zeros(5))


def test_interest_per_borrower_multiple_loans() -> None:
    """Test interest_per_borrower with multiple loans."""
    lb = LoanBook()

    # Borrower 0: 100 @ 5% = 5 interest
    # Borrower 1: 200 @ 6% = 12 interest
    # Borrower 2: 150 @ 4% = 6 interest
    lb.append_loans_for_lender(
        0,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    result = lb.interest_per_borrower(n_borrowers=3)

    expected = np.array([5.0, 12.0, 6.0])
    np.testing.assert_allclose(result, expected)


def test_principal_per_borrower_empty() -> None:
    """Test principal_per_borrower with no loans."""
    lb = LoanBook()
    result = lb.principal_per_borrower(n_borrowers=5)

    assert result.size == 5
    np.testing.assert_array_equal(result, np.zeros(5))


def test_principal_per_borrower_single_loan() -> None:
    """Test principal_per_borrower with single loan."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0, np.array([2], dtype=np.int64), np.array([100.0]), np.array([0.05])
    )

    result = lb.principal_per_borrower(n_borrowers=5)

    expected = np.array([0.0, 0.0, 100.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_principal_per_borrower_multiple_loans_same_borrower() -> None:
    """Test principal_per_borrower when borrower has multiple loans."""
    lb = LoanBook()

    # Borrower 1 has two loans from different lenders
    lb.append_loans_for_lender(
        0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([0.05])
    )
    lb.append_loans_for_lender(
        1, np.array([1], dtype=np.int64), np.array([200.0]), np.array([0.06])
    )

    result = lb.principal_per_borrower(n_borrowers=3)

    # Borrower 1 principal = 100 + 200 = 300
    expected = np.array([0.0, 300.0, 0.0])
    np.testing.assert_allclose(result, expected)


# ============================================================================
# Purge Operations
# ============================================================================


def test_purge_borrowers_empty_list() -> None:
    """Test purge_borrowers with empty list does nothing."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    removed = lb.purge_borrowers(np.array([], dtype=np.int64))

    assert removed == 0
    assert lb.size == 3


def test_purge_borrowers_single() -> None:
    """Test purging a single borrower."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    removed = lb.purge_borrowers(np.array([2], dtype=np.int64))

    assert removed == 1
    assert lb.size == 2
    # Remaining borrowers should be 1 and 3
    np.testing.assert_array_equal(lb.source_ids[:2], [1, 3])


def test_purge_borrowers_multiple() -> None:
    """Test purging multiple borrowers."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([100.0, 200.0, 150.0, 175.0, 225.0]),
        np.array([0.05, 0.06, 0.04, 0.07, 0.05]),
    )

    removed = lb.purge_borrowers(np.array([2, 4], dtype=np.int64))

    assert removed == 2
    assert lb.size == 3
    # Remaining borrowers should be 1, 3, 5
    np.testing.assert_array_equal(lb.source_ids[:3], [1, 3, 5])


def test_purge_borrowers_nonexistent() -> None:
    """Test purging borrowers that don't exist."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    removed = lb.purge_borrowers(np.array([99], dtype=np.int64))

    assert removed == 0
    assert lb.size == 3


def test_purge_lenders_single() -> None:
    """Test purging a single lender."""
    lb = LoanBook()
    # Lender 0: borrowers [1, 2]
    lb.append_loans_for_lender(
        0,
        np.array([1, 2], dtype=np.int64),
        np.array([100.0, 200.0]),
        np.array([0.05, 0.06]),
    )
    # Lender 1: borrowers [3]
    lb.append_loans_for_lender(
        1, np.array([3], dtype=np.int64), np.array([150.0]), np.array([0.04])
    )

    removed = lb.purge_lenders(np.array([0], dtype=np.int64))

    assert removed == 2
    assert lb.size == 1
    # Remaining loan should be from lender 1
    assert lb.target_ids[0] == 1
    assert lb.source_ids[0] == 3


def test_purge_lenders_multiple() -> None:
    """Test purging multiple lenders."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([0.05])
    )
    lb.append_loans_for_lender(
        1, np.array([2], dtype=np.int64), np.array([200.0]), np.array([0.06])
    )
    lb.append_loans_for_lender(
        2, np.array([3], dtype=np.int64), np.array([150.0]), np.array([0.04])
    )

    removed = lb.purge_lenders(np.array([0, 2], dtype=np.int64))

    assert removed == 2
    assert lb.size == 1
    # Only lender 1 should remain
    assert lb.target_ids[0] == 1


# ============================================================================
# Drop Operations
# ============================================================================


def test_drop_rows_empty_mask() -> None:
    """Test drop_rows with all-False mask does nothing."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    mask = np.array([False, False, False])
    removed = lb.drop_rows(mask)

    assert removed == 0
    assert lb.size == 3


def test_drop_rows_single() -> None:
    """Test dropping a single row."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    mask = np.array([False, True, False])  # Drop middle row
    removed = lb.drop_rows(mask)

    assert removed == 1
    assert lb.size == 2
    # Should keep rows 0 and 2
    np.testing.assert_array_equal(lb.source_ids[:2], [1, 3])
    np.testing.assert_allclose(lb.principal[:2], [100.0, 150.0])


def test_drop_rows_multiple() -> None:
    """Test dropping multiple rows."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([100.0, 200.0, 150.0, 175.0, 225.0]),
        np.array([0.05, 0.06, 0.04, 0.07, 0.05]),
    )

    mask = np.array([True, False, True, False, True])  # Drop rows 0, 2, 4
    removed = lb.drop_rows(mask)

    assert removed == 3
    assert lb.size == 2
    # Should keep rows 1 and 3 (borrowers 2 and 4)
    np.testing.assert_array_equal(lb.source_ids[:2], [2, 4])


def test_drop_rows_all() -> None:
    """Test dropping all rows."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    mask = np.array([True, True, True])  # Drop all
    removed = lb.drop_rows(mask)

    assert removed == 3
    assert lb.size == 0


def test_drop_rows_on_empty_loanbook() -> None:
    """Test drop_rows on empty LoanBook."""
    lb = LoanBook()

    mask = np.array([False, False, False])
    removed = lb.drop_rows(mask)

    assert removed == 0
    assert lb.size == 0


# ============================================================================
# Edge Cases
# ============================================================================


def test_zero_interest_rate() -> None:
    """Test loan with zero interest rate."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([0.0])
    )

    assert lb.interest[0] == 0.0
    assert lb.debt[0] == 100.0  # principal only


def test_very_high_interest_rate() -> None:
    """Test loan with very high interest rate."""
    lb = LoanBook()
    lb.append_loans_for_lender(
        0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([2.0])
    )  # 200% interest

    assert lb.interest[0] == 200.0
    assert lb.debt[0] == 300.0  # 100 * (1 + 2.0)


def test_many_loans_same_borrower_lender_pair() -> None:
    """Test multiple loans between same borrower-lender pair."""
    lb = LoanBook()

    # Same borrower-lender pair, multiple loans
    for _ in range(3):
        lb.append_loans_for_lender(
            0, np.array([1], dtype=np.int64), np.array([100.0]), np.array([0.05])
        )

    assert lb.size == 3
    # All three loans should coexist
    np.testing.assert_array_equal(lb.source_ids[:3], [1, 1, 1])
    np.testing.assert_array_equal(lb.target_ids[:3], [0, 0, 0])

    # Total debt should be 3 * 105 = 315
    result = lb.debt_per_borrower(n_borrowers=2)
    np.testing.assert_allclose(result[1], 315.0)


def test_borrower_indices_out_of_order() -> None:
    """Test that borrower indices don't need to be sequential."""
    lb = LoanBook()

    # Non-sequential borrower IDs
    lb.append_loans_for_lender(
        0,
        np.array([10, 5, 100], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    assert lb.size == 3
    np.testing.assert_array_equal(lb.source_ids[:3], [10, 5, 100])

    # Aggregation should still work correctly
    result = lb.debt_per_borrower(n_borrowers=101)
    assert result[10] > 0
    assert result[5] > 0
    assert result[100] > 0


def test_mixed_purge_and_append() -> None:
    """Test alternating purge and append operations."""
    lb = LoanBook()

    # Initial loans
    lb.append_loans_for_lender(
        0,
        np.array([1, 2, 3], dtype=np.int64),
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
    )

    # Purge borrower 2
    lb.purge_borrowers(np.array([2], dtype=np.int64))
    assert lb.size == 2

    # Add new loans
    lb.append_loans_for_lender(
        1,
        np.array([4, 5], dtype=np.int64),
        np.array([175.0, 225.0]),
        np.array([0.07, 0.05]),
    )

    assert lb.size == 4
    # Should have borrowers 1, 3, 4, 5
    expected_borrowers = [1, 3, 4, 5]
    np.testing.assert_array_equal(lb.source_ids[:4], expected_borrowers)
