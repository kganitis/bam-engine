"""
Credit-market events internal implementation unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bamengine import Rng, make_rng
from bamengine.events._internal.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    credit_market_round,
    firms_calc_financial_fragility,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
)
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Lender
from bamengine.utils import select_top_k_indices_sorted
from tests.helpers.factories import (
    mock_borrower,
    mock_employer,
    mock_lender,
    mock_loanbook,
    mock_worker,
)

CAP_FRAG = 1.0e-9


def _mini_state(
    *,
    n_borrowers: int = 4,
    n_lenders: int = 2,
    H: int = 2,
    seed: int = 7,
) -> tuple[Borrower, Lender, LoanBook, Rng, int]:
    """
    Build minimal Borrower, Lender, and empty LoanBook components plus an RNG.

    * Borrowers: wage_bill > net_worth for the first 2 borrowers so they demand credit.
    * Lenders:   non-zero credit_supply, distinct interest rates.
    """
    rng = make_rng(seed)

    bor = mock_borrower(
        n=n_borrowers,
        queue_h=H,
        wage_bill=np.concatenate((np.full(2, 15.0), np.full(n_borrowers - 2, 5.0))),
        net_worth=np.full(n_borrowers, 10.0),
    )
    lend = mock_lender(
        n=n_lenders,
        queue_h=H,
        equity_base=np.linspace(8_000, 12_000, n_lenders, dtype=np.float64),
        credit_supply=np.full(n_lenders, 4_000.0),
        interest_rate=np.linspace(0.08, 0.11, n_lenders, dtype=np.float64),
        recv_loan_apps=np.full((n_lenders, n_borrowers), -1, dtype=np.int64),
    )
    # Provide per-bank opex shock (normally set by banks_decide_interest_rate)
    lend.opex_shock = np.full(n_lenders, 0.10)
    ledger = mock_loanbook()

    return bor, lend, ledger, rng, H


def test_decide_credit_supply_basic() -> None:
    lend = mock_lender(
        n=3,
        queue_h=2,
        equity_base=np.array([10.0, 20.0, 5.0]),
    )
    banks_decide_credit_supply(lend, v=0.1)
    np.testing.assert_allclose(lend.credit_supply, lend.equity_base / 0.1)


def test_interest_rate_basic() -> None:
    lend = mock_lender(n=4, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.1, rng=make_rng(0))
    assert (lend.interest_rate >= 0.05 - 1e-12).all()
    assert (lend.interest_rate <= 0.05 * 1.1 + 1e-12).all()


def test_interest_rate_zero_shock() -> None:
    lend = mock_lender(n=2, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.0, rng=make_rng(0))
    assert np.allclose(lend.interest_rate, 0.05, atol=1e-12)


def test_interest_rate_reuses_scratch() -> None:
    lend = mock_lender(n=3, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.1, rng=make_rng(0))
    buf0 = lend.opex_shock
    banks_decide_interest_rate(lend, r_bar=0.06, h_phi=0.1, rng=make_rng(1))
    assert buf0 is lend.opex_shock
    # mypy: lend.opex_shock is Optional[…]; prove it's not None first
    assert lend.opex_shock is not None and lend.opex_shock.flags.writeable


@pytest.mark.parametrize(
    ("wage_bill", "net_worth", "expected_B"),
    [
        ([12.0], [10.0], [2.0]),  # positive demand
        ([5.0], [10.0], [0.0]),  # no demand when net worth covers wages
        ([0.0], [0.0], [0.0]),  # both zero – guard
    ],
)
def test_credit_demand_basic(
    wage_bill: NDArray[np.float64],
    net_worth: NDArray[np.float64],
    expected_B: NDArray[np.float64],
) -> None:
    bor = mock_borrower(
        n=1,
        queue_h=2,
        wage_bill=np.array(wage_bill),
        net_worth=np.array(net_worth),
    )
    firms_decide_credit_demand(bor)
    np.testing.assert_allclose(bor.credit_demand, expected_B)


def test_calc_credit_metrics_fragility() -> None:
    bor = mock_borrower(
        n=3,
        queue_h=1,
        credit_demand=np.array([5.0, 8.0, 0.0]),
        net_worth=np.array([10.0, 1.0, 5.0]),
    )
    firms_calc_financial_fragility(bor)
    expected = np.array([0.5, 8.0, 0.0])
    np.testing.assert_allclose(bor.projected_fragility, expected, rtol=1e-12)


def test_calc_credit_metrics_allocates_buffer() -> None:
    """
    Branch: projected_fragility is None (or wrong shape) → function
    must allocate a fresh buffer and write results in-place.
    """
    bor = mock_borrower(
        n=2,
        queue_h=1,
        credit_demand=np.array([4.0, 6.0]),
        net_worth=np.array([8.0, 2.0]),
    )
    # give it the wrong shape so the allocation branch triggers
    bor.projected_fragility = np.empty(0, dtype=np.float64)

    firms_calc_financial_fragility(bor)

    assert bor.projected_fragility is not None
    # fragility = min(B/A, B): [4/8=0.5, 6/2=3.0]
    np.testing.assert_allclose(
        bor.projected_fragility, np.array([0.5, 3.0]), rtol=1e-12
    )
    # buffer has correct shape & is writable
    assert bor.projected_fragility.shape == bor.net_worth.shape
    assert bor.projected_fragility.flags.writeable


def test_calc_credit_metrics_negative_net_worth() -> None:
    """Firms with NW <= 0 get max_leverage as fragility."""
    bor = mock_borrower(
        n=4,
        queue_h=1,
        credit_demand=np.array([5.0, 8.0, 3.0, 10.0]),
        net_worth=np.array([10.0, 0.0, -5.0, 1.0]),
    )
    max_lev = 15.0
    firms_calc_financial_fragility(bor, max_leverage=max_lev)
    expected = np.array([0.5, max_lev, max_lev, 10.0])
    np.testing.assert_allclose(bor.projected_fragility, expected, rtol=1e-12)


def test_calc_credit_metrics_stale_buffer_cleared() -> None:
    """Second call with NW changing from positive to zero doesn't leak stale values."""
    bor = mock_borrower(
        n=2,
        queue_h=1,
        credit_demand=np.array([4.0, 6.0]),
        net_worth=np.array([8.0, 2.0]),
    )
    max_lev = 10.0
    # First call: both firms have positive NW
    firms_calc_financial_fragility(bor, max_leverage=max_lev)
    np.testing.assert_allclose(
        bor.projected_fragility, np.array([0.5, 3.0]), rtol=1e-12
    )

    # Second call: firm 1's NW drops to zero
    bor.net_worth[:] = np.array([8.0, 0.0])
    firms_calc_financial_fragility(bor, max_leverage=max_lev)
    # Firm 1 should get max_leverage, not the stale 3.0
    np.testing.assert_allclose(
        bor.projected_fragility, np.array([0.5, max_lev]), rtol=1e-12
    )


def test_topk_lowest_rate_partial_sort() -> None:
    vals = np.array([0.09, 0.07, 0.12, 0.08])
    k = 2
    idx = select_top_k_indices_sorted(vals, k=k, descending=False)
    chosen = set(vals[idx])
    assert chosen == {0.07, 0.08} and idx.shape == (k,)


def test_prepare_applications_basic() -> None:
    bor, lend, ledger, rng, H = _mini_state()
    firms_decide_credit_demand(bor)  # ensure positive demand
    firms_prepare_loan_applications(bor, lend, ledger, max_H=H, rng=rng)

    active = np.where(bor.credit_demand > 0.0)[0]
    # every demanding firm receives H targets and a valid head pointer
    for f in active:
        assert bor.loan_apps_head[f] >= 0
        row = bor.loan_apps_targets[f]
        assert ((row >= 0) & (row < lend.interest_rate.size)).all()


def test_prepare_applications_single_trial() -> None:
    bor, lend, ledger, rng, _ = _mini_state(H=1)
    firms_decide_credit_demand(bor)
    firms_prepare_loan_applications(bor, lend, ledger, max_H=1, rng=rng)
    assert np.all(bor.loan_apps_head[bor.credit_demand > 0] % 1 == 0)


def test_prepare_applications_no_demand() -> None:
    bor = mock_borrower(n=2, queue_h=2, credit_demand=np.zeros(2))
    lend = mock_lender(n=2, queue_h=2)
    ledger = mock_loanbook()
    firms_prepare_loan_applications(bor, lend, ledger, max_H=2, rng=make_rng(0))
    assert np.all(bor.loan_apps_head == -1)
    assert np.all(bor.loan_apps_targets == -1)


def test_decide_credit_supply_clips_negative_equity_to_zero() -> None:
    """
    banks_decide_credit_supply must floor credit_supply to 0 for non-positive equity.
    """
    lend = mock_lender(
        n=3,
        queue_h=2,
        equity_base=np.array([10.0, -5.0, 0.0]),
    )
    banks_decide_credit_supply(lend, v=0.2)
    expected = np.array([50.0, 0.0, 0.0])  # 10/0.2, clipped negatives/zeros to 0
    np.testing.assert_allclose(lend.credit_supply, expected, atol=1e-12)


def test_prepare_applications_no_lenders_early_exit() -> None:
    """
    No lenders with supply → function should set heads to -1 and return.
    """
    bor = mock_borrower(n=3, queue_h=2)
    bor.credit_demand[:] = np.array([5.0, 0.0, 2.0])  # some borrowers demand
    lend = mock_lender(n=2, queue_h=2)
    lend.credit_supply[:] = 0.0  # no lenders available
    ledger = mock_loanbook()

    firms_prepare_loan_applications(bor, lend, ledger, max_H=2, rng=make_rng(0))

    # All heads must be -1; targets remain all -1 (initialized by factory)
    assert np.all(bor.loan_apps_head == -1)
    assert np.all(bor.loan_apps_targets == -1)


def test_prepare_applications_Heff_lt_H_and_sorted_by_rate() -> None:
    """
    H_eff < max_H: trailing slots must be -1;
    chosen targets must be sorted by interest rate.
    """
    H = 3
    bor = mock_borrower(n=2, queue_h=H)
    bor.credit_demand[:] = np.array([5.0, 7.0])

    # Only two lenders exist → H_eff = min(H, 2) = 2
    lend = mock_lender(
        n=2,
        queue_h=H,
        # Distinct rates so sort order is observable
        interest_rate=np.array([0.12, 0.05]),
    )
    lend.credit_supply[:] = 100.0
    ledger = mock_loanbook()

    firms_prepare_loan_applications(bor, lend, ledger, max_H=H, rng=make_rng(0))

    demanding = np.where(bor.credit_demand > 0)[0]
    for f_id in demanding:
        row = bor.loan_apps_targets[f_id]
        # First two entries must be the lower-rate lender first, then the higher
        assert np.all(row[:2] == np.array([1, 0]))
        # Trailing slot(s) must be sentinels since H_eff < H
        assert np.all(row[2:] == -1)
        # And the head is set to f_id * stride
        assert bor.loan_apps_head[f_id] == f_id * H


# ============================================================================
# credit_market_round edge cases
# ============================================================================


def test_credit_round_no_active_borrowers() -> None:
    """All borrowers have exhausted application queues (head = -1)."""
    H = 2
    bor = mock_borrower(n=2, queue_h=H, credit_demand=np.array([5.0, 3.0]))
    # Head pointers exhausted — no pending applications
    bor.loan_apps_head[:] = -1
    lend = mock_lender(n=1, queue_h=H, credit_supply=np.array([1000.0]))
    lend.opex_shock = np.array([0.10])
    lb = mock_loanbook()

    credit_market_round(bor, lend, lb, r_bar=0.05)
    assert lb.size == 0


def test_credit_round_no_supply_at_targets() -> None:
    """Borrowers have pending apps, but target banks have zero supply."""
    H = 2
    n_bor = 2
    bor = mock_borrower(n=n_bor, queue_h=H, credit_demand=np.array([5.0, 3.0]))
    # Set up valid head pointers pointing to bank 0
    bor.loan_apps_targets[0, :] = np.array([0, 0])
    bor.loan_apps_targets[1, :] = np.array([0, 0])
    bor.loan_apps_head[:] = np.arange(n_bor) * H  # valid heads

    lend = mock_lender(n=1, queue_h=H, credit_supply=np.array([0.0]))
    lend.opex_shock = np.array([0.10])
    lb = mock_loanbook()

    credit_market_round(bor, lend, lb, r_bar=0.05)
    assert lb.size == 0


def test_credit_round_nw_cap_zeros_all_grants() -> None:
    """Net-worth cap makes all grant amounts zero."""
    H = 2
    n_bor = 2
    bor = mock_borrower(
        n=n_bor,
        queue_h=H,
        credit_demand=np.array([5.0, 3.0]),
        net_worth=np.array([0.0, 0.0]),  # zero NW → nw_cap = 0
        projected_fragility=np.array([1.0, 1.0]),
    )
    bor.loan_apps_targets[0, :] = np.array([0, 0])
    bor.loan_apps_targets[1, :] = np.array([0, 0])
    bor.loan_apps_head[:] = np.arange(n_bor) * H

    lend = mock_lender(n=1, queue_h=H, credit_supply=np.array([1000.0]))
    lend.opex_shock = np.array([0.10])
    lb = mock_loanbook()

    credit_market_round(bor, lend, lb, r_bar=0.05, max_loan_to_net_worth=1.0)
    assert lb.size == 0


# ============================================================================
# firms_fire_workers edge cases
# ============================================================================


def test_fire_workers_no_employed_workers() -> None:
    """Firms have gaps but nobody is employed anywhere."""
    emp = mock_employer(
        n=2,
        wage_bill=np.array([100.0, 50.0]),
        total_funds=np.array([10.0, 5.0]),  # gaps exist
    )
    wrk = mock_worker(n=4)
    # All workers unemployed (default employer=-1)

    firms_fire_workers(emp, wrk, rng=make_rng(0))
    # No crash, no state changes
    assert np.all(wrk.employer == -1)


def test_fire_workers_no_workers_at_gap_firms() -> None:
    """Workers are employed, but only at firms without gaps."""
    emp = mock_employer(
        n=2,
        wage_bill=np.array([100.0, 10.0]),
        total_funds=np.array([10.0, 100.0]),  # firm 0 has gap, firm 1 does not
        current_labor=np.array([0, 2], dtype=np.int64),
    )
    wrk = mock_worker(
        n=3,
        employer=np.array([1, 1, -1], dtype=np.intp),  # workers at firm 1 only
        wage=np.array([5.0, 5.0, 0.0]),
    )

    labor_before = int(emp.current_labor[1])
    firms_fire_workers(emp, wrk, rng=make_rng(0))
    # Workers at firm 1 (no gap) should be unaffected
    assert emp.current_labor[1] == labor_before
    assert np.all(wrk.employer[:2] == 1)
