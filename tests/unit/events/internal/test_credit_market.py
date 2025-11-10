"""
Credit-market events internal implementation unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from bamengine import Rng, make_rng
from bamengine.roles import Borrower, Lender, LoanBook
from bamengine.events._internal.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
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
        net_worth=np.array([10.0, 0.0, 5.0]),  # second firm zero ⇒ CAP_FRAG
        rnd_intensity=np.array([1.0, 0.5, 2.0]),
    )
    firms_calc_credit_metrics(bor)
    expected = np.array([0.5, 4.0, 0.0])
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
        rnd_intensity=np.array([1.0, 2.0]),
    )
    # give it the wrong shape so the allocation branch triggers
    bor.projected_fragility = np.empty(0, dtype=np.float64)

    firms_calc_credit_metrics(bor)

    assert bor.projected_fragility is not None
    np.testing.assert_allclose(
        bor.projected_fragility, np.array([0.5, 6.0]), rtol=1e-12
    )
    # buffer has correct shape & is writable
    assert bor.projected_fragility.shape == bor.net_worth.shape
    assert bor.projected_fragility.flags.writeable


# TODO move to loanbook tests
def test_topk_lowest_rate_partial_sort() -> None:
    vals = np.array([0.09, 0.07, 0.12, 0.08])
    k = 2
    idx = select_top_k_indices_sorted(vals, k=k, descending=False)
    chosen = set(vals[idx])
    assert chosen == {0.07, 0.08} and idx.shape == (k,)


def test_prepare_applications_basic() -> None:
    bor, lend, _, rng, H = _mini_state()
    firms_decide_credit_demand(bor)  # ensure positive demand
    firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)

    active = np.where(bor.credit_demand > 0.0)[0]
    # every demanding firm receives H targets and a valid head pointer
    for f in active:
        assert bor.loan_apps_head[f] >= 0
        row = bor.loan_apps_targets[f]
        assert ((0 <= row) & (row < lend.interest_rate.size)).all()


def test_prepare_applications_single_trial() -> None:
    bor, lend, _, rng, _ = _mini_state(H=1)
    firms_decide_credit_demand(bor)
    firms_prepare_loan_applications(bor, lend, max_H=1, rng=rng)
    assert np.all((bor.loan_apps_head[bor.credit_demand > 0] % 1 == 0))


def test_prepare_applications_no_demand() -> None:
    bor = mock_borrower(n=2, queue_h=2, credit_demand=np.zeros(2))
    lend = mock_lender(n=2, queue_h=2)
    firms_prepare_loan_applications(bor, lend, max_H=2, rng=make_rng(0))
    assert np.all((bor.loan_apps_head == -1))
    assert np.all(bor.loan_apps_targets == -1)


def test_send_one_loan_app_queue_insert() -> None:
    bor, lend, _, rng, H = _mini_state()
    firms_decide_credit_demand(bor)
    firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)
    firms_send_one_loan_app(bor, lend)

    # At least one bank must have a non-empty queue
    assert (lend.recv_loan_apps_head >= 0).any()


def test_borrower_with_empty_list_is_skipped() -> None:
    bor = mock_borrower(n=1, queue_h=2, credit_demand=np.array([10.0]))
    lend = mock_lender(n=1, queue_h=2)
    bor.loan_apps_head[0] = -1  # no targets
    firms_send_one_loan_app(bor, lend)
    assert lend.recv_loan_apps_head[0] == -1  # still empty


def test_send_one_loan_app_exhausted_target() -> None:
    bor = mock_borrower(n=1, queue_h=1, credit_demand=np.array([10.0]))
    lend = mock_lender(n=1, queue_h=1)
    bor.loan_apps_head[0] = 0
    bor.loan_apps_targets[0, 0] = -1  # exhausted sentinel
    firms_send_one_loan_app(bor, lend)
    assert bor.loan_apps_head[0] == -1
    assert lend.recv_loan_apps_head[0] == -1


def _run_basic_loan_cycle(
    bor: Borrower,
    lend: Lender,
    ledger: LoanBook,
    rng: Rng,
    H: int,
) -> NDArray[np.float64]:
    """helper used by many tests"""
    firms_decide_credit_demand(bor)
    orig_demand = np.asarray(bor.credit_demand.copy())  # snapshot for later assertions
    firms_calc_credit_metrics(bor)
    firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)
    for _ in range(H):
        firms_send_one_loan_app(bor, lend)
        banks_provide_loans(bor, ledger, lend, r_bar=0.07, h_phi=0.10)
    return orig_demand


def test_banks_provide_loans_basic() -> None:
    bor, lend, ledger, rng, H = _mini_state()
    orig_demand = _run_basic_loan_cycle(bor, lend, ledger, rng, H)

    # ledger updated and invariants hold
    assert ledger.size > 0
    assert (bor.credit_demand >= -1e-12).all()
    assert (lend.credit_supply >= -1e-12).all()

    # no firm funded above demand
    loans_granted = bor.total_funds - bor.net_worth
    assert np.all(loans_granted <= orig_demand + 1e-9)


def test_banks_provide_loans_demand_exceeds_supply() -> None:
    """
    Give each bank only 1 monetary unit of capacity –
    at least one will deplete while some demand remains unmet.
    """
    bor, lend, ledger, rng, H = _mini_state()
    lend.credit_supply[:] = 1.0  # tiny pot; will run out
    _run_basic_loan_cycle(bor, lend, ledger, rng, H)

    # SOME – not necessarily all – banks must be exhausted
    assert (lend.credit_supply < 1e-9).any()
    # Global supply dropped
    assert lend.credit_supply.sum() < len(lend.credit_supply)
    # Still unmet demand somewhere
    assert (bor.credit_demand > 0).any()


def test_banks_provide_loans_bank_zero_supply() -> None:
    """
    Applications sent to a bank with zero credit_supply must leave the queue
    untouched and never create negative balances.
    """
    H = 2
    bor, lend, ledger, rng, _ = _mini_state(H=H, n_lenders=1)
    lend.credit_supply[0] = 0.0
    _run_basic_loan_cycle(bor, lend, ledger, rng, H)
    assert ledger.size == 0
    assert lend.recv_loan_apps_head[0] == -1
    assert lend.credit_supply[0] == 0.0


def test_banks_provide_loans_skip_invalid_slots() -> None:
    """
    If the queue contains only -1 sentinels banks_provide_loans should do nothing.
    """
    bor, lend, ledger, _, _ = _mini_state()
    k = 0
    lend.recv_loan_apps_head[k] = 2
    lend.recv_loan_apps[k, :3] = -1  # all invalid sentinels
    banks_provide_loans(bor, ledger, lend, r_bar=0.07, h_phi=0.10)
    assert ledger.size == 0
    assert lend.recv_loan_apps_head[k] == -1  # flushed by implementation


def test_ledger_capacity_auto_grows() -> None:
    """
    Start with capacity 1 so the very first append triggers a resize.
    """
    bor, lend, ledger, rng, H = _mini_state(n_borrowers=12, n_lenders=2, H=3)
    ledger.capacity = 1  # intentionally tiny
    _run_basic_loan_cycle(bor, lend, ledger, rng, H)

    # at least two rows appended, and capacity grew beyond the sentinel 1
    assert ledger.size >= 2
    assert ledger.capacity > 1
    assert ledger.capacity >= ledger.size  # internal consistency


def test_firms_fire_workers_gap_closed() -> None:
    # one firm with wage-bill > funds → must fire enough workers
    emp = mock_employer(
        n=1,
        current_labor=np.array([5]),
        wage_offer=np.array([1.0]),
        wage_bill=np.array([5.0]),
        total_funds=np.array([2.5]),  # gap 2.5
    )
    wrk = mock_worker(n=5)
    wrk.employed[:] = True
    wrk.employer[:] = 0

    firms_fire_workers(emp, wrk, rng=make_rng(0))

    assert emp.wage_bill[0] <= emp.total_funds[0] + 1e-12
    assert emp.current_labor[0] == wrk.employed.sum()


def test_firms_fire_workers_zero_needed() -> None:
    """
    Firm has no financing gap (gap <= 0) → no one should be fired.
    """
    emp = mock_employer(
        n=1,
        current_labor=np.array([3]),
        wage_bill=np.array([10.0]),
        total_funds=np.array([10.0]),  # no gap → 0
    )
    wrk = mock_worker(n=3)
    wrk.employed[:] = 1
    wrk.employer[:] = 0
    wrk.wage[:] = np.array([4.0, 3.0, 3.0])

    before_labor = emp.current_labor.copy()
    before_wage_bill = emp.wage_bill.copy()
    before_employed = wrk.employed.copy()
    before_employer = wrk.employer.copy()
    before_wage = wrk.wage.copy()

    firms_fire_workers(emp, wrk, rng=make_rng(42))

    # Nothing should change when there is no financing gap
    np.testing.assert_array_equal(emp.current_labor, before_labor)
    np.testing.assert_array_equal(emp.wage_bill, before_wage_bill)
    np.testing.assert_array_equal(wrk.employed, before_employed)
    np.testing.assert_array_equal(wrk.employer, before_employer)
    np.testing.assert_array_equal(wrk.wage, before_wage)
    assert wrk.fired.sum() == 0


def test_firms_fire_workers_no_workforce() -> None:
    """
    Branch: ``workforce.size == 0``.
    The firm appears to employ labour according to the accounting
    vector, but no matching workers are actually flagged → skip.
    """
    emp = mock_employer(
        n=1,
        current_labor=np.array([2]),  # says 2 workers employed
        wage_offer=np.array([1.0]),
        wage_bill=np.array([4.0]),
        total_funds=np.array([1.0]),  # gap = 3.0 ⇒ needs layoffs
    )
    wrk = mock_worker(n=2)  # but both workers are unemployed

    firms_fire_workers(emp, wrk, rng=make_rng(0))

    assert wrk.fired.sum() == 0


@settings(max_examples=200, deadline=None)
@given(
    n_borrowers=st.integers(4, 12),
    n_lenders=st.integers(2, 6),
    H=st.integers(1, 3),
)
def test_banks_provide_loans_properties(
    n_borrowers: int, n_lenders: int, H: int
) -> None:
    rng = make_rng(999)
    bor = mock_borrower(n=n_borrowers, queue_h=H)
    lend = mock_lender(n=n_lenders, queue_h=H)

    # make sure some demand & supply exist
    bor.wage_bill[:] = rng.uniform(5.0, 20.0, size=n_borrowers)
    bor.net_worth[:] = rng.uniform(0.0, 15.0, size=n_borrowers)
    lend.credit_supply[:] = rng.uniform(1_000.0, 5_000.0, size=n_lenders)

    firms_decide_credit_demand(bor)
    assume((bor.credit_demand > 0).any())

    ledger = mock_loanbook()
    _run_basic_loan_cycle(bor, lend, ledger, rng, H)

    # invariants – no negative balances / supplies, and ledger rows consistent
    assert (bor.credit_demand >= -1e-9).all()
    assert (lend.credit_supply >= -1e-9).all()
    assert ledger.size <= ledger.capacity
    # every ledger row indices within bounds
    assert (ledger.borrower[: ledger.size] < n_borrowers).all()
    assert (ledger.lender[: ledger.size] < n_lenders).all()


def test_full_credit_round() -> None:
    bor, lend, ledger, rng, H = _mini_state()
    banks_decide_credit_supply(lend, v=0.25)
    banks_decide_interest_rate(lend, r_bar=0.07, h_phi=0.05, rng=rng)

    _run_basic_loan_cycle(bor, lend, ledger, rng, H)

    # cross-component invariants
    assert (bor.credit_demand >= -1e-12).all()
    assert (lend.credit_supply >= -1e-12).all()
    assert ledger.size > 0
    # borrower total_funds increased by at least the amount of principal granted
    assert (bor.total_funds >= bor.net_worth).all()


def test_send_one_loan_app_drops_when_bank_has_no_credit() -> None:
    """
    firms_send_one_loan_app: branch where targeted bank has no credit.
    App should be dropped, borrower advances head, queue unchanged.
    """
    bor = mock_borrower(n=1, queue_h=1, credit_demand=np.array([10.0]))
    lend = mock_lender(
        n=1,
        queue_h=1,
        recv_loan_apps=np.full((1, 1), -1, dtype=np.int64),
    )
    lend.credit_supply[0] = 0.0  # no credit at the bank

    # borrower has one target -> bank 0
    bor.loan_apps_head[0] = 0
    bor.loan_apps_targets[0, 0] = 0

    pre_queue = lend.recv_loan_apps.copy()
    firms_send_one_loan_app(bor, lend)

    # Not inserted
    assert np.array_equal(lend.recv_loan_apps, pre_queue)
    assert lend.recv_loan_apps_head[0] == -1
    # Borrower advanced and slot cleared
    assert bor.loan_apps_targets[0, 0] == -1
    assert bor.loan_apps_head[0] == 1  # advanced past its single slot


def test_send_one_loan_app_drops_when_bank_queue_full() -> None:
    """
    firms_send_one_loan_app: branch where bank's application queue is full.
    App should be dropped, head advances, and bank queue is unchanged.
    """
    bor = mock_borrower(n=1, queue_h=1, credit_demand=np.array([10.0]))
    lend = mock_lender(
        n=1,
        queue_h=1,
        recv_loan_apps=np.full((1, 1), -1, dtype=np.int64),
    )
    lend.credit_supply[0] = 100.0  # bank *does* have credit

    # Fill the bank queue so the next insert would overflow
    lend.recv_loan_apps_head[0] = lend.recv_loan_apps.shape[1] - 1  # last slot used
    lend.recv_loan_apps[0, 0] = 999  # sentinel to make sure it doesn't change

    # borrower targets that bank
    bor.loan_apps_head[0] = 0
    bor.loan_apps_targets[0, 0] = 0

    pre_head = lend.recv_loan_apps_head[0]
    pre_queue = lend.recv_loan_apps.copy()

    firms_send_one_loan_app(bor, lend)

    # Bank queue is untouched and head didn't advance (still "full")
    assert lend.recv_loan_apps_head[0] == pre_head
    assert np.array_equal(lend.recv_loan_apps, pre_queue)
    # Borrower advanced and slot cleared
    assert bor.loan_apps_targets[0, 0] == -1
    assert bor.loan_apps_head[0] == 1


def test_firms_fire_workers_random_sufficient_mask_branch() -> None:
    """
    firms_fire_workers (random): cover branch where the cumulative wage
    of the *first few shuffled workers* reaches the gap -> fire minimal number.
    Using equal wages makes the outcome deterministic regardless of shuffle.
    """
    # 5 workers, each wage = 1.0; wage bill 5.0; funds 2.6 => gap = 2.4
    emp = mock_employer(
        n=1,
        current_labor=np.array([5]),
        wage_offer=np.array([1.0]),
        wage_bill=np.array([5.0]),
        total_funds=np.array([2.6]),
    )
    wrk = mock_worker(n=5)
    wrk.employed[:] = 1
    wrk.employer[:] = 0
    wrk.wage[:] = 1.0

    firms_fire_workers(emp, wrk, rng=make_rng(123), method="random")

    # Minimal number that covers 2.4 with 1.0 wages is 3 workers
    assert wrk.fired.sum() == 3
    assert emp.current_labor[0] == 2
    assert emp.wage_bill[0] == pytest.approx(2.0)


def test_send_one_loan_app_exhausted_then_cleanup_next_round() -> None:
    """
    Cover the branch: head >= (i + 1) * stride  → mark borrower done (head = -1).
    We let a borrower consume all their slots; on the *next* call they should be
    detected as exhausted and cleaned up without queue writes.
    """
    # One borrower, two banks, stride=2
    bor = mock_borrower(n=1, queue_h=2, credit_demand=np.array([10.0]))
    lend = mock_lender(n=2, queue_h=2)

    # Give banks capacity and supply so the apps succeed
    lend.credit_supply[:] = 100.0

    # Pre-populate the borrower’s two targets and set head to the start
    bor.loan_apps_targets[0, 0] = 0
    bor.loan_apps_targets[0, 1] = 1
    bor.loan_apps_head[0] = 0  # row=0, col=0

    # First call → consumes col=0, advances head to 1
    firms_send_one_loan_app(bor, lend)
    assert bor.loan_apps_head[0] == 1
    # Second call → consumes col=1, advances head to 2 == (i+1)*stride
    firms_send_one_loan_app(bor, lend)
    assert bor.loan_apps_head[0] == 2

    # Third call → triggers the updated branch and cleans up
    firms_send_one_loan_app(bor, lend)
    assert bor.loan_apps_head[0] == -1

    # Sanity: two apps actually landed (one per bank), no extra writes happened
    assert lend.recv_loan_apps_head[0] >= 0
    assert lend.recv_loan_apps_head[1] >= 0


def test_firms_fire_workers_expensive_picks_top_wages() -> None:
    """
    'expensive' method should fire the highest-wage workers first until gap is covered.
    Wages = [1, 3, 4, 5], gap = 9 → fire 5 and 4 (2 workers), keep 1 and 3.
    """
    # Employer: one firm, 4 workers, wage bill equals sum(wages)
    wages = np.array([1.0, 3.0, 4.0, 5.0])
    emp = mock_employer(
        n=1,
        current_labor=np.array([4]),
        wage_bill=np.array([wages.sum()]),  # 13
        total_funds=np.array([wages.sum() - 9.0]),  # gap = 9
    )
    wrk = mock_worker(n=4)
    wrk.employed[:] = 1
    wrk.employer[:] = 0
    wrk.wage[:] = wages

    firms_fire_workers(emp, wrk, method="expensive", rng=make_rng(0))

    fired_idx = set(np.flatnonzero(wrk.fired))
    # Expect the two most expensive: indices 3 (wage 5) and 2 (wage 4)
    assert fired_idx == {2, 3}
    # Current labor decreased by exactly 2
    assert emp.current_labor[0] == 2
    # Wage bill recomputed from remaining workers: 1 + 3 = 4
    assert emp.wage_bill[0] == pytest.approx(4.0, rel=0, abs=1e-12)
    # Fired workers cleared
    assert np.all(wrk.employed[list(fired_idx)] == 0)
    assert np.all(wrk.employer[list(fired_idx)] == -1)
    assert np.all(wrk.wage[list(fired_idx)] == 0.0)


def test_firms_fire_workers_expensive_fire_all_if_insufficient() -> None:
    """
    If even all workers together can't cover the gap, 'expensive' should fire everyone.
    Wages = [5, 1], gap = 10 → cumsum [5, 6] < 10 ⇒ fire both.
    """
    wages = np.array([5.0, 1.0])
    emp = mock_employer(
        n=1,
        current_labor=np.array([2]),
        wage_bill=np.array([wages.sum()]),  # 6
        total_funds=np.array([wages.sum() - 10.0]),  # gap = 10
    )
    wrk = mock_worker(n=2)
    wrk.employed[:] = 1
    wrk.employer[:] = 0
    wrk.wage[:] = wages

    firms_fire_workers(emp, wrk, method="expensive", rng=make_rng(0))

    # Both fired
    assert wrk.fired.sum() == 2
    assert emp.current_labor[0] == 0
    assert emp.wage_bill[0] == pytest.approx(0.0, rel=0, abs=1e-12)
    assert np.all(wrk.employed == 0)
    assert np.all(wrk.employer == -1)
    assert np.all(wrk.wage == 0.0)


def test_send_one_loan_app_head_negative_after_shuffle_branch() -> None:
    """
    Reach the branch:
        if head < 0:  # Borrower was considered active but head flipped negative
            ... continue
    We simulate an external mutation by providing a custom RNG whose shuffle()
    both shuffles and sets one borrower's head to -1 *after* borrowers_applying
    is constructed but *before* the loop reads head.
    """
    # One borrower with positive demand and a valid head pointer
    bor = mock_borrower(n=1, queue_h=2, credit_demand=np.array([10.0]))
    lend = mock_lender(n=1, queue_h=2)
    # Provide a valid target so the function would otherwise proceed
    bor.loan_apps_head[0] = 0
    bor.loan_apps_targets[0, 0] = 0  # only lender

    class EvilRng:
        @staticmethod
        def shuffle(arr: np.ndarray) -> None:
            # Flip the head of the first borrower in the shuffled list to -1
            # (mutates state between mask/selection and loop body)
            if arr.size > 0:
                idx = int(arr[0])
                bor.loan_apps_head[idx] = -1
            # do a trivial deterministic "shuffle" to satisfy the API
            arr[:] = arr[::-1]

    # noinspection PyTypeChecker
    firms_send_one_loan_app(bor, lend, rng=EvilRng())

    # Since the only borrower got head < 0 right before the loop,
    # nothing should have been queued and head should remain -1.
    assert lend.recv_loan_apps_head[0] == -1
    assert bor.loan_apps_head[0] == -1
    # And the lender's queue should still be all sentinels.
    assert np.all(lend.recv_loan_apps[0] == -1)


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


def test_calc_credit_metrics_general_cap_for_tiny_positive_net_worth() -> None:
    """
    When A is tiny but positive ⇒ B/A very large; function must cap fragility to B, then
    multiply by μ. Here B=1, A=1e-12 ⇒ raw l≈1e12 ⇒ capped to B=1 ⇒ f = μ * 1.
    """
    bor = mock_borrower(
        n=1,
        queue_h=1,
        credit_demand=np.array([1.0]),  # B
        net_worth=np.array([1e-12]),  # tiny positive A
        rnd_intensity=np.array([0.3]),  # μ
    )
    firms_calc_credit_metrics(bor)
    assert bor.projected_fragility is not None
    np.testing.assert_allclose(bor.projected_fragility, np.array([0.3]), atol=1e-12)


def test_prepare_applications_no_lenders_early_exit() -> None:
    """
    No lenders with supply → function should set heads to -1 and return.
    """
    bor = mock_borrower(n=3, queue_h=2)
    bor.credit_demand[:] = np.array([5.0, 0.0, 2.0])  # some borrowers demand
    lend = mock_lender(n=2, queue_h=2)
    lend.credit_supply[:] = 0.0  # no lenders available

    firms_prepare_loan_applications(bor, lend, max_H=2, rng=make_rng(0))

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

    firms_prepare_loan_applications(bor, lend, max_H=H, rng=make_rng(0))

    demanding = np.where(bor.credit_demand > 0)[0]
    for f_id in demanding:
        row = bor.loan_apps_targets[f_id]
        # First two entries must be the lower-rate lender first, then the higher
        assert np.all(row[:2] == np.array([1, 0]))
        # Trailing slot(s) must be sentinels since H_eff < H
        assert np.all(row[2:] == -1)
        # And the head is set to f_id * stride
        assert bor.loan_apps_head[f_id] == f_id * H


def test_banks_provide_loans_sets_contract_rate_formula() -> None:
    """
    Ledger rates must be computed as r_bar * (1 + h_phi * fragility_i).
    Construct a tiny deterministic case with one lender and one borrower.
    """
    # One borrower with demand; set fragility manually to a known value.
    bor = mock_borrower(n=1, queue_h=1, credit_demand=np.array([10.0]))
    bor.projected_fragility[:] = np.array([0.4])  # f_i

    # One lender with ample supply; queue the borrower directly.
    lend = mock_lender(
        n=1, queue_h=1, recv_loan_apps=np.full((1, 1), -1, dtype=np.int64)
    )
    lend.credit_supply[0] = 100.0
    lend.recv_loan_apps_head[0] = 0
    lend.recv_loan_apps[0, 0] = 0  # borrower 0 queued at lender 0

    ledger = mock_loanbook()

    r_bar = 0.07
    h_phi = 0.10
    expected_rate = r_bar * (1.0 + h_phi * bor.projected_fragility[0])

    banks_provide_loans(bor, ledger, lend, r_bar=r_bar, h_phi=h_phi)

    # Exactly one loan row must exist and rate must match the formula.
    assert ledger.size == 1
    np.testing.assert_allclose(ledger.rate[0], expected_rate, atol=1e-12)
