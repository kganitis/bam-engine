# tests/unit/systems/test_credit_market.py
"""
Credit-market systems unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from bamengine.components import Borrower, Lender, LoanBook
from bamengine.helpers import select_top_k_indices_sorted

# noinspection PyProtectedMember
from bamengine.systems.credit_market import (  # systems under test
    CAP_FRAG,
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
from tests.helpers.factories import (
    mock_borrower,
    mock_employer,
    mock_lender,
    mock_loanbook,
    mock_worker,
)

# --------------------------------------------------------------------------- #
#  deterministic micro-scenario helper                                        #
# --------------------------------------------------------------------------- #


def _mini_state(
    *,
    n_borrowers: int = 4,
    n_lenders: int = 2,
    H: int = 2,
    seed: int = 7,
) -> tuple[Borrower, Lender, LoanBook, Generator, int]:
    """
    Build minimal Borrower, Lender, and empty LoanBook components plus an RNG.

    * Borrowers: wage_bill > net_worth for the first 2 borrowers so they demand credit.
    * Lenders:   non-zero credit_supply, distinct interest rates.
    """
    rng = default_rng(seed)

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


# --------------------------------------------------------------------------- #
#  banks_decide_credit_supply                                                 #
# --------------------------------------------------------------------------- #


def test_decide_credit_supply_basic() -> None:
    lend = mock_lender(
        n=3,
        queue_h=2,
        equity_base=np.array([10.0, 20.0, 5.0]),
    )
    banks_decide_credit_supply(lend, v=0.1)
    np.testing.assert_allclose(lend.credit_supply, lend.equity_base / 0.1)


# --------------------------------------------------------------------------- #
#  banks_decide_interest_rate                                                 #
# --------------------------------------------------------------------------- #


def test_interest_rate_basic() -> None:
    lend = mock_lender(n=4, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.1, rng=default_rng(0))
    assert (lend.interest_rate >= 0.05 - 1e-12).all()
    assert (lend.interest_rate <= 0.05 * 1.1 + 1e-12).all()


def test_interest_rate_zero_shock() -> None:
    lend = mock_lender(n=2, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.0, rng=default_rng(0))
    assert np.allclose(lend.interest_rate, 0.05, atol=1e-12)


def test_interest_rate_reuses_scratch() -> None:
    lend = mock_lender(n=3, queue_h=2)
    banks_decide_interest_rate(lend, r_bar=0.05, h_phi=0.1, rng=default_rng(0))
    buf0 = lend.opex_shock
    banks_decide_interest_rate(lend, r_bar=0.06, h_phi=0.1, rng=default_rng(1))
    assert buf0 is lend.opex_shock
    # mypy: lend.opex_shock is Optional[…]; prove it's not None first
    assert lend.opex_shock is not None and lend.opex_shock.flags.writeable


# --------------------------------------------------------------------------- #
#  firms_decide_credit_demand                    #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
#  firms_calc_credit_metrics                                                  #
# --------------------------------------------------------------------------- #


def test_calc_credit_metrics_fragility() -> None:
    bor = mock_borrower(
        n=3,
        queue_h=1,
        credit_demand=np.array([5.0, 8.0, 0.0]),
        net_worth=np.array([10.0, 0.0, 5.0]),  # second firm zero ⇒ CAP_FRAG
        rnd_intensity=np.array([1.0, 0.5, 2.0]),
    )
    firms_calc_credit_metrics(bor)
    expected = np.array([0.5, CAP_FRAG * 0.5, 0.0])
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


# --------------------------------------------------------------------------- #
#  _topk_lowest_rate helper                                                   #
# --------------------------------------------------------------------------- #


# TODO move to loanbook tests
def test_topk_lowest_rate_partial_sort() -> None:
    vals = np.array([0.09, 0.07, 0.12, 0.08])
    k = 2
    idx = select_top_k_indices_sorted(vals, k=k, descending=False)
    chosen = set(vals[idx])
    assert chosen == {0.07, 0.08} and idx.shape == (k,)


# --------------------------------------------------------------------------- #
#  firms_prepare_loan_applications                                            #
# --------------------------------------------------------------------------- #


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
    firms_prepare_loan_applications(bor, lend, max_H=2, rng=default_rng(0))
    assert np.all((bor.loan_apps_head == -1))
    assert np.all(bor.loan_apps_targets == -1)


# --------------------------------------------------------------------------- #
#  firms_send_one_loan_app                                                    #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
#  banks_provide_loans                                                        #
# --------------------------------------------------------------------------- #


def _run_basic_loan_cycle(
    bor: Borrower,
    lend: Lender,
    ledger: LoanBook,
    rng: Generator,
    H: int,
) -> NDArray[np.float64]:
    """helper used by many tests"""
    firms_decide_credit_demand(bor)
    orig_demand = np.asarray(bor.credit_demand.copy())  # snapshot for later assertions
    firms_calc_credit_metrics(bor)
    firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)
    for _ in range(H):
        firms_send_one_loan_app(bor, lend)
        banks_provide_loans(bor, ledger, lend)
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
    banks_provide_loans(bor, ledger, lend)
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


# --------------------------------------------------------------------------- #
#  firms_fire_workers                               #
# --------------------------------------------------------------------------- #


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

    firms_fire_workers(emp, wrk, rng=default_rng(0))

    assert emp.wage_bill[0] <= emp.total_funds[0] + 1e-12
    assert emp.current_labor[0] == wrk.employed.sum()


def test_firms_fire_workers_zero_needed() -> None:
    """
    Branch: ``n_fire == 0``.
    Make the wage gap > 0 but set ``wage_offer = inf`` so
    gap / wage_offer → 0 → ceil(0) → 0.
    """
    emp = mock_employer(
        n=1,
        current_labor=np.array([3]),
        wage_offer=np.array([np.inf]),
        wage_bill=np.array([10.0]),  # gap = 10
        total_funds=np.array([0.0]),
    )
    wrk = mock_worker(n=3)
    wrk.employed[:] = True
    wrk.employer[:] = 0

    before = emp.current_labor.copy()
    firms_fire_workers(emp, wrk, rng=default_rng(42))
    # nothing should change – the "float quirks" early-exit hit
    np.testing.assert_array_equal(emp.current_labor, before)
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

    firms_fire_workers(emp, wrk, rng=default_rng(0))

    assert wrk.fired.sum() == 0


# --------------------------------------------------------------------------- #
#  Property-based invariant: banks_provide_loans                              #
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(
    n_borrowers=st.integers(4, 12),
    n_lenders=st.integers(2, 6),
    H=st.integers(1, 3),
)
def test_banks_provide_loans_properties(
    n_borrowers: int, n_lenders: int, H: int
) -> None:
    rng = default_rng(999)
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


# --------------------------------------------------------------------------- #
#  End-to-end micro integration of one credit-market event                    #
# --------------------------------------------------------------------------- #


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
