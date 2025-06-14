# test/unit/systems/test_bankruptcy.py
from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from bamengine.systems.bankruptcy import (
    firms_update_net_worth,
    mark_bankrupt_banks,
    mark_bankrupt_firms,
    spawn_replacement_banks,
    spawn_replacement_firms,
)
from tests.helpers.factories import (
    mock_borrower,
    mock_economy,
    mock_employer,
    mock_lender,
    mock_loanbook,
    mock_producer,
    mock_worker,
)


def test_firms_update_net_worth_syncs_cash() -> None:
    bor = mock_borrower(
        n=3,
        net_worth=np.array([10.0, 5.0, 0.0]),
        total_funds=np.array([10.0, 5.0, 0.0]),
        retained_profit=np.array([+2.0, -1.0, 0.0]),
    )
    firms_update_net_worth(bor)

    np.testing.assert_allclose(bor.net_worth, [12.0, 4.0, 0.0])
    np.testing.assert_allclose(bor.total_funds, bor.net_worth)


def test_mark_bankrupt_firms_fires_workers_and_purges_loans() -> None:
    # minimal 3-firm economy
    prod = mock_producer(3)
    emp = mock_employer(3)
    bor = mock_borrower(3, net_worth=np.array([5.0, -2.0, 8.0]))
    wrk = mock_worker(5)

    # five workers – 3 of them employed by firm-1 which will go bust
    wrk.employed[:] = [1, 1, 1, 0, 0]
    wrk.employer[:] = [1, 1, 1, -1, -1]
    emp.current_labor[:] = [0, 3, 0]
    emp.wage_bill[:] = [0.0, 6.0, 0.0]

    # LoanBook: rows 0,1 → firm-1 ; rows 2,3 → other firms
    lb = mock_loanbook(capacity=4, size=4)
    lb.borrower[:4] = [1, 1, 0, 2]
    lb.lender[:4] = [0, 1, 0, 1]
    lb.debt[:4] = [7.0, 3.0, 4.0, 2.0]

    ec = mock_economy()

    mark_bankrupt_firms(ec, emp, bor, prod, wrk, lb)

    # exiting list
    assert np.array_equal(ec.exiting_firms, [1])

    # all workers of firm-1 flags set
    assert np.all((wrk.employed[[0, 1, 2]] == 0))
    assert np.all((wrk.fired[[0, 1, 2]] == 0))
    assert np.all((wrk.employer[[0, 1, 2]] == -1))

    # employer books wiped
    assert emp.current_labor[1] == 0
    assert emp.wage_bill[1] == 0.0

    # LoanBook compacted → rows with borrower==1 gone
    assert lb.size == 2
    assert not np.isin(lb.borrower[: lb.size], 1).any()


def test_mark_bankrupt_firms_exits_zero_production() -> None:
    """Positive equity but zero output ⇒ exit (ghost-firm rule)."""
    prod = mock_producer(2, production=np.array([0.0, 3.0]))  # firm-0 is a ghost
    emp = mock_employer(2)
    bor = mock_borrower(2, net_worth=np.array([10.0, 4.0]))
    wrk = mock_worker(0)

    ec = mock_economy()
    mark_bankrupt_firms(ec, emp, bor, prod, wrk, mock_loanbook(size=0))

    assert np.array_equal(ec.exiting_firms, np.array([0]))


def test_mark_bankrupt_firms_noop_when_all_viable() -> None:
    """All firms have positive equity **and** production."""
    prod = mock_producer(2, production=np.array([4.0, 3.0]))
    bor = mock_borrower(2, net_worth=np.array([3.0, 1.0]))
    ec = mock_economy()
    before = bor.net_worth.copy()

    mark_bankrupt_firms(
        ec,
        mock_employer(2),
        bor,
        prod,
        mock_worker(0),
        mock_loanbook(size=0),
    )

    assert ec.exiting_firms.size == 0
    np.testing.assert_array_equal(bor.net_worth, before)


def test_mark_bankrupt_banks_purges_loans() -> None:
    ec = mock_economy()
    lend = mock_lender(2, equity_base=np.array([10_000.0, -5.0]))
    lb = mock_loanbook(capacity=3, size=3)
    lb.lender[:3] = [1, 0, 1]  # rows 0 & 2 tied to bankrupt bank-1
    lb.debt[:3] = [2.0, 3.0, 4.0]

    mark_bankrupt_banks(ec, lend, lb)

    assert np.array_equal(ec.exiting_banks, [1])
    assert lb.size == 1 and lb.lender[0] == 0


def test_spawn_replacement_firms_restores_positive_equity() -> None:
    rng = default_rng(42)

    ec = mock_economy()
    ec.exiting_firms = np.array([0, 2], dtype=np.int64)

    prod = mock_producer(3, production=np.array([0.0, 10.0, 0.0]))
    emp = mock_employer(3, wage_offer=np.array([1.2, 1.0, 1.5]))
    bor = mock_borrower(
        3,
        net_worth=np.array([-5.0, 8.0, -1.0]),
        total_funds=np.array([-5.0, 8.0, -1.0]),
    )

    spawn_replacement_firms(ec, prod, emp, bor, rng=rng)

    # exiting list cleared
    assert ec.exiting_firms.size == 0

    # bankrupt slots resurrected with positive equity & cash=equity
    assert (bor.net_worth >= 0).all()
    np.testing.assert_allclose(bor.net_worth, bor.total_funds)

    # labour reset to zero -> for *replaced* firms: inventory zeroed out
    assert emp.current_labor[[0, 2]].sum() == 0
    assert np.array_equal(prod.inventory[[0, 2]], [0, 0])


def test_spawn_replacement_banks_clone_and_fallback() -> None:
    rng = default_rng(7)

    # case-A: clone from healthy peer
    ec = mock_economy()
    ec.exiting_banks = np.array([1], dtype=np.int64)
    lend = mock_lender(3, equity_base=np.array([20_000.0, -1.0, 30_000.0]))
    spawn_replacement_banks(ec, lend, rng=rng)

    assert ec.exiting_banks.size == 0
    assert lend.equity_base[1] > 0.0 and lend.credit_supply[1] == 0.0

    # case-B: all banks bankrupt → fallback path
    ec.exiting_banks = np.array([0, 1, 2], dtype=np.int64)
    lend.equity_base[:] = -1.0
    spawn_replacement_banks(ec, lend, rng=rng)

    assert ec.destroyed
