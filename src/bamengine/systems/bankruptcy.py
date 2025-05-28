# src/bamengine/systems/bankruptcy.py
"""
Event-7  ─  Bankruptcy (mark exits, fire workers, purge loans)
Event-8  ─  Entry (spawn replacements for those exits)

Each event is exposed as an *independent* system so a user can
override / reorder them freely.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bamengine.components import (
    Borrower,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.helpers import sample_beta_with_mean

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_EPS = 1.0e-12


# ───────────────────────── helpers ──────────────────────────
def _trim_mean(x: NDArray[np.float64], p: float = 0.05) -> float:
    """Return the ``p`` % two-sided trimmed mean ( SciPy-style )."""
    if x.size == 0:
        return 0.0
    k = int(round(p * x.size))
    if k == 0:
        return float(x.mean())
    idx = np.argpartition(x, (k, x.size - k - 1))
    core = x[idx[k : x.size - k]]
    return float(core.mean())


# ───────────────────────── Event-7  ─  Bankruptcy ───────────
def firms_update_net_worth(bor: Borrower) -> None:
    """Retained profit is added to equity; syncs the cash column too."""
    retained_profit_sum = bor.retained_profit.sum()
    log.info(
        f"Total retained profits being added to net worth: {retained_profit_sum:,.2f}"
    )

    np.add(bor.net_worth, bor.retained_profit, out=bor.net_worth)
    bor.total_funds[:] = bor.net_worth

    log.debug(
        f"  Detailed Net Worths after update:\n"
        f"{np.array2string(bor.net_worth, precision=2)}"
    )


def mark_bankrupt_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    wrk: Worker,
    lb: LoanBook,
) -> None:
    """
    • Detect A_i < 0.
    • Fire every employee of those firms.
    • Remove their loan rows from the ledger.
    • Record indices in `ec.exiting_firms`.
    """
    bankrupt = np.where(bor.net_worth < -_EPS)[0]
    ec.exiting_firms = bankrupt.astype(np.int64)

    if bankrupt.size == 0:
        log.info("No new firm bankruptcies this period.")
        return

    log.warning(f"--> {bankrupt.size} FIRM(S) HAVE GONE BANKRUPT: {bankrupt}")

    # ---- fire *all* workers whose employer just went bust --------------
    mask = np.isin(wrk.employer, bankrupt)
    if mask.any():
        wrk.employed[mask] = 0
        wrk.employer_prev[mask] = -1
        wrk.employer[mask] = -1
        wrk.wage[mask] = 0.0
        wrk.periods_left[mask] = 0
        wrk.contract_expired[mask] = 0
        wrk.fired[mask] = 0

    emp.current_labor[bankrupt] = 0
    emp.wage_bill[bankrupt] = 0.0

    # ---- purge loans ---------------------------------------------------
    if lb.size:
        bad = np.isin(lb.borrower[: lb.size], bankrupt)
        if bad.any():
            keep = ~bad
            keep_size = int(keep.sum())
            for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
                arr = getattr(lb, name)
                if arr.size == 0:  # column not initialised
                    continue
                arr[:keep_size] = arr[: lb.size][keep]
            lb.size = keep_size


def mark_bankrupt_banks(
    ec: Economy,
    lend: Lender,
    lb: LoanBook,
) -> None:
    """
    • Detect E_k < 0.
    • Purge every loan row that references the bankrupt bank.
    • Record indices in `ec.exiting_banks`.
    """
    bankrupt = np.where(lend.equity_base < -_EPS)[0]
    ec.exiting_banks = bankrupt.astype(np.int64)

    if bankrupt.size == 0:
        log.info("No new bank bankruptcies this period.")
        return

    log.critical(f"!!! {bankrupt.size} BANK(S) HAVE GONE BANKRUPT: {bankrupt} !!!")

    if lb.size:
        bad = np.isin(lb.lender[: lb.size], bankrupt)
        if bad.any():
            keep = ~bad
            keep_size = int(keep.sum())
            for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
                arr = getattr(lb, name)
                if arr.size == 0:
                    continue
                arr[:keep_size] = arr[: lb.size][keep]
            lb.size = keep_size


# ───────────────────────── Event-8  ─  Entry  ─────────────────
def spawn_replacement_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    *,
    rng: Generator,
) -> None:
    """Create one brand-new firm *per* index stored in `ec.exiting_firms`."""
    exiting = ec.exiting_firms
    if exiting.size == 0:
        return

    if exiting.size == bor.net_worth.size:
        log.critical("ALL FIRMS BANKRUPT. Attempting to respawn from defaults.")
        # Fallback to avoid division by zero if all firms fail
        mean_net = 10.0
        mean_prod = 1.0
        mean_wage = ec.min_wage
    else:
        # trimmed means of survivors
        survivors = np.setdiff1d(
            np.arange(bor.net_worth.size), exiting, assume_unique=True
        )
        mean_net = _trim_mean(bor.net_worth[survivors])
        mean_prod = _trim_mean(prod.production[survivors])
        mean_wage = _trim_mean(emp.wage_offer[survivors])
        log.debug(
            f"  New firms initialized based on survivor averages: "
            f"mean_net={mean_net:.2f}, mean_prod={mean_prod:.2f}, "
            f"mean_wage={mean_wage:.2f}"
        )

    for i in exiting:
        s = sample_beta_with_mean(0.5, concentration=6, rng=rng)

        bor.net_worth[i] = mean_net * s
        bor.total_funds[i] = bor.net_worth[i]
        bor.gross_profit[i] = bor.net_profit[i] = 0.0
        bor.retained_profit[i] = 0.0
        bor.credit_demand[i] = 0.0
        bor.projected_fragility[i] = 0.0

        prod.production[i] = mean_prod
        prod.inventory[i] = 0.0
        prod.expected_demand[i] = 0.0
        prod.desired_production[i] = 0.0
        prod.labor_productivity[i] = 1.0
        prod.price[i] = ec.avg_mkt_price

        emp.current_labor[i] = 0
        emp.desired_labor[i] = 0
        emp.wage_offer[i] = mean_wage
        emp.n_vacancies[i] = 0
        emp.total_funds[i] = bor.total_funds[i]
        emp.wage_bill[i] = 0.0

    ec.exiting_firms = np.empty(0, np.int64)  # clear list


def spawn_replacement_banks(
    ec: Economy,
    lend: Lender,
    *,
    rng: Generator,
) -> None:
    """Clone parameters from a random healthy peer for each exiting bank."""
    exiting = ec.exiting_banks
    if exiting.size == 0:
        return

    alive = np.setdiff1d(np.arange(lend.equity_base.size), exiting, assume_unique=True)

    for k in exiting:
        if alive.size:  # clone from peer
            src = int(rng.choice(alive))
            log.debug(f"  Cloning healthy bank {src} to replace bankrupt bank {k}.")
            lend.equity_base[k] = lend.equity_base[src]
            lend.interest_rate[k] = lend.interest_rate[src]
        else:  # fallback
            log.warning(
                "  No surviving banks to clone! Spawning new bank from default values."
            )
            lend.equity_base[k] = rng.poisson(10_000.0) + 10.0
            lend.interest_rate[k] = 0.0

        # Reset state for the new bank
        lend.credit_supply[k] = 0.0
        lend.recv_apps_head[k] = -1
        lend.recv_apps[k, :] = -1

    ec.exiting_banks = np.empty(0, np.int64)  # clear list
