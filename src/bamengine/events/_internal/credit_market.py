"""
System functions for credit market phase events.

This module contains the internal implementation functions for credit market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.credit_market : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Employer, Lender, Worker
from bamengine.utils import (
    EPS,
    _flatten_and_shuffle_groups,
    grouped_cumsum,
    select_top_k_indices_sorted,
)

log = logging.getLogger(__name__)


def banks_decide_credit_supply(lend: Lender, *, v: float) -> None:
    """
    Calculate bank credit supply from equity base.

    See Also
    --------
    bamengine.events.credit_market.BanksDecideCreditSupply : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Banks Deciding Credit Supply ---")
        log.info(
            f"  Inputs: Capital Requirement (v)={v:.3f} (Max Leverage={1 / v:.2f}x)"
        )

    # Core Rule
    np.divide(lend.equity_base, v, out=lend.credit_supply)
    np.maximum(lend.credit_supply, 0.0, out=lend.credit_supply)

    # Logging
    if info_enabled:
        total_supply = lend.credit_supply.sum()
        log.info(f"  Total credit supply in the economy: {total_supply:,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Equity Base (E) (first 10 borrowers): "
            f"{np.array2string(lend.equity_base[:10], precision=2)}"
        )
        log.debug(
            f"  Credit Supply (C = E / v) (first 10 borrowers): "
            f"{np.array2string(lend.credit_supply[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Credit Supply Decision complete ---")


def banks_decide_interest_rate(
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    rng: Rng = make_rng(),
) -> None:
    """
    Set bank interest rates with random markup over base rate.

    See Also
    --------
    bamengine.events.credit_market.BanksDecideInterestRate : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Banks Deciding Interest Rate ---")
        log.info(
            f"  Inputs: Base Rate (r_bar)={r_bar:.4f}  |"
            f"  Max Markup Shock (h_phi)={h_phi:.4f}"
        )
    shape = lend.interest_rate.shape

    # Permanent scratch buffer
    shock = lend.opex_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        lend.opex_shock = shock

    # Core Rule
    shock[:] = rng.uniform(0.0, h_phi, size=shape)
    lend.interest_rate[:] = r_bar * (1.0 + shock)

    # Logging
    if info_enabled:
        avg_rate = lend.interest_rate.mean() * 100
        log.info(f"  Interest rates set. Average rate: {avg_rate:.3f}%")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Generated shocks (first 10 borrowers): "
            f"{np.array2string(shock[:10], precision=4)}"
        )
        log.debug(
            f"  Interest Rates (%) (first 10 borrowers): "
            f"{np.array2string(lend.interest_rate[:10] * 100, precision=4)}"
        )
    if info_enabled:
        log.info("--- Interest Rate Decision complete ---")


def firms_decide_credit_demand(bor: Borrower) -> None:
    """
    Calculate firm credit demand from wage bill and total funds gap.

    See Also
    --------
    bamengine.events.credit_market.FirmsDecideCreditDemand : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Deciding Credit Demand ---")
        log.info(
            f"  Inputs: Total Wage Bill={bor.wage_bill.sum():,.2f}  |"
            f"  Total Net Worth={bor.total_funds.sum():,.2f}"
        )

    # Core Rule
    np.subtract(bor.wage_bill, bor.total_funds, out=bor.credit_demand)
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)

    # Logging
    if info_enabled:
        total_demand = bor.credit_demand.sum()
        num_borrowers = np.sum(bor.credit_demand > 0)
        log.info(
            f"  {num_borrowers} borrowers demand credit, for a total of {total_demand:,.2f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Credit Demand per borrower (B = max(0, W-A)) (first 10 borrowers): "
            f"{np.array2string(bor.credit_demand[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Credit Demand Decision complete ---")


def firms_calc_financial_fragility(
    bor: Borrower, *, max_leverage: float = 10.0
) -> None:
    """
    Calculate firm projected financial fragility metric for loan applications.

    See Also
    --------
    bamengine.events.credit_market.FirmsCalcFinancialFragility : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Calculating Credit Metrics ---")
    shape = bor.net_worth.shape

    # Permanent scratch buffer (may be None on first call)
    frag = bor.projected_fragility
    if frag is None or frag.shape != shape:  # type: ignore[redundant-expr]
        frag = np.empty(shape, dtype=np.float64)
        bor.projected_fragility = frag

    # Pre-fill with max_leverage so firms with NW <= 0 get a deterministic,
    # economically meaningful default (worst credit priority).
    frag[:] = max_leverage

    # Core Rule
    np.divide(bor.credit_demand, bor.net_worth, out=frag, where=bor.net_worth > 0.0)

    # Logging
    if info_enabled:
        valid_frag = frag[np.isfinite(frag)]
        avg_fragility = valid_frag.mean() if valid_frag.size > 0 else 0.0
        log.info(
            f"  Average projected fragility across all borrowers: {avg_fragility:.4f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Projected Fragility per borrower (first 10 borrowers): "
            f"{np.array2string(frag[:10], precision=3)}"
        )
    if info_enabled:
        log.info("--- Credit Metrics Calculation complete ---")


def firms_prepare_loan_applications(
    bor: Borrower,
    lend: Lender,
    lb: LoanBook,
    *,
    max_H: int,
    rng: Rng = make_rng(),
) -> None:
    """
    Firms build loan application queue sorted by interest rate.

    Clears any stale loans before fresh credit matching begins (safety
    guard). The loan book should normally be empty at this point because
    all loans are resolved in the revenue phase, but this protects
    against edge cases.

    See Also
    --------
    bamengine.events.credit_market.FirmsPrepareLoanApplications : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Preparing Loan Applications ---")

    # Clear all settled loans before fresh credit matching begins.
    # Loan records are intentionally retained through the planning and labor
    # phases (after financial settlement in the revenue phase) so that
    # planning-phase breakeven calculations can access previous-period
    # interest obligations via lb.interest_per_borrower().
    if lb.size > 0:
        if info_enabled:
            log.info(f"  Clearing {lb.size} settled loans from previous period.")
        lb.size = 0

    lenders = np.where(lend.credit_supply > 0)[0]
    borrowers = np.where(bor.credit_demand > 0.0)[0]

    if info_enabled:
        log.info(
            f"  {borrowers.size} borrowers are seeking loans "
            f"from {lenders.size} available lenders (max apps per borrower, H={max_H})."
        )

    if borrowers.size == 0 or lenders.size == 0:
        if info_enabled:
            log.info(
                "  No borrowers or no available lenders. "
                "Skipping loan application preparation."
            )
        bor.loan_apps_head.fill(-1)
        if info_enabled:
            log.info("--- Loan Application Preparation complete ---")
        return

    # Sample H random lending banks per borrower (vectorized)
    H_eff = min(max_H, lenders.size)
    if info_enabled:
        log.info(f"  Effective applications per borrower (H_eff): {H_eff}")
    # Generate one random score per (borrower, lender) pair, then pick the H_eff
    # smallest per row via argpartition — equivalent to sampling without replacement
    if H_eff == lenders.size:
        # All lenders selected — no need to sample
        sample = np.broadcast_to(lenders, (borrowers.size, lenders.size)).copy()
    else:
        rand_scores = rng.random((borrowers.size, lenders.size))
        top_indices = np.argpartition(rand_scores, H_eff, axis=1)[:, :H_eff]
        sample = lenders[top_indices]
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Initial random bank sample (first 10 borrowers):\n{sample[:10]}")

    # Sort applications by ascending interest rate
    topk = select_top_k_indices_sorted(
        lend.interest_rate[sample], k=H_eff, descending=False
    )
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted bank sample by interest rate (first 5 borrowers):\n"
            f"{sorted_sample[:5]}"
        )

    # Write buffers
    log.debug("  Writing application targets and head pointers for all borrowers...")
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)
    stride = max_H

    # Vectorized buffer write — fancy indexing replaces per-borrower loop
    bor.loan_apps_targets[borrowers, :H_eff] = sorted_sample
    bor.loan_apps_head[borrowers] = borrowers * stride

    if log.isEnabledFor(logging.TRACE):
        for i in range(min(10, borrowers.size)):
            f_id = borrowers[i]
            log.trace(
                f"    Borrower {f_id}: targets={bor.loan_apps_targets[f_id]}, "
                f"head_ptr={bor.loan_apps_head[f_id]}"
            )

    if info_enabled:
        log.info("--- Loan Application Preparation complete ---")


def credit_market_round(
    bor: Borrower,
    lend: Lender,
    lb: LoanBook,
    *,
    r_bar: float,
    max_leverage: float = 0.0,
    max_loan_to_net_worth: float = 0.0,
    rng: Rng = make_rng(),
) -> None:
    """One vectorized round of credit market matching.

    All borrowers simultaneously send their next loan application,
    applicants are grouped by bank, ranked by fragility (ascending),
    and loans are provisioned using grouped cumsum to track supply
    exhaustion per bank.

    Parameters
    ----------
    bor : Borrower
        Borrower role (firms).
    lend : Lender
        Lender role (banks).
    lb : LoanBook
        Loan relationship ledger.
    r_bar : float
        Baseline policy rate.
    max_leverage : float
        Cap on fragility for interest rate calculation.
    max_loan_to_net_worth : float
        Maximum loan-to-net-worth ratio (0 = no cap).
    rng : Rng
        Random generator for tie-breaking.
    """
    assert lend.opex_shock is not None, (
        "lend.opex_shock must be set before credit_market_round() — "
        "run banks_decide_interest_rate() first"
    )
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Credit Market Round ---")

    stride = bor.loan_apps_targets.shape[1]  # max_H
    n_banks = lend.credit_supply.size

    # 1. Find active borrowers with credit demand > 0 and pending apps
    borrowers = np.where(bor.credit_demand > 0.0)[0]
    if borrowers.size == 0:
        if info_enabled:
            log.info("  No borrowers with credit demand. Skipping round.")
            log.info("--- Credit Market Round complete ---")
        return

    active_mask = bor.loan_apps_head[borrowers] >= 0
    active = borrowers[active_mask]

    if active.size == 0:
        if info_enabled:
            log.info("  No borrowers with pending applications. Skipping round.")
            log.info("--- Credit Market Round complete ---")
        return

    # 2. Decode targets from head pointers
    heads = bor.loan_apps_head[active]
    rows, cols = np.divmod(heads, stride)
    target_banks = bor.loan_apps_targets[rows, cols]

    # 3. Filter: valid targets (not -1) with credit supply
    valid_mask = target_banks >= 0
    valid_with_supply = valid_mask.copy()
    valid_with_supply[valid_mask] &= lend.credit_supply[target_banks[valid_mask]] > EPS

    # 4. Advance ALL head pointers
    new_heads = heads + 1
    exhausted = new_heads >= (active + 1) * stride
    new_heads[exhausted | (~valid_mask)] = -1
    bor.loan_apps_head[active] = new_heads

    # 5. Get valid senders
    valid_firms = active[valid_with_supply]
    valid_banks = target_banks[valid_with_supply]

    if valid_firms.size == 0:
        if info_enabled:
            log.info("  No valid applications this round.")
            log.info("--- Credit Market Round complete ---")
        return

    # 6. Group by target bank, sort within groups by fragility (ascending)
    #    np.lexsort sorts by last key first, so (fragility, bank) sorts by
    #    bank first, then by fragility within each bank.
    frag = bor.projected_fragility[valid_firms]
    sort_order = np.lexsort((frag, valid_banks))
    sorted_firms = valid_firms[sort_order]
    sorted_banks = valid_banks[sort_order]
    sorted_frag = frag[sort_order]

    # 7. Compute loan amounts per applicant
    max_grant = bor.credit_demand[sorted_firms].copy()

    # Apply loan-to-net-worth cap if configured
    if max_loan_to_net_worth > 0.0:
        nw_cap = bor.net_worth[sorted_firms] * max_loan_to_net_worth
        max_grant = np.minimum(max_grant, nw_cap)

    # 8. Grouped cumsum to track supply exhaustion per bank
    #    Find group boundaries (where bank ID changes)
    bank_change = np.empty(sorted_banks.size, dtype=np.bool_)
    bank_change[0] = True
    bank_change[1:] = sorted_banks[1:] != sorted_banks[:-1]
    group_starts = np.where(bank_change)[0]

    # Per-bank supply lookup for each applicant's bank
    per_applicant_supply = lend.credit_supply[sorted_banks]

    # Grouped cumsum of max_grant per bank
    cum_demand = grouped_cumsum(max_grant, group_starts)

    # Mask where cumulative demand exceeds supply
    exceeds = cum_demand > per_applicant_supply
    # For the boundary applicant: partial grant
    # Amount available = supply - (cum_demand - max_grant) = supply - cum_demand_before
    cum_before = cum_demand - max_grant
    partial_amount = np.maximum(per_applicant_supply - cum_before, 0.0)
    # Final amounts: full grant where not exceeding, partial at boundary, 0 beyond
    final_amounts = np.where(exceeds, partial_amount, max_grant)

    # Only keep positive amounts
    pos_mask = final_amounts > EPS
    if not pos_mask.any():
        if info_enabled:
            log.info("  No loans granted this round.")
            log.info("--- Credit Market Round complete ---")
        return

    loan_firms = sorted_firms[pos_mask]
    loan_banks = sorted_banks[pos_mask]
    loan_amounts = final_amounts[pos_mask]
    loan_frag = sorted_frag[pos_mask]

    # 9. Calculate interest rates
    frag_for_rate = loan_frag
    if max_leverage > 0.0:
        frag_for_rate = np.minimum(frag_for_rate, max_leverage)
    loan_rates = r_bar * (1.0 + lend.opex_shock[loan_banks] * frag_for_rate)

    # 10. Batch append to LoanBook
    lb.append_loans_batch(loan_banks, loan_firms, loan_amounts, loan_rates)

    # 11. Update borrower state
    np.add.at(bor.total_funds, loan_firms, loan_amounts)
    np.subtract.at(bor.credit_demand, loan_firms, loan_amounts)
    # Clamp any tiny negative values from floating point
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)

    # 12. Update lender state: deduct per-bank totals
    bank_totals = np.bincount(loan_banks, weights=loan_amounts, minlength=n_banks)
    lend.credit_supply -= bank_totals

    if info_enabled:
        log.info(
            f"  Granted {loan_firms.size} loans totaling {loan_amounts.sum():,.2f}"
        )
        log.info("--- Credit Market Round complete ---")


def firms_fire_workers(
    emp: Employer,
    wrk: Worker,
    *,
    rng: Rng = make_rng(),
) -> None:
    """Vectorized firing of workers when credit is insufficient.

    Groups workers by employer, identifies firms with financing gaps,
    randomly selects victims using cumulative wage sums, and batch-updates
    all worker/firm state.

    Parameters
    ----------
    emp : Employer
        Employer role (firms).
    wrk : Worker
        Worker role (households).
    rng : Rng
        Random generator for random firing selection.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Firing Workers ---")

    n_firms = emp.current_labor.size
    gaps = emp.wage_bill - emp.total_funds
    firing_ids = np.where(gaps > EPS)[0]

    if firing_ids.size == 0:
        if info_enabled:
            log.info("  No firms need to fire workers.")
            log.info("--- Firms Firing Workers complete ---")
        return

    if info_enabled:
        total_gap = gaps[firing_ids].sum()
        log.info(
            f"  {firing_ids.size} firms have financing gaps totaling {total_gap:,.2f}"
        )

    # Build worker-to-employer index for all employed workers
    employed_mask = wrk.employed == 1
    all_employed = np.where(employed_mask)[0]

    if all_employed.size == 0:
        if info_enabled:
            log.info("  No employed workers. Skipping.")
            log.info("--- Firms Firing Workers complete ---")
        return

    employers_of_employed = wrk.employer[all_employed]

    # Sort workers by employer
    sort_order = np.argsort(employers_of_employed, kind="stable")
    sorted_workers = all_employed[sort_order]
    sorted_employers = employers_of_employed[sort_order]

    # Group boundaries per firm
    boundaries_lo = np.searchsorted(sorted_employers, firing_ids, side="left")
    boundaries_hi = np.searchsorted(sorted_employers, firing_ids, side="right")

    # Vectorized firing: flatten workers at firing firms, shuffle within
    # groups, compute grouped wage cumsum, fire until cumsum covers gap.
    items, group_idx, rank, group_sizes, group_starts = _flatten_and_shuffle_groups(
        sorted_workers, boundaries_lo, boundaries_hi, rng
    )

    if items.size == 0:
        if info_enabled:
            log.info("  No workers at firing firms.")
            log.info("--- Firms Firing Workers complete ---")
        return

    # Grouped cumulative sum of wages in shuffled order
    shuffled_wages = wrk.wage[items]
    grouped_cumwage = grouped_cumsum(shuffled_wages, group_starts[:-1])

    # Per-worker gap threshold
    gap_per_worker = np.repeat(gaps[firing_ids], group_sizes)

    # Find first position per group where cumulative wage >= gap
    sufficient = grouped_cumwage >= gap_per_worker
    INF = int(items.size) + 1
    rank_if_sufficient = np.where(sufficient, rank, INF)

    # np.minimum.reduceat finds earliest sufficient rank per group
    # (only over non-empty groups)
    non_empty = group_sizes > 0
    first_sufficient = np.full(firing_ids.size, INF, dtype=np.intp)
    if non_empty.any():
        ne_starts = group_starts[:-1][non_empty]
        ne_result = np.minimum.reduceat(rank_if_sufficient, ne_starts)
        first_sufficient[non_empty] = ne_result

    # Fire count: first_sufficient + 1, or group_size if gap never covered
    fire_count = np.where(
        first_sufficient < INF,
        first_sufficient + 1,
        group_sizes,
    )
    threshold = np.repeat(fire_count, group_sizes)
    fire_mask = rank < threshold

    victims = items[fire_mask]
    victim_employers = firing_ids[group_idx[fire_mask]]
    total_fired = int(victims.size)

    if (
        victims.size == 0
    ):  # pragma: no cover — defensive; unreachable when items.size > 0
        if info_enabled:
            log.info("  No workers fired.")
            log.info("--- Firms Firing Workers complete ---")
        return

    # Capture fired wages before zeroing
    fired_wages = wrk.wage[victims].copy()

    # Batch worker-side updates
    wrk.employer[victims] = -1
    wrk.employer_prev[victims] = victim_employers
    wrk.wage[victims] = 0.0
    wrk.periods_left[victims] = 0
    wrk.contract_expired[victims] = 0
    wrk.fired[victims] = 1

    # Batch firm-side updates
    fire_counts = np.bincount(victim_employers, minlength=n_firms)
    emp.current_labor -= fire_counts.astype(emp.current_labor.dtype)

    # Subtract fired wages from wage bill (avoids full-workforce scan)
    fired_wage_sum = np.bincount(
        victim_employers, weights=fired_wages, minlength=n_firms
    )
    emp.wage_bill -= fired_wage_sum

    if info_enabled:
        log.info(f"  Fired {total_fired} workers across {firing_ids.size} firms.")
        log.info("--- Firms Firing Workers complete ---")
