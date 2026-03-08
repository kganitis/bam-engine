"""
Vectorized market matching system functions.

These functions replace the sequential for-loop market matching in
``labor_market.py``, ``credit_market.py``, and ``goods_market.py`` with
batch NumPy operations.  Each function implements one *round* of matching
and is called ``max_M`` / ``max_H`` / ``max_Z`` times by the pipeline.

Key design choices
------------------
- **Batch-send + conflict resolution** for labor and credit markets:
  all agents act simultaneously, conflicts resolved by random selection.
- **Batch-sequential processing** for goods market: consumers are
  processed in randomized batches, each completing all visits before
  the next batch starts, preserving sequential depletion dynamics.
- **Cross-round state**: each round sees updated vacancies/supply/inventory
  from previous rounds, preserving search friction.
"""

from __future__ import annotations

import logging

import numpy as np

from bamengine import Rng, make_rng
from bamengine.events._internal.vectorized_utils import (
    grouped_cumsum,
    resolve_conflicts,
)
from bamengine.relationships.loanbook import LoanBook
from bamengine.roles.borrower import Borrower
from bamengine.roles.consumer import Consumer
from bamengine.roles.employer import Employer
from bamengine.roles.lender import Lender
from bamengine.roles.producer import Producer
from bamengine.roles.worker import Worker

log = logging.getLogger(__name__)

EPS = 1e-12


# ═════════════════════════════════════════════════════════════════════════════
#  LABOR MARKET
# ═════════════════════════════════════════════════════════════════════════════


def labor_market_round_vec(
    emp: Employer,
    wrk: Worker,
    *,
    theta: int,
    rng: Rng = make_rng(),
) -> None:
    """One vectorized round of labor market matching.

    All unemployed applicants simultaneously send their next application,
    conflicts are resolved randomly, and accepted workers are batch-hired.

    Parameters
    ----------
    emp : Employer
        Employer role (firms).
    wrk : Worker
        Worker role (households).
    theta : int
        Minimum contract duration.
    rng : Rng
        Random generator for conflict resolution.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Vectorized Labor Market Round ---")

    stride = wrk.job_apps_targets.shape[1]  # max_M
    n_firms = emp.current_labor.size

    # 1. Find active unemployed applicants with pending applications
    unemp_ids = np.where(wrk.employed == 0)[0]
    if unemp_ids.size == 0:
        if info_enabled:
            log.info("  No unemployed workers. Skipping round.")
            log.info("--- Vectorized Labor Market Round complete ---")
        return

    active_mask = wrk.job_apps_head[unemp_ids] >= 0
    active = unemp_ids[active_mask]

    if active.size == 0:
        if info_enabled:
            log.info("  No workers with pending applications. Skipping round.")
            log.info("--- Vectorized Labor Market Round complete ---")
        return

    # 2. Decode targets from head pointers
    heads = wrk.job_apps_head[active]
    rows, cols = np.divmod(heads, stride)
    target_firms = wrk.job_apps_targets[rows, cols]

    # 3. Filter: valid targets (not -1)
    valid_target_mask = target_firms >= 0
    # Also filter for firms with vacancies
    has_vacancy_mask = valid_target_mask.copy()
    has_vacancy_mask[valid_target_mask] &= (
        emp.n_vacancies[target_firms[valid_target_mask]] > 0
    )

    # 4. Advance ALL head pointers (even for invalid/no-vacancy targets)
    new_heads = heads + 1
    # Mark exhausted queues
    exhausted = (new_heads >= (active + 1) * stride) | (~valid_target_mask)
    new_heads[exhausted | (~valid_target_mask)] = -1
    # For valid targets that had no vacancy, still advance
    no_vacancy_valid = valid_target_mask & ~has_vacancy_mask
    new_heads_nv = heads[no_vacancy_valid] + 1
    nv_exhausted = new_heads_nv >= (active[no_vacancy_valid] + 1) * stride
    new_heads[no_vacancy_valid] = new_heads_nv
    new_heads[np.where(no_vacancy_valid)[0][nv_exhausted]] = -1

    wrk.job_apps_head[active] = new_heads
    # Clear consumed slots
    wrk.job_apps_targets[rows, cols] = -1

    # 5. Get senders with valid, vacancy-having targets
    valid_senders = active[has_vacancy_mask]
    valid_targets = target_firms[has_vacancy_mask]

    if valid_senders.size == 0:
        if info_enabled:
            log.info("  No valid applications this round.")
            log.info("--- Vectorized Labor Market Round complete ---")
        return

    # 6. Conflict resolution
    accepted_mask = resolve_conflicts(
        valid_senders, valid_targets, emp.n_vacancies, n_firms, rng
    )

    hired_workers = valid_senders[accepted_mask]
    hired_firms = valid_targets[accepted_mask]

    if hired_workers.size == 0:
        if info_enabled:
            log.info("  No hires this round after conflict resolution.")
            log.info("--- Vectorized Labor Market Round complete ---")
        return

    if info_enabled:
        log.info(f"  Hiring {hired_workers.size} worker(s) this round.")

    # 7. Batch hire: update worker state
    wrk.employer[hired_workers] = hired_firms
    wrk.wage[hired_workers] = emp.wage_offer[hired_firms]
    wrk.periods_left[hired_workers] = theta
    wrk.contract_expired[hired_workers] = 0
    wrk.fired[hired_workers] = 0
    wrk.job_apps_head[hired_workers] = -1
    wrk.job_apps_targets[hired_workers, :] = -1

    # 8. Batch hire: update firm state
    hire_counts = np.bincount(hired_firms, minlength=n_firms).astype(
        emp.current_labor.dtype
    )
    emp.current_labor += hire_counts
    emp.n_vacancies -= hire_counts

    if info_enabled:
        log.info("--- Vectorized Labor Market Round complete ---")


# ═════════════════════════════════════════════════════════════════════════════
#  CREDIT MARKET
# ═════════════════════════════════════════════════════════════════════════════


def credit_market_round_vec(
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
        "lend.opex_shock must be set before credit_market_round_vec() — "
        "run banks_decide_interest_rate() first"
    )
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Vectorized Credit Market Round ---")

    stride = bor.loan_apps_targets.shape[1]  # max_H
    n_banks = lend.credit_supply.size

    # 1. Find active borrowers with credit demand > 0 and pending apps
    borrowers = np.where(bor.credit_demand > 0.0)[0]
    if borrowers.size == 0:
        if info_enabled:
            log.info("  No borrowers with credit demand. Skipping round.")
            log.info("--- Vectorized Credit Market Round complete ---")
        return

    active_mask = bor.loan_apps_head[borrowers] >= 0
    active = borrowers[active_mask]

    if active.size == 0:
        if info_enabled:
            log.info("  No borrowers with pending applications. Skipping round.")
            log.info("--- Vectorized Credit Market Round complete ---")
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
    # Valid but no supply: still advance
    no_supply = valid_mask & ~valid_with_supply
    ns_new = heads[no_supply] + 1
    ns_exhausted = ns_new >= (active[no_supply] + 1) * stride
    new_heads[no_supply] = ns_new
    new_heads[np.where(no_supply)[0][ns_exhausted]] = -1
    bor.loan_apps_head[active] = new_heads

    # 5. Get valid senders
    valid_firms = active[valid_with_supply]
    valid_banks = target_banks[valid_with_supply]

    if valid_firms.size == 0:
        if info_enabled:
            log.info("  No valid applications this round.")
            log.info("--- Vectorized Credit Market Round complete ---")
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
    cd = bor.credit_demand[sorted_firms].copy()
    max_grant = cd.copy()

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
            log.info("--- Vectorized Credit Market Round complete ---")
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
        log.info("--- Vectorized Credit Market Round complete ---")


# ═════════════════════════════════════════════════════════════════════════════
#  FIRING (post credit market)
# ═════════════════════════════════════════════════════════════════════════════


def firms_fire_workers_vec(
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
        log.info("--- Vectorized Firms Firing Workers ---")

    n_firms = emp.current_labor.size
    gaps = emp.wage_bill - emp.total_funds
    firing_ids = np.where(gaps > EPS)[0]

    if firing_ids.size == 0:
        if info_enabled:
            log.info("  No firms need to fire workers.")
            log.info("--- Vectorized Firms Firing Workers complete ---")
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
            log.info("--- Vectorized Firms Firing Workers complete ---")
        return

    employers_of_employed = wrk.employer[all_employed]

    # Sort workers by employer
    sort_order = np.argsort(employers_of_employed, kind="stable")
    sorted_workers = all_employed[sort_order]
    sorted_employers = employers_of_employed[sort_order]

    # Group boundaries per firm
    boundaries_lo = np.searchsorted(sorted_employers, firing_ids, side="left")
    boundaries_hi = np.searchsorted(sorted_employers, firing_ids, side="right")

    # Process each firing firm
    all_victims = []
    all_victim_employers = []
    total_fired = 0

    for idx in range(firing_ids.size):
        i = firing_ids[idx]
        lo, hi = boundaries_lo[idx], boundaries_hi[idx]
        group = sorted_workers[lo:hi]

        if group.size == 0:
            continue

        gap = gaps[i]
        worker_wages = wrk.wage[group]

        # Random firing until gap covered
        shuffled = rng.permutation(group.size)
        shuffled_wages = worker_wages[shuffled]
        cumsum_wages = np.cumsum(shuffled_wages)

        sufficient = cumsum_wages >= gap
        if sufficient.any():
            n_fire = int(sufficient.argmax()) + 1
        else:
            n_fire = group.size

        victims = group[shuffled[:n_fire]]
        all_victims.append(victims)
        all_victim_employers.append(np.full(victims.size, i, dtype=np.intp))
        total_fired += victims.size

    if not all_victims:
        if info_enabled:
            log.info("  No workers fired.")
            log.info("--- Vectorized Firms Firing Workers complete ---")
        return

    victims = np.concatenate(all_victims)
    victim_employers = np.concatenate(all_victim_employers)

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

    # Recalculate wage bill for firing firms
    # (need fresh bincount after firing)
    employed_after = wrk.employed == 1
    employed_idx = np.where(employed_after)[0]
    if employed_idx.size > 0:
        wb = np.bincount(
            wrk.employer[employed_idx],
            weights=wrk.wage[employed_idx],
            minlength=n_firms,
        )
        emp.wage_bill[firing_ids] = wb[firing_ids]
    else:
        emp.wage_bill[firing_ids] = 0.0

    if info_enabled:
        log.info(f"  Fired {total_fired} workers across {firing_ids.size} firms.")
        log.info("--- Vectorized Firms Firing Workers complete ---")


# ═════════════════════════════════════════════════════════════════════════════
#  GOODS MARKET
# ═════════════════════════════════════════════════════════════════════════════


def goods_market_round_vec(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Rng = make_rng(),
) -> None:
    """Vectorized goods market matching via batch-sequential processing.

    Processes consumers in randomized batches, where each batch completes
    ALL shopping visits before the next batch starts — mirroring the
    sequential version's dynamics where early consumers deplete inventory
    that later consumers must work around.  Each batch is processed using
    vectorized NumPy operations for performance.

    Parameters
    ----------
    con : Consumer
        Consumer role (households).
    prod : Producer
        Producer role (firms).
    max_Z : int
        Maximum shopping visits per consumer.
    rng : Rng
        Random generator for consumer ordering.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Vectorized Goods Market Round ---")

    stride = con.shop_visits_targets.shape[1]

    # Identify consumers with budget
    buyers = np.where(con.income_to_spend > EPS)[0]
    if buyers.size == 0:
        if info_enabled:
            log.info("  No consumers with budget. Skipping.")
            log.info("--- Vectorized Goods Market Round complete ---")
        return

    if prod.inventory.sum() <= EPS:
        if info_enabled:
            log.info("  No inventory available. Skipping.")
            log.info("--- Vectorized Goods Market Round complete ---")
        return

    # Shuffle consumers — single global ordering like sequential version
    rng.shuffle(buyers)

    total_purchases = 0
    total_qty = 0.0
    total_revenue = 0.0

    # Process in batches — each batch completes all Z visits before
    # the next batch starts, preserving sequential depletion dynamics
    batch_size = max(1, buyers.size // 10)  # ~10 batches
    n_batches = (buyers.size + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch = buyers[b * batch_size : (b + 1) * batch_size]

        # Each consumer in this batch processes all Z visits
        for _visit in range(max_Z):
            # Filter to batch members with budget and valid head
            has_budget = con.income_to_spend[batch] > EPS
            has_head = con.shop_visits_head[batch] >= 0
            active_mask = has_budget & has_head
            active = batch[active_mask]

            if active.size == 0:
                break

            # Decode targets
            heads = con.shop_visits_head[active]
            rows, cols = np.divmod(heads, stride)
            target_firms = con.shop_visits_targets[rows, cols]
            valid_mask = target_firms >= 0

            # Advance head pointers
            new_heads = heads + 1
            past_end = new_heads >= (active + 1) * stride
            new_heads[past_end | (~valid_mask)] = -1
            con.shop_visits_head[active] = new_heads
            con.shop_visits_targets[rows, cols] = -1

            # Filter to valid targets with inventory
            valid_shoppers = active[valid_mask]
            valid_targets = target_firms[valid_mask]

            if valid_shoppers.size == 0:
                continue

            has_inv = prod.inventory[valid_targets] > EPS
            shoppers = valid_shoppers[has_inv]
            targets = valid_targets[has_inv]

            if shoppers.size == 0:
                continue

            # Compute purchases — cap at available inventory
            prices = prod.price[targets]
            qty_wanted = con.income_to_spend[shoppers] / prices
            qty_actual = np.minimum(qty_wanted, prod.inventory[targets])

            # Execute purchases
            spent = qty_actual * prices
            con.income_to_spend[shoppers] -= spent
            np.subtract.at(prod.inventory, targets, qty_actual)
            np.maximum(con.income_to_spend, 0.0, out=con.income_to_spend)
            np.maximum(prod.inventory, 0.0, out=prod.inventory)

            total_purchases += shoppers.size
            total_qty += qty_actual.sum()
            total_revenue += spent.sum()

    if info_enabled:
        log.info(
            f"  {total_purchases} purchases, "
            f"qty={total_qty:,.2f}, revenue={total_revenue:,.2f}"
        )
        log.info("--- Vectorized Goods Market Round complete ---")
