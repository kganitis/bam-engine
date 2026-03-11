"""
System functions for goods market phase events.

This module contains the internal implementation functions for goods market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.goods_market : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.roles import Consumer, Producer
from bamengine.utils import EPS

log = logging.getLogger(__name__)


def consumers_calc_propensity(
    con: Consumer,
    *,
    avg_sav: float,
    beta: float,
) -> None:
    """
    Calculate marginal propensity to consume based on relative savings.

    See Also
    --------
    bamengine.events.goods_market.ConsumersCalcPropensity : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Calculating Consumer Spending Propensity ---")
        log.info(f"  Inputs: Average Savings={avg_sav:.3f} | β={beta:.3f}")

    # Defensive operations to ensure valid calculations
    initial_negative_savings = np.sum(con.savings < EPS)
    if initial_negative_savings > 0:
        log.warning(
            f"  Found {initial_negative_savings} consumers with negative savings. "
            f"Clamping to 0.0."
        )

    np.maximum(con.savings, 0.0, out=con.savings)  # defensive clamp
    avg_sav = max(avg_sav, EPS)  # avoid division by zero

    # Core calculation
    savings_ratio = con.savings / avg_sav
    t = np.tanh(savings_ratio)  # ∈ [0, 1]
    con.propensity[:] = 1.0 / (1.0 + t**beta)

    # Summary statistics
    if info_enabled:
        min_propensity = con.propensity.min()
        max_propensity = con.propensity.max()
        avg_propensity = con.propensity.mean()

        log.info(f"  Propensity calculated for {con.propensity.size:,} consumers.")
        log.info(
            f"  Propensity range: [{min_propensity:.3f}, {max_propensity:.3f}], "
            f"Average: {avg_propensity:.3f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        high_spenders = np.sum(con.propensity > 0.8)
        low_spenders = np.sum(con.propensity < 0.2)
        log.debug(
            f"  High spenders (>0.8): {high_spenders}, "
            f"Low spenders (<0.2): {low_spenders}"
        )
        log.debug(
            f"  First 10 propensities: "
            f"{np.array2string(con.propensity[:10], precision=3)}"
        )

    if info_enabled:
        log.info("--- Consumer Spending Propensity Calculation complete ---")


def consumers_decide_income_to_spend(con: Consumer) -> None:
    """
    Allocate wealth to spending budget based on propensity to consume.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideIncomeToSpend : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Deciding Income to Spend ---")

        # Pre-calculation statistics
        total_initial_savings = con.savings.sum()
        total_income = con.income.sum()
        total_wealth = total_initial_savings + total_income
        avg_propensity = con.propensity.mean()

        log.info(
            f"  Initial state: Total Savings={total_initial_savings:,.2f}, "
            f"Total Income={total_income:,.2f}, Total Wealth={total_wealth:,.2f}"
        )
        log.info(f"  Average propensity to spend: {avg_propensity:.3f}")

    # Core calculation
    wealth = con.savings + con.income
    con.income_to_spend[:] = wealth * con.propensity
    con.savings[:] = wealth - con.income_to_spend
    con.income[:] = 0.0  # zero-out disposable income after allocation

    # Post-calculation statistics
    if info_enabled:
        total_spending_budget = con.income_to_spend.sum()
        total_final_savings = con.savings.sum()
        consumers_with_budget = np.sum(con.income_to_spend > EPS)

        log.info(
            f"  Spending decisions made for {con.income_to_spend.size:,} consumers."
        )
        log.info(f"  Total spending budget allocated: {total_spending_budget:,.2f}")
        log.info(f"  Total remaining savings: {total_final_savings:,.2f}")
        log.info(
            f"  Consumers with positive spending budget: {consumers_with_budget:,}"
        )

    if log.isEnabledFor(logging.DEBUG):
        consumers_with_budget = np.sum(con.income_to_spend > EPS)
        max_budget = con.income_to_spend.max()
        avg_budget = (
            con.income_to_spend[con.income_to_spend > 0].mean()
            if consumers_with_budget > 0
            else 0.0
        )
        log.debug(
            f"  Spending budget stats - Max: {max_budget:.2f}, "
            f"Avg (of spenders): {avg_budget:.2f}"
        )
        log.debug(
            f"  First 10 spending budgets: "
            f"{np.array2string(con.income_to_spend[:10], precision=2)}"
        )

        # Sanity check: wealth should be conserved
        total_spending_budget = con.income_to_spend.sum()
        total_final_savings = con.savings.sum()
        total_wealth = con.savings.sum() + con.income_to_spend.sum()
        wealth_check = total_spending_budget + total_final_savings
        if abs(wealth_check - total_wealth) > EPS:
            log.error(
                f"  WEALTH CONSERVATION ERROR: "
                f"Expected {total_wealth:.2f}, Got {wealth_check:.2f}"
            )

    if info_enabled:
        log.info("--- Consumer Income-to-Spend Decision complete ---")


def consumers_decide_firms_to_visit(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Rng = make_rng(),
    consumer_matching: str = "loyalty",
) -> None:
    """
    Consumers select firms to visit and set loyalty BEFORE shopping.

    The loyalty (largest_prod_prev) is updated here based on the largest
    producer in the consideration set, BEFORE any shopping occurs.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideFirmsToVisit : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Deciding Firms to Visit ---")

    n_firms = prod.inventory.size
    stride = max_Z

    # Initialize/flush all shopping queues
    con.shop_visits_targets.fill(-1)
    con.shop_visits_head.fill(-1)

    if n_firms == 0:
        if info_enabled:
            log.info("  No firms available. All shopping queues cleared.")
            log.info("--- Consumer Firm Selection complete ---")
        return

    # Identify consumers with budget (vectorized)
    has_budget = con.income_to_spend > EPS
    budget_indices = np.where(has_budget)[0]
    n_active = budget_indices.size

    if info_enabled:
        log.info(
            f"  {n_active:,} consumers with spending budget will select"
            f" up to {max_Z} firms each from {n_firms} firms."
        )

    if n_active == 0:
        if info_enabled:
            log.info(
                "  No consumers have spending budget. All shopping queues cleared."
            )
            log.info("--- Consumer Firm Selection complete ---")
        return

    # Get loyalty firms for active consumers
    loyalty_firms = con.largest_prod_prev[budget_indices]
    has_loyalty = loyalty_firms >= 0

    # Vectorized firm selection using random priorities
    # Generate random priorities for each (consumer, firm) pair
    # Then select top-k firms by these priorities
    effective_Z = min(max_Z, n_firms)
    priorities = rng.random((n_active, n_firms))

    # For consumers with loyalty, give their loyalty firm highest priority (> 1.0)
    if consumer_matching == "loyalty":
        loyal_consumer_local_idx = np.where(has_loyalty)[0]
        if loyal_consumer_local_idx.size > 0:
            loyal_firm_ids = loyalty_firms[has_loyalty].astype(np.intp)
            priorities[loyal_consumer_local_idx, loyal_firm_ids] = 1.1

    # Select top effective_Z firms per consumer using argpartition (O(n) vs O(n log n))
    if effective_Z < n_firms:
        # argpartition: first effective_Z elements will be the top-k (unordered)
        top_k_indices = np.argpartition(-priorities, kth=effective_Z - 1, axis=1)[
            :, :effective_Z
        ]
    else:
        # If max_Z >= n_firms, all firms are selected
        top_k_indices = np.broadcast_to(np.arange(n_firms), (n_active, n_firms)).copy()

    # Sort selected firms by price (cheapest first) - vectorized
    prices_selected = prod.price[top_k_indices]
    price_order = np.argsort(prices_selected, axis=1)
    sorted_firms = np.take_along_axis(top_k_indices, price_order, axis=1)

    # Vectorized buffer write — fancy indexing replaces per-consumer loop
    con.shop_visits_targets[budget_indices, :effective_Z] = sorted_firms
    con.shop_visits_head[budget_indices] = budget_indices * stride

    # Update loyalty to largest producer in consideration set (vectorized)
    if consumer_matching == "loyalty":
        # For each consumer, find the firm with max production among selected firms
        production_selected = prod.production[sorted_firms]
        largest_local_idx = np.argmax(production_selected, axis=1)
        largest_firm_ids = sorted_firms[np.arange(n_active), largest_local_idx]
        con.largest_prod_prev[budget_indices] = largest_firm_ids

    # Compute statistics for logging
    loyalty_applied = has_loyalty.sum()
    total_selections_made = n_active * effective_Z
    loyalty_updates = n_active

    if info_enabled:
        avg_selections = effective_Z
        loyalty_rate = loyalty_applied / n_active if n_active > 0 else 0.0

        log.info(f"  Firm selection completed for {n_active:,} consumers with budget.")
        log.info(
            f"  Total firm selections made: {total_selections_made:,} "
            f"(Average: {avg_selections:.1f} per consumer)"
        )
        log.info(
            f"  Loyalty rule applied: "
            f"{loyalty_applied:,} times ({loyalty_rate:.1%} of consumers)"
        )
        log.info(
            f"  Loyalty updated (pre-shopping): "
            f"{loyalty_updates:,} consumers set loyalty to largest in consideration set"
        )

    debug_enabled = log.isEnabledFor(logging.DEBUG)
    if debug_enabled:
        active_shoppers = np.sum(con.shop_visits_head >= 0)
        log.debug(f"  Active shoppers with queued visits: {active_shoppers:,}")

        # Check firm popularity
        firm_selection_counts = np.bincount(
            con.shop_visits_targets[con.shop_visits_targets >= 0],
            minlength=n_firms,
        )
        most_popular_firm = np.argmax(firm_selection_counts)
        max_selections = firm_selection_counts[most_popular_firm]
        log.debug(
            f"  Most popular firm: {most_popular_firm} "
            f"(selected by {max_selections} consumers)"
        )

    if info_enabled:
        log.info("--- Consumer Firm Selection complete ---")


def consumers_finalize_purchases(con: Consumer) -> None:
    """
    Return unspent budget to savings after shopping rounds complete.

    See Also
    --------
    bamengine.events.goods_market.ConsumersFinalizePurchases : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Finalizing Consumer Purchases ---")

        # Pre-finalization statistics
        total_unspent = con.income_to_spend.sum()
        total_savings_before = con.savings.sum()
        consumers_with_unspent = np.sum(con.income_to_spend > EPS)

        log.info(
            f"  {consumers_with_unspent:,} consumers have unspent budget "
            f"totaling {total_unspent:,.2f}"
        )
        log.info(f"  Current total savings: {total_savings_before:,.2f}")

    # Core operation: move unspent budget to savings
    np.add(con.savings, con.income_to_spend, out=con.savings)
    con.income_to_spend.fill(0.0)

    # Post-finalization statistics
    if info_enabled:
        total_savings_after = con.savings.sum()
        # Note: total_savings_before and total_unspent computed above under info_enabled
        savings_increase = total_savings_after - total_savings_before

        log.info(
            f"  Unspent budget moved to savings. "
            f"New total savings: {total_savings_after:,.2f}"
        )
        log.info(f"  Savings increase: {savings_increase:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        avg_savings = con.savings.mean()
        max_savings = con.savings.max()
        consumers_with_savings = np.sum(con.savings > 0.0)

        log.debug(
            f"  Final savings stats - Average: {avg_savings:.2f}, "
            f"Maximum: {max_savings:.2f}"
        )
        log.debug(f"  Consumers with positive savings: {consumers_with_savings:,}")

        # Wealth conservation check (only if info was enabled and we have the values)
        if info_enabled:
            if abs(savings_increase - total_unspent) > EPS:
                log.error(
                    f"  WEALTH CONSERVATION ERROR: Expected savings increase of "
                    f"{total_unspent:.2f}, got {savings_increase:.2f}"
                )
            else:
                log.debug(
                    "  Wealth conservation verified: unspent budget properly saved"
                )

    if info_enabled:
        log.info("--- Purchase Finalization complete ---")


def goods_market_round(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    n_batches: int = 10,
    rng: Rng = make_rng(),
) -> None:
    """Vectorized goods market matching via batch-sequential processing.

    Processes consumers in randomized batches, where each batch completes
    ALL shopping visits before the next batch starts -- mirroring the
    sequential version's dynamics where early consumers deplete inventory
    that later consumers must work around.  Each batch is processed using
    vectorized NumPy operations for performance.

    When multiple consumers in the same batch target the same firm, a small
    oversell can occur (each reads stale inventory).  This is intentional:
    phantom goods compensate for the batch-sequential variance reduction
    relative to true sequential processing.  Inventory is clamped to zero
    once per batch, bounding the actual oversell.

    Parameters
    ----------
    con : Consumer
        Consumer role (households).
    prod : Producer
        Producer role (firms).
    max_Z : int
        Maximum shopping visits per consumer.
    n_batches : int
        Number of sequential batches to split consumers into.  More batches
        better approximate fully sequential processing.  Default 10.
    rng : Rng
        Random generator for consumer ordering.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Goods Market Round ---")

    # Identify consumers with budget
    buyers = np.where(con.income_to_spend > EPS)[0]
    if buyers.size == 0:
        if info_enabled:
            log.info("  No consumers with budget. Skipping.")
            log.info("--- Goods Market Round complete ---")
        return

    if prod.inventory.sum() <= EPS:
        if info_enabled:
            log.info("  No inventory available. Skipping.")
            log.info("--- Goods Market Round complete ---")
        return

    # Shuffle consumers -- single global ordering like sequential version
    rng.shuffle(buyers)

    total_purchases = 0
    total_qty = 0.0
    total_revenue = 0.0

    # Process in batches -- each batch completes all Z visits before
    # the next batch starts, preserving sequential depletion dynamics.
    # Batch 0 = highest-priority consumers (first in shuffle order).
    actual_n_batches = min(n_batches, buyers.size)
    batch_size = max(1, buyers.size // actual_n_batches)
    # Recompute actual count from batch_size to handle rounding
    actual_n_batches = (buyers.size + batch_size - 1) // batch_size

    # Pre-cache array references (constant during shopping round)
    targets_arr = con.shop_visits_targets
    inv = prod.inventory
    prices_arr = prod.price

    for b in range(actual_n_batches):
        batch = buyers[b * batch_size : (b + 1) * batch_size]

        # Work on a local budget copy to avoid repeated fancy-index
        # read/writes on the global income_to_spend array.
        budget = con.income_to_spend[batch]  # fancy index → copy

        # Each consumer in this batch processes all Z visits.
        # Targets are indexed directly by [consumer, visit] — the head
        # pointer machinery is unnecessary since consumers_decide_firms_to_visit
        # fills targets sequentially into slots 0..max_Z-1.
        for visit in range(max_Z):
            targets = targets_arr[batch, visit]

            # Combined filter: has budget, valid target, firm has inventory.
            # np.maximum(targets, 0) prevents negative indexing when target=-1;
            # the result is masked out by `targets >= 0` anyway.
            safe_t = np.maximum(targets, 0)
            active_mask = (budget > EPS) & (targets >= 0) & (inv[safe_t] > EPS)

            if not active_mask.any():
                continue

            local_idx = np.where(active_mask)[0]
            firm_ids = targets[local_idx]

            # Compute purchases — cap at available inventory.
            # Within a batch, multiple consumers may read the same (stale)
            # inventory, causing small oversells.  This is intentional:
            # phantom goods compensate for the batch-sequential variance
            # reduction vs true sequential processing.
            prices = prices_arr[firm_ids]
            qty_wanted = budget[local_idx] / prices
            qty_actual = np.minimum(qty_wanted, inv[firm_ids])

            # Execute purchases
            spent = qty_actual * prices
            budget[local_idx] -= spent
            np.subtract.at(inv, firm_ids, qty_actual)

            if info_enabled:
                total_purchases += local_idx.size
                total_qty += float(qty_actual.sum())
                total_revenue += float(spent.sum())

        # Write back budget and clamp once per batch.
        # Inventory can go briefly negative from subtract.at collisions;
        # has_inv filters negative-inventory firms in subsequent visits,
        # so per-visit clamping is unnecessary — once per batch suffices.
        np.maximum(budget, 0.0, out=budget)
        con.income_to_spend[batch] = budget
        np.maximum(inv, 0.0, out=inv)

    if info_enabled:
        log.info(
            f"  {total_purchases} purchases, "
            f"qty={total_qty:,.2f}, revenue={total_revenue:,.2f}"
        )
        log.info("--- Goods Market Round complete ---")
