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

    # Write sorted firms to shop_visits_targets for each active consumer
    for i, h in enumerate(budget_indices):
        con.shop_visits_targets[h, :effective_Z] = sorted_firms[i]
        con.shop_visits_head[h] = h * stride

    # Update loyalty to largest producer in consideration set (vectorized)
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


def consumers_shop_one_round(
    con: Consumer, prod: Producer, rng: Rng = make_rng()
) -> None:
    """
    Execute one shopping round where consumers purchase from one firm each.

    See Also
    --------
    bamengine.events.goods_market.ConsumersShopOneRound : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Shopping One Round ---")

    stride = con.shop_visits_targets.shape[1]
    buyers_indices = np.where(con.income_to_spend > EPS)[0]

    if buyers_indices.size == 0:
        if info_enabled:
            log.info(
                "  No consumers with remaining spending budget. Shopping round skipped."
            )
            log.info("--- Shopping Round complete ---")
        return

    # Pre-round statistics
    total_budget_before = con.income_to_spend.sum()
    total_inventory_before = prod.inventory.sum()

    if total_inventory_before <= EPS:
        if info_enabled:
            log.info("  No firms with remaining inventory. Shopping round skipped.")
            log.info("--- Shopping Round complete ---")
        return

    if info_enabled:
        log.info(
            f"  {buyers_indices.size:,} consumers with remaining budget "
            f"(Total: {total_budget_before:,.2f}) are shopping."
        )
        log.info(f"  Total available inventory: {total_inventory_before:,.2f}")

    # Randomize shopping order for fairness
    rng.shuffle(buyers_indices)
    if info_enabled:
        log.info("  Shopping order randomized for fairness.")

    # Track round statistics
    successful_purchases = 0
    total_quantity_sold = 0.0
    total_revenue = 0.0
    consumers_exhausted_budget = 0
    consumers_exhausted_queue = 0
    firms_sold_out = 0

    # Cache log level check outside loop for performance
    trace_enabled = log.isEnabledFor(logging.TRACE)

    for h in buyers_indices:
        ptr = con.shop_visits_head[h]
        if ptr < 0:
            continue  # Consumer has no more firms to visit

        row, col = divmod(ptr, stride)
        firm_idx = con.shop_visits_targets[row, col]

        if firm_idx < 0:  # Reached end of queue
            con.shop_visits_head[h] = -1
            consumers_exhausted_queue += 1
            if trace_enabled:
                log.trace(f"    Consumer {h} exhausted firm queue at col {col}")
            continue

        # Check if firm still has inventory
        if prod.inventory[firm_idx] <= EPS:
            # Firm sold out - skip but advance pointer
            # Note: Loyalty was already set BEFORE shopping in consumers_decide_firms_to_visit
            con.shop_visits_head[h] = ptr + 1
            con.shop_visits_targets[row, col] = -1
            if trace_enabled:
                log.trace(f"    Consumer {h}: Firm {firm_idx} sold out, skipping")
            continue

        # Calculate purchase quantity and cost
        price = prod.price[firm_idx]
        max_qty_by_budget = con.income_to_spend[h] / price
        max_qty_by_inventory = float(prod.inventory[firm_idx])
        qty = min(max_qty_by_budget, max_qty_by_inventory)
        spent = qty * price

        # Execute purchase
        prod.inventory[firm_idx] -= qty
        con.income_to_spend[h] -= spent

        # Track if firm sold out
        if prod.inventory[firm_idx] <= EPS:
            firms_sold_out += 1

        # Note: Loyalty is NOT updated here - it was set BEFORE shopping
        # in consumers_decide_firms_to_visit based on the consideration set

        # Update statistics
        successful_purchases += 1
        total_quantity_sold += qty
        total_revenue += spent

        if trace_enabled:
            log.trace(
                f"    Consumer {h} bought {qty:.2f} from firm {firm_idx} "
                f"for {spent:.2f} (price={price:.2f})"
            )

        # Advance shopping queue
        con.shop_visits_head[h] = ptr + 1
        con.shop_visits_targets[row, col] = -1

        # Check if consumer exhausted budget
        if con.income_to_spend[h] <= EPS:  # Effectively zero
            consumers_exhausted_budget += 1
            con.shop_visits_head[h] = -1  # Stop shopping
            if trace_enabled:
                log.trace(f"    Consumer {h} exhausted spending budget")

    # Post-round statistics
    if info_enabled:
        total_budget_after = con.income_to_spend.sum()
        total_inventory_after = prod.inventory.sum()
        budget_spent = total_budget_before - total_budget_after
        inventory_sold = total_inventory_before - total_inventory_after

        log.info(
            f"  Shopping round completed: {successful_purchases:,} purchases made."
        )
        log.info(
            f"  Total quantity sold: {total_quantity_sold:,.2f}, "
            f"Total revenue: {total_revenue:,.2f}"
        )
        log.info(
            f"  Budget spent: {budget_spent:,.2f} of {total_budget_before:,.2f} "
            f"({budget_spent / total_budget_before:.1%} utilization)"
        )
        log.info(
            f"  Inventory sold: {inventory_sold:,.2f} of {total_inventory_before:,.2f} "
            f"({inventory_sold / total_inventory_before:.1%} depletion)"
        )

    if log.isEnabledFor(logging.DEBUG):
        total_budget_after = con.income_to_spend.sum()
        total_inventory_after = prod.inventory.sum()
        budget_spent = total_budget_before - total_budget_after
        inventory_sold = total_inventory_before - total_inventory_after

        log.debug(
            f"  Consumer outcomes: {consumers_exhausted_budget:,} exhausted budget, "
            f"{consumers_exhausted_queue:,} exhausted firm queue"
        )
        log.debug(f"  Firm outcomes: {firms_sold_out:,} firms sold out completely")

        # Validation check
        if abs(budget_spent - total_revenue) > EPS:
            log.error(
                f"  ACCOUNTING ERROR: Budget spent ({budget_spent:.2f}) != "
                f"Revenue generated ({total_revenue:.2f})"
            )
        if abs(inventory_sold - total_quantity_sold) > EPS:
            log.error(
                f"  INVENTORY ERROR: Inventory sold ({inventory_sold:.2f}) != "
                f"Quantity purchased ({total_quantity_sold:.2f})"
            )

    if info_enabled:
        log.info("--- Shopping Round complete ---")


def consumers_shop_sequential(
    con: Consumer, prod: Producer, *, max_Z: int, rng: Rng = make_rng()
) -> None:
    """
    Execute sequential shopping where each consumer completes all visits.

    Unlike round-robin shopping (consumers_shop_one_round), this function
    processes consumers one at a time. Each consumer completes all their
    shopping visits before the next consumer starts. This matches NetLogo
    and ABCredit behavior and makes the goods market less efficient:
    early consumers can deplete inventory, leaving late consumers with
    wasted visits on sold-out firms.

    See Also
    --------
    bamengine.events.goods_market.ConsumersShopSequential : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Consumers Shopping Sequential ---")

    stride = con.shop_visits_targets.shape[1]
    buyers = np.where(con.income_to_spend > EPS)[0]

    if buyers.size == 0:
        if info_enabled:
            log.info("  No consumers with remaining spending budget. Shopping skipped.")
            log.info("--- Sequential Shopping complete ---")
        return

    # Pre-shopping statistics
    total_budget_before = con.income_to_spend.sum()
    total_inventory_before = prod.inventory.sum()

    if total_inventory_before <= EPS:
        if info_enabled:
            log.info("  No firms with remaining inventory. Shopping skipped.")
            log.info("--- Sequential Shopping complete ---")
        return

    if info_enabled:
        log.info(
            f"  {buyers.size:,} consumers with remaining budget "
            f"(Total: {total_budget_before:,.2f}) will shop sequentially."
        )
        log.info(f"  Total available inventory: {total_inventory_before:,.2f}")
        log.info(f"  Max visits per consumer: {max_Z}")

    # Randomize consumer order (like NetLogo's ask workers)
    rng.shuffle(buyers)
    if info_enabled:
        log.info("  Consumer order randomized for fairness.")

    # Track statistics
    successful_purchases = 0
    total_quantity_sold = 0.0
    total_revenue = 0.0
    wasted_visits = 0
    consumers_exhausted_budget = 0
    consumers_exhausted_queue = 0

    # Cache log level check outside loop for performance
    trace_enabled = log.isEnabledFor(logging.TRACE)

    for h in buyers:
        # Each consumer completes all Z visits before next consumer starts
        for _ in range(max_Z):
            if con.income_to_spend[h] <= EPS:
                consumers_exhausted_budget += 1
                break  # Budget exhausted

            ptr = con.shop_visits_head[h]
            if ptr < 0:
                consumers_exhausted_queue += 1
                break  # No more firms to visit

            row, col = divmod(ptr, stride)
            firm_idx = con.shop_visits_targets[row, col]

            if firm_idx < 0:
                con.shop_visits_head[h] = -1
                consumers_exhausted_queue += 1
                break  # End of queue

            # Advance pointer regardless of purchase success (like NetLogo)
            con.shop_visits_head[h] = ptr + 1
            con.shop_visits_targets[row, col] = -1

            # Check if firm has inventory
            if prod.inventory[firm_idx] <= EPS:
                wasted_visits += 1
                if trace_enabled:
                    log.trace(
                        f"    Consumer {h}: Firm {firm_idx} sold out, wasted visit"
                    )
                continue  # Wasted visit - firm sold out

            # Calculate purchase quantity and cost
            price = prod.price[firm_idx]
            max_qty_by_budget = con.income_to_spend[h] / price
            max_qty_by_inventory = float(prod.inventory[firm_idx])
            qty = min(max_qty_by_budget, max_qty_by_inventory)
            spent = qty * price

            # Execute purchase
            prod.inventory[firm_idx] -= qty
            con.income_to_spend[h] -= spent

            # Update statistics
            successful_purchases += 1
            total_quantity_sold += qty
            total_revenue += spent

            if trace_enabled:
                log.trace(
                    f"    Consumer {h} bought {qty:.2f} from firm {firm_idx} "
                    f"for {spent:.2f} (price={price:.2f})"
                )

    # Post-shopping statistics
    if info_enabled:
        total_budget_after = con.income_to_spend.sum()
        total_inventory_after = prod.inventory.sum()
        budget_spent = total_budget_before - total_budget_after
        inventory_sold = total_inventory_before - total_inventory_after

        log.info(
            f"  Sequential shopping completed: {successful_purchases:,} purchases made."
        )
        log.info(
            f"  Total quantity sold: {total_quantity_sold:,.2f}, "
            f"Total revenue: {total_revenue:,.2f}"
        )
        log.info(
            f"  Wasted visits (firms sold out): {wasted_visits:,} "
            f"({wasted_visits / max(1, successful_purchases + wasted_visits):.1%} of attempts)"
        )
        log.info(
            f"  Budget spent: {budget_spent:,.2f} of {total_budget_before:,.2f} "
            f"({budget_spent / total_budget_before:.1%} utilization)"
        )
        log.info(
            f"  Inventory sold: {inventory_sold:,.2f} of {total_inventory_before:,.2f} "
            f"({inventory_sold / total_inventory_before:.1%} depletion)"
        )

    if log.isEnabledFor(logging.DEBUG):
        total_budget_after = con.income_to_spend.sum()
        total_inventory_after = prod.inventory.sum()
        budget_spent = total_budget_before - total_budget_after
        inventory_sold = total_inventory_before - total_inventory_after

        log.debug(
            f"  Consumer outcomes: {consumers_exhausted_budget:,} exhausted budget, "
            f"{consumers_exhausted_queue:,} exhausted firm queue"
        )

        # Validation check
        if abs(budget_spent - total_revenue) > EPS:
            log.error(
                f"  ACCOUNTING ERROR: Budget spent ({budget_spent:.2f}) != "
                f"Revenue generated ({total_revenue:.2f})"
            )
        if abs(inventory_sold - total_quantity_sold) > EPS:
            log.error(
                f"  INVENTORY ERROR: Inventory sold ({inventory_sold:.2f}) != "
                f"Quantity purchased ({total_quantity_sold:.2f})"
            )

    if info_enabled:
        log.info("--- Sequential Shopping complete ---")


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
