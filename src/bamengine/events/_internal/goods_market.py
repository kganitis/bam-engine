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

    log.info("--- Consumer Spending Propensity Calculation complete ---")


def consumers_decide_income_to_spend(con: Consumer) -> None:
    """
    Allocate wealth to spending budget based on propensity to consume.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideIncomeToSpend : Full documentation
    """
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
    total_spending_budget = con.income_to_spend.sum()
    total_final_savings = con.savings.sum()
    consumers_with_budget = np.sum(con.income_to_spend > EPS)

    log.info(f"  Spending decisions made for {con.income_to_spend.size:,} consumers.")
    log.info(f"  Total spending budget allocated: {total_spending_budget:,.2f}")
    log.info(f"  Total remaining savings: {total_final_savings:,.2f}")
    log.info(f"  Consumers with positive spending budget: {consumers_with_budget:,}")

    if log.isEnabledFor(logging.DEBUG):
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
        wealth_check = total_spending_budget + total_final_savings
        if abs(wealth_check - total_wealth) > EPS:
            log.error(
                f"  WEALTH CONSERVATION ERROR: "
                f"Expected {total_wealth:.2f}, Got {wealth_check:.2f}"
            )

    log.info("--- Consumer Income-to-Spend Decision complete ---")


def consumers_decide_firms_to_visit(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Rng = make_rng(),
) -> None:
    """
    Consumers select firms to visit, sorted by price.

    See Also
    --------
    bamengine.events.goods_market.ConsumersDecideFirmsToVisit : Full documentation
    """
    log.info("--- Consumers Deciding Firms to Visit ---")

    stride = max_Z
    avail = np.where(prod.inventory > EPS)[0]
    consumers_with_budget = np.sum(con.income_to_spend > EPS)

    log.info(
        f"  {consumers_with_budget:,} consumers with spending budget will select"
        f" up to {max_Z} firms each from {avail.size} firms with inventory."
    )

    # Initialize/flush all shopping queues
    con.shop_visits_targets.fill(-1)
    con.shop_visits_head.fill(-1)

    if avail.size == 0:
        log.info("  No firms have inventory available. All shopping queues cleared.")
        log.info("--- Consumer Firm Selection complete ---")
        return

    if consumers_with_budget == 0:
        log.info("  No consumers have spending budget. All shopping queues cleared.")
        log.info("--- Consumer Firm Selection complete ---")
        return

    # Track loyalty statistics
    loyalty_applied = 0
    total_selections_made = 0
    consumers_processed = 0

    log.info("  Processing firm selection for each consumer...")

    for h in range(con.income_to_spend.size):
        if con.income_to_spend[h] <= EPS:
            continue  # Skip consumers with no spending budget

        consumers_processed += 1
        row = con.shop_visits_targets[h]
        filled = 0

        # Apply loyalty rule for slot 0
        prev = con.largest_prod_prev[h]
        loyal = (prev >= 0) and (prod.inventory[prev] > 0.0)
        if loyal:
            row[0] = prev
            filled = 1
            loyalty_applied += 1
            if log.isEnabledFor(logging.TRACE):
                log.trace(f"    Consumer {h}: Applied loyalty to firm {prev} (slot 0)")

        # Fill remaining slots with random sampling
        n_draw = min(stride - filled, avail.size - int(loyal))
        if n_draw > 0:
            # Ensure we don't re-sample the loyal firm
            choices = avail if not loyal else avail[avail != prev]
            if choices.size >= n_draw:
                sample = rng.choice(choices, size=n_draw, replace=False)
                # Sort by price (cheapest first) for optimal shopping order
                order = np.argsort(prod.price[sample])
                row[filled : filled + n_draw] = sample[order]
                filled += n_draw

                if log.isEnabledFor(logging.TRACE):
                    log.trace(
                        f"    Consumer {h}: Added {n_draw} firms, "
                        f"sorted by price: {sample[order]}"
                    )

        # Defensive loyalty enforcement (should never trigger in practice)
        if loyal and filled > 1 and row[0] != prev:  # pragma: no cover
            log.warning(f"    Consumer {h}: Loyalty violation detected, correcting...")
            j = np.where(row[:filled] == prev)[0][0]
            row[0], row[j] = row[j], row[0]

        # Activate shopping queue if any firms selected
        if filled > 0:
            con.shop_visits_head[h] = h * stride
            total_selections_made += filled

            if log.isEnabledFor(logging.DEBUG) and h < 10:  # Log first 10 consumers
                selected_firms = row[:filled]
                prices = prod.price[selected_firms[selected_firms >= 0]]
                log.debug(
                    f"    Consumer {h}: Selected {filled} firms: {selected_firms}, "
                    f"Prices: {np.array2string(prices, precision=2)}"
                )

    # Summary statistics
    avg_selections = (
        total_selections_made / consumers_processed if consumers_processed > 0 else 0.0
    )
    loyalty_rate = (
        loyalty_applied / consumers_processed if consumers_processed > 0 else 0.0
    )

    log.info(
        f"  Firm selection completed for "
        f"{consumers_processed:,} consumers with budget."
    )
    log.info(
        f"  Total firm selections made: {total_selections_made:,} "
        f"(Average: {avg_selections:.1f} per consumer)"
    )
    log.info(
        f"  Loyalty rule applied: "
        f"{loyalty_applied:,} times ({loyalty_rate:.1%} of consumers)"
    )

    if log.isEnabledFor(logging.DEBUG):
        active_shoppers = np.sum(con.shop_visits_head >= 0)
        log.debug(f"  Active shoppers with queued visits: {active_shoppers:,}")

        # Check firm popularity
        firm_selection_counts = np.bincount(
            con.shop_visits_targets[con.shop_visits_targets >= 0],
            minlength=prod.inventory.size,
        )
        most_popular_firm = np.argmax(firm_selection_counts)
        max_selections = firm_selection_counts[most_popular_firm]
        log.debug(
            f"  Most popular firm: {most_popular_firm} "
            f"(selected by {max_selections} consumers)"
        )

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
    log.info("--- Consumers Shopping One Round ---")

    stride = con.shop_visits_targets.shape[1]
    buyers_indices = np.where(con.income_to_spend > EPS)[0]

    if buyers_indices.size == 0:
        log.info(
            "  No consumers with remaining spending budget. Shopping round skipped."
        )
        log.info("--- Shopping Round complete ---")
        return

    # Pre-round statistics
    total_budget_before = con.income_to_spend.sum()
    total_inventory_before = prod.inventory.sum()

    if total_inventory_before <= EPS:
        log.info("  No firms with remaining inventory. Shopping round skipped.")
        log.info("--- Shopping Round complete ---")
        return

    log.info(
        f"  {buyers_indices.size:,} consumers with remaining budget "
        f"(Total: {total_budget_before:,.2f}) are shopping."
    )
    log.info(f"  Total available inventory: {total_inventory_before:,.2f}")

    # Randomize shopping order for fairness
    rng.shuffle(buyers_indices)
    log.info("  Shopping order randomized for fairness.")

    # Track round statistics
    successful_purchases = 0
    total_quantity_sold = 0.0
    total_revenue = 0.0
    consumers_exhausted_budget = 0
    consumers_exhausted_queue = 0
    firms_sold_out = 0
    loyalty_updates = 0

    for h in buyers_indices:
        ptr = con.shop_visits_head[h]
        if ptr < 0:
            continue  # Consumer has no more firms to visit

        row, col = divmod(ptr, stride)
        firm_idx = con.shop_visits_targets[row, col]

        if firm_idx < 0:  # Reached end of queue
            con.shop_visits_head[h] = -1
            consumers_exhausted_queue += 1
            if log.isEnabledFor(logging.TRACE):
                log.trace(f"    Consumer {h} exhausted firm queue at col {col}")
            continue

        # Check if firm still has inventory
        if prod.inventory[firm_idx] <= EPS:
            # Firm sold out - skip but advance pointer
            con.shop_visits_head[h] = ptr + 1
            con.shop_visits_targets[row, col] = -1
            if log.isEnabledFor(logging.TRACE):
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

        # Update loyalty tracking
        prev = con.largest_prod_prev[h]
        if (prev < 0) or (prod.production[firm_idx] > prod.production[prev]):
            con.largest_prod_prev[h] = firm_idx
            loyalty_updates += 1

        # Update statistics
        successful_purchases += 1
        total_quantity_sold += qty
        total_revenue += spent

        if log.isEnabledFor(logging.TRACE):
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
            if log.isEnabledFor(logging.TRACE):
                log.trace(f"    Consumer {h} exhausted spending budget")

    # Post-round statistics
    total_budget_after = con.income_to_spend.sum()
    total_inventory_after = prod.inventory.sum()
    budget_spent = total_budget_before - total_budget_after
    inventory_sold = total_inventory_before - total_inventory_after

    log.info(f"  Shopping round completed: {successful_purchases:,} purchases made.")
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
        log.debug(
            f"  Consumer outcomes: {consumers_exhausted_budget:,} exhausted budget, "
            f"{consumers_exhausted_queue:,} exhausted firm queue"
        )
        log.debug(f"  Firm outcomes: {firms_sold_out:,} firms sold out completely")
        log.debug(
            f"  Loyalty updates: "
            f"{loyalty_updates:,} consumers updated largest producer"
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

    log.info("--- Shopping Round complete ---")


def consumers_finalize_purchases(con: Consumer) -> None:
    """
    Return unspent budget to savings after shopping rounds complete.

    See Also
    --------
    bamengine.events.goods_market.ConsumersFinalizePurchases : Full documentation
    """
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
    total_savings_after = con.savings.sum()
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

        # Wealth conservation check
        if abs(savings_increase - total_unspent) > EPS:
            log.error(
                f"  WEALTH CONSERVATION ERROR: Expected savings increase of "
                f"{total_unspent:.2f}, got {savings_increase:.2f}"
            )
        else:
            log.debug("  Wealth conservation verified: unspent budget properly saved")

    log.info("--- Purchase Finalization complete ---")
