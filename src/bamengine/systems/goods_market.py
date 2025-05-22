# src/bamengine/systems/goods_market.py
"""
Event-5 – Goods-market systems
Vectorised, allocation-free during the hot path.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator

from bamengine.components import Consumer, Producer


# ------------------------------------------------------------------ #
# 1.  Households: budget rule                                         #
# ------------------------------------------------------------------ #
def consumers_decide_income_to_spend(
    con: Consumer,
    *,
    avg_sav: float,
    beta: float,
) -> None:
    """
    Share of disposable wealth turned into consumption expenditure:

        w_h        = savings_h + income_h
        prop_h     = 1 / (1 + tanh(savings_h / avg_sav) ** beta)
        rem_inc_h  = prop_h · w_h
        savings_h  = w_h − rem_inc_h

    All vectors updated in-place; no temporaries.
    """
    avg_sav = max(avg_sav, 1.0e-12)  # guard division

    wealth = con.savings + con.income
    prop = 1.0 / (1.0 + np.tanh(con.savings / avg_sav) ** beta)
    con.remaining_income[:] = wealth * prop
    con.savings[:] = wealth - con.remaining_income
    con.income[:] = 0.0  # spent or saved – income cleared


# ------------------------------------------------------------------ #
# 2.  Households: pick firms to visit                                 #
# ------------------------------------------------------------------ #
def consumers_decide_firms_to_visit(
    con: Consumer,
    prod: Producer,
    *,
    max_Z: int,
    rng: Generator,
) -> None:
    """
    Each household draws Z candidate firms (with `inventory>0`).

    Loyalty rule
    ------------
    Previous-period “largest producer visited” stays in slot 0
    *iff* it still holds inventory.

    Remaining candidates are sampled **without replacement** and
    sorted ascending by price.
    """
    stride = max_Z
    avail = np.where(prod.inventory > 0.0)[0]

    # flush queues first
    con.shop_visits_targets.fill(-1)
    con.shop_visits_head.fill(-1)
    if avail.size == 0:
        return  # nothing to buy this period

    for h in range(con.remaining_income.size):
        row = con.shop_visits_targets[h]
        filled = 0

        prev = con.largest_prod_prev[h]
        # loyalty slot
        loyal = (prev >= 0) and (prod.inventory[prev] > 0.0)
        if loyal:
            row[0] = prev
            filled = 1

        n_draw = min(stride - filled, avail.size - int(loyal))
        if n_draw > 0:
            # ensure we don’t re-sample *prev*
            choices = avail if not loyal else avail[avail != prev]
            sample = rng.choice(choices, size=n_draw, replace=False)
            order = np.argsort(prod.price[sample])  # cheapest first
            row[filled : filled + n_draw] = sample[order]
            filled += n_draw

        if loyal and filled > 1 and row[0] != prev:
            # guarantee loyalty stays at slot-0 (rare race with price ties)
            j = np.where(row[:filled] == prev)[0][0]
            row[0], row[j] = row[j], row[0]

        if filled > 0:
            con.shop_visits_head[h] = h * stride


# ------------------------------------------------------------------ #
# 3.  One “shopping round”                                            #
# ------------------------------------------------------------------ #
def consumers_visit_one_round(con: Consumer, prod: Producer) -> None:
    """
    Execute *one* round of purchases for **all** households.
    """
    stride = con.shop_visits_targets.shape[1]

    for h in np.where(con.remaining_income > 0.0)[0]:
        ptr = con.shop_visits_head[h]
        if ptr < 0:
            continue

        row, col = divmod(ptr, stride)
        firm_idx = con.shop_visits_targets[row, col]
        if firm_idx < 0:  # exhausted queue
            con.shop_visits_head[h] = -1
            continue

        if prod.inventory[firm_idx] <= 0.0:
            # sold out – skip but advance pointer
            con.shop_visits_head[h] = ptr + 1
            con.shop_visits_targets[row, col] = -1
            continue

        price = prod.price[firm_idx]
        qty = min(prod.inventory[firm_idx], con.remaining_income[h] / price)
        spent = qty * price
        prod.inventory[firm_idx] -= qty
        con.remaining_income[h] -= spent

        # loyalty update
        prev = con.largest_prod_prev[h]
        if (prev < 0) or (prod.production[firm_idx] > prod.production[prev]):
            con.largest_prod_prev[h] = firm_idx

        # advance pointer & clear slot
        con.shop_visits_head[h] = ptr + 1
        con.shop_visits_targets[row, col] = -1


# ------------------------------------------------------------------ #
# 4.  Finalise: stash leftovers back to savings                       #
# ------------------------------------------------------------------ #
def consumers_finalize_purchases(con: Consumer) -> None:
    """Unspent income → savings; reset scratch vectors."""
    np.add(con.savings, con.remaining_income, out=con.savings)
    con.remaining_income.fill(0.0)
