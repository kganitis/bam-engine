# src/bamengine/components/consumer.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Consumer:
    """
    Dense state for *all* households.
    ─────────────────────────────────────────────────────────────
    income            : wage income earned **this** period
    savings           : accumulated unspent wealth (≥ 0)
    income_to_spend   : budget still available to spend this period
    propensity        : π_h  ≡ share of wealth that will be spent (0-1)
    largest_prod_prev : firm with highest output visited in t-1  (-1 ⇒ none)

    Scratch columns (cleared every period)
    ──────────────────────────────────────
    shop_visits_head     : pointer into ``shop_visits_targets`` (-1 ⇒ exhausted)
    shop_visits_targets  : shopping queue, shape = (N_households, Z)
    """

    # persistent
    income: Float1D
    savings: Float1D
    income_to_spend: Float1D
    propensity: Float1D
    largest_prod_prev: Idx1D

    # scratch (reset every period)
    shop_visits_head: Idx1D
    shop_visits_targets: Idx2D
