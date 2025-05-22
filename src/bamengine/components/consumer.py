# src/bamengine/components/consumer.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Consumer:
    """
    Dense state for *all* households.

    Persistent columns
    ------------------
    income   : wage income earned this period (cleared after ‘‘goods-market’’)
    savings  : accumulated, unspent wealth carried across periods
    remaining_income : income still available to spend this period
    largest_prod_prev: firm index with the highest output visited in *t-1*
                       (-1 ⇒ none)

    Scratch columns (reset every period)
    ------------------------------------
    shop_visits_head : pointer into ``shop_visits_targets`` (-1 ⇒ exhausted)
    shop_visits_targets : planned shopping queue
                          shape = (N_households,  Z)
    """

    # persistent
    income: Float1D
    savings: Float1D
    remaining_income: Float1D
    largest_prod_prev: Idx1D

    # scratch queues (reset every period)
    shop_visits_head: Idx1D
    shop_visits_targets: Idx2D
