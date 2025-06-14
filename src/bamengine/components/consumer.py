# src/bamengine/components/consumer.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Consumer:

    income: Float1D
    savings: Float1D
    income_to_spend: Float1D
    propensity: Float1D
    largest_prod_prev: Idx1D

    # Scratch queues
    shop_visits_head: Idx1D
    shop_visits_targets: Idx2D  # shape (n_households, Z)
