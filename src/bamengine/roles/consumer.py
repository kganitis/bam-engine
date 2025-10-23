# src/bamengine/roles/consumer.py
from dataclasses import dataclass

from bamengine.core import Role
from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Consumer(Role):
    """
    Consumer role for agents.

    Represents an entity that is able to receive income and trade it for consumer goods.
    """

    income: Float1D
    savings: Float1D
    income_to_spend: Float1D
    propensity: Float1D
    largest_prod_prev: Idx1D

    # Scratch queues
    shop_visits_head: Idx1D
    shop_visits_targets: Idx2D  # shape (n_households, Z)
