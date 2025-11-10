from bamengine.core.decorators import role
from bamengine.typing import Float1D, Idx1D, Idx2D


@role
class Consumer:
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
