# src/bamengine/roles/producer.py
from dataclasses import field
from typing import Optional

from bamengine.core.decorators import role
from bamengine.typing import Bool1D, Float1D


@role
class Producer:
    """
    Producer role for agents.

    Represents an entity that is able to produce consumer goods
    using labor and trade them for an offered price.
    """

    production: Float1D
    inventory: Float1D
    expected_demand: Float1D
    desired_production: Float1D
    labor_productivity: Float1D
    breakeven_price: Float1D
    price: Float1D

    # Scratch buffers (optional for performance)
    prod_shock: Optional[Float1D] = field(default=None, repr=False)
    prod_mask_up: Optional[Bool1D] = field(default=None, repr=False)
    prod_mask_dn: Optional[Bool1D] = field(default=None, repr=False)
    price_shock: Optional[Float1D] = field(default=None, repr=False)
