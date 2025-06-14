# src/bamengine/components/producer.py
from dataclasses import dataclass, field

from bamengine.typing import Bool1D, Float1D


@dataclass(slots=True)
class Producer:

    production: Float1D
    inventory: Float1D
    expected_demand: Float1D
    desired_production: Float1D
    labor_productivity: Float1D
    breakeven_price: Float1D
    price: Float1D

    # Scratch buffers
    prod_shock: Float1D | None = field(default=None, repr=False)
    prod_mask_up: Bool1D | None = field(default=None, repr=False)
    prod_mask_dn: Bool1D | None = field(default=None, repr=False)
    price_shock: Float1D | None = field(default=None, repr=False)
