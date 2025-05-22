# src/bamengine/components/producer.py
from dataclasses import dataclass, field

from bamengine.typing import Bool1D, Float1D


@dataclass(slots=True)
class Producer:
    """Dense state needed for production firms."""

    production: Float1D  # Y_i  (carried from t-1)
    inventory: Float1D  # S_i  (carried from t-1)
    expected_demand: Float1D  # D̂_i
    desired_production: Float1D  # Yd_i
    labor_productivity: Float1D  # a_i   (can change with R&D later)

    price: Float1D  # p_i  (carried from t-1)

    # ---- permanent scratch buffers ----
    prod_shock: Float1D | None = field(default=None, repr=False)
    prod_mask_up: Bool1D | None = field(default=None, repr=False)
    prod_mask_dn: Bool1D | None = field(default=None, repr=False)
    price_shock: Float1D | None = field(default=None, repr=False)
