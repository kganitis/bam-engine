from dataclasses import dataclass, field

from bamengine.typing import Bool1D, Float1D, Int1D


@dataclass(slots=True)
class FirmProductionPlan:
    """Dense state needed for production-planning."""

    price: Float1D  # current price P_i
    inventory: Float1D  # inventory S_i  (carried from t-1)
    prev_production: Float1D  # Y_{i,t-1}
    expected_demand: Float1D  # D̂_i  (output, same size)
    desired_production: Float1D  # Yd_i (output, same size)

    # ---- permanent scratch buffers ----
    prod_shock: Float1D | None = field(default=None, repr=False)
    prod_mask_up: Bool1D | None = field(default=None, repr=False)
    prod_mask_dn: Bool1D | None = field(default=None, repr=False)


@dataclass(slots=True)
class FirmLaborPlan:
    """Dense state needed for labor-demand decisions."""

    desired_production: Float1D  # Yd_i  (read-only here)
    labor_productivity: Float1D  # a_i   (can change with R&D later)
    desired_labor: Int1D  # Ld_i  (output)


@dataclass(slots=True)
class FirmVacancies:
    """Dense state needed for calculating number of n_vacancies."""

    desired_labor: Int1D  # Ld_i  (read-only)
    current_labor: Int1D  # L_i   (input, updated by matching later)
    n_vacancies: Int1D  # V_i   (output)
