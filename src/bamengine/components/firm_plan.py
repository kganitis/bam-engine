from dataclasses import dataclass, field

from bamengine.typing import BoolA, FloatA, IntA


@dataclass(slots=True)
class FirmProductionPlan:
    """Dense state needed for production-planning."""

    price: FloatA  # current price P_i
    inventory: FloatA  # inventory S_i  (carried from t-1)
    prev_production: FloatA  # Y_{i,t-1}
    expected_demand: FloatA  # D̂_i  (output, same size)
    desired_production: FloatA  # Yd_i (output, same size)

    # ---- permanent scratch buffers ----
    prod_shock: FloatA | None = field(default=None, repr=False)
    prod_mask_up: BoolA | None = field(default=None, repr=False)
    prod_mask_dn: BoolA | None = field(default=None, repr=False)


@dataclass(slots=True)
class FirmLaborPlan:
    """Dense state needed for labor-demand decisions."""

    desired_production: FloatA  # Yd_i  (read-only here)
    labor_productivity: FloatA  # a_i   (can change with R&D later)
    desired_labor: IntA  # Ld_i  (output)


@dataclass(slots=True)
class FirmVacancies:
    """Dense state needed for calculating number of n_vacancies."""

    desired_labor: IntA  # Ld_i  (read-only)
    current_labor: IntA  # L_i   (input, updated by matching later)
    n_vacancies: IntA  # V_i   (output)
