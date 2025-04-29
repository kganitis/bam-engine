"""
Event-level orchestration for period-t production planning.

Calls:
  - decide_desired_production
  - decide_desired_labor
"""

from __future__ import annotations

from numpy.random import Generator

from bamengine.components.firm_plan import FirmLaborPlan, FirmProductionPlan
from bamengine.systems.planning import decide_desired_labor, decide_desired_production


def firms_planning(
    prod: FirmProductionPlan,
    lab: FirmLaborPlan,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """Run Event 1 for all firms (pure, vectorised)."""
    decide_desired_production(prod, p_avg, h_rho, rng)
    decide_desired_labor(lab)
