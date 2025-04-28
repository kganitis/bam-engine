"""
Event-level orchestration for period-t production planning.

Calls:
  • decide_desired_production
  • decide_desired_labor
"""

from __future__ import annotations

from numpy.random import Generator

from bamengine.components.firm_labor import FirmLabor
from bamengine.components.firm_production import FirmProduction
from bamengine.systems.labor import decide_desired_labor
from bamengine.systems.production import decide_desired_production


def event_plan_production(
    prod: FirmProduction,
    lab: FirmLabor,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """Run Event 1 for all firms (pure, vectorised)."""
    decide_desired_production(prod, p_avg, h_rho, rng)
    decide_desired_labor(lab)
