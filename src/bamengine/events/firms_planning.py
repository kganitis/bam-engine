from numpy.random import Generator

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.systems.planning import (
    decide_desired_labor,
    decide_desired_production,
    decide_vacancies,
)


def firms_planning(
    prod: FirmProductionPlan,
    lab: FirmLaborPlan,
    vac: FirmVacancies,
    *,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """Event 1: plan production + labor + vacancy posting (vectorised)."""
    decide_desired_production(prod, p_avg, h_rho, rng)
    decide_desired_labor(lab)
    decide_vacancies(vac)
