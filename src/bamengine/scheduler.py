from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng

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
from bamengine.components.economy import Economy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Scheduler:
    """Very small driver that owns the component arrays and calls systems."""

    rng: Generator
    prod: FirmProductionPlan
    lab: FirmLaborPlan
    h_rho: float  # max growth rate for production
    vac: FirmVacancies

    @classmethod
    def init(cls, n_firms: int, h_rho: float, seed: int = 0) -> "Scheduler":
        rng = default_rng(seed)

        # ---- create shared / independent arrays ---------------------------
        # fundamental arrays
        price = np.full(n_firms, 1.5)  # currency units (float32)
        production = np.ones(n_firms)  # product units (float32)
        labor = np.zeros(n_firms, dtype=np.int64)  # worker units (int64)

        inventory = np.zeros_like(production)
        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labor_productivity = np.ones_like(production)
        desired_labor = np.zeros_like(labor)
        n_vacancies = np.zeros_like(labor)

        prod = FirmProductionPlan(
            price=price,
            inventory=inventory,
            prev_production=production,
            expected_demand=expected_demand,
            desired_production=desired_production,
        )
        lab = FirmLaborPlan(
            desired_production=desired_production,
            labor_productivity=labor_productivity,
            desired_labor=desired_labor,
        )
        vac = FirmVacancies(
            desired_labor=desired_labor,
            current_labor=labor,
            n_vacancies=n_vacancies,
        )

        return cls(rng=rng, prod=prod, lab=lab, vac=vac, h_rho=h_rho)

    # ---------------------------------------------------------------------

    def step(self) -> None:
        """Run one period with the current systems."""
        p_avg = float(self.prod.price.mean())

        decide_desired_production(self.prod, p_avg, self.h_rho, self.rng)
        decide_desired_labor(self.lab)
        decide_vacancies(self.vac)

        # ---- minimal state advance (stub) --------------------------------
        # For repeated steps we advance prod.prev_production so the next loop
        # sees updated baseline output. We also jitter price/inventory so
        # the conditions vary between periods.
        self.prod.prev_production[:] = self.prod.desired_production
        self.prod.inventory[:] = self.rng.integers(0, 6, self.prod.inventory.shape)
        self.prod.price[:] *= self.rng.uniform(0.98, 1.02, self.prod.price.shape)

    # ---------------------------------------------------------------------

    # Convenience getters for tests / reports
    @property
    def mean_Yd(self) -> float:
        return float(self.prod.desired_production.mean())

    @property
    def mean_Ld(self) -> float:
        return float(self.lab.desired_labor.mean())
