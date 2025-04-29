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
from bamengine.events.firms_planning import firms_planning

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
    def init_random(cls, n_firms: int, h_rho: float, seed: int = 0) -> "Scheduler":
        rng = default_rng(seed)

        # ---- create shared / independent arrays ---------------------------
        price = np.full(n_firms, 1.5)
        inventory = rng.integers(0, 6, n_firms).astype(float)
        prev_production = np.full(n_firms, 10.0)
        current_labor = np.zeros(n_firms, dtype=np.int64)

        desired_production = np.zeros(n_firms)  # shared ndarray

        prod = FirmProductionPlan(
            price=price,
            inventory=inventory,
            prev_production=prev_production,
            expected_demand=np.zeros(n_firms),
            desired_production=desired_production,
        )

        lab = FirmLaborPlan(
            desired_production=desired_production,  # share!
            labor_productivity=np.ones(n_firms),
            desired_labor=np.zeros(n_firms, dtype=np.int64),
        )

        vac = FirmVacancies(
            desired_labor=lab.desired_labor,
            current_labor=current_labor,
            n_vacancies=np.zeros(n_firms, dtype=np.int64),
        )

        return cls(rng=rng, prod=prod, lab=lab, vac=vac, h_rho=h_rho)

    # ---------------------------------------------------------------------

    def step(self) -> None:
        """Run one period with the current systems."""
        p_avg = float(self.prod.price.mean())
        firms_planning(
            self.prod, self.lab, self.vac, p_avg=p_avg, h_rho=self.h_rho, rng=self.rng
        )

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
