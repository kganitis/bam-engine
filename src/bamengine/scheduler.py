"""BAM Engine – tiny driver that wires components ↔ systems for 1 period."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer
from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.components.worker_job import WorkerJobSearch
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_prepare_applications,
    workers_send_one_round,
)
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Scheduler:
    """Owns all arrays and calls the systems in canonical BAM order."""

    rng: Generator
    ec: Economy

    prod: FirmProductionPlan
    lab: FirmLaborPlan
    vac: FirmVacancies

    fw: FirmWageOffer
    ws: WorkerJobSearch
    fh: FirmHiring

    # global parameters
    h_rho: float  # max output-growth shock
    h_xi: float  # max wage-growth shock
    max_M: int  # applications per employed worker
    theta: int  # contract length (not yet stored per worker)

    # --------------------------------------------------------------------- #
    #                            constructor                                #
    # --------------------------------------------------------------------- #
    @classmethod
    def init(
        cls,
        *,
        n_firms: int,
        n_households: int,
        h_rho: float = 0.1,
        h_xi: float = 0.05,
        max_M: int = 4,
        theta: int = 8,
        seed: int = 0,
    ) -> "Scheduler":
        rng = default_rng(seed)

        price = np.full(n_firms, 1.5)
        production = np.ones(n_firms)
        inventory = np.zeros_like(production)

        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labour_productivity = np.ones_like(production)
        desired_labor = np.zeros(n_firms, dtype=np.int64)
        current_labor = np.zeros_like(desired_labor)
        n_vacancies = np.zeros_like(desired_labor)

        wage_prev = np.full(n_firms, 1.0)
        wage_offer = np.zeros_like(price)

        employed = np.zeros(n_households, dtype=np.int64)
        employer_prev = np.full(n_households, -1, dtype=np.int64)
        contract_expired = np.zeros_like(employed)
        fired = np.zeros_like(employed)

        apps_head = np.full(n_households, -1, dtype=np.int64)
        apps_targets = np.full((n_households, max_M), -1, dtype=np.int64)
        recv_apps_head = np.full(n_firms, -1, dtype=np.int64)
        recv_apps = np.full((n_firms, max_M), -1, dtype=np.int64)

        # ---------- wrap into components ----------------------------------
        ec = Economy(
            min_wage=1.0,
            min_wage_rev_period=4,
            avg_mrkt_price_history=np.array([1.5]),
        )

        prod = FirmProductionPlan(
            price=price,
            inventory=inventory,
            prev_production=production,
            expected_demand=expected_demand,
            desired_production=desired_production,
        )
        lab = FirmLaborPlan(
            desired_production=desired_production,  # shared view
            labor_productivity=labour_productivity,
            desired_labor=desired_labor,
        )
        vac = FirmVacancies(
            desired_labor=desired_labor,  # shared view
            current_labor=current_labor,
            n_vacancies=n_vacancies,
        )
        fw = FirmWageOffer(
            wage_prev=wage_prev,
            n_vacancies=n_vacancies,  # shared view
            wage_offer=wage_offer,
        )
        ws = WorkerJobSearch(
            employed=employed,
            employer_prev=employer_prev,
            contract_expired=contract_expired,
            fired=fired,
            apps_head=apps_head,
            apps_targets=apps_targets,
        )
        fh = FirmHiring(
            wage_offer=wage_offer,  # shared view
            n_vacancies=n_vacancies,  # shared view
            current_labor=current_labor,  # shared view
            recv_apps_head=recv_apps_head,
            recv_apps=recv_apps,
        )

        return cls(
            rng=rng,
            ec=ec,
            prod=prod,
            lab=lab,
            vac=vac,
            fw=fw,
            ws=ws,
            fh=fh,
            h_rho=h_rho,
            h_xi=h_xi,
            max_M=max_M,
            theta=theta,
        )

    # --------------------------------------------------------------------- #
    #                               one step                                #
    # --------------------------------------------------------------------- #
    def step(
        self,
        *,
        before_planning: callable | None = None,
        after_labor_market: callable | None = None,
        after_stub: callable | None = None,
    ) -> None:
        """Advance the economy by one period.

        Parameters
        ----------
        before_planning : callable(self) | None, optional
            Called **before** planning systems run.
        after_labor_market  : callable(self) | None, optional
            Called **after** labor market systems finish.
        after_stub:  callable(self) | None, optional
            Called **after** stub bookkeeping.
        """

        # Optional hook
        if before_planning is not None:
            before_planning(self)

        # ===== Event 1 – firms plan =======================================
        p_avg = float(self.prod.price.mean())

        firms_decide_desired_production(self.prod, p_avg, self.h_rho, self.rng)
        firms_decide_desired_labor(self.lab)
        firms_decide_vacancies(self.vac)

        # ===== Event 2 – labor-market =====================================
        adjust_minimum_wage(self.ec)
        firms_decide_wage_offer(
            self.fw, w_min=self.ec.min_wage, h_xi=self.h_xi, rng=self.rng
        )
        workers_prepare_applications(self.ws, self.fw, max_M=self.max_M, rng=self.rng)
        for _ in range(self.max_M):  # round‐robin M times
            workers_send_one_round(self.ws, self.fh)
            firms_hire_workers(self.ws, self.fh, contract_theta=self.theta)

        # Optional hook
        if after_labor_market is not None:
            after_labor_market(self)

        self._advance_stub_state(p_avg)

        # Final hook – deterministic tweaks that must survive into t+1
        if after_stub is not None:
            after_stub(self)

    # --------------------------------------------------------------------- #
    #                         stub state advance                            #
    # --------------------------------------------------------------------- #
    def _advance_stub_state(self, p_avg: float) -> None:
        """
        Temporary helper used while future events are not implemented.

        It performs three minimal tasks so the simulation can run multiple
        periods and the tests have fresh – but *plausible* – arrays:

        1. Append the realised average market price to the global series.
        2. Roll `prev_production` forward to last period’s `desired_production`.
        3. Jitter inventory and price so future periods don’t see constants.
        """
        # add price to history (needed for minimum-wage inflation rule)
        self.ec.avg_mrkt_price_history = np.append(
            self.ec.avg_mrkt_price_history, p_avg
        )

        # carry forward production level
        self.prod.prev_production[:] = self.prod.desired_production

        # stub: random new inventory and mild price noise
        self.prod.inventory[:] = self.rng.integers(0, 6, size=self.prod.inventory.shape)
        self.prod.price[:] *= self.rng.uniform(0.98, 1.02, size=self.prod.price.shape)

    # ------------------------------------------------------------------ #
    # convenience                                                       #
    # ------------------------------------------------------------------ #
    @property
    def mean_Yd(self) -> float:  # noqa: D401
        return float(self.prod.desired_production.mean())

    @property
    def mean_Ld(self) -> float:  # noqa: D401
        return float(self.lab.desired_labor.mean())
