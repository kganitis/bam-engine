"""BAM Engine – tiny driver that wires components ↔ systems for 1 period."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, TypeAlias

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer
from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.components.worker_labor import WorkerJobSearch
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
from bamengine.typing import FloatA, IntA

__all__ = [
    "Scheduler",
]

log = logging.getLogger(__name__)

# A hook function that receives the Scheduler and returns nothing
SchedulerHook: TypeAlias = Callable[["Scheduler"], None] | None


@dataclass(slots=True)
class Scheduler:
    """
    Facade that drives a BAM economy for one or more periods.
    """

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
        seed: int | np.random.Generator = 0,
    ) -> "Scheduler":
        rng = seed if isinstance(seed, np.random.Generator) else default_rng(seed)

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

        work_shock = np.empty_like(price, dtype=np.float64)
        work_mask_up = np.empty_like(price, dtype=np.bool_)
        work_mask_dn = np.empty_like(price, dtype=np.bool_)

        # ---------- wrap into components ----------------------------------
        ec = Economy(
            min_wage=1.0,
            min_wage_rev_period=4,
            avg_mrkt_price=1.5,
            avg_mrkt_price_history=np.array([1.5]),
        )

        prod = FirmProductionPlan(
            price=price,
            inventory=inventory,
            prev_production=production,
            expected_demand=expected_demand,
            desired_production=desired_production,
            work_shock=work_shock,
            work_mask_up=work_mask_up,
            work_mask_dn=work_mask_dn,
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
        before_planning: SchedulerHook = None,
        after_labor_market: SchedulerHook = None,
        after_stub: SchedulerHook = None,
    ) -> None:
        """Advance the economy by one period.

        Parameters
        ----------
        before_planning : callable(self), optional
            Called **before** planning systems run.
        after_labor_market: callable(self), optional
            Called **after** labor market systems finish.
        after_stub: callable(self), optional
            Called **after** stub bookkeeping.
        """

        # Optional hook
        if before_planning is not None:
            before_planning(self)

        # ===== Event 1 – firms plan =======================================
        avg_mrkt_price = float(self.prod.price.mean())

        firms_decide_desired_production(self.prod, avg_mrkt_price, self.h_rho, self.rng)
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

        # Stub state advance
        import _testing

        _testing.advance_stub_state(self, avg_mrkt_price)

        # Final hook – deterministic tweaks that must survive into t+1
        if after_stub is not None:
            after_stub(self)

    # --------------------------------------------------------------------- #
    #                               snapshot                                #
    # --------------------------------------------------------------------- #
    def snapshot(self, *, deep: bool = False) -> Dict[str, FloatA | IntA | float]:
        """ "
        Return a read‑only view (or copy) of key state arrays.

        Parameters
        ----------
        deep : bool, default False
            * False – return **views**; cheap but mutation‑unsafe.
            * True  – return **copies** so the caller can mutate freely.
        """
        cp = np.copy if deep else lambda x: x  # cheap inline helper

        return {
            "price": cp(self.prod.price),
            "inventory": cp(self.prod.inventory),
            "desired_production": cp(self.prod.desired_production),
            "desired_labor": cp(self.lab.desired_labor),
            "current_labor": cp(self.fh.current_labor),
            "min_wage": float(self.ec.min_wage),
            "avg_price": float(self.ec.avg_mrkt_price),
        }

    # ------------------------------------------------------------------ #
    # convenience                                                       #
    # ------------------------------------------------------------------ #
    @property
    def mean_Yd(self) -> float:  # noqa: D401
        return float(self.prod.desired_production.mean())

    @property
    def mean_Ld(self) -> float:  # noqa: D401
        return float(self.lab.desired_labor.mean())
