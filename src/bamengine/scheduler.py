"""BAM Engine – tiny driver that wires components ↔ systems for 1 period."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, TypeAlias

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components.bank_credit import (
    BankCreditSupply,
    BankInterestRate,
    BankProvideLoan,
)
from bamengine.components.credit import LoanBook
from bamengine.components.economy import Economy
from bamengine.components.firm_credit import (
    FirmCreditDemand,
    FirmCreditMetrics,
    FirmLoanApplication,
)
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer, FirmWageBill
from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.components.worker_labor import WorkerJobSearch
from bamengine.systems.credit_market import banks_decide_credit_supply, \
    banks_decide_interest_rate, firms_decide_credit_demand, firms_calc_credit_metrics, \
    firms_prepare_loan_applications, firms_send_one_loan_app, banks_provide_loans
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_prepare_job_applications,
    workers_send_one_round, firms_calc_wage_bill,
)
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.typing import Float1D, Int1D

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
    wb: FirmWageBill

    cd: FirmCreditDemand
    cm: FirmCreditMetrics
    la: FirmLoanApplication
    cs: BankCreditSupply
    ir: BankInterestRate
    pl: BankProvideLoan

    ledger: LoanBook

    # global parameters
    h_rho: float  # max production-growth shock
    h_xi: float  # max wage-growth shock
    h_phi: float  # max bank operational costs shock
    max_M: int  # max job applications per unemployed worker
    max_H: int  # max loan applications per firm
    theta: int  # job contract length

    # --------------------------------------------------------------------- #
    #                            constructor                                #
    # --------------------------------------------------------------------- #
    @classmethod
    def init(
        cls,
        *,
        n_firms: int,
        n_households: int,
        n_banks: int,
        h_rho: float = 0.1,
        h_xi: float = 0.05,
        h_phi: float = 0.1,
        max_M: int = 4,
        max_H: int = 2,
        theta: int = 8,
        seed: int | np.random.Generator = 0,
    ) -> "Scheduler":
        rng = seed if isinstance(seed, np.random.Generator) else default_rng(seed)

        net_worth = np.full(n_firms, 10.0)
        price = np.full(n_firms, 1.5)
        production = np.ones(n_firms)
        inventory = np.zeros_like(production)
        wage = np.ones(n_firms)

        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labour_productivity = np.ones_like(production)
        desired_labor = np.zeros(n_firms, dtype=np.int64)
        current_labor = np.zeros_like(desired_labor)
        n_vacancies = np.zeros_like(desired_labor)

        wage_prev = np.copy(wage)
        wage_bill = np.zeros_like(wage)

        employed = np.zeros(n_households, dtype=np.int64)
        employer_prev = np.full(n_households, -1, dtype=np.int64)
        contract_expired = np.zeros_like(employed)
        fired = np.zeros_like(employed)

        job_apps_head = np.full(n_households, -1, dtype=np.int64)
        job_apps_targets = np.full((n_households, max_M), -1, dtype=np.int64)
        recv_job_apps_head = np.full(n_firms, -1, dtype=np.int64)
        recv_job_apps = np.full((n_firms, max_M), -1, dtype=np.int64)

        credit_demand = np.zeros_like(net_worth)
        rnd_intensity = np.ones(n_firms)
        projected_fragility = np.zeros(n_firms)

        equity_base = np.full(n_banks, 10_000.00)
        credit_supply = np.zeros_like(equity_base)
        interest_rate = np.zeros(n_banks)

        loan_apps_head = np.full(n_firms, -1, dtype=np.int64)
        loan_apps_targets = np.full((n_firms, max_H), -1, dtype=np.int64)
        recv_loan_apps_head = np.full(n_banks, -1, dtype=np.int64)
        recv_loan_apps = np.full((n_banks, max_H), -1, dtype=np.int64)


        # ---------- wrap into components ----------------------------------
        ec = Economy(
            min_wage=1.0,
            min_wage_rev_period=4,
            avg_mkt_price=1.5,
            avg_mkt_price_history=np.array([1.5]),
            r_bar=0.07,
            v=0.23,
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
            wage_offer=wage,
        )
        ws = WorkerJobSearch(
            employed=employed,
            employer_prev=employer_prev,
            contract_expired=contract_expired,
            fired=fired,
            apps_head=job_apps_head,
            apps_targets=job_apps_targets,
        )
        fh = FirmHiring(
            wage_offer=wage,  # shared view
            n_vacancies=n_vacancies,  # shared view
            current_labor=current_labor,  # shared view
            recv_apps_head=recv_job_apps_head,
            recv_apps=recv_job_apps,
        )
        wb = FirmWageBill(
            current_labor=current_labor,
            wage=wage,
            wage_bill=wage_bill,
        )
        cd = FirmCreditDemand(
            net_worth=net_worth,
            wage_bill=wage_bill,
            credit_demand=credit_demand,
        )
        cm = FirmCreditMetrics(
            credit_demand=credit_demand,
            net_worth=net_worth,
            rnd_intensity=rnd_intensity,
            projected_fragility=projected_fragility,
        )
        la = FirmLoanApplication(
            credit_demand=credit_demand,
            projected_fragility=projected_fragility,
            loan_apps_head=loan_apps_head,
            loan_apps_targets=loan_apps_targets,
        )
        cs = BankCreditSupply(
            equity_base=equity_base,
            credit_supply=credit_supply,
        )
        ir = BankInterestRate(
            interest_rate=interest_rate,
        )
        pl = BankProvideLoan(
            credit_supply=credit_supply,
            recv_apps_head=recv_loan_apps_head,
            recv_apps=recv_loan_apps,
        )
        ledger = LoanBook()

        return cls(
            rng=rng,
            ec=ec,
            prod=prod,
            lab=lab,
            vac=vac,
            fw=fw,
            ws=ws,
            fh=fh,
            wb=wb,
            cd=cd,
            cm=cm,
            la=la,
            cs=cs,
            ir=ir,
            pl=pl,
            ledger=ledger,
            h_rho=h_rho,
            h_xi=h_xi,
            h_phi=h_phi,
            max_M=max_M,
            max_H=max_H,
            theta=theta,
        )

    # --------------------------------------------------------------------- #
    #                               one step                                #
    # --------------------------------------------------------------------- #
    def step(
        self,
        *,
        before_planning: SchedulerHook = None,
        before_labor_market: SchedulerHook = None,
        before_credit_market: SchedulerHook = None,
        after_stub: SchedulerHook = None,
    ) -> None:
        """Advance the economy by one period.

        Parameters
        ----------
        before_planning : callable(self), optional
            Called **before** planning systems run.
        before_labor_market: callable(self) optional
            Called **before** labor market systems run.
        before_credit_market: callable(self), optional
            Called **after** labor market systems finish.
        after_stub: callable(self), optional
            Called **after** stub bookkeeping.
        """

        # Optional hook
        if before_planning is not None:
            before_planning(self)

        # ===== Event 1 – firms plan =======================================
        avg_mkt_price = float(self.prod.price.mean())

        firms_decide_desired_production(
            self.prod, p_avg=avg_mkt_price, h_rho=self.h_rho, rng=self.rng
        )
        firms_decide_desired_labor(self.lab)
        firms_decide_vacancies(self.vac)
        # ==================================================================

        # Optional hook
        if before_labor_market is not None:
            before_labor_market(self)

        # ===== Event 2 – labor-market =====================================
        adjust_minimum_wage(self.ec)
        firms_decide_wage_offer(
            self.fw, w_min=self.ec.min_wage, h_xi=self.h_xi, rng=self.rng
        )
        workers_prepare_job_applications(
            self.ws, self.fw, max_M=self.max_M, rng=self.rng
        )
        for _ in range(self.max_M):  # round‐robin M times
            workers_send_one_round(self.ws, self.fh)
            firms_hire_workers(self.ws, self.fh, contract_theta=self.theta)
        firms_calc_wage_bill(self.wb)
        # ==================================================================

        # Optional hook
        if before_credit_market is not None:
            before_credit_market(self)

        # ===== Event 3 – credit-market ====================================
        banks_decide_credit_supply(self.cs, v=self.ec.v)
        banks_decide_interest_rate(
            self.ir, r_bar=self.ec.r_bar, h_phi=self.h_phi, rng=self.rng,
        )
        firms_decide_credit_demand(self.cd)
        firms_calc_credit_metrics(self.cm)
        firms_prepare_loan_applications(
            self.la, self.ir, max_H=self.max_H, rng=self.rng
        )
        for _ in range(self.max_H):  # round‐robin H times
            firms_send_one_loan_app(self.la, self.pl)
            banks_provide_loans(self.la, self.ledger, self.pl, r_bar=self.ec.r_bar)
        # ==================================================================

        # Stub state advance
        import _testing
        _testing.advance_stub_state(self, avg_mkt_price)

        # Final hook – deterministic tweaks that must survive into t+1
        if after_stub is not None:
            after_stub(self)

    # --------------------------------------------------------------------- #
    #                               snapshot                                #
    # --------------------------------------------------------------------- #
    def snapshot(self, *, copy: bool = False) -> Dict[str, Float1D | Int1D | float]:
        """ "
        Return a read‑only view (or copy) of key state arrays.

        Parameters
        ----------
        copy : bool, default False
            * False – return **views**; cheap but mutation‑unsafe.
            * True  – return **copies** so the caller can mutate freely.
        """
        cp = np.copy if copy else lambda x: x  # cheap inline helper

        return {
            "price": cp(self.prod.price),
            "inventory": cp(self.prod.inventory),
            "desired_production": cp(self.prod.desired_production),
            "desired_labor": cp(self.lab.desired_labor),
            "current_labor": cp(self.fh.current_labor),
            "min_wage": float(self.ec.min_wage),
            "avg_price": float(self.ec.avg_mkt_price),
        }

    # ------------------------------------------------------------------ #
    # convenience                                                        #
    # ------------------------------------------------------------------ #
    @property
    def mean_Yd(self) -> float:  # noqa: D401
        return float(self.prod.desired_production.mean())

    @property
    def mean_Ld(self) -> float:  # noqa: D401
        return float(self.lab.desired_labor.mean())
