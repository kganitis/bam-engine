# src/bamengine/scheduler.py
"""BAM Engine – tiny driver that wires components ↔ systems for 1 period."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import (
    Borrower,
    Consumer,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.systems.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
from bamengine.systems.goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_visit_one_round,
)
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    firms_calc_wage_bill,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.systems.production import (
    firms_decide_price,
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)
from bamengine.systems.revenue import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)

__all__ = [
    "Scheduler",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
#                               Scheduler                               #
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class Scheduler:
    """
    Facade that drives a BAM economy for one or more periods.
    """

    rng: Generator
    ec: Economy
    prod: Producer
    wrk: Worker
    emp: Employer
    bor: Borrower
    lend: Lender
    con: Consumer
    lb: LoanBook

    # global parameters
    n_firms: int
    n_households: int
    n_banks: int
    h_rho: float  # max production-growth shock
    h_xi: float  # max wage-growth shock
    h_phi: float  # max bank operational costs shock
    h_eta: float  # max price-growth shock
    max_M: int  # max job applications per unemployed worker
    max_H: int  # max loan applications per firm
    max_Z: int  # max firm visits per consumer
    theta: int  # job contract length
    beta: float  # propensity to consume parameter
    delta: float  # dividend payout ratio (DPR)

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
        h_eta: float = 0.1,
        max_M: int = 4,
        max_H: int = 2,
        max_Z: int = 2,
        theta: int = 8,
        beta: float = 0.87,
        delta: float = 0.15,
        seed: int | Generator = 0,
    ) -> "Scheduler":
        rng = seed if isinstance(seed, Generator) else default_rng(seed)

        # finance vectors
        net_worth = np.full(n_firms, 10.0)
        total_funds = np.copy(net_worth)
        rnd_intensity = np.ones(n_firms)

        # producer vectors
        production = np.ones(n_firms)
        inventory = np.zeros_like(production)
        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labor_productivity = np.ones_like(production)

        price = np.full(n_firms, 1.5)

        # employer vectors
        labor = np.zeros(n_firms, dtype=np.int64)
        desired_labor = np.zeros_like(labor)
        wage_offer = np.ones(n_firms)
        wage_bill = np.zeros_like(wage_offer)
        n_vacancies = np.zeros_like(desired_labor)
        recv_job_apps_head = np.full(n_firms, -1, dtype=np.int64)
        recv_job_apps = np.full((n_firms, max_M), -1, dtype=np.int64)

        # worker vectors
        employed = np.zeros(n_households, dtype=np.bool_)
        employer = np.full(n_households, -1, dtype=np.int64)
        employer_prev = np.full_like(employer, -1)
        periods_left = np.zeros(n_households, dtype=np.int64)
        contract_expired = np.zeros_like(employed)
        fired = np.zeros_like(employed)
        wage = np.zeros(n_households)
        job_apps_head = np.full(n_households, -1, dtype=np.int64)
        job_apps_targets = np.full((n_households, max_M), -1, dtype=np.int64)

        # borrower vectors
        credit_demand = np.zeros_like(net_worth)
        projected_fragility = np.zeros(n_firms)
        gross_profit = np.zeros_like(net_worth)
        net_profit = np.zeros_like(net_worth)
        retained_profit = np.zeros_like(net_worth)
        loan_apps_head = np.full(n_firms, -1, dtype=np.int64)
        loan_apps_targets = np.full((n_firms, max_H), -1, dtype=np.int64)

        # lender vectors
        equity_base = np.full(n_banks, 10_000.00)
        credit_supply = np.zeros_like(equity_base)
        interest_rate = np.zeros(n_banks)
        recv_loan_apps_head = np.full(n_banks, -1, dtype=np.int64)
        recv_loan_apps = np.full((n_banks, max_H), -1, dtype=np.int64)

        # consumer vectors
        income = np.zeros_like(wage)
        savings = np.zeros_like(wage)
        income_to_spend = np.zeros_like(wage)
        propensity = np.zeros(n_households)
        largest_prod_prev = np.full(n_households, -1, dtype=np.int64)
        shop_visits_head = np.full(n_households, -1, dtype=np.int64)
        shop_visits_targets = np.full((n_households, max_Z), -1, dtype=np.int64)

        # ---------- wrap into components ----------------------------------
        ec = Economy(
            min_wage=1.0,
            min_wage_rev_period=4,
            avg_mkt_price=1.5,
            avg_mkt_price_history=np.array([1.5]),
            r_bar=0.07,
            v=0.23,
        )
        prod = Producer(
            price=price,
            production=production,
            inventory=inventory,
            expected_demand=expected_demand,
            desired_production=desired_production,
            labor_productivity=labor_productivity,
        )
        wrk = Worker(
            employed=employed,
            employer=employer,
            employer_prev=employer_prev,
            wage=wage,
            periods_left=periods_left,
            contract_expired=contract_expired,
            fired=fired,
            job_apps_head=job_apps_head,
            job_apps_targets=job_apps_targets,
        )
        emp = Employer(
            desired_labor=desired_labor,
            current_labor=labor,
            wage_offer=wage_offer,
            wage_bill=wage_bill,
            n_vacancies=n_vacancies,
            total_funds=total_funds,
            recv_job_apps_head=recv_job_apps_head,
            recv_job_apps=recv_job_apps,
        )
        bor = Borrower(
            net_worth=net_worth,
            total_funds=total_funds,
            wage_bill=wage_bill,
            credit_demand=credit_demand,
            rnd_intensity=rnd_intensity,
            gross_profit=gross_profit,
            net_profit=net_profit,
            retained_profit=retained_profit,
            projected_fragility=projected_fragility,
            loan_apps_head=loan_apps_head,
            loan_apps_targets=loan_apps_targets,
        )
        lend = Lender(
            equity_base=equity_base,
            credit_supply=credit_supply,
            interest_rate=interest_rate,
            recv_apps_head=recv_loan_apps_head,
            recv_apps=recv_loan_apps,
        )
        lb = LoanBook()
        con = Consumer(
            income=income,
            savings=savings,
            income_to_spend=income_to_spend,
            propensity=propensity,
            largest_prod_prev=largest_prod_prev,
            shop_visits_head=shop_visits_head,
            shop_visits_targets=shop_visits_targets,
        )

        return cls(
            n_firms=n_firms,
            n_households=n_households,
            n_banks=n_banks,
            rng=rng,
            ec=ec,
            prod=prod,
            wrk=wrk,
            emp=emp,
            bor=bor,
            lend=lend,
            lb=lb,
            con=con,
            h_rho=h_rho,
            h_xi=h_xi,
            h_phi=h_phi,
            h_eta=h_eta,
            max_M=max_M,
            max_H=max_H,
            max_Z=max_Z,
            theta=theta,
            beta=beta,
            delta=delta,
        )

    # ------------------------------------------------------------------ #
    #                               one step                             #
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """Advance the economy by one period."""

        # ===== Event 1 – planning ======================================

        firms_decide_desired_production(
            self.prod, p_avg=self.ec.avg_mkt_price, h_rho=self.h_rho, rng=self.rng
        )
        firms_decide_desired_labor(self.prod, self.emp)
        firms_decide_vacancies(self.emp)

        # ===== Event 2 – labor-market ==================================

        adjust_minimum_wage(self.ec)
        firms_decide_wage_offer(
            self.emp, w_min=self.ec.min_wage, h_xi=self.h_xi, rng=self.rng
        )
        workers_decide_firms_to_apply(
            self.wrk, self.emp, max_M=self.max_M, rng=self.rng
        )
        for _ in range(self.max_M):
            workers_send_one_round(self.wrk, self.emp)
            firms_hire_workers(self.wrk, self.emp, theta=self.theta)
        firms_calc_wage_bill(self.emp)

        # ===== Event 3 – credit-market =================================

        banks_decide_credit_supply(self.lend, v=self.ec.v)
        banks_decide_interest_rate(
            self.lend, r_bar=self.ec.r_bar, h_phi=self.h_phi, rng=self.rng
        )
        firms_decide_credit_demand(self.bor)
        firms_calc_credit_metrics(self.bor)
        firms_prepare_loan_applications(
            self.bor, self.lend, max_H=self.max_H, rng=self.rng
        )
        for _ in range(self.max_H):
            firms_send_one_loan_app(self.bor, self.lend)
            banks_provide_loans(self.bor, self.lb, self.lend, r_bar=self.ec.r_bar)
        firms_fire_workers(self.emp, self.wrk, rng=self.rng)

        # ===== Event 4 – production ====================================

        firms_decide_price(
            self.prod,
            self.emp,
            self.lb,
            p_avg=self.ec.avg_mkt_price,
            h_eta=self.h_eta,
            rng=self.rng,
        )
        update_avg_mkt_price(self.ec, self.prod)

        firms_pay_wages(self.emp)
        workers_receive_wage(self.con, self.wrk)
        firms_run_production(self.prod, self.emp)
        workers_update_contracts(self.wrk, self.emp)

        # ===== Event 5 – goods-market ==================================

        _avg_sav = float(self.con.savings.mean())
        consumers_calc_propensity(self.con, avg_sav=_avg_sav, beta=self.beta)
        consumers_decide_income_to_spend(self.con)
        consumers_decide_firms_to_visit(
            self.con, self.prod, max_Z=self.max_Z, rng=self.rng
        )
        for _ in range(self.max_Z):
            consumers_visit_one_round(self.con, self.prod)
        consumers_finalize_purchases(self.con)

        # ===== Event 6 – revenue =======================================

        firms_collect_revenue(self.prod, self.bor)
        firms_validate_debt_commitments(self.bor, self.lend, self.lb)
        firms_pay_dividends(self.bor, delta=self.delta)
