"""Agent classes for the Mesa implementation of the baseline BAM model."""

from __future__ import annotations

import math
from typing import NamedTuple

import mesa


class Loan(NamedTuple):
    """A single loan record held by a Firm."""

    principal: float
    rate: float


class Firm(mesa.Agent):
    """Firm agent (Producer + Employer + Borrower roles)."""

    def __init__(
        self,
        model,
        *,
        price,
        production_prev,
        net_worth,
        wage_offer,
        labor_productivity,
    ):
        super().__init__(model)
        # Producer
        self.production = 0.0
        self.production_prev = production_prev
        self.inventory = 0.0
        self.expected_demand = 1.0
        self.desired_production = 0.0
        self.labor_productivity = labor_productivity
        self.breakeven_price = price
        self.price = price
        # Employer
        self.desired_labor = 0
        self.current_labor = 0
        self.wage_offer = wage_offer
        self.wage_bill = 0.0
        self.n_vacancies = 0
        self.total_funds = net_worth
        # Borrower
        self.net_worth = net_worth
        self.credit_demand = 0.0
        self.projected_fragility = 0.0
        self.gross_profit = 0.0
        self.net_profit = 0.0
        self.retained_profit = 0.0
        # scratch for market rounds (set during phases)
        self.loan_apps = []  # ranked bank ids (objects), consumed per round
        self.loans: list[Loan] = []  # outstanding loans
        self.employees = (
            set()
        )  # Household objects employed here (maintained on hire/fire)

    # ------------------------------------------------------------------
    # Planning phase methods (events 1-6)
    # ------------------------------------------------------------------

    def decide_desired_production(self) -> None:
        """Event 1: set production target based on inventory and market position."""
        model = self.model
        p_avg = model.avg_mkt_price
        h_rho = model.p["h_rho"]

        self.production = 0.0
        shock = model.random.uniform(0, h_rho)
        up = self.inventory == 0 and self.price >= p_avg
        dn = self.inventory > 0 and self.price < p_avg

        self.expected_demand = self.production_prev
        if up:
            self.expected_demand *= 1 + shock
        elif dn:
            self.expected_demand *= 1 - shock

        self.desired_production = self.expected_demand

    def plan_breakeven_price(self) -> None:
        """Event 2: compute breakeven price from wage bill and loan interest."""
        eps = self.model.EPS
        interest = sum(loan.rate * loan.principal for loan in self.loans)
        self.breakeven_price = (self.wage_bill + interest) / max(
            self.desired_production, eps
        )

    def plan_price(self) -> None:
        """Event 3: adjust price based on inventory and market position."""
        model = self.model
        p_avg = model.avg_mkt_price
        h_eta = model.p["h_eta"]

        shock = model.random.uniform(0, h_eta)
        up = self.inventory == 0 and self.price < p_avg
        dn = self.inventory > 0 and self.price >= p_avg

        if up:
            self.price *= 1 + shock
            self.price = max(self.price, self.breakeven_price)
        elif dn:
            self.price *= 1 - shock
            self.price = max(self.price, self.breakeven_price)

    def decide_desired_labor(self) -> None:
        """Event 4: compute desired labor as ceil(desired_production / labor_productivity)."""
        eps = self.model.EPS
        self.desired_labor = math.ceil(
            self.desired_production / max(self.labor_productivity, eps)
        )

    def decide_vacancies(self) -> None:
        """Event 5: open vacancies = max(desired_labor - current_labor, 0)."""
        self.n_vacancies = max(self.desired_labor - self.current_labor, 0)

    def fire_excess_workers(self) -> None:
        """Event 6: fire excess workers chosen uniformly at random."""
        excess = self.current_labor - self.desired_labor
        if excess <= 0:
            return

        k = min(excess, len(self.employees))
        victims = self.model.random.sample(list(self.employees), k)
        for h in victims:
            h.employer = None
            h.employer_prev = self
            h.wage = 0.0
            h.periods_left = 0
            h.contract_expired = False
            h.fired = True
            self.employees.discard(h)
            self.current_labor -= 1

    # ------------------------------------------------------------------
    # Credit market methods (events 15-17, 19)
    # ------------------------------------------------------------------

    def decide_credit_demand(self) -> None:
        """Event 15: credit demand = max(wage_bill - total_funds, 0)."""
        self.credit_demand = max(self.wage_bill - self.total_funds, 0.0)

    def calc_fragility(self) -> None:
        """Event 16: projected fragility = credit_demand / net_worth (or max_leverage if NW<=0)."""
        max_leverage = self.model.p["max_leverage"]
        if self.net_worth > 0:
            self.projected_fragility = self.credit_demand / self.net_worth
        else:
            self.projected_fragility = max_leverage

    def prepare_loan_applications(self, model) -> None:
        """Event 17: sample H_eff banks with supply, sort by interest_rate ASC."""
        if self.credit_demand <= 0:
            self.loan_apps = []
            return
        max_H = model.p["max_H"]
        lenders = [b for b in model.banks if b.credit_supply > 0]
        H_eff = min(max_H, len(lenders))
        if H_eff == 0:
            self.loan_apps = []
            return
        sample = model.random.sample(lenders, H_eff)
        sample.sort(key=lambda b: b.interest_rate)
        self.loan_apps = sample

    def fire_workers_for_gap(self, model) -> None:
        """Event 19: fire workers until wage_bill is covered by total_funds."""
        if self.wage_bill <= self.total_funds:
            return
        gap = self.wage_bill - self.total_funds
        employees_list = list(self.employees)
        model.random.shuffle(employees_list)
        cumulative = 0.0
        for h in employees_list:
            wage = h.wage
            h.employer = None
            h.employer_prev = self
            h.wage = 0.0
            h.periods_left = 0
            h.contract_expired = False
            h.fired = True
            self.employees.discard(h)
            self.current_labor -= 1
            self.wage_bill -= wage
            cumulative += wage
            if cumulative >= gap:
                break

    # ------------------------------------------------------------------
    # Labor market methods (events 9, 12)
    # ------------------------------------------------------------------

    def decide_wage_offer(self) -> None:
        """Event 9: set wage offer with random markup (zero if no vacancies)."""
        model = self.model
        h_xi = model.p["h_xi"]

        shock = model.random.uniform(0, h_xi) if self.n_vacancies > 0 else 0.0
        self.wage_offer *= 1.0 + shock
        self.wage_offer = max(self.wage_offer, model.min_wage)

    def calc_wage_bill(self) -> None:
        """Event 12: sum wages of all employees."""
        self.wage_bill = sum(w.wage for w in self.employees)


class Household(mesa.Agent):
    """Household agent (Worker + Consumer + Shareholder roles)."""

    def __init__(self, model, *, savings):
        super().__init__(model)
        # Worker
        self.employer = None  # Firm object or None (unemployed)
        self.employer_prev = None
        self.wage = 0.0
        self.periods_left = 0
        self.contract_expired = False
        self.fired = False
        # Consumer
        self.income = 0.0
        self.savings = savings
        self.income_to_spend = 0.0
        self.propensity = 0.0
        self.largest_prod_prev = None  # Firm object or None (none visited)
        self.job_apps = []
        self.shop_visits = []
        # Shareholder
        self.dividends = 0.0

    @property
    def employed(self) -> bool:
        """Derived property: whether this household is employed."""
        return self.employer is not None

    # ------------------------------------------------------------------
    # Labor market methods (event 10)
    # ------------------------------------------------------------------

    def decide_firms_to_apply(self) -> None:
        """Event 10: build ranked application queue for unemployed workers.

        Only unemployed workers participate.  Samples min(max_M, |pool|) firms
        without replacement, sorts by wage_offer DESC, and applies the loyalty
        rule (previous employer to front if contract expired without being fired
        and employer_prev is still in the pool).
        """
        if self.employed:
            return

        model = self.model
        max_M = model.p["max_M"]

        pool = list(model.firms)
        M_eff = min(max_M, len(pool))
        sample = model.random.sample(pool, M_eff)

        # Sort by wage_offer DESC.
        sample.sort(key=lambda f: f.wage_offer, reverse=True)

        # Loyalty: move employer_prev to front if eligible.
        if self.contract_expired and not self.fired and self.employer_prev in pool:
            prev = self.employer_prev
            if prev in sample:
                sample.remove(prev)
            else:
                # Drop last to keep M_eff length.
                if len(sample) == M_eff:
                    sample = sample[: M_eff - 1]
            sample.insert(0, prev)

        self.job_apps = sample
        self.contract_expired = False
        self.fired = False


class Bank(mesa.Agent):
    """Bank agent (Lender role)."""

    def __init__(self, model, *, equity_base):
        super().__init__(model)
        self.equity_base = equity_base
        self.credit_supply = 0.0
        self.interest_rate = 0.0
        self.opex_shock = 0.0

    # ------------------------------------------------------------------
    # Credit market methods (events 13-14)
    # ------------------------------------------------------------------

    def decide_credit_supply(self) -> None:
        """Event 13: credit_supply = max(equity_base / v, 0)."""
        v = self.model.p["v"]
        self.credit_supply = max(self.equity_base / v, 0.0)

    def decide_interest_rate(self) -> None:
        """Event 14: draw opex_shock ~ U(0, h_phi); interest_rate = r_bar*(1+opex_shock)."""
        model = self.model
        h_phi = model.p["h_phi"]
        r_bar = model.p["r_bar"]
        self.opex_shock = model.random.uniform(0, h_phi)
        self.interest_rate = r_bar * (1.0 + self.opex_shock)
