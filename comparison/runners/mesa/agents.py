"""Agent classes for the Mesa implementation of the baseline BAM model."""

from __future__ import annotations

import mesa


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
        self.employees = (
            set()
        )  # Household objects employed here (maintained on hire/fire)


class Household(mesa.Agent):
    """Household agent (Worker + Consumer + Shareholder roles)."""

    def __init__(self, model, *, savings):
        super().__init__(model)
        # Worker
        self.employer = -1  # int: Firm object or -1 (unemployed)
        self.employer_prev = -1
        self.wage = 0.0
        self.periods_left = 0
        self.contract_expired = False
        self.fired = False
        # Consumer
        self.income = 0.0
        self.savings = savings
        self.income_to_spend = 0.0
        self.propensity = 0.0
        self.largest_prod_prev = -1  # int: Firm object or -1 (none)
        self.job_apps = []
        self.shop_visits = []
        # Shareholder
        self.dividends = 0.0

    @property
    def employed(self) -> bool:
        """Derived property: whether this household is employed."""
        return self.employer != -1


class Bank(mesa.Agent):
    """Bank agent (Lender role)."""

    def __init__(self, model, *, equity_base):
        super().__init__(model)
        self.equity_base = equity_base
        self.credit_supply = 0.0
        self.interest_rate = 0.0
        self.opex_shock = 0.0
