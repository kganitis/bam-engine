"""BamModel: Mesa implementation of the baseline BAM model."""

from __future__ import annotations

import mesa

from comparison.runners.mesa.agents import Bank, Firm, Household
from comparison.runners.mesa.markets import run_credit_market, run_labor_market

EPS = 1e-9


class BamModel(mesa.Model):
    """The baseline BAM agent-based macroeconomic model in Mesa 3.x."""

    EPS = EPS

    def __init__(self, n_firms, n_households, n_banks, params, seed=0):
        super().__init__(rng=seed)
        self.n_firms = n_firms
        self.n_households = n_households
        self.n_banks = n_banks
        self.p = dict(params)
        self.period = 0

        # derived init values (REF §3)
        lp = self.p["labor_productivity"]
        price_init = self.p["price_init"]
        production_init = n_households * lp / n_firms
        wage_offer_init = price_init / 3.0
        net_worth_init = production_init * price_init * self.p["net_worth_ratio"]

        for _ in range(n_firms):
            Firm(
                self,
                price=price_init,
                production_prev=production_init,
                net_worth=net_worth_init,
                wage_offer=wage_offer_init,
                labor_productivity=lp,
            )
        for _ in range(n_households):
            Household(self, savings=self.p["savings_init"])
        for _ in range(n_banks):
            Bank(self, equity_base=self.p["equity_base_init"])

        # economy state (REF §3)
        self.avg_mkt_price = price_init
        self.min_wage = wage_offer_init * self.p["min_wage_ratio"]
        self.min_wage_rev_period = self.p["min_wage_rev_period"]
        self.avg_mkt_price_history = [float(price_init)]
        self.inflation_history = [0.0]
        self.collapsed = False

    @property
    def firms(self):
        return self.agents_by_type[Firm]

    @property
    def households(self):
        return self.agents_by_type[Household]

    @property
    def banks(self):
        return self.agents_by_type[Bank]

    def _planning(self) -> None:
        """Phase 1: planning (events 1-6)."""
        self.firms.do("decide_desired_production")
        self.firms.do("plan_breakeven_price")
        self.firms.do("plan_price")
        self.firms.do("decide_desired_labor")
        self.firms.do("decide_vacancies")
        self.firms.do("fire_excess_workers")

    def _calc_inflation(self) -> None:
        """Event 7: YoY inflation rate appended to inflation_history."""
        hist = self.avg_mkt_price_history
        if len(hist) <= 4:
            self.inflation_history.append(0.0)
            return
        p_now = hist[-1]
        p_prev = hist[-5]
        if p_prev <= 0:
            self.inflation_history.append(0.0)
        else:
            self.inflation_history.append((p_now - p_prev) / p_prev)

    def _adjust_min_wage(self) -> None:
        """Event 8: periodically index minimum wage to inflation."""
        m = self.min_wage_rev_period
        hist_len = len(self.avg_mkt_price_history)
        if hist_len <= m:
            return
        if (hist_len - 1) % m != 0:
            return
        inflation = self.inflation_history[-1]
        self.min_wage *= 1.0 + inflation
        # Bump employed workers below the new floor.
        for h in self.households:
            if h.employed and h.wage < self.min_wage:
                h.wage = self.min_wage

    def _labor_market(self) -> None:
        """Phase 2: labor market (events 7-12)."""
        self._calc_inflation()
        self._adjust_min_wage()
        self.firms.do("decide_wage_offer")
        self.households.do("decide_firms_to_apply")
        run_labor_market(self)
        self.firms.do("calc_wage_bill")

    def _credit_market(self) -> None:
        """Phase 3: credit market (events 13-19)."""
        # Purge previous-period loans (retained through planning/labor for breakeven).
        for f in self.firms:
            f.loans = []
        self.banks.do("decide_credit_supply")
        self.banks.do("decide_interest_rate")
        self.firms.do("decide_credit_demand")
        self.firms.do("calc_fragility")
        for f in self.firms:
            f.prepare_loan_applications(self)
        run_credit_market(self)
        for f in self.firms:
            f.fire_workers_for_gap(self)

    def step(self):
        """Execute one simulation period."""
        if self.collapsed:
            return
        self.period += 1
        self._planning()
        self._labor_market()
        self._credit_market()
