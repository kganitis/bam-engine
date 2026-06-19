"""BamModel: Mesa implementation of the baseline BAM model."""

from __future__ import annotations

import mesa

from comparison.runners.mesa.agents import Bank, Firm, Household


class BamModel(mesa.Model):
    """The baseline BAM agent-based macroeconomic model in Mesa 3.x."""

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

    def step(self):
        """Execute one simulation period.

        For now, only increments the period counter.
        Phases added in later tasks.
        """
        if self.collapsed:
            return
        self.period += 1
