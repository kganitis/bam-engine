"""BamModel: Mesa implementation of the baseline BAM model."""

from __future__ import annotations

import mesa

from comparison.runners.mesa.agents import Bank, Firm, Household
from comparison.runners.mesa.markets import (
    run_credit_market,
    run_goods_market,
    run_labor_market,
)

EPS = 1e-9


def trim_mean(values, pct=0.05):
    """Compute trimmed mean, removing pct from each tail."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = int(n * pct)
    trimmed = sorted_vals[k : n - k] if n - 2 * k > 0 else sorted_vals
    return sum(trimmed) / len(trimmed)


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

    def _update_avg_mkt_price(self) -> None:
        """Event 23: production-weighted average price; keep previous if result <= 0."""
        total_prod = 0.0
        weighted_sum = 0.0
        for f in self.firms:
            if f.production >= 1e-3:
                weighted_sum += f.price * f.production
                total_prod += f.production
        if total_prod > 0:
            new_price = weighted_sum / total_prod
        else:
            new_price = 0.0
        if new_price > 0:
            self.avg_mkt_price = new_price
        # else: keep previous avg_mkt_price
        self.avg_mkt_price_history.append(self.avg_mkt_price)

    def _production(self) -> None:
        """Phase 4: production (events 20-24)."""
        self.firms.do("pay_wages")
        self.households.do("receive_wage")
        self.firms.do("run_production")
        self._update_avg_mkt_price()
        # Snapshot employer before clearing so we can update firm bookkeeping.
        for h in list(self.households):
            h.update_contract()

    def _goods_market(self) -> None:
        """Phase 5: goods market (events 25-29)."""
        all_savings = [h.savings for h in self.households]
        avg_sav = (
            max(sum(all_savings) / len(all_savings), self.EPS)
            if all_savings
            else self.EPS
        )
        for h in self.households:
            h.calc_propensity(avg_sav)
        self.households.do("decide_income_to_spend")
        for h in self.households:
            h.decide_firms_to_visit(self)
        run_goods_market(self)
        self.households.do("finalize_purchases")

    def _pay_dividends(self) -> None:
        """Event 32: distribute dividends from profitable firms to all households."""
        delta = self.p["delta"]
        total_dividends = 0.0
        for f in self.firms:
            retained = f.net_profit
            if f.net_profit > 0:
                retained *= 1.0 - delta
            dividends_paid = f.net_profit - retained
            f.total_funds -= dividends_paid
            f.retained_profit = retained
            total_dividends += dividends_paid

        n_households = len(self.households)
        div_per_hh = total_dividends / n_households if n_households > 0 else 0.0
        for h in self.households:
            h.savings += div_per_hh
            h.dividends = div_per_hh

    def _mark_bankrupt_and_replace(self) -> None:
        """Events 34-37: detect and replace bankrupt firms and banks."""
        eps = self.EPS

        # Identify exiting firms: insolvent OR ghost (production_prev <= 0).
        exiting_firms = [
            f for f in self.firms if f.net_worth < eps or f.production_prev <= eps
        ]

        # Fire all employees of exiting firms and clear their loans.
        for f in exiting_firms:
            for h in list(f.employees):
                h.employer = None
                h.employer_prev = None
                h.wage = 0.0
                h.periods_left = 0
                h.contract_expired = False
                h.fired = False
            f.employees.clear()
            f.current_labor = 0
            f.wage_bill = 0.0
            f.loans = []

        # Identify exiting banks: negative equity.
        exiting_banks = [b for b in self.banks if b.equity_base < eps]

        # Drop loans issued by exiting banks (from surviving firms).
        if exiting_banks:
            exiting_bank_set = set(exiting_banks)
            for f in self.firms:
                f.loans = [
                    loan for loan in f.loans if loan.lender not in exiting_bank_set
                ]

        # Collapse check: all firms or all banks exiting.
        if len(exiting_firms) == len(self.firms) or len(exiting_banks) == len(
            self.banks
        ):
            self.collapsed = True
            return

        # Compute survivor statistics for firm replacement.
        survivor_firms = [f for f in self.firms if f not in set(exiting_firms)]
        survivor_net_worths = [f.net_worth for f in survivor_firms]
        # Use production_prev (last period's actual output) as the production signal.
        # At replacement time (post production event 22), production is the current
        # period's output; production_prev holds the same value. Using production_prev
        # ensures a non-zero reference even in unit-test contexts where production
        # has not been set explicitly.
        survivor_productions = [f.production_prev for f in survivor_firms]

        employed_wages = [h.wage for h in self.households if h.employed and h.wage > 0]

        mean_net = trim_mean(survivor_net_worths)
        mean_prod = trim_mean(survivor_productions)
        mean_wage = trim_mean(employed_wages)

        avg_price = self.avg_mkt_price
        min_wage = self.min_wage

        # Replace each exiting firm in place.
        for f in exiting_firms:
            nw_val = mean_net * self.p["new_firm_size_factor"]
            f.net_worth = nw_val
            f.total_funds = nw_val
            f.gross_profit = 0.0
            f.net_profit = 0.0
            f.retained_profit = 0.0
            f.credit_demand = 0.0
            f.projected_fragility = 0.0
            prod_val = mean_prod * self.p["new_firm_production_factor"]
            f.production = prod_val
            f.production_prev = prod_val
            f.inventory = 0.0
            f.expected_demand = 0.0
            f.desired_production = 0.0
            f.price = avg_price * self.p["new_firm_price_markup"]
            f.current_labor = 0
            f.desired_labor = 0
            f.n_vacancies = 0
            f.wage_bill = 0.0
            f.wage_offer = max(mean_wage * self.p["new_firm_wage_factor"], min_wage)
            f.loans = []

        # Replace each exiting bank by cloning a random survivor's equity.
        survivor_banks = [b for b in self.banks if b not in set(exiting_banks)]
        for b in exiting_banks:
            src = self.random.choice(survivor_banks)
            b.equity_base = src.equity_base
            b.credit_supply = 0.0
            b.interest_rate = 0.0

    def _revenue(self) -> None:
        """Phase 6: revenue (events 30-32)."""
        self.firms.do("collect_revenue")
        for f in self.firms:
            f.validate_debt(self)
        self._pay_dividends()

    def _bankruptcy_entry(self) -> None:
        """Phase 7-8: net worth update, bankruptcy detection, and entry (events 33-37)."""
        self.firms.do("update_net_worth")
        self._mark_bankrupt_and_replace()

    def step(self):
        """Execute one simulation period."""
        if self.collapsed:
            return
        self.period += 1
        self._planning()
        self._labor_market()
        self._credit_market()
        self._production()
        self._goods_market()
        self._revenue()
        self._bankruptcy_entry()
