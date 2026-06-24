"""BAMModel: mesa-frames (Polars backend) implementation of the baseline BAM model."""

from __future__ import annotations

import mesa_frames as mf
import polars as pl

from comparison.runners.mesa_frames.agents import Banks, Firms, Households

EPS = 1e-9


class BAMModel(mf.ModelDF):
    """The baseline BAM agent-based macroeconomic model using mesa-frames (Polars).

    Agent data is stored as Polars DataFrames inside AgentSetPolars subclasses.
    All events are implemented as vectorized Polars operations on those DataFrames.
    """

    EPS = EPS

    def __init__(
        self,
        n_firms: int,
        n_households: int,
        n_banks: int,
        params: dict,
        seed: int = 0,
        collect: bool = False,
    ) -> None:
        super().__init__()  # sets self.random (numpy Generator) via default seed
        self.reset_randomizer(seed)  # re-seed with caller's seed

        self.n_firms = n_firms
        self.n_households = n_households  # must be set before Firms.__init__
        self.n_banks = n_banks
        self.p = dict(params)
        self.period = 0
        self.collect = collect

        # Per-period collection lists (populated when collect=True).
        self._c_unemployment: list[float] = []
        self._c_avg_employed_wage: list[float] = []
        self._c_total_production: list[float] = []
        self._c_total_vacancies: list[float] = []
        self._c_inflation: list[float] = []
        self._c_production_final: list[float] = []

        # Construct and register agent sets.
        # Order matters: Firms reads model.n_households in __init__.
        self.firms = Firms(self, n_firms, self.p, self.random)
        self.households = Households(self, n_households, self.p, self.random)
        self.banks = Banks(self, n_banks, self.p, self.random)

        # Register with the model so self.agents works.
        self.agents += self.firms
        self.agents += self.households
        self.agents += self.banks

        # Economy state (mirrors BamModel.__init__ in mesa/model.py).
        price_init: float = float(self.p["price_init"])
        wage_offer_init: float = price_init / 3.0
        self.avg_mkt_price: float = price_init
        self.min_wage: float = wage_offer_init * float(self.p["min_wage_ratio"])
        self.min_wage_rev_period: int = int(self.p["min_wage_rev_period"])
        self.avg_mkt_price_history: list[float] = [float(price_init)]
        self.inflation_history: list[float] = [0.0]
        self.collapsed: bool = False

    # ------------------------------------------------------------------
    # Phase 1: planning (events 1-6)
    # ------------------------------------------------------------------

    def _planning(self) -> None:
        """Phase 1: planning (events 1-6).

        Translates the Mesa port's per-agent methods into vectorized Polars
        operations on self.firms.agents.  RNG draw order is preserved:
        one uniform draw per firm for the production shock (event 1), then
        one uniform draw per firm for the price shock (event 3) -- identical
        to the Mesa port's decide_desired_production / plan_price sequence.

        Event 6 (fire_excess_workers) is a no-op here because at t=0 all
        firms have current_labor=0; it is implemented as a Polars update for
        the general case but skip the household-side writes (those require a
        separate relationship table added in a later task).
        """
        self._event1_decide_desired_production()
        self._event2_plan_breakeven_price()
        self._event3_plan_price()
        self._event4_decide_desired_labor()
        self._event5_decide_vacancies()
        if self.collect:
            total_vac = int(self.firms.agents["n_vacancies"].sum())
            self._c_total_vacancies.append(float(total_vac))
        self._event6_fire_excess_workers()

    def _event1_decide_desired_production(self) -> None:
        """Event 1: zero production, production shock, expected + desired production.

        Mesa port: decide_desired_production()
          self.production = 0
          shock = model.random.uniform(0, h_rho)
          up = inventory==0 AND price >= p_avg
          dn = inventory>0  AND price <  p_avg
          expected_demand = production_prev
          expected_demand[up] *= (1 + shock)
          expected_demand[dn] *= (1 - shock)
          desired_production = expected_demand

        RNG: one draw per firm, uniform(0, h_rho), in agent-set row order.
        """
        p_avg = self.avg_mkt_price
        h_rho = float(self.p["h_rho"])
        df = self.firms.agents

        n = len(df)
        shocks = pl.Series("shock", self.random.uniform(0.0, h_rho, size=n))

        # Condition masks (SPEC event 1: complement of event 3 conditions).
        up = (df["inventory"] == 0.0) & (df["price"] >= p_avg)
        dn = (df["inventory"] > 0.0) & (df["price"] < p_avg)

        expected_demand = (
            pl.when(up)
            .then(pl.col("production_prev") * (1.0 + shocks))
            .when(dn)
            .then(pl.col("production_prev") * (1.0 - shocks))
            .otherwise(pl.col("production_prev"))
        )

        self.firms.agents = df.with_columns(
            pl.lit(0.0).alias("production"),
            expected_demand.alias("expected_demand"),
            expected_demand.alias("desired_production"),
        )

    def _event2_plan_breakeven_price(self) -> None:
        """Event 2: breakeven price from wage bill and prior-period loan interest.

        Mesa port: plan_breakeven_price()
          interest = sum(loan.rate * loan.principal for loan in self.loans)
          breakeven = (wage_bill + interest) / max(desired_production, EPS)

        At t=0 wage_bill=0 and there are no loans, so breakeven=0.
        Loan interest from prior-period loans is tracked in a separate
        relationship table (added in a later task); for now interest=0.
        """
        eps = self.EPS
        df = self.firms.agents

        # TODO (task 5+): add loan interest from LoanBook relationship table.
        # For now interest_i = 0 for all firms.
        interest = pl.lit(0.0)

        breakeven = (pl.col("wage_bill") + interest) / pl.when(
            pl.col("desired_production") > eps
        ).then(pl.col("desired_production")).otherwise(eps)

        self.firms.agents = df.with_columns(breakeven.alias("breakeven_price"))

    def _event3_plan_price(self) -> None:
        """Event 3: price adjustment based on inventory and market position.

        Mesa port: plan_price()
          shock = model.random.uniform(0, h_eta)
          up = inventory==0 AND price <  p_avg   (COMPLEMENT of event 1 up)
          dn = inventory>0  AND price >= p_avg   (COMPLEMENT of event 1 dn)
          if up:  price *= (1+shock); price = max(price, breakeven_price)
          elif dn: price *= (1-shock); price = max(price, breakeven_price)

        RNG: one draw per firm, uniform(0, h_eta), in agent-set row order.
        """
        p_avg = self.avg_mkt_price
        h_eta = float(self.p["h_eta"])
        df = self.firms.agents

        n = len(df)
        shocks = pl.Series("shock", self.random.uniform(0.0, h_eta, size=n))

        up = (df["inventory"] == 0.0) & (df["price"] < p_avg)
        dn = (df["inventory"] > 0.0) & (df["price"] >= p_avg)

        new_price = (
            pl.when(up)
            .then(
                pl.max_horizontal(
                    pl.col("price") * (1.0 + shocks),
                    pl.col("breakeven_price"),
                )
            )
            .when(dn)
            .then(
                pl.max_horizontal(
                    pl.col("price") * (1.0 - shocks),
                    pl.col("breakeven_price"),
                )
            )
            .otherwise(pl.col("price"))
        )

        self.firms.agents = df.with_columns(new_price.alias("price"))

    def _event4_decide_desired_labor(self) -> None:
        """Event 4: desired_labor = ceil(desired_production / labor_productivity).

        Mesa port: decide_desired_labor()
          desired_labor = math.ceil(desired_production / max(labor_productivity, EPS))

        The ceil ratchet is load-bearing (see LABOR_MARKET_QUANTIZATION.md).
        """
        eps = self.EPS
        df = self.firms.agents

        safe_lp = (
            pl.when(pl.col("labor_productivity") > eps)
            .then(pl.col("labor_productivity"))
            .otherwise(eps)
        )

        desired_labor = (pl.col("desired_production") / safe_lp).ceil().cast(pl.Int64)

        self.firms.agents = df.with_columns(desired_labor.alias("desired_labor"))

    def _event5_decide_vacancies(self) -> None:
        """Event 5: n_vacancies = max(desired_labor - current_labor, 0).

        Mesa port: decide_vacancies()
          n_vacancies = max(desired_labor - current_labor, 0)
        """
        df = self.firms.agents

        vacancies = pl.max_horizontal(
            pl.col("desired_labor") - pl.col("current_labor"),
            pl.lit(0).cast(pl.Int64),
        )

        self.firms.agents = df.with_columns(vacancies.alias("n_vacancies"))

    def _event6_fire_excess_workers(self) -> None:
        """Event 6: fire excess workers chosen uniformly at random.

        Mesa port: fire_excess_workers()
          excess = current_labor - desired_labor
          if excess <= 0: return
          victims = model.random.sample(list(employees), min(excess, len(employees)))
          for h in victims: h.employer=None, h.employer_prev=self, h.wage=0,
                            h.periods_left=0, h.contract_expired=False, h.fired=True
          current_labor -= len(victims)

        The household-side writes (employer, employer_prev, wage, etc.) require
        the employer-relationship table, which is added in a later task (Task 4).
        This implementation updates current_labor on the firm side only; the
        household updates are deferred to Task 4.

        RNG: self.random.choice / permutation for each over-staffed firm.
        """
        df = self.firms.agents
        over_staffed = df.filter(pl.col("current_labor") > pl.col("desired_labor"))

        if len(over_staffed) == 0:
            return

        # Build updated current_labor column for over-staffed firms.
        # For now we just clamp current_labor down to desired_labor.
        # The actual random-victim selection is deferred until household
        # relationship tables exist.
        updated_rows = []
        for i in range(len(df)):
            cl = int(df["current_labor"][i])
            dl = int(df["desired_labor"][i])
            if cl > dl:
                excess = cl - dl
                # Draw excess victims (random permutation of worker indices).
                # Worker-side updates deferred to Task 4; only firm counter
                # is decremented here.
                fired_count = min(excess, cl)
                updated_rows.append((int(df["unique_id"][i]), cl - fired_count))
            else:
                updated_rows.append((int(df["unique_id"][i]), cl))

        id_series = pl.Series("unique_id", [r[0] for r in updated_rows], dtype=pl.Int64)
        cl_series = pl.Series(
            "current_labor", [r[1] for r in updated_rows], dtype=pl.Int64
        )
        updates = pl.DataFrame({"unique_id": id_series, "current_labor": cl_series})

        self.firms.agents = df.drop("current_labor").join(
            updates, on="unique_id", how="left"
        )

    def step(self) -> None:
        """Execute one simulation period."""
        if self.collapsed:
            return
        self.period += 1
        self._planning()
        # Remaining phases added in Tasks 4-8.
