"""BAMModel: mesa-frames (Polars backend) implementation of the baseline BAM model."""

from __future__ import annotations

import mesa_frames as mf
import polars as pl

from comparison.runners.mesa_frames.agents import Banks, Firms, Households
from comparison.runners.mesa_frames.markets import run_labor_market

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

        Household-side writes go through the columnar employer relationship:
        ``employer`` is the firm unique_id (-1 = unemployed); employed is
        derived from ``employer >= 0``.  For each over-staffed firm (in firm
        row order, matching the Mesa port's firm iteration), the firm's current
        employees are gathered as worker rows where ``employer == firm_id`` (in
        worker row order, mirroring the Mesa employees dict insertion order),
        and ``min(excess, n_employees)`` victims are drawn WITHOUT replacement.

        RNG: one ``self.random.choice(..., replace=False)`` per over-staffed
        firm, in firm row order -- the same draw structure as the Mesa port's
        per-firm ``model.random.sample``.
        """
        fdf = self.firms.agents
        rng = self.random

        firm_ids = fdf["unique_id"].to_list()
        cur_labor = [int(c) for c in fdf["current_labor"]]
        desired = [int(d) for d in fdf["desired_labor"]]

        # Map firm_id -> list of worker rows currently employed there (row order).
        hdf = self.households.agents
        hh_employer = [int(e) for e in hdf["employer"]]
        employees_of: dict[int, list[int]] = {}
        for i, emp in enumerate(hh_employer):
            if emp >= 0:
                employees_of.setdefault(emp, []).append(i)

        # Worker-side mutable copies (row-indexed).
        new_employer = list(hh_employer)
        new_employer_prev = [int(e) for e in hdf["employer_prev"]]
        new_wage = [float(w) for w in hdf["wage"]]
        new_periods = [int(p) for p in hdf["periods_left"]]
        new_contract_expired = list(hdf["contract_expired"])
        new_fired = list(hdf["fired"])

        any_fired = False
        for fi, firm_id in enumerate(firm_ids):
            excess = cur_labor[fi] - desired[fi]
            if excess <= 0:
                continue
            emps = employees_of.get(firm_id, [])
            k = min(excess, len(emps))
            if k <= 0:
                continue
            any_fired = True
            # Draw k victims without replacement, in this firm's pass.
            victims = rng.choice(emps, size=k, replace=False)
            for vi in victims:
                vi = int(vi)
                new_employer[vi] = -1
                new_employer_prev[vi] = firm_id
                new_wage[vi] = 0.0
                new_periods[vi] = 0
                new_contract_expired[vi] = False
                new_fired[vi] = True
            cur_labor[fi] -= k

        if not any_fired:
            return

        self.firms.agents = fdf.with_columns(
            pl.Series("current_labor", cur_labor, dtype=pl.Int64)
        )
        self.households.agents = hdf.with_columns(
            pl.Series("employer", new_employer, dtype=pl.Int64),
            pl.Series("employer_prev", new_employer_prev, dtype=pl.Int64),
            pl.Series("wage", new_wage, dtype=pl.Float64),
            pl.Series("periods_left", new_periods, dtype=pl.Int64),
            pl.Series("contract_expired", new_contract_expired, dtype=pl.Boolean),
            pl.Series("fired", new_fired, dtype=pl.Boolean),
        )

    # ------------------------------------------------------------------
    # Phase 2: labor market (events 7-12)
    # ------------------------------------------------------------------

    def _labor_market(self) -> None:
        """Phase 2: labor market (events 7-12).

        Mirrors BamModel._labor_market in mesa/model.py:
          _calc_inflation -> _adjust_min_wage -> firms_decide_wage_offer
          -> workers_decide_firms_to_apply -> run_labor_market -> calc_wage_bill
        """
        self._calc_inflation()
        if self.collect:
            self._c_inflation.append(self.inflation_history[-1])
        self._adjust_min_wage()
        self._event9_decide_wage_offer()
        self._event10_decide_firms_to_apply()
        run_labor_market(self)
        self._event12_calc_wage_bill()

    def _calc_inflation(self) -> None:
        """Event 7: YoY inflation rate appended to inflation_history.

        Mesa port: _calc_inflation() (identical scalar logic).
        """
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
        """Event 8: periodically index minimum wage to inflation.

        Mesa port: _adjust_min_wage().  Employed workers below the new floor are
        bumped up to it.  Employed is derived from ``employer >= 0``.
        """
        m = self.min_wage_rev_period
        hist_len = len(self.avg_mkt_price_history)
        if hist_len <= m:
            return
        if (hist_len - 1) % m != 0:
            return
        inflation = self.inflation_history[-1]
        self.min_wage *= 1.0 + inflation

        # Bump employed workers below the new floor (vectorized).
        hdf = self.households.agents
        new_wage = (
            pl.when((pl.col("employer") >= 0) & (pl.col("wage") < self.min_wage))
            .then(pl.lit(self.min_wage))
            .otherwise(pl.col("wage"))
        )
        self.households.agents = hdf.with_columns(new_wage.alias("wage"))

    def _event9_decide_wage_offer(self) -> None:
        """Event 9: firms set wage offer with random markup (zero if no vacancies).

        Mesa port: decide_wage_offer()
          shock = model.random.uniform(0, h_xi) if n_vacancies > 0 else 0.0
          wage_offer *= (1 + shock); wage_offer = max(wage_offer, min_wage)

        RNG: one draw per firm, uniform(0, h_xi), in firm row order, masked to
        zero where n_vacancies == 0 (mirrors bamengine's draw-then-mask, which
        the Mesa port replicates per firm).
        """
        h_xi = float(self.p["h_xi"])
        df = self.firms.agents
        n = len(df)

        shocks = pl.Series("shock", self.random.uniform(0.0, h_xi, size=n))
        masked_shock = (
            pl.when(pl.col("n_vacancies") > 0).then(shocks).otherwise(pl.lit(0.0))
        )
        new_offer = pl.max_horizontal(
            pl.col("wage_offer") * (1.0 + masked_shock),
            pl.lit(self.min_wage),
        )
        self.firms.agents = df.with_columns(new_offer.alias("wage_offer"))

    def _event10_decide_firms_to_apply(self) -> None:
        """Event 10: unemployed workers build a ranked job-application queue.

        Mesa port: decide_firms_to_apply()
          only unemployed workers participate
          pool = all firms; M_eff = min(max_M, |pool|)
          sample = model.random.sample(pool, M_eff)  (without replacement)
          sort sample by wage_offer DESC
          loyalty: if contract_expired AND not fired AND employer_prev in pool:
            move employer_prev to front (remove if present else drop last)
          job_apps = sample; contract_expired=False; fired=False

        The queue is stored as -1-padded columns job_app_0..job_app_{max_M-1}
        plus job_app_head (reset to 0).

        RNG: one without-replacement sample of M_eff firms per UNEMPLOYED worker,
        in worker row order -- the same draw structure as the Mesa port (which
        iterates households in row order and draws only for unemployed ones).
        """
        max_M = int(self.p["max_M"])
        rng = self.random

        fdf = self.firms.agents
        firm_ids = fdf["unique_id"].to_numpy()
        wage_offer = {
            int(fid): float(wo)
            for fid, wo in zip(
                fdf["unique_id"].to_list(),
                fdf["wage_offer"].to_list(),
                strict=True,
            )
        }
        firm_id_set = set(int(f) for f in firm_ids)
        pool_size = len(firm_ids)
        M_eff = min(max_M, pool_size)

        hdf = self.households.agents
        worker_employer = [int(e) for e in hdf["employer"]]
        worker_emp_prev = [int(e) for e in hdf["employer_prev"]]
        worker_contract_expired = list(hdf["contract_expired"])
        worker_fired = list(hdf["fired"])
        n_workers = len(worker_employer)

        # Output queue matrix (n_workers x max_M), -1-padded.
        queue = [[-1] * max_M for _ in range(n_workers)]
        new_contract_expired = list(worker_contract_expired)
        new_fired = list(worker_fired)

        for i in range(n_workers):
            if worker_employer[i] >= 0:
                continue  # employed workers do not apply
            # Sample M_eff firms without replacement (in firm-id space).
            # One choice(replace=False) per unemployed worker -- same draw
            # structure as the Mesa port's model.random.sample(pool, M_eff).
            sample = [int(f) for f in rng.choice(firm_ids, size=M_eff, replace=False)]
            # Sort by wage_offer DESC (stable on sampled order).
            sample.sort(key=lambda fid: wage_offer[fid], reverse=True)

            # Loyalty: move employer_prev to front if eligible.
            prev = worker_emp_prev[i]
            if (
                worker_contract_expired[i]
                and not worker_fired[i]
                and prev in firm_id_set
            ):
                if prev in sample:
                    sample.remove(prev)
                elif len(sample) == M_eff:
                    sample = sample[: M_eff - 1]
                sample.insert(0, prev)

            for k, fid in enumerate(sample[:max_M]):
                queue[i][k] = fid

            new_contract_expired[i] = False
            new_fired[i] = False

        cols = []
        for k in range(max_M):
            cols.append(
                pl.Series(
                    f"job_app_{k}",
                    [queue[i][k] for i in range(n_workers)],
                    dtype=pl.Int64,
                )
            )
        cols.append(pl.Series("job_app_head", [0] * n_workers, dtype=pl.Int64))
        cols.append(
            pl.Series("contract_expired", new_contract_expired, dtype=pl.Boolean)
        )
        cols.append(pl.Series("fired", new_fired, dtype=pl.Boolean))

        self.households.agents = hdf.with_columns(*cols)

    def _event12_calc_wage_bill(self) -> None:
        """Event 12: wage_bill_i = sum of wages over firm i's employees.

        Mesa port: calc_wage_bill().  Computed via a group-by join on the
        employer relationship (employed workers grouped by employer firm id).
        """
        hdf = self.households.agents
        fdf = self.firms.agents

        wage_by_firm = (
            hdf.filter(pl.col("employer") >= 0)
            .group_by("employer")
            .agg(pl.col("wage").sum().alias("_wage_bill_new"))
            .rename({"employer": "unique_id"})
        )

        joined = fdf.join(wage_by_firm, on="unique_id", how="left")
        self.firms.agents = joined.with_columns(
            pl.col("_wage_bill_new").fill_null(0.0).alias("wage_bill")
        ).drop("_wage_bill_new")

    def step(self) -> None:
        """Execute one simulation period."""
        if self.collapsed:
            return
        self.period += 1
        self._planning()
        self._labor_market()
        # Remaining phases added in Tasks 5-8.
