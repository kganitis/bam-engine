"""BAMModel: mesa-frames (Polars backend) implementation of the baseline BAM model."""

from __future__ import annotations

import mesa_frames as mf
import polars as pl

from comparison.runners.mesa_frames.agents import Banks, Firms, Households
from comparison.runners.mesa_frames.markets import (
    run_credit_market,
    run_goods_market,
    run_labor_market,
)

EPS = 1e-9


def empty_loan_book() -> pl.DataFrame:
    """An empty loan relationship table with the canonical schema.

    Mirrors bamengine's ``LoanBook`` / the Mesa port's per-firm ``list[Loan]``.
    Each row is one firm<->bank loan.  ``interest`` and ``debt`` are derived
    (``principal * interest_rate`` and ``principal * (1 + interest_rate)``) and
    recomputed where needed rather than stored, matching the Mesa ``Loan``
    NamedTuple properties.
    """
    return pl.DataFrame(
        schema={
            "borrower_id": pl.Int64,  # firm unique_id
            "lender_id": pl.Int64,  # bank unique_id
            "principal": pl.Float64,
            "interest_rate": pl.Float64,
        }
    )


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

        # Loan relationship table (firm<->bank loans).  Mirrors bamengine's
        # LoanBook and the Mesa port's per-firm ``Firm.loans`` lists, aggregated
        # into one columnar table keyed by ``borrower_id``.  Loans PERSIST across
        # periods (retained after settlement so the planning-phase breakeven in
        # event 2 can read last period's interest); they are purged at the start
        # of the credit market (``_credit_market``), matching the Mesa port's
        # ``for f in self.firms: f.loans = []`` at the top of ``_credit_market``.
        self.loans: pl.DataFrame = empty_loan_book()

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

        Event 2 runs in the planning phase, BEFORE the credit market opens, so
        ``self.loans`` still holds last period's loans (they are purged at the
        start of ``_credit_market``).  The per-firm interest is therefore the
        prior-period interest ``Σ principal * interest_rate`` over the firm's
        loans -- identical to the Mesa port reading ``self.loans`` carried over
        from the previous period.  At t=0 the table is empty and wage_bill=0, so
        breakeven=0.
        """
        eps = self.EPS
        df = self.firms.agents

        # Prior-period interest per borrowing firm: Σ principal * interest_rate.
        interest_by_firm = (
            self.loans.with_columns(
                (pl.col("principal") * pl.col("interest_rate")).alias("_interest")
            )
            .group_by("borrower_id")
            .agg(pl.col("_interest").sum().alias("_interest_sum"))
            .rename({"borrower_id": "unique_id"})
        )

        df = df.join(interest_by_firm, on="unique_id", how="left").with_columns(
            pl.col("_interest_sum").fill_null(0.0).alias("_interest_sum")
        )

        breakeven = (pl.col("wage_bill") + pl.col("_interest_sum")) / pl.when(
            pl.col("desired_production") > eps
        ).then(pl.col("desired_production")).otherwise(eps)

        self.firms.agents = df.with_columns(breakeven.alias("breakeven_price")).drop(
            "_interest_sum"
        )

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

    # ------------------------------------------------------------------
    # Phase 3: credit market (events 13-19)
    # ------------------------------------------------------------------

    def _credit_market(self) -> None:
        """Phase 3: credit market (events 13-19).

        Mirrors BamModel._credit_market in mesa/model.py:
          purge prior-period loans -> banks_decide_credit_supply
          -> banks_decide_interest_rate -> firms_decide_credit_demand
          -> firms_calc_fragility -> firms_prepare_loan_applications
          -> run_credit_market -> firms_fire_workers_for_gap
        """
        # Purge previous-period loans (retained through planning/labor for the
        # event-2 breakeven), matching the Mesa port clearing f.loans at the top
        # of _credit_market.
        self.loans = empty_loan_book()
        self._event13_decide_credit_supply()
        self._event14_decide_interest_rate()
        self._event15_decide_credit_demand()
        self._event16_calc_fragility()
        self._event17_prepare_loan_applications()
        run_credit_market(self)
        self._event19_fire_workers_for_gap()

    def _event13_decide_credit_supply(self) -> None:
        """Event 13: credit_supply = max(equity_base / v, 0).

        Mesa port: Bank.decide_credit_supply().  No RNG.
        """
        v = float(self.p["v"])
        bdf = self.banks.agents
        supply = pl.max_horizontal(pl.col("equity_base") / v, pl.lit(0.0))
        self.banks.agents = bdf.with_columns(supply.alias("credit_supply"))

    def _event14_decide_interest_rate(self) -> None:
        """Event 14: opex_shock ~ U(0, h_phi); interest_rate = r_bar*(1+shock).

        Mesa port: Bank.decide_interest_rate()
          opex_shock = model.random.uniform(0, h_phi)
          interest_rate = r_bar * (1 + opex_shock)

        RNG: one draw per bank, uniform(0, h_phi), in bank row order -- the same
        draw structure as the Mesa port (banks.do("decide_interest_rate")).
        The posted ``interest_rate`` ranks banks; ``opex_shock`` is reused in the
        fragility-scaled loan contract rate (event 18).
        """
        h_phi = float(self.p["h_phi"])
        r_bar = float(self.p["r_bar"])
        bdf = self.banks.agents
        n = len(bdf)

        shocks = pl.Series("opex_shock", self.random.uniform(0.0, h_phi, size=n))
        self.banks.agents = bdf.with_columns(
            shocks.alias("opex_shock"),
            (r_bar * (1.0 + shocks)).alias("interest_rate"),
        )

    def _event15_decide_credit_demand(self) -> None:
        """Event 15: credit_demand = max(wage_bill - total_funds, 0).

        Mesa port: Firm.decide_credit_demand().  No RNG.
        """
        fdf = self.firms.agents
        demand = pl.max_horizontal(
            pl.col("wage_bill") - pl.col("total_funds"), pl.lit(0.0)
        )
        self.firms.agents = fdf.with_columns(demand.alias("credit_demand"))

    def _event16_calc_fragility(self) -> None:
        """Event 16: projected_fragility = credit_demand/net_worth (or max_leverage).

        Mesa port: Firm.calc_fragility()
          if net_worth > 0: projected_fragility = credit_demand / net_worth
          else:             projected_fragility = max_leverage

        No RNG.  net_worth<=0 firms get max_leverage (worst fragility).
        """
        max_leverage = float(self.p["max_leverage"])
        fdf = self.firms.agents
        fragility = (
            pl.when(pl.col("net_worth") > 0.0)
            .then(pl.col("credit_demand") / pl.col("net_worth"))
            .otherwise(pl.lit(max_leverage))
        )
        self.firms.agents = fdf.with_columns(fragility.alias("projected_fragility"))

    def _event17_prepare_loan_applications(self) -> None:
        """Event 17: each firm with credit_demand>0 samples banks, sorts by rate.

        Mesa port: Firm.prepare_loan_applications()
          if credit_demand <= 0: loan_apps = []; return
          lenders = [b for b in banks if b.credit_supply > 0]
          H_eff = min(max_H, len(lenders))
          if H_eff == 0: loan_apps = []; return
          sample = model.random.sample(lenders, H_eff)
          sample.sort(key=lambda b: b.interest_rate)
          loan_apps = sample

        The application queue is stored on the Households-style pattern: per-firm
        ``-1``-padded columns ``loan_app_0 .. loan_app_{max_H-1}`` (bank
        unique_ids) plus ``loan_app_head`` (reset to 0).

        RNG: one without-replacement sample of H_eff banks per firm with
        credit_demand>0, in firm row order -- the same draw structure as the
        Mesa port (firms.do iterates firms in row order; only demanding firms
        draw).  Banks eligible = those with credit_supply>0 at the snapshot
        taken before any round runs (matching the Mesa port).
        """
        max_H = int(self.p["max_H"])
        rng = self.random

        bdf = self.banks.agents
        # Lenders with supply > 0 (snapshot in bank row order), and their rates.
        lender_ids = [
            int(bid)
            for bid, sup in zip(
                bdf["unique_id"].to_list(),
                bdf["credit_supply"].to_list(),
                strict=True,
            )
            if sup > 0.0
        ]
        rate_of = {
            int(bid): float(r)
            for bid, r in zip(
                bdf["unique_id"].to_list(),
                bdf["interest_rate"].to_list(),
                strict=True,
            )
        }
        lender_id_arr = list(lender_ids)
        n_lenders = len(lender_id_arr)
        H_eff = min(max_H, n_lenders)

        fdf = self.firms.agents
        credit_demand = [float(d) for d in fdf["credit_demand"]]
        n_firms = len(credit_demand)

        # Output queue matrix (n_firms x max_H), -1-padded bank ids.
        queue = [[-1] * max_H for _ in range(n_firms)]

        for i in range(n_firms):
            if credit_demand[i] <= 0.0 or H_eff == 0:
                continue
            # One without-replacement sample of H_eff banks per demanding firm --
            # same draw structure as the Mesa port's model.random.sample.
            sample = [
                int(b) for b in rng.choice(lender_id_arr, size=H_eff, replace=False)
            ]
            # Sort by interest_rate ASC (cheapest bank first).
            sample.sort(key=lambda bid: rate_of[bid])
            for k, bid in enumerate(sample[:max_H]):
                queue[i][k] = bid

        cols = []
        for k in range(max_H):
            cols.append(
                pl.Series(
                    f"loan_app_{k}",
                    [queue[i][k] for i in range(n_firms)],
                    dtype=pl.Int64,
                )
            )
        cols.append(pl.Series("loan_app_head", [0] * n_firms, dtype=pl.Int64))
        self.firms.agents = fdf.with_columns(*cols)

    def _event19_fire_workers_for_gap(self) -> None:
        """Event 19: firms with wage_bill > total_funds fire workers to close gap.

        Mesa port: Firm.fire_workers_for_gap()
          if wage_bill <= total_funds: return
          gap = wage_bill - total_funds
          employees_list = list(self.employees)
          model.random.shuffle(employees_list)
          cumulative = 0
          for h in employees_list:
            fire h; wage_bill -= h.wage; cumulative += h.wage; current_labor -= 1
            if cumulative >= gap: break

        Household-side writes mirror event 6 (employer relation): fired worker
        gets employer=-1, employer_prev=firm, wage=0, periods_left=0,
        contract_expired=False, fired=True.

        RNG: one ``model.random.shuffle`` per firm with a financing gap, in firm
        row order -- identical draw structure to the Mesa port (firms iterated in
        row order; only gap firms shuffle).  Workers of a firm are gathered as
        rows where employer==firm_id, in worker row order (mirroring the Mesa
        employees-dict insertion order, which is worker creation/row order).
        """
        fdf = self.firms.agents
        rng = self.random

        firm_ids = fdf["unique_id"].to_list()
        wage_bill = [float(w) for w in fdf["wage_bill"]]
        total_funds = [float(t) for t in fdf["total_funds"]]
        cur_labor = [int(c) for c in fdf["current_labor"]]

        hdf = self.households.agents
        hh_employer = [int(e) for e in hdf["employer"]]
        hh_wage = [float(w) for w in hdf["wage"]]

        # Map firm_id -> worker rows employed there, in worker row order.
        employees_of: dict[int, list[int]] = {}
        for i, emp in enumerate(hh_employer):
            if emp >= 0:
                employees_of.setdefault(emp, []).append(i)

        # Worker-side mutable copies.
        new_employer = list(hh_employer)
        new_employer_prev = [int(e) for e in hdf["employer_prev"]]
        new_wage = list(hh_wage)
        new_periods = [int(p) for p in hdf["periods_left"]]
        new_contract_expired = list(hdf["contract_expired"])
        new_fired = list(hdf["fired"])

        any_fired = False
        for fi, firm_id in enumerate(firm_ids):
            if wage_bill[fi] <= total_funds[fi]:
                continue
            emps = employees_of.get(firm_id, [])
            if not emps:
                continue
            gap = wage_bill[fi] - total_funds[fi]
            # ONE shuffle per gap firm, in firm row order -- mirrors the Mesa
            # port's model.random.shuffle(employees_list).
            order = list(emps)
            rng.shuffle(order)
            cumulative = 0.0
            for vi in order:
                wage = new_wage[vi]
                new_employer[vi] = -1
                new_employer_prev[vi] = firm_id
                new_wage[vi] = 0.0
                new_periods[vi] = 0
                new_contract_expired[vi] = False
                new_fired[vi] = True
                cur_labor[fi] -= 1
                wage_bill[fi] -= wage
                cumulative += wage
                any_fired = True
                if cumulative >= gap:
                    break

        if not any_fired:
            return

        # Write firm-side results (current_labor, wage_bill).
        self.firms.agents = fdf.with_columns(
            pl.Series("current_labor", cur_labor, dtype=pl.Int64),
            pl.Series("wage_bill", wage_bill, dtype=pl.Float64),
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
    # Phase 4: production (events 20-24)
    # ------------------------------------------------------------------

    def _production(self) -> None:
        """Phase 4: production (events 20-24).

        Mirrors BamModel._production in mesa/model.py:
          firms_pay_wages -> workers_receive_wage -> firms_run_production
          -> _update_avg_mkt_price -> workers_update_contracts
        """
        self._event20_pay_wages()
        self._event21_receive_wage()
        self._event22_run_production()
        if self.collect:
            hdf = self.households.agents
            fdf = self.firms.agents
            employed_mask = hdf["employer"] >= 0
            employed_count = int(employed_mask.sum())
            n_hh = self.n_households
            unemployment = 1.0 - employed_count / n_hh if n_hh > 0 else 1.0
            wage_sum = float(hdf.filter(employed_mask)["wage"].sum())
            avg_wage = wage_sum / employed_count if employed_count > 0 else 0.0
            total_prod = float(fdf["production"].sum())
            self._c_unemployment.append(unemployment)
            self._c_avg_employed_wage.append(avg_wage)
            self._c_total_production.append(total_prod)
            self._c_production_final = fdf["production"].to_list()
        self._update_avg_mkt_price()
        self._event24_update_contracts()

    def _event20_pay_wages(self) -> None:
        """Event 20: total_funds -= wage_bill for all firms.

        Mesa port: Firm.pay_wages()
          self.total_funds -= self.wage_bill

        Fully vectorized: no RNG.
        """
        fdf = self.firms.agents
        self.firms.agents = fdf.with_columns(
            (pl.col("total_funds") - pl.col("wage_bill")).alias("total_funds")
        )

    def _event21_receive_wage(self) -> None:
        """Event 21: employed workers add their wage to income.

        Mesa port: Household.receive_wage()
          if self.employed: self.income += self.wage

        Fully vectorized: employed derived from employer >= 0.
        """
        hdf = self.households.agents
        new_income = (
            pl.when(pl.col("employer") >= 0)
            .then(pl.col("income") + pl.col("wage"))
            .otherwise(pl.col("income"))
        )
        self.households.agents = hdf.with_columns(new_income.alias("income"))

    def _event22_run_production(self) -> None:
        """Event 22: production, production_prev, inventory update.

        Mesa port: Firm.run_production()
          self.production = self.labor_productivity * self.current_labor
          self.production_prev = self.production   # unconditional
          self.inventory = self.production          # OVERWRITE, not accumulate

        No RNG.
        """
        fdf = self.firms.agents
        production = pl.col("labor_productivity") * pl.col("current_labor").cast(
            pl.Float64
        )
        self.firms.agents = fdf.with_columns(
            production.alias("production"),
            production.alias("production_prev"),
            production.alias("inventory"),
        )

    def _update_avg_mkt_price(self) -> None:
        """Event 23: production-weighted average price; keep previous if result <= 0.

        Mesa port: BamModel._update_avg_mkt_price()
          only firms with production >= 1e-3 contribute
          p_avg = weighted_sum / total_prod  (if total_prod > 0)
          if new_price > 0: self.avg_mkt_price = new_price
          append self.avg_mkt_price to avg_mkt_price_history

        No RNG.
        """
        fdf = self.firms.agents
        active = fdf.filter(pl.col("production") >= 1e-3)
        total_prod = float(active["production"].sum())
        if total_prod > 0:
            weighted_sum = float((active["price"] * active["production"]).sum())
            new_price = weighted_sum / total_prod
        else:
            new_price = 0.0
        if new_price > 0:
            self.avg_mkt_price = new_price
        # else: keep previous avg_mkt_price
        self.avg_mkt_price_history.append(self.avg_mkt_price)

    def _event24_update_contracts(self) -> None:
        """Event 24: employed workers decrement periods_left; expire at 0.

        Mesa port: Household.update_contract()
          if not self.employed: return
          self.periods_left -= 1
          if self.periods_left == 0:
            employer = self.employer
            self.employer_prev = employer; self.employer = None; self.wage = 0
            self.contract_expired = True; self.fired = False
            employer.employees.pop(self); employer.current_labor -= 1

        contract_expired=True + fired=False marks the worker as loyal next period.
        wage_bill is NOT recomputed after contract expiry (matches spec event 24 note).

        Firm-side: current_labor decremented for each expiry.
        """
        hdf = self.households.agents
        fdf = self.firms.agents

        employed = pl.col("employer") >= 0

        # Decrement periods_left for employed workers only.
        new_periods = (
            pl.when(employed)
            .then(pl.col("periods_left") - 1)
            .otherwise(pl.col("periods_left"))
        )

        # A contract expires when it was at 1 period remaining (becomes 0).
        expires = employed & (pl.col("periods_left") == 1)

        new_employer_prev = (
            pl.when(expires).then(pl.col("employer")).otherwise(pl.col("employer_prev"))
        )
        new_employer = (
            pl.when(expires)
            .then(pl.lit(-1, dtype=pl.Int64))
            .otherwise(pl.col("employer"))
        )
        new_wage = pl.when(expires).then(pl.lit(0.0)).otherwise(pl.col("wage"))
        new_contract_expired = (
            pl.when(expires).then(pl.lit(True)).otherwise(pl.col("contract_expired"))
        )
        new_fired = pl.when(expires).then(pl.lit(False)).otherwise(pl.col("fired"))

        self.households.agents = hdf.with_columns(
            new_periods.alias("periods_left"),
            new_employer_prev.alias("employer_prev"),
            new_employer.alias("employer"),
            new_wage.alias("wage"),
            new_contract_expired.alias("contract_expired"),
            new_fired.alias("fired"),
        )

        # Firm-side: decrement current_labor by the number of expired contracts.
        # Group expired workers by employer_prev (the firm they just left),
        # count expirations per firm, join and decrement.
        hdf_after = self.households.agents
        expired_workers = hdf_after.filter(pl.col("contract_expired"))
        if len(expired_workers) > 0:
            expirations_by_firm = (
                expired_workers.group_by("employer_prev")
                .agg(pl.len().alias("n_expired"))
                .rename({"employer_prev": "unique_id"})
            )
            self.firms.agents = (
                fdf.join(expirations_by_firm, on="unique_id", how="left")
                .with_columns(
                    (pl.col("current_labor") - pl.col("n_expired").fill_null(0)).alias(
                        "current_labor"
                    )
                )
                .drop("n_expired")
            )

    # ------------------------------------------------------------------
    # Phase 5: goods market (events 25-29)
    # ------------------------------------------------------------------

    def _goods_market(self) -> None:
        """Phase 5: goods market (events 25-29).

        Mirrors BamModel._goods_market in mesa/model.py:
          consumers_calc_propensity -> consumers_decide_income_to_spend
          -> consumers_decide_firms_to_visit -> run_goods_market
          -> consumers_finalize_purchases
        """
        self._event25_calc_propensity()
        self._event26_decide_income_to_spend()
        self._event27_decide_firms_to_visit()
        run_goods_market(self)
        self._event29_finalize_purchases()

    def _event25_calc_propensity(self) -> None:
        """Event 25: marginal propensity to consume from relative savings.

        Mesa port: Household.calc_propensity(avg_savings)
          savings = max(self.savings, 0)
          avg_sav = max(avg_savings, EPS)
          propensity = 1 / (1 + tanh(savings / avg_sav) ** beta)

        Where avg_sav = max(mean(savings over ALL consumers), EPS) -- computed
        once before iterating, matching the Mesa port's pre-loop calculation:
          all_savings = [h.savings for h in self.households]
          avg_sav = max(sum(all_savings)/len(all_savings), EPS) if all_savings else EPS

        No RNG.
        """

        eps = self.EPS
        beta = float(self.p["beta"])
        hdf = self.households.agents

        # avg_sav = max(mean(savings), EPS) -- matches Mesa port.
        avg_sav = max(float(hdf["savings"].mean()), eps)

        # propensity = 1 / (1 + tanh(max(savings,0)/avg_sav)^beta)
        # Fully vectorized Polars expression.
        clamped_savings = pl.max_horizontal(pl.col("savings"), pl.lit(0.0))
        ratio = clamped_savings / avg_sav
        # tanh(ratio)^beta: Polars has .tanh() on expressions.
        tanh_ratio = ratio.tanh()
        propensity = pl.lit(1.0) / (pl.lit(1.0) + tanh_ratio.pow(beta))

        self.households.agents = hdf.with_columns(propensity.alias("propensity"))

    def _event26_decide_income_to_spend(self) -> None:
        """Event 26: split wealth into spending budget and savings.

        Mesa port: Household.decide_income_to_spend()
          wealth = self.savings + self.income
          self.income_to_spend = wealth * self.propensity
          self.savings = wealth - self.income_to_spend
          self.income = 0

        Fully vectorized, no RNG.
        """
        hdf = self.households.agents
        wealth = pl.col("savings") + pl.col("income")
        income_to_spend = wealth * pl.col("propensity")
        savings = wealth - income_to_spend

        self.households.agents = hdf.with_columns(
            income_to_spend.alias("income_to_spend"),
            savings.alias("savings"),
            pl.lit(0.0).alias("income"),
        )

    def _event27_decide_firms_to_visit(self) -> None:
        """Event 27: each consumer selects firms to visit; loyalty rule applied.

        Mesa port: Household.decide_firms_to_visit(model)
          if income_to_spend <= EPS: shop_visits = []; return
          Z = min(max_Z, n_firms)
          selected = model.random.sample(all_firms, Z)  (without replacement)
          if loyalty and largest_prod_prev not in selected:
            selected[-1] = largest_prod_prev
          sort selected by price ASC
          shop_visits = selected
          if loyalty: largest_prod_prev = max(selected, key=lambda f: f.production)

        The visit list is stored as ``-1``-padded columns
        ``shop_visit_0 .. shop_visit_{max_Z-1}`` on the Households DataFrame.
        ``largest_prod_prev`` is updated BEFORE shopping (same as Mesa port).

        RNG: one without-replacement sample of Z firm ids per consumer with
        budget > EPS, in consumer row order -- matching the Mesa port's
        per-household ``model.random.sample``.
        """
        eps = self.EPS
        max_Z = int(self.p["max_Z"])
        consumer_matching = self.p.get("consumer_matching", "loyalty")
        rng = self.random

        fdf = self.firms.agents
        firm_ids_arr = fdf["unique_id"].to_numpy()
        firm_id_list = [int(f) for f in firm_ids_arr]
        n_firms = len(firm_id_list)
        Z = min(max_Z, n_firms)

        price_of: dict[int, float] = dict(
            zip(
                fdf["unique_id"].to_list(),
                (float(p) for p in fdf["price"]),
                strict=True,
            )
        )
        production_of: dict[int, float] = dict(
            zip(
                fdf["unique_id"].to_list(),
                (float(p) for p in fdf["production"]),
                strict=True,
            )
        )
        firm_id_set: set[int] = set(firm_id_list)

        hdf = self.households.agents
        n_hh = len(hdf)
        income_to_spend: list[float] = [float(b) for b in hdf["income_to_spend"]]
        largest_prod_prev: list[int] = [int(lp) for lp in hdf["largest_prod_prev"]]

        # Output: shop visit matrix (n_hh x max_Z), -1-padded.
        visit_queue: list[list[int]] = [[-1] * max_Z for _ in range(n_hh)]
        new_largest: list[int] = list(largest_prod_prev)

        for i in range(n_hh):
            if income_to_spend[i] <= eps:
                # No budget: leave shop_visits all -1 (already initialised).
                continue

            # Sample Z firms without replacement in firm_id_list order --
            # one rng.choice per consumer with budget, matching the Mesa port's
            # model.random.sample(all_firms, Z).
            sample = [int(f) for f in rng.choice(firm_ids_arr, size=Z, replace=False)]

            # Loyalty: force largest_prod_prev into set if not already present.
            if consumer_matching == "loyalty":
                prev = largest_prod_prev[i]
                if prev >= 0 and prev in firm_id_set and prev not in sample:
                    sample[-1] = prev

            # Sort by price ASC (cheapest first, mirroring Mesa port).
            sample.sort(key=lambda fid: price_of[fid])

            # Update loyalty BEFORE shopping: track firm with max production in set.
            if consumer_matching == "loyalty":
                new_largest[i] = max(sample, key=lambda fid: production_of[fid])

            # Store in visit queue (truncated to max_Z slots).
            for k, fid in enumerate(sample[:max_Z]):
                visit_queue[i][k] = fid

        # Build Polars columns for shop_visit_0..max_Z-1 and largest_prod_prev.
        cols = []
        for k in range(max_Z):
            cols.append(
                pl.Series(
                    f"shop_visit_{k}",
                    [visit_queue[i][k] for i in range(n_hh)],
                    dtype=pl.Int64,
                )
            )
        cols.append(pl.Series("largest_prod_prev", new_largest, dtype=pl.Int64))

        self.households.agents = hdf.with_columns(*cols)

    def _event29_finalize_purchases(self) -> None:
        """Event 29: return unspent budget to savings; zero income_to_spend.

        Mesa port: Household.finalize_purchases()
          self.savings += self.income_to_spend
          self.income_to_spend = 0

        Fully vectorized, no RNG.
        """
        hdf = self.households.agents
        self.households.agents = hdf.with_columns(
            (pl.col("savings") + pl.col("income_to_spend")).alias("savings"),
            pl.lit(0.0).alias("income_to_spend"),
        )

    # ------------------------------------------------------------------
    # Phase 6: revenue (events 30-32)
    # ------------------------------------------------------------------

    def _revenue(self) -> None:
        """Phase 6: revenue (events 30-32).

        Mirrors BamModel._revenue in mesa/model.py:
          firms_collect_revenue -> firms_validate_debt_commitments
          -> firms_pay_dividends
        """
        self._event30_collect_revenue()
        self._event31_validate_debt()
        self._event32_pay_dividends()

    def _event30_collect_revenue(self) -> None:
        """Event 30: collect revenue from sales; compute gross_profit.

        Mesa port: Firm.collect_revenue()
          qty_sold = self.production - self.inventory
          revenue = self.price * qty_sold
          self.total_funds += revenue
          self.gross_profit = revenue - self.wage_bill

        After event 22, inventory == production (full overwrite), so qty_sold starts
        at 0 until the goods market (Task 7) depletes inventory.  The formula is
        faithfully translated; revenue is non-zero once Task 7 is implemented.

        No RNG.
        """
        fdf = self.firms.agents
        qty_sold = pl.col("production") - pl.col("inventory")
        revenue = pl.col("price") * qty_sold
        self.firms.agents = fdf.with_columns(
            (pl.col("total_funds") + revenue).alias("total_funds"),
            (revenue - pl.col("wage_bill")).alias("gross_profit"),
        )

    def _event31_validate_debt(self) -> None:
        """Event 31: repay loans if solvent; write off bad debt if not; compute net_profit.

        Mesa port: Firm.validate_debt(model)
          total_debt     = sum(loan.principal*(1+rate) for loan in self.loans)
          total_interest = sum(loan.principal*rate      for loan in self.loans)
          total_principal= sum(loan.principal           for loan in self.loans)
          if total_debt > eps:
            if total_funds - total_debt >= -eps:   # solvent
              total_funds -= total_debt
              for loan: loan.lender.equity_base += loan.interest
            else:                                  # default
              total_funds = 0
              for loan:
                frac = principal / max(total_principal, eps)
                recovery = clip(frac * net_worth, 0, principal)
                loss = principal - recovery
                loan.lender.equity_base -= loss
          net_profit = gross_profit - total_interest   # ALL firms

        Notes:
        - Uses CURRENT (pre-update) net_worth for recovery (spec flag 7).
        - Banks book interest ONLY on fully-repaid loans.
        - Loans are retained in the table here; purged at next credit market open.
        - self.loans refers to the columnar self.model.loans in the mesa-frames port.
        """
        eps = self.EPS
        fdf = self.firms.agents
        bdf = self.banks.agents
        loans = self.loans

        if len(loans) == 0:
            # No outstanding loans: net_profit == gross_profit for all firms.
            self.firms.agents = fdf.with_columns(
                pl.col("gross_profit").alias("net_profit")
            )
            return

        # Per-firm loan aggregates.
        agg = (
            loans.with_columns(
                (pl.col("principal") * pl.col("interest_rate")).alias("_interest"),
                (pl.col("principal") * (1.0 + pl.col("interest_rate"))).alias("_debt"),
            )
            .group_by("borrower_id")
            .agg(
                pl.col("_debt").sum().alias("_total_debt"),
                pl.col("_interest").sum().alias("_total_interest"),
                pl.col("principal").sum().alias("_total_principal"),
            )
            .rename({"borrower_id": "unique_id"})
        )

        # Join onto firms (non-borrowers get 0).
        fdf_ext = fdf.join(agg, on="unique_id", how="left").with_columns(
            pl.col("_total_debt").fill_null(0.0),
            pl.col("_total_interest").fill_null(0.0),
            pl.col("_total_principal").fill_null(0.0),
        )

        # Classify each firm: has_debt, solvent, defaulting.
        has_debt = pl.col("_total_debt") > eps
        is_solvent = has_debt & (
            (pl.col("total_funds") - pl.col("_total_debt")) >= -eps
        )
        is_defaulting = has_debt & (
            (pl.col("total_funds") - pl.col("_total_debt")) < -eps
        )

        new_total_funds = (
            pl.when(is_solvent)
            .then(pl.col("total_funds") - pl.col("_total_debt"))
            .when(is_defaulting)
            .then(pl.lit(0.0))
            .otherwise(pl.col("total_funds"))
        )
        net_profit = pl.col("gross_profit") - pl.col("_total_interest")

        # Build a lookup for bank equity updates: need firm solvency + per-loan info.
        # Capture the classification before dropping auxiliary columns.
        firm_meta = fdf_ext.select(
            ["unique_id", "total_funds", "net_worth", "_total_debt", "_total_principal"]
        )
        firm_meta_map = {
            row["unique_id"]: row for row in firm_meta.iter_rows(named=True)
        }

        self.firms.agents = fdf_ext.with_columns(
            new_total_funds.alias("total_funds"),
            net_profit.alias("net_profit"),
        ).drop(["_total_debt", "_total_interest", "_total_principal"])

        # Apply bank equity updates per loan (mirrors the Mesa port's per-loan loop).
        bank_delta: dict[int, float] = {}
        for loan_row in loans.iter_rows(named=True):
            fid = int(loan_row["borrower_id"])
            bid = int(loan_row["lender_id"])
            principal = float(loan_row["principal"])
            rate = float(loan_row["interest_rate"])
            interest = principal * rate

            meta = firm_meta_map.get(fid)
            if meta is None:
                continue
            total_debt_f = float(meta["_total_debt"])
            if total_debt_f <= eps:
                continue
            total_funds_f = float(meta["total_funds"])

            if (total_funds_f - total_debt_f) >= -eps:
                # Solvent: lender earns interest.
                bank_delta[bid] = bank_delta.get(bid, 0.0) + interest
            else:
                # Default: lender takes proportional loss.
                total_principal_f = float(meta["_total_principal"])
                nw = float(meta["net_worth"])
                frac = principal / max(total_principal_f, eps)
                recovery = min(max(frac * nw, 0.0), principal)
                loss = principal - recovery
                bank_delta[bid] = bank_delta.get(bid, 0.0) - loss

        if bank_delta:
            new_equity = [
                row["equity_base"] + bank_delta.get(int(row["unique_id"]), 0.0)
                for row in bdf.iter_rows(named=True)
            ]
            self.banks.agents = bdf.with_columns(
                pl.Series("equity_base", new_equity, dtype=pl.Float64)
            )

    def _event32_pay_dividends(self) -> None:
        """Event 32: profitable firms pay a fraction of net_profit as dividends.

        Mesa port: BamModel._pay_dividends()
          delta = self.p["delta"]
          total_dividends = 0
          for f in self.firms:
            retained = f.net_profit
            if f.net_profit > 0: retained *= (1 - delta)
            dividends_paid = f.net_profit - retained
            f.total_funds -= dividends_paid
            f.retained_profit = retained
            total_dividends += dividends_paid
          div_per_hh = total_dividends / n_households
          for h in self.households:
            h.savings += div_per_hh; h.dividends = div_per_hh

        Only profitable firms (net_profit > 0) pay dividends (spec flag 8).
        div_per_hh is split EQUALLY across ALL households.
        No RNG.
        """
        delta = float(self.p["delta"])
        fdf = self.firms.agents

        # retained_profit = net_profit * (1-delta) if net_profit > 0 else net_profit.
        retained = (
            pl.when(pl.col("net_profit") > 0.0)
            .then(pl.col("net_profit") * (1.0 - delta))
            .otherwise(pl.col("net_profit"))
        )
        dividends_paid = pl.col("net_profit") - retained

        self.firms.agents = fdf.with_columns(
            retained.alias("retained_profit"),
            (pl.col("total_funds") - dividends_paid).alias("total_funds"),
        )

        # Total dividends pooled and split equally across all households.
        fdf_after = self.firms.agents
        total_dividends = float(
            fdf_after.select(
                (pl.col("net_profit") - pl.col("retained_profit")).sum()
            ).item()
        )
        n_hh = self.n_households
        div_per_hh = total_dividends / n_hh if n_hh > 0 else 0.0

        hdf = self.households.agents
        self.households.agents = hdf.with_columns(
            (pl.col("savings") + div_per_hh).alias("savings"),
            pl.lit(div_per_hh).alias("dividends"),
        )

    def step(self) -> None:
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
        # Bankruptcy + entry added in Task 8.
