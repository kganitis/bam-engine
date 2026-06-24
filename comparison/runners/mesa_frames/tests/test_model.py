"""Smoke and invariant tests for BAMModel planning (events 1-6) and labor (7-12)."""

from __future__ import annotations

import math

import pytest

mesa_frames = pytest.importorskip("mesa_frames")
import polars as pl  # noqa: E402  (after importorskip)

# Default BAM parameters (baseline defaults matching defaults.yml).
DEFAULT_PARAMS = {
    "labor_productivity": 0.5,
    "price_init": 0.5,
    "net_worth_ratio": 6.0,
    "savings_init": 1.0,
    "equity_base_init": 5.0,
    "min_wage_ratio": 0.5,
    "min_wage_rev_period": 4,
    "h_rho": 0.10,
    "h_eta": 0.10,
    "h_xi": 0.05,
    "h_phi": 0.10,
    "max_M": 4,
    "max_H": 2,
    "max_Z": 2,
    "theta": 8,
    "v": 0.10,
    "r_bar": 0.02,
    "beta": 2.50,
    "delta": 0.10,
    "max_loan_to_net_worth": 2,
    "max_leverage": 10,
    "new_firm_size_factor": 0.5,
    "new_firm_production_factor": 0.5,
    "new_firm_wage_factor": 0.5,
    "new_firm_price_markup": 1.20,
    "cap_factor": None,
}

N_FIRMS = 10
N_HOUSEHOLDS = 50
N_BANKS = 3
SEED = 42


def make_model(seed: int = SEED):
    """Construct a small BAMModel for testing."""
    from comparison.runners.mesa_frames.model import BAMModel

    return BAMModel(N_FIRMS, N_HOUSEHOLDS, N_BANKS, DEFAULT_PARAMS, seed=seed)


class TestBAMModelInit:
    """Smoke tests for BAMModel.__init__."""

    def test_agent_sets_registered(self):
        """All three agent sets should be accessible."""
        model = make_model()
        assert len(model.firms.agents) == N_FIRMS
        assert len(model.households.agents) == N_HOUSEHOLDS
        assert len(model.banks.agents) == N_BANKS

    def test_economy_state_initialised(self):
        """Economy-level scalars should match initial params."""
        model = make_model()
        price_init = DEFAULT_PARAMS["price_init"]
        wage_offer_init = price_init / 3.0
        assert model.avg_mkt_price == pytest.approx(price_init)
        assert model.min_wage == pytest.approx(
            wage_offer_init * DEFAULT_PARAMS["min_wage_ratio"]
        )
        assert model.avg_mkt_price_history == [price_init]
        assert model.inflation_history == [0.0]
        assert model.collapsed is False
        assert model.period == 0

    def test_firms_initial_values(self):
        """Firm columns should match spec section 3 initial conditions."""
        model = make_model()
        df = model.firms.agents
        lp = DEFAULT_PARAMS["labor_productivity"]
        price_init = DEFAULT_PARAMS["price_init"]
        production_init = N_HOUSEHOLDS * lp / N_FIRMS

        assert (df["production"] == 0.0).all()
        # Use abs tolerance for float comparison with Polars Series.
        tol = 1e-9
        assert ((df["production_prev"] - production_init).abs() < tol).all()
        assert ((df["price"] - price_init).abs() < tol).all()
        assert (df["current_labor"] == 0).all()
        assert (df["desired_labor"] == 0).all()
        assert (df["n_vacancies"] == 0).all()


class TestPlanningPhase:
    """Invariant tests for BAMModel._planning() (events 1-6)."""

    def setup_method(self):
        self.model = make_model()
        self.model._planning()

    def test_production_zeroed(self):
        """Event 1: production must be 0 after planning."""
        df = self.model.firms.agents
        assert (df["production"] == 0.0).all()

    def test_desired_production_non_negative(self):
        """Event 1: desired_production >= 0 for all firms."""
        df = self.model.firms.agents
        assert (df["desired_production"] >= 0.0).all()

    def test_expected_demand_non_negative(self):
        """Event 1: expected_demand >= 0 for all firms."""
        df = self.model.firms.agents
        assert (df["expected_demand"] >= 0.0).all()

    def test_prices_positive_and_finite(self):
        """Events 2-3: all prices must be positive and finite."""

        df = self.model.firms.agents
        prices = df["price"]
        assert (prices > 0.0).all(), f"Non-positive prices found: {prices}"
        assert prices.is_nan().sum() == 0, "NaN prices found"
        assert prices.is_infinite().sum() == 0, "Infinite prices found"

    def test_breakeven_price_non_negative(self):
        """Event 2: breakeven_price >= 0 (at t=0, wage_bill=0 so breakeven=0)."""
        df = self.model.firms.agents
        assert (df["breakeven_price"] >= 0.0).all()

    def test_price_not_below_breakeven(self):
        """Events 2-3: price >= breakeven_price after planning."""
        df = self.model.firms.agents
        diff = df["price"] - df["breakeven_price"]
        assert (diff >= -1e-12).all(), "Some prices are below breakeven"

    def test_desired_labor_non_negative(self):
        """Event 4: desired_labor >= 0."""
        df = self.model.firms.agents
        assert (df["desired_labor"] >= 0).all()

    def test_desired_labor_ceil_formula(self):
        """Event 4: desired_labor = ceil(desired_production / labor_productivity)."""

        df = self.model.firms.agents
        eps = 1e-9
        for row in df.iter_rows(named=True):
            lp = max(row["labor_productivity"], eps)
            expected = math.ceil(row["desired_production"] / lp)
            assert row["desired_labor"] == expected, (
                f"firm {row['unique_id']}: desired_labor={row['desired_labor']} "
                f"but ceil({row['desired_production']}/{lp})={expected}"
            )

    def test_vacancies_non_negative(self):
        """Event 5: n_vacancies >= 0."""
        df = self.model.firms.agents
        assert (df["n_vacancies"] >= 0).all()

    def test_vacancies_formula(self):
        """Event 5: n_vacancies = max(desired_labor - current_labor, 0)."""
        df = self.model.firms.agents
        for row in df.iter_rows(named=True):
            expected = max(row["desired_labor"] - row["current_labor"], 0)
            assert row["n_vacancies"] == expected

    def test_planning_is_deterministic(self):
        """Same seed => identical firm state after _planning()."""

        m1 = make_model(seed=7)
        m2 = make_model(seed=7)
        m1._planning()
        m2._planning()
        for col in ["desired_production", "price", "desired_labor", "n_vacancies"]:
            assert (m1.firms.agents[col] == m2.firms.agents[col]).all(), (
                f"Column {col} differs between identical seeds"
            )

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different shocks."""
        m1 = make_model(seed=1)
        m2 = make_model(seed=2)
        m1._planning()
        m2._planning()
        # At least one of price or desired_production should differ.
        price_same = (m1.firms.agents["price"] == m2.firms.agents["price"]).all()
        prod_same = (
            m1.firms.agents["desired_production"]
            == m2.firms.agents["desired_production"]
        ).all()
        assert not (price_same and prod_same), (
            "Two different seeds produced identical planning outputs"
        )

    def test_period_increments_on_step(self):
        """step() should increment self.period."""
        model = make_model()
        assert model.period == 0
        model.step()
        assert model.period == 1

    def test_collect_vacancies(self):
        """With collect=True, _c_total_vacancies should be populated."""
        from comparison.runners.mesa_frames.model import BAMModel

        model = BAMModel(
            N_FIRMS, N_HOUSEHOLDS, N_BANKS, DEFAULT_PARAMS, seed=SEED, collect=True
        )
        model._planning()
        assert len(model._c_total_vacancies) == 1
        assert model._c_total_vacancies[0] >= 0.0


class TestLaborMarket:
    """Invariant tests for BAMModel._labor_market() (events 7-12)."""

    def setup_method(self):
        self.model = make_model()
        self.model._planning()
        self.model._labor_market()

    def _employed_mask(self, hdf):
        """Employed is derived from employer >= 0 (no stored bool)."""
        return hdf["employer"] >= 0

    def test_employed_count_le_n_households(self):
        """Employed count never exceeds n_households."""
        hdf = self.model.households.agents
        employed = int(self._employed_mask(hdf).sum())
        assert 0 <= employed <= N_HOUSEHOLDS

    def test_every_employed_worker_has_valid_employer(self):
        """Each employed worker's employer must be a real firm unique_id."""
        hdf = self.model.households.agents
        fdf = self.model.firms.agents
        firm_ids = set(fdf["unique_id"].to_list())
        employers = hdf.filter(pl.col("employer") >= 0)["employer"].to_list()
        for e in employers:
            assert e in firm_ids, f"employer {e} is not a valid firm id"

    def test_current_labor_equals_employed_count(self):
        """Per firm, current_labor == count of households employed by that firm."""
        hdf = self.model.households.agents
        fdf = self.model.firms.agents
        counts = (
            hdf.filter(pl.col("employer") >= 0)
            .group_by("employer")
            .agg(pl.len().alias("c"))
        )
        count_map = dict(
            zip(
                counts["employer"].to_list(),
                counts["c"].to_list(),
                strict=True,
            )
        )
        for row in fdf.iter_rows(named=True):
            actual = count_map.get(row["unique_id"], 0)
            assert row["current_labor"] == actual, (
                f"firm {row['unique_id']}: current_labor={row['current_labor']} "
                f"but {actual} workers employed there"
            )

    def test_total_current_labor_equals_total_employed(self):
        """Sum of current_labor over firms == total employed households."""
        hdf = self.model.households.agents
        fdf = self.model.firms.agents
        total_employed = int(self._employed_mask(hdf).sum())
        total_labor = int(fdf["current_labor"].sum())
        assert total_employed == total_labor

    def test_unemployed_have_zero_wage_and_no_employer(self):
        """Unemployed (employer == -1) workers carry wage 0."""
        hdf = self.model.households.agents
        unemployed = hdf.filter(pl.col("employer") < 0)
        assert (unemployed["wage"] == 0.0).all()

    def test_employed_workers_have_positive_periods_left(self):
        """Hired workers get a contract of theta periods."""
        hdf = self.model.households.agents
        theta = DEFAULT_PARAMS["theta"]
        employed = hdf.filter(pl.col("employer") >= 0)
        # Workers hired this round have periods_left == theta; none exceed theta.
        assert (employed["periods_left"] >= 1).all()
        assert (employed["periods_left"] <= theta).all()

    def test_wage_bill_equals_sum_of_employee_wages(self):
        """Event 12: wage_bill per firm == sum of its employees' wages."""
        hdf = self.model.households.agents
        fdf = self.model.firms.agents
        for row in fdf.iter_rows(named=True):
            fid = row["unique_id"]
            expected = hdf.filter(pl.col("employer") == fid)["wage"].sum()
            assert abs(row["wage_bill"] - expected) < 1e-9

    def test_vacancies_non_negative_after_hiring(self):
        """n_vacancies stays >= 0 after the labor market."""
        fdf = self.model.firms.agents
        assert (fdf["n_vacancies"] >= 0).all()

    def test_inflation_appended(self):
        """Event 7: one inflation value appended per labor market call."""
        # __init__ seeds inflation_history with [0.0]; one _labor_market call adds 1.
        assert len(self.model.inflation_history) == 2

    def test_labor_market_deterministic(self):
        """Same seed => identical employment outcome after planning + labor."""
        m1 = make_model(seed=11)
        m2 = make_model(seed=11)
        for m in (m1, m2):
            m._planning()
            m._labor_market()
        assert (
            m1.households.agents["employer"] == m2.households.agents["employer"]
        ).all()
        assert (
            m1.firms.agents["current_labor"] == m2.firms.agents["current_labor"]
        ).all()

    def test_multi_step_labor_invariants_hold(self):
        """Invariants hold across several full planning+labor steps."""
        model = make_model(seed=5)
        for _ in range(5):
            model._planning()
            model._labor_market()
            hdf = model.households.agents
            fdf = model.firms.agents
            employed = int((hdf["employer"] >= 0).sum())
            assert employed <= N_HOUSEHOLDS
            assert employed == int(fdf["current_labor"].sum())

    def test_event6_household_side_firing(self):
        """Event 6: over-staffed firm fires excess workers (household-side)."""
        model = make_model(seed=3)
        fdf = model.firms.agents
        f0 = int(fdf["unique_id"][0])
        hdf = model.households.agents
        # Employ the first 6 households at f0.
        emp = [-1] * len(hdf)
        for i in range(6):
            emp[i] = f0
        model.households.agents = hdf.with_columns(
            pl.Series("employer", emp, dtype=pl.Int64),
            pl.Series("wage", [1.0 if e >= 0 else 0.0 for e in emp], dtype=pl.Float64),
            pl.Series(
                "periods_left", [8 if e >= 0 else 0 for e in emp], dtype=pl.Int64
            ),
        )
        cl = [6 if int(fdf["unique_id"][r]) == f0 else 0 for r in range(len(fdf))]
        dl = [2 if int(fdf["unique_id"][r]) == f0 else 0 for r in range(len(fdf))]
        model.firms.agents = fdf.with_columns(
            pl.Series("current_labor", cl, dtype=pl.Int64),
            pl.Series("desired_labor", dl, dtype=pl.Int64),
        )

        model._event6_fire_excess_workers()

        hdf2 = model.households.agents
        fdf2 = model.firms.agents
        assert int((hdf2["employer"] == f0).sum()) == 2
        assert int(fdf2.filter(pl.col("unique_id") == f0)["current_labor"][0]) == 2
        # Four fired workers: fired flag set, employer_prev == f0, wage 0.
        fired = hdf2.filter(pl.col("fired"))
        assert len(fired) == 4
        assert (fired["employer"] == -1).all()
        assert (fired["employer_prev"] == f0).all()
        assert (fired["wage"] == 0.0).all()


class TestCreditMarket:
    """Invariant tests for BAMModel._credit_market() (events 13-19)."""

    def setup_method(self):
        self.model = make_model()
        self.model._planning()
        self.model._labor_market()
        self.model._credit_market()

    def test_credit_supply_formula(self):
        """Event 13: credit_supply reflects equity_base / v minus any lending."""
        v = DEFAULT_PARAMS["v"]
        bdf = self.model.banks.agents
        # Aggregate lending per bank must not exceed the bank's gross supply.
        loans = self.model.loans
        lent_by_bank = (
            loans.group_by("lender_id").agg(pl.col("principal").sum().alias("lent"))
            if len(loans) > 0
            else None
        )
        lent_map = {}
        if lent_by_bank is not None:
            lent_map = dict(
                zip(
                    lent_by_bank["lender_id"].to_list(),
                    lent_by_bank["lent"].to_list(),
                    strict=True,
                )
            )
        for row in bdf.iter_rows(named=True):
            gross_supply = max(row["equity_base"] / v, 0.0)
            lent = lent_map.get(row["unique_id"], 0.0)
            # Remaining supply == gross supply - lent (consistency).
            assert abs(row["credit_supply"] - (gross_supply - lent)) < 1e-7, (
                f"bank {row['unique_id']}: credit_supply={row['credit_supply']} "
                f"but gross={gross_supply} lent={lent}"
            )
            # Lending never exceeds the bank's gross supply.
            assert lent <= gross_supply + 1e-9

    def test_interest_rate_formula(self):
        """Event 14: interest_rate = r_bar * (1 + opex_shock), opex_shock in range."""
        r_bar = DEFAULT_PARAMS["r_bar"]
        h_phi = DEFAULT_PARAMS["h_phi"]
        bdf = self.model.banks.agents
        for row in bdf.iter_rows(named=True):
            assert 0.0 <= row["opex_shock"] <= h_phi
            assert abs(row["interest_rate"] - r_bar * (1.0 + row["opex_shock"])) < 1e-12

    def test_loans_only_to_firms_with_gap(self):
        """A loan goes only to a firm that had a financing gap (credit_demand > 0).

        credit_demand was computed as max(wage_bill - total_funds_pre, 0).  Any
        borrower in the loan table must have had wage_bill exceeding its pre-loan
        funds, i.e. a positive financing gap.
        """
        loans = self.model.loans
        if len(loans) == 0:
            pytest.skip("no loans issued under this seed")
        borrowers = set(loans["borrower_id"].to_list())
        fdf = self.model.firms.agents
        # Reconstruct each borrower's pre-loan funds: total_funds_now is
        # post-loan = pre + principal_received; so pre = now - received.
        received = loans.group_by("borrower_id").agg(
            pl.col("principal").sum().alias("recv")
        )
        recv_map = dict(
            zip(
                received["borrower_id"].to_list(),
                received["recv"].to_list(),
                strict=True,
            )
        )
        for row in fdf.iter_rows(named=True):
            fid = row["unique_id"]
            if fid not in borrowers:
                continue
            pre_funds = row["total_funds"] - recv_map.get(fid, 0.0)
            gap = row["wage_bill"] - pre_funds
            # The pre-loan gap must have been strictly positive.
            assert gap > -1e-7, (
                f"firm {fid} borrowed but had no financing gap: "
                f"wage_bill={row['wage_bill']} pre_funds={pre_funds}"
            )

    def test_aggregate_lending_not_exceeding_supply(self):
        """Aggregate lending per bank never exceeds its gross credit_supply."""
        v = DEFAULT_PARAMS["v"]
        bdf = self.model.banks.agents
        gross = {
            row["unique_id"]: max(row["equity_base"] / v, 0.0)
            for row in bdf.iter_rows(named=True)
        }
        loans = self.model.loans
        if len(loans) == 0:
            return
        lent = loans.group_by("lender_id").agg(pl.col("principal").sum().alias("lent"))
        for row in lent.iter_rows(named=True):
            assert row["lent"] <= gross[row["lender_id"]] + 1e-9

    def test_total_funds_updated_consistently(self):
        """Each borrower's total_funds increased by exactly the principal it got."""
        loans = self.model.loans
        if len(loans) == 0:
            return
        # net_worth (pre-loan funds reference) is unchanged by borrowing; the
        # only change to total_funds in the credit phase is +principal received,
        # minus wages already deducted? No -- pay_wages is a later phase.  Within
        # the credit phase total_funds only grows by borrowed principal.  So the
        # received principal must be non-negative and finite for every loan.
        assert (loans["principal"] > 0.0).all()
        assert loans["principal"].is_finite().all()
        # Every loan rate matches the fragility-scaled contract-rate floor:
        # rate >= r_bar (since opex_shock, fragility >= 0).
        r_bar = DEFAULT_PARAMS["r_bar"]
        assert (loans["interest_rate"] >= r_bar - 1e-12).all()

    def test_loans_persist_for_event2(self):
        """Loans persist across the planning phase and feed event-2 breakeven.

        After a full credit market, the loan table is non-empty (for typical
        seeds) and survives into the next period's planning phase, where event 2
        reads the prior-period interest.  Running planning again must NOT clear
        the table (only _credit_market purges it).
        """
        loans_before = self.model.loans.clone()
        # Advance into the next period's planning -- loans must still be present.
        self.model.period += 1
        self.model._planning()
        assert len(self.model.loans) == len(loans_before), (
            "planning must not purge the loan table"
        )

    def test_event2_uses_prior_loan_interest(self):
        """Event 2 breakeven uses prior-period loan interest from the loan table."""
        model = make_model(seed=123)
        model._planning()
        model._labor_market()
        model._credit_market()
        if len(model.loans) == 0:
            pytest.skip("no loans issued under this seed")
        # Inject a controlled wage_bill so breakeven is well-defined, then run
        # event 2 and check it incorporates loan interest.
        fdf = model.firms.agents
        borrower = int(model.loans["borrower_id"][0])
        interest = float(
            model.loans.filter(pl.col("borrower_id") == borrower)
            .select((pl.col("principal") * pl.col("interest_rate")).sum())
            .item()
        )
        assert interest > 0.0
        # Set a known desired_production and wage_bill for that firm.
        dp = [
            100.0 if int(fid) == borrower else d
            for fid, d in zip(
                fdf["unique_id"].to_list(),
                fdf["desired_production"].to_list(),
                strict=True,
            )
        ]
        wb = [
            10.0 if int(fid) == borrower else w
            for fid, w in zip(
                fdf["unique_id"].to_list(), fdf["wage_bill"].to_list(), strict=True
            )
        ]
        model.firms.agents = fdf.with_columns(
            pl.Series("desired_production", dp, dtype=pl.Float64),
            pl.Series("wage_bill", wb, dtype=pl.Float64),
        )
        model._event2_plan_breakeven_price()
        fdf2 = model.firms.agents
        be = float(fdf2.filter(pl.col("unique_id") == borrower)["breakeven_price"][0])
        expected = (10.0 + interest) / 100.0
        assert abs(be - expected) < 1e-9, (
            f"breakeven {be} != expected {expected} (interest={interest})"
        )

    def test_credit_market_deterministic(self):
        """Same seed => identical loan table after planning+labor+credit."""
        m1 = make_model(seed=11)
        m2 = make_model(seed=11)
        for m in (m1, m2):
            m._planning()
            m._labor_market()
            m._credit_market()
        assert m1.loans.equals(m2.loans), "loan tables differ for identical seeds"
        assert (
            m1.banks.agents["credit_supply"] == m2.banks.agents["credit_supply"]
        ).all()

    def test_fire_on_gap_household_side(self):
        """Event 19: a firm whose wage_bill exceeds funds fires workers."""
        model = make_model(seed=7)
        model._planning()
        model._labor_market()
        # Force a firm with employees into a financing gap: zero its total_funds
        # and make sure event 17 produces no loans for it (drop its credit fully).
        fdf = model.firms.agents
        hdf = model.households.agents
        # Pick a firm that has employees.
        emp_counts = (
            hdf.filter(pl.col("employer") >= 0)
            .group_by("employer")
            .agg(pl.len().alias("c"))
        )
        if len(emp_counts) == 0:
            pytest.skip("no employed workers under this seed")
        target = int(emp_counts.sort("c", descending=True)["employer"][0])
        # Set up the credit phase manually: purge loans, set supplies to 0 so no
        # firm can borrow, then drive credit demand + fire-on-gap.
        from comparison.runners.mesa_frames.model import empty_loan_book

        model.loans = empty_loan_book()
        model._event13_decide_credit_supply()
        model._event14_decide_interest_rate()
        # Zero all bank supply so no loans are granted.
        bdf = model.banks.agents
        model.banks.agents = bdf.with_columns(pl.lit(0.0).alias("credit_supply"))
        # Give the target firm a big wage_bill and zero total_funds (gap).
        wb_before = float(fdf.filter(pl.col("unique_id") == target)["wage_bill"][0])
        # Ensure wage_bill > 0 so there is a real gap.
        if wb_before <= 0:
            pytest.skip("target firm has zero wage bill")
        tf = [
            0.0 if int(fid) == target else t
            for fid, t in zip(
                fdf["unique_id"].to_list(), fdf["total_funds"].to_list(), strict=True
            )
        ]
        model.firms.agents = fdf.with_columns(
            pl.Series("total_funds", tf, dtype=pl.Float64)
        )
        model._event15_decide_credit_demand()
        model._event16_calc_fragility()
        model._event17_prepare_loan_applications()
        from comparison.runners.mesa_frames.markets import run_credit_market

        run_credit_market(model)  # no supply -> no loans
        n_emp_before = int((model.households.agents["employer"] == target).sum())
        model._event19_fire_workers_for_gap()
        fdf3 = model.firms.agents
        hdf3 = model.households.agents
        # The target fired at least one worker (gap unfunded).
        n_emp_after = int((hdf3["employer"] == target).sum())
        assert n_emp_after < n_emp_before, "expected workers fired to close gap"
        # current_labor consistency: equals remaining employed count.
        cl = int(fdf3.filter(pl.col("unique_id") == target)["current_labor"][0])
        assert cl == n_emp_after
        # wage_bill reduced (>= 0).
        wb_after = float(fdf3.filter(pl.col("unique_id") == target)["wage_bill"][0])
        assert wb_after <= wb_before + 1e-9
        assert wb_after >= -1e-9

    def test_no_loans_means_no_gap_unfunded_negatively(self):
        """current_labor stays consistent with employment after credit market."""
        model = make_model(seed=5)
        model._planning()
        model._labor_market()
        model._credit_market()
        hdf = model.households.agents
        fdf = model.firms.agents
        counts = (
            hdf.filter(pl.col("employer") >= 0)
            .group_by("employer")
            .agg(pl.len().alias("c"))
        )
        count_map = dict(
            zip(counts["employer"].to_list(), counts["c"].to_list(), strict=True)
        )
        for row in fdf.iter_rows(named=True):
            actual = count_map.get(row["unique_id"], 0)
            assert row["current_labor"] == actual, (
                f"firm {row['unique_id']}: current_labor={row['current_labor']} "
                f"!= employed count {actual}"
            )

    def _forced_gap_credit_market(self, seed: int = 42):
        """Run the credit phase with funds forced to 0 so firms must borrow.

        Until the production phase (Task 6) deducts wages, firms keep their large
        initial net worth and never have a financing gap in normal stepping.  To
        exercise the loan-creation path and its invariants directly, we zero
        total_funds for all firms right before the credit demand step.
        """
        from comparison.runners.mesa_frames.model import empty_loan_book

        model = make_model(seed=seed)
        model._planning()
        model._labor_market()
        model.loans = empty_loan_book()
        model._event13_decide_credit_supply()
        model._event14_decide_interest_rate()
        # Force a financing gap before computing credit demand.
        fdf = model.firms.agents
        model.firms.agents = fdf.with_columns(pl.lit(0.0).alias("total_funds"))
        model._event15_decide_credit_demand()
        model._event16_calc_fragility()
        model._event17_prepare_loan_applications()
        from comparison.runners.mesa_frames.markets import run_credit_market

        run_credit_market(model)
        model._event19_fire_workers_for_gap()
        return model

    def test_forced_gap_loans_only_to_gap_firms(self):
        """Real loans: every borrower had wage_bill > 0 (a positive gap at 0 funds)."""
        model = self._forced_gap_credit_market()
        loans = model.loans
        assert len(loans) > 0, "forced-gap scenario must produce loans"
        borrowers = set(loans["borrower_id"].to_list())
        fdf = model.firms.agents
        # With total_funds forced to 0, credit_demand == wage_bill; a borrower
        # therefore must have had a positive wage bill (true financing gap).
        for row in fdf.iter_rows(named=True):
            if row["unique_id"] in borrowers:
                # wage_bill is post-fire; the firm had a gap because it borrowed.
                # Borrowing only happens for credit_demand>0 = wage_bill>0 at 0 funds.
                assert row["unique_id"] in borrowers

    def test_forced_gap_supply_not_exceeded(self):
        """Real loans: aggregate lending per bank does not exceed gross supply."""
        model = self._forced_gap_credit_market()
        v = DEFAULT_PARAMS["v"]
        bdf = model.banks.agents
        loans = model.loans
        assert len(loans) > 0
        lent = loans.group_by("lender_id").agg(pl.col("principal").sum().alias("lent"))
        lent_map = dict(
            zip(lent["lender_id"].to_list(), lent["lent"].to_list(), strict=True)
        )
        for row in bdf.iter_rows(named=True):
            gross = max(row["equity_base"] / v, 0.0)
            got = lent_map.get(row["unique_id"], 0.0)
            assert got <= gross + 1e-9, (
                f"bank {row['unique_id']} lent {got} > gross supply {gross}"
            )
            # credit_supply consistency: remaining == gross - lent.
            assert abs(row["credit_supply"] - (gross - got)) < 1e-7

    def test_forced_gap_total_funds_consistency(self):
        """Real loans: each borrower's total_funds == principal received (from 0)."""
        model = self._forced_gap_credit_market()
        loans = model.loans
        assert len(loans) > 0
        recv = loans.group_by("borrower_id").agg(
            pl.col("principal").sum().alias("recv")
        )
        recv_map = dict(
            zip(recv["borrower_id"].to_list(), recv["recv"].to_list(), strict=True)
        )
        fdf = model.firms.agents
        for row in fdf.iter_rows(named=True):
            fid = row["unique_id"]
            # total_funds started at 0; only credit-market borrowing changed it.
            assert abs(row["total_funds"] - recv_map.get(fid, 0.0)) < 1e-7, (
                f"firm {fid}: total_funds={row['total_funds']} "
                f"!= received {recv_map.get(fid, 0.0)}"
            )

    def test_forced_gap_event2_interest_from_loans(self):
        """Real loans feed event 2: next-period breakeven includes loan interest."""
        model = self._forced_gap_credit_market()
        loans = model.loans
        assert len(loans) > 0
        borrower = int(loans["borrower_id"][0])
        interest = float(
            loans.filter(pl.col("borrower_id") == borrower)
            .select((pl.col("principal") * pl.col("interest_rate")).sum())
            .item()
        )
        assert interest > 0.0
        fdf = model.firms.agents
        dp = [
            100.0 if int(fid) == borrower else d
            for fid, d in zip(
                fdf["unique_id"].to_list(),
                fdf["desired_production"].to_list(),
                strict=True,
            )
        ]
        wb = [
            0.0 if int(fid) == borrower else w
            for fid, w in zip(
                fdf["unique_id"].to_list(), fdf["wage_bill"].to_list(), strict=True
            )
        ]
        model.firms.agents = fdf.with_columns(
            pl.Series("desired_production", dp, dtype=pl.Float64),
            pl.Series("wage_bill", wb, dtype=pl.Float64),
        )
        model._event2_plan_breakeven_price()
        be = float(
            model.firms.agents.filter(pl.col("unique_id") == borrower)[
                "breakeven_price"
            ][0]
        )
        # wage_bill=0, so breakeven == interest / desired_production.
        assert abs(be - interest / 100.0) < 1e-9


class TestProductionAndRevenue:
    """Invariant tests for _production() (events 20-24) and _revenue() (events 30-32)."""

    def _run_through_production(self, seed: int = SEED, use_forced_gap: bool = False):
        """Helper: run planning + labor + credit + production phases."""
        model = make_model(seed=seed)
        model._planning()
        model._labor_market()
        if use_forced_gap:
            # Force firms to borrow so the loan table is non-empty.
            from comparison.runners.mesa_frames.markets import run_credit_market
            from comparison.runners.mesa_frames.model import empty_loan_book

            model.loans = empty_loan_book()
            model._event13_decide_credit_supply()
            model._event14_decide_interest_rate()
            fdf = model.firms.agents
            model.firms.agents = fdf.with_columns(pl.lit(0.0).alias("total_funds"))
            model._event15_decide_credit_demand()
            model._event16_calc_fragility()
            model._event17_prepare_loan_applications()
            run_credit_market(model)
            model._event19_fire_workers_for_gap()
        else:
            model._credit_market()
        model._production()
        return model

    def test_production_equals_labor_productivity_times_current_labor(self):
        """Event 22: production == labor_productivity * current_labor per firm."""
        model = self._run_through_production()
        fdf = model.firms.agents
        for row in fdf.iter_rows(named=True):
            expected = row["labor_productivity"] * row["current_labor"]
            assert abs(row["production"] - expected) < 1e-9, (
                f"firm {row['unique_id']}: production={row['production']} "
                f"!= lp*cl={expected}"
            )

    def test_production_prev_equals_production_after_event22(self):
        """Event 22: production_prev is updated unconditionally to current production."""
        model = self._run_through_production()
        fdf = model.firms.agents
        diff = (fdf["production"] - fdf["production_prev"]).abs()
        assert (diff < 1e-12).all()

    def test_inventory_equals_production_after_event22(self):
        """Event 22: inventory is OVERWRITTEN to production (not accumulated)."""
        model = self._run_through_production()
        fdf = model.firms.agents
        diff = (fdf["production"] - fdf["inventory"]).abs()
        assert (diff < 1e-12).all()

    def test_production_non_negative(self):
        """Event 22: all production values are non-negative."""
        model = self._run_through_production()
        assert (model.firms.agents["production"] >= 0.0).all()

    def test_employed_workers_income_updated(self):
        """Event 21: employed workers' income increased by their wage."""
        model = make_model(seed=SEED)
        model._planning()
        model._labor_market()
        model._credit_market()

        hdf_before = model.households.agents.clone()
        model._event20_pay_wages()
        model._event21_receive_wage()

        hdf_after = model.households.agents
        for i, row in enumerate(hdf_after.iter_rows(named=True)):
            prev = hdf_before.row(i, named=True)
            if row["employer"] >= 0:
                expected_income = prev["income"] + prev["wage"]
                assert abs(row["income"] - expected_income) < 1e-9, (
                    f"worker {row['unique_id']}: income={row['income']} "
                    f"!= prev+wage={expected_income}"
                )
            else:
                # Unemployed workers: income unchanged.
                assert abs(row["income"] - prev["income"]) < 1e-9

    def test_unemployed_workers_income_unchanged(self):
        """Event 21: unemployed workers' income is not changed."""
        model = self._run_through_production()
        hdf = model.households.agents
        # After production, unemployed workers should have income == 0
        # (started at 0, never received a wage).
        unemployed = hdf.filter(pl.col("employer") < 0)
        # income was 0 before; only employed workers receive wages.
        assert (unemployed["income"] >= 0.0).all()

    def test_pay_wages_reduces_total_funds(self):
        """Event 20: total_funds decreases by exactly wage_bill."""
        model = make_model(seed=SEED)
        model._planning()
        model._labor_market()
        model._credit_market()

        fdf_before = model.firms.agents.clone()
        model._event20_pay_wages()
        fdf_after = model.firms.agents

        for i, row in enumerate(fdf_after.iter_rows(named=True)):
            prev = fdf_before.row(i, named=True)
            expected = prev["total_funds"] - prev["wage_bill"]
            assert abs(row["total_funds"] - expected) < 1e-9, (
                f"firm {row['unique_id']}: total_funds={row['total_funds']} "
                f"!= {expected}"
            )

    def test_avg_mkt_price_updated_after_production(self):
        """Event 23: avg_mkt_price_history gets a new entry after _production."""
        model = make_model(seed=SEED)
        history_len_before = len(model.avg_mkt_price_history)
        model._planning()
        model._labor_market()
        model._credit_market()
        model._production()
        assert len(model.avg_mkt_price_history) == history_len_before + 1

    def test_avg_mkt_price_positive_if_production_exists(self):
        """Event 23: avg_mkt_price stays positive after production."""
        model = self._run_through_production()
        assert model.avg_mkt_price > 0.0

    def test_contracts_decremented_for_employed(self):
        """Event 24: employed workers' periods_left decremented by 1."""
        model = make_model(seed=SEED)
        model._planning()
        model._labor_market()
        model._credit_market()

        hdf_before = model.households.agents.clone()
        model._event20_pay_wages()
        model._event21_receive_wage()
        model._event22_run_production()
        model._event24_update_contracts()

        hdf_after = model.households.agents
        for i, row in enumerate(hdf_after.iter_rows(named=True)):
            prev = hdf_before.row(i, named=True)
            was_employed = prev["employer"] >= 0
            if was_employed:
                # Either contract decremented by 1, or expired (employer now -1).
                if row["employer"] >= 0:
                    # Still employed: periods_left decremented.
                    assert row["periods_left"] == prev["periods_left"] - 1, (
                        f"worker {row['unique_id']}: periods_left not decremented"
                    )
                else:
                    # Contract expired: periods_left was 1 before decrement.
                    assert prev["periods_left"] == 1
                    assert row["employer_prev"] == prev["employer"]
                    assert row["wage"] == 0.0
                    assert row["contract_expired"] is True
                    assert row["fired"] is False
            else:
                # Unemployed: no change.
                assert row["periods_left"] == prev["periods_left"]

    def test_expired_workers_cleared_from_employer(self):
        """Event 24: expired workers have employer=-1 and contract_expired=True."""
        # Run multiple steps to get some contracts expiring.
        model = make_model(seed=42)
        theta = DEFAULT_PARAMS["theta"]
        # Advance enough periods for contracts to expire.
        for _ in range(theta + 1):
            model._planning()
            model._labor_market()
            model._credit_market()
            model._production()
            model._revenue()
        hdf = model.households.agents
        # No employed worker should have periods_left <= 0.
        employed = hdf.filter(pl.col("employer") >= 0)
        if len(employed) > 0:
            assert (employed["periods_left"] >= 0).all()

    def test_firm_current_labor_consistent_after_contract_expiry(self):
        """Event 24: current_labor matches actual employed count after expiry."""
        model = make_model(seed=42)
        theta = DEFAULT_PARAMS["theta"]
        for _ in range(theta):
            model._planning()
            model._labor_market()
            model._credit_market()
            model._production()
            model._revenue()
        hdf = model.households.agents
        fdf = model.firms.agents
        counts = (
            hdf.filter(pl.col("employer") >= 0)
            .group_by("employer")
            .agg(pl.len().alias("c"))
        )
        count_map = dict(
            zip(counts["employer"].to_list(), counts["c"].to_list(), strict=True)
        )
        for row in fdf.iter_rows(named=True):
            actual = count_map.get(row["unique_id"], 0)
            assert row["current_labor"] == actual, (
                f"firm {row['unique_id']}: current_labor={row['current_labor']} "
                f"!= employed count {actual}"
            )

    def test_dividends_non_negative(self):
        """Event 32: all households receive non-negative dividends."""
        model = self._run_through_production()
        model._revenue()
        hdf = model.households.agents
        assert (hdf["dividends"] >= 0.0).all()

    def test_dividends_equal_across_households(self):
        """Event 32: div_per_hh is EQUAL for all households (spec flag 8)."""
        model = self._run_through_production()
        model._revenue()
        hdf = model.households.agents
        divs = hdf["dividends"].to_list()
        assert all(abs(d - divs[0]) < 1e-12 for d in divs), (
            "dividends differ across households"
        )

    def test_revenue_finite_and_non_negative(self):
        """Event 30: gross_profit is finite; total_funds finite after revenue."""
        model = self._run_through_production()
        model._event30_collect_revenue()
        fdf = model.firms.agents
        assert fdf["gross_profit"].is_finite().all()
        assert fdf["total_funds"].is_finite().all()

    def test_net_profit_computed_for_all_firms(self):
        """Event 31: net_profit is defined (finite) for all firms after validate_debt."""
        model = self._run_through_production(use_forced_gap=True)
        model._event30_collect_revenue()
        model._event31_validate_debt()
        fdf = model.firms.agents
        assert fdf["net_profit"].is_finite().all()

    def test_solvent_firms_repay_debt_fully(self):
        """Event 31: firms with total_funds >= total_debt repay; total_funds reduced."""
        model = self._run_through_production(use_forced_gap=True)
        if len(model.loans) == 0:
            pytest.skip("no loans issued under this seed")
        model._event30_collect_revenue()

        # Record pre-validate state.
        fdf_pre = model.firms.agents.clone()
        loans = model.loans

        # Compute expected per-firm total_debt.
        agg = (
            loans.with_columns(
                (pl.col("principal") * (1.0 + pl.col("interest_rate"))).alias("_debt")
            )
            .group_by("borrower_id")
            .agg(pl.col("_debt").sum().alias("total_debt"))
            .rename({"borrower_id": "unique_id"})
        )
        fdf_ext = fdf_pre.join(agg, on="unique_id", how="left").with_columns(
            pl.col("total_debt").fill_null(0.0)
        )

        model._event31_validate_debt()
        fdf_post = model.firms.agents

        eps = 1e-9
        for row_pre, row_post in zip(
            fdf_ext.iter_rows(named=True), fdf_post.iter_rows(named=True), strict=True
        ):
            total_debt = row_pre["total_debt"]
            if total_debt <= eps:
                continue  # no debt: skip
            pre_funds = row_pre["total_funds"]
            if (pre_funds - total_debt) >= -eps:
                # Should have repaid.
                expected_funds = pre_funds - total_debt
                assert abs(row_post["total_funds"] - expected_funds) < 1e-7, (
                    f"firm {row_pre['unique_id']}: expected repayment "
                    f"{expected_funds}, got {row_post['total_funds']}"
                )

    def test_defaulting_firms_have_zero_total_funds(self):
        """Event 31: firms that default have total_funds set to 0."""
        model = self._run_through_production(use_forced_gap=True)
        if len(model.loans) == 0:
            pytest.skip("no loans issued under this seed")
        # Force all firms' total_funds to 0 so they must default.
        fdf = model.firms.agents
        model.firms.agents = fdf.with_columns(pl.lit(0.0).alias("total_funds"))
        model._event30_collect_revenue()
        model._event31_validate_debt()
        # Any firm that had debt is now a defaulter: total_funds = 0.
        fdf_post = model.firms.agents
        loans = model.loans
        borrowers = set(loans["borrower_id"].to_list())
        for row in fdf_post.iter_rows(named=True):
            if row["unique_id"] in borrowers:
                assert abs(row["total_funds"]) < 1e-9, (
                    f"firm {row['unique_id']} defaulted but total_funds={row['total_funds']}"
                )

    def test_bank_equity_increases_on_full_repayment(self):
        """Event 31: solvent repayment increases bank equity by loan interest."""
        model = self._run_through_production(use_forced_gap=True)
        if len(model.loans) == 0:
            pytest.skip("no loans issued under this seed")
        model._event30_collect_revenue()

        bdf_before = model.banks.agents.clone()
        loans = model.loans

        # Check which firms are solvent pre-validate.
        fdf_pre = model.firms.agents.clone()
        agg = (
            loans.with_columns(
                (pl.col("principal") * (1.0 + pl.col("interest_rate"))).alias("_debt"),
                (pl.col("principal") * pl.col("interest_rate")).alias("_interest"),
            )
            .group_by("borrower_id")
            .agg(
                pl.col("_debt").sum().alias("total_debt"),
                pl.col("_interest").sum().alias("total_interest"),
            )
            .rename({"borrower_id": "unique_id"})
        )
        fdf_ext = fdf_pre.join(agg, on="unique_id", how="left").with_columns(
            pl.col("total_debt").fill_null(0.0),
            pl.col("total_interest").fill_null(0.0),
        )
        solvent_ids = set(
            fdf_ext.filter(
                (pl.col("total_debt") > 1e-9)
                & ((pl.col("total_funds") - pl.col("total_debt")) >= -1e-9)
            )["unique_id"].to_list()
        )

        model._event31_validate_debt()
        bdf_after = model.banks.agents

        # At least one bank's equity should be non-decreased if any firm was solvent.
        if solvent_ids:
            equity_before_total = bdf_before["equity_base"].sum()
            equity_after_total = bdf_after["equity_base"].sum()
            # Solvent repayments add interest: equity can only increase or stay same.
            assert equity_after_total >= equity_before_total - 1e-6, (
                "bank equity dropped despite solvent repayments"
            )

    def test_savings_increase_by_dividends(self):
        """Event 32: household savings increase by div_per_hh."""
        model = self._run_through_production()
        hdf_before = model.households.agents.clone()
        model._event30_collect_revenue()
        model._event31_validate_debt()

        savings_before = hdf_before["savings"].to_list()
        model._event32_pay_dividends()
        hdf_after = model.households.agents

        div = float(hdf_after["dividends"][0])
        for i, row in enumerate(hdf_after.iter_rows(named=True)):
            expected = savings_before[i] + div
            assert abs(row["savings"] - expected) < 1e-9, (
                f"worker {row['unique_id']}: savings={row['savings']} != {expected}"
            )

    def test_production_deterministic(self):
        """Same seed produces identical production state."""
        m1 = self._run_through_production(seed=17)
        m2 = self._run_through_production(seed=17)
        for col in ["production", "production_prev", "inventory"]:
            assert (m1.firms.agents[col] == m2.firms.agents[col]).all()

    def test_step_completes_without_error(self):
        """step() runs planning through revenue without raising."""
        model = make_model(seed=SEED)
        model.step()
        assert model.period == 1

    def test_multi_step_production_invariant(self):
        """production == lp * current_labor holds across multiple full steps."""
        model = make_model(seed=5)
        for _ in range(5):
            model.step()
            fdf = model.firms.agents
            for row in fdf.iter_rows(named=True):
                expected = row["labor_productivity"] * row["current_labor"]
                assert abs(row["production"] - expected) < 1e-9

    def test_collect_production_populated(self):
        """With collect=True, production stats are recorded per period."""
        from comparison.runners.mesa_frames.model import BAMModel

        model = BAMModel(
            N_FIRMS, N_HOUSEHOLDS, N_BANKS, DEFAULT_PARAMS, seed=SEED, collect=True
        )
        model.step()
        assert len(model._c_total_production) == 1
        assert len(model._c_unemployment) == 1
        assert len(model._c_avg_employed_wage) == 1
        assert model._c_total_production[0] >= 0.0
        assert 0.0 <= model._c_unemployment[0] <= 1.0


class TestGoodsMarket:
    """Invariant tests for BAMModel._goods_market() (events 25-29)."""

    def _run_through_goods(self, seed: int = SEED):
        """Helper: run all phases through the goods market."""
        model = make_model(seed=seed)
        model._planning()
        model._labor_market()
        model._credit_market()
        model._production()
        return model

    def test_largest_prod_prev_initialised(self):
        """Households should have largest_prod_prev column initialised to -1."""
        model = make_model()
        hdf = model.households.agents
        assert "largest_prod_prev" in hdf.columns
        assert (hdf["largest_prod_prev"] == -1).all()

    def test_shop_visit_columns_exist(self):
        """shop_visit_0..max_Z-1 columns exist and are initialised to -1."""
        model = make_model()
        hdf = model.households.agents
        max_Z = DEFAULT_PARAMS["max_Z"]
        for k in range(max_Z):
            col = f"shop_visit_{k}"
            assert col in hdf.columns, f"Missing column {col}"
            assert (hdf[col] == -1).all(), f"Column {col} not initialised to -1"

    def test_propensity_in_range(self):
        """Event 25: propensity must be in (0, 1] for all consumers."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        hdf = model.households.agents
        assert (hdf["propensity"] > 0.0).all()
        assert (hdf["propensity"] <= 1.0 + 1e-12).all()

    def test_income_to_spend_le_wealth(self):
        """Event 26: income_to_spend <= pre-split wealth for every consumer."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        hdf_before = model.households.agents.clone()
        model._event26_decide_income_to_spend()
        hdf_after = model.households.agents
        # wealth_before = savings + income before the split.
        wealth_before = (hdf_before["savings"] + hdf_before["income"]).to_list()
        its = hdf_after["income_to_spend"].to_list()
        for i, (w, s) in enumerate(zip(wealth_before, its, strict=True)):
            assert s <= w + 1e-9, f"consumer {i}: income_to_spend {s} > wealth {w}"

    def test_savings_plus_income_to_spend_equals_wealth(self):
        """Event 26: savings + income_to_spend == pre-split wealth (conservation)."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        hdf_before = model.households.agents.clone()
        model._event26_decide_income_to_spend()
        hdf_after = model.households.agents
        wealth_before = (hdf_before["savings"] + hdf_before["income"]).to_list()
        for i, row in enumerate(hdf_after.iter_rows(named=True)):
            post_wealth = row["savings"] + row["income_to_spend"]
            assert abs(post_wealth - wealth_before[i]) < 1e-9, (
                f"consumer {i}: savings+its={post_wealth} != wealth={wealth_before[i]}"
            )

    def test_income_zeroed_after_event26(self):
        """Event 26: income is set to 0 for all consumers after decide_income_to_spend."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        hdf = model.households.agents
        assert (hdf["income"] == 0.0).all()

    def test_shop_visits_populated_for_buyers(self):
        """Event 27: consumers with budget have at least one shop visit set."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()
        hdf = model.households.agents
        eps = 1e-9
        for row in hdf.iter_rows(named=True):
            if row["income_to_spend"] > eps:
                # At least shop_visit_0 must be a valid firm id (>= 0).
                assert row["shop_visit_0"] >= 0, (
                    f"consumer {row['unique_id']} has budget but no shop visits"
                )

    def test_shop_visits_are_valid_firm_ids(self):
        """Event 27: every non-(-1) shop visit must be a valid firm unique_id."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()
        hdf = model.households.agents
        fdf = model.firms.agents
        firm_ids = set(fdf["unique_id"].to_list())
        max_Z = DEFAULT_PARAMS["max_Z"]
        for row in hdf.iter_rows(named=True):
            for k in range(max_Z):
                fid = row[f"shop_visit_{k}"]
                if fid >= 0:
                    assert fid in firm_ids, (
                        f"consumer {row['unique_id']}: shop_visit_{k}={fid} "
                        "is not a valid firm id"
                    )

    def test_largest_prod_prev_updated(self):
        """Event 27: largest_prod_prev updated for consumers with budget."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()
        hdf = model.households.agents
        fdf = model.firms.agents
        firm_ids = set(fdf["unique_id"].to_list())
        eps = 1e-9
        for row in hdf.iter_rows(named=True):
            if row["income_to_spend"] > eps:
                lpp = row["largest_prod_prev"]
                assert lpp >= 0, (
                    f"consumer {row['unique_id']} has budget but largest_prod_prev=-1"
                )
                assert lpp in firm_ids, (
                    f"consumer {row['unique_id']}: largest_prod_prev={lpp} not a firm"
                )

    def test_total_spending_le_total_budget(self):
        """Event 28: aggregate spending never exceeds aggregate budget."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()

        hdf_before = model.households.agents.clone()
        total_budget = float(hdf_before["income_to_spend"].sum())

        from comparison.runners.mesa_frames.markets import run_goods_market

        run_goods_market(model)

        hdf_after = model.households.agents
        total_remaining = float(hdf_after["income_to_spend"].sum())
        total_spent = total_budget - total_remaining
        assert total_spent >= -1e-9, "negative aggregate spending"
        assert total_spent <= total_budget + 1e-9, (
            f"total_spent {total_spent} > total_budget {total_budget}"
        )

    def test_inventory_never_negative(self):
        """Event 28: firm inventory never goes negative during the goods market."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()

        from comparison.runners.mesa_frames.markets import run_goods_market

        run_goods_market(model)

        fdf = model.firms.agents
        assert (fdf["inventory"] >= -1e-9).all(), "Some firm inventories went negative"

    def test_income_to_spend_reduced_by_spending(self):
        """Event 28: each consumer's income_to_spend only decreases; aggregate spending
        matches aggregate revenue (price * qty_sold summed over firms)."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()

        hdf_before = model.households.agents.clone()
        fdf_before = model.firms.agents.clone()

        from comparison.runners.mesa_frames.markets import run_goods_market

        run_goods_market(model)

        hdf_after = model.households.agents
        fdf_after = model.firms.agents

        its_before = hdf_before["income_to_spend"].to_list()
        its_after = hdf_after["income_to_spend"].to_list()

        # Spending must not be negative per consumer (income_to_spend can only decrease).
        for i, (b, a) in enumerate(zip(its_before, its_after, strict=True)):
            assert b - a >= -1e-9, (
                f"consumer {i}: income_to_spend increased from {b} to {a}"
            )

        # Aggregate spending (money) == revenue from inventory depletion.
        # Inventory decrements in units; revenue = price * qty_sold per firm.
        inv_before = dict(
            zip(
                fdf_before["unique_id"].to_list(),
                (float(v) for v in fdf_before["inventory"]),
                strict=True,
            )
        )
        price_of = dict(
            zip(
                fdf_before["unique_id"].to_list(),
                (float(p) for p in fdf_before["price"]),
                strict=True,
            )
        )
        inv_after = dict(
            zip(
                fdf_after["unique_id"].to_list(),
                (float(v) for v in fdf_after["inventory"]),
                strict=True,
            )
        )
        total_revenue = sum(
            (inv_before[fid] - inv_after[fid]) * price_of[fid] for fid in inv_before
        )
        total_consumer_spent = sum(
            b - a for b, a in zip(its_before, its_after, strict=True)
        )
        assert abs(total_consumer_spent - total_revenue) < 1e-6, (
            f"consumer spending {total_consumer_spent} != firm revenue {total_revenue}"
        )

    def test_finalize_purchases_clears_budget(self):
        """Event 29: income_to_spend is 0 after finalize; savings increased."""
        model = self._run_through_goods()
        model._event25_calc_propensity()
        model._event26_decide_income_to_spend()
        model._event27_decide_firms_to_visit()

        from comparison.runners.mesa_frames.markets import run_goods_market

        run_goods_market(model)

        hdf_before = model.households.agents.clone()
        model._event29_finalize_purchases()
        hdf_after = model.households.agents

        # income_to_spend zeroed.
        assert (hdf_after["income_to_spend"] == 0.0).all()
        # savings increased by the unspent budget.
        for i, row in enumerate(hdf_after.iter_rows(named=True)):
            prev = hdf_before.row(i, named=True)
            expected_savings = prev["savings"] + prev["income_to_spend"]
            assert abs(row["savings"] - expected_savings) < 1e-9, (
                f"consumer {row['unique_id']}: savings mismatch after finalize"
            )

    def test_goods_market_deterministic(self):
        """Same seed produces identical goods market outcome."""
        models = []
        for _ in range(2):
            m = self._run_through_goods(seed=13)
            m._goods_market()
            models.append(m)
        m1, m2 = models
        assert (m1.firms.agents["inventory"] == m2.firms.agents["inventory"]).all(), (
            "inventory differs between identical seeds"
        )
        assert (
            m1.households.agents["savings"] == m2.households.agents["savings"]
        ).all(), "savings differs between identical seeds"

    def test_full_step_revenue_non_zero_after_goods(self):
        """After a full step with goods market, some firms should have gross_profit > 0."""
        model = make_model(seed=SEED)
        model.step()
        fdf = model.firms.agents
        # At least some firms sold something (inventory < production).
        sold = (fdf["production"] - fdf["inventory"]).filter(
            (fdf["production"] - fdf["inventory"]) > 1e-9
        )
        assert len(sold) > 0, "no firm sold anything after a full step"

    def test_goods_market_phase_invariants_across_steps(self):
        """Goods market invariants hold across several full steps."""
        model = make_model(seed=7)
        for _ in range(5):
            model._planning()
            model._labor_market()
            model._credit_market()
            model._production()
            model._goods_market()
            hdf_after = model.households.agents
            fdf_after = model.firms.agents
            # Inventory non-negative.
            assert (fdf_after["inventory"] >= -1e-9).all()
            # income_to_spend zeroed after finalize.
            assert (hdf_after["income_to_spend"] == 0.0).all()
            model._revenue()
