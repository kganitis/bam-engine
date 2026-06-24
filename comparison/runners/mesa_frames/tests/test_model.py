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
