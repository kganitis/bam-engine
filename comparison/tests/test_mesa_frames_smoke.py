"""Smoke tests for the mesa-frames runner registration and requirements.

These tests do NOT require the .venv-mf virtual environment to exist, so they
remain fast in CI without triggering a full environment build.

Agent construction tests (test_*_agent_set_*) require the mesa-frames venv and
are guarded with pytest.importorskip("mesa_frames").  Run them with:

    comparison/runners/mesa_frames/.venv-mf/bin/python -m pytest \
        comparison/tests/test_mesa_frames_smoke.py --no-cov -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comparison.orchestrator.run import RUNNER_CMD

_RUNNERS_DIR = Path(__file__).resolve().parent.parent / "runners" / "mesa_frames"
_REQS_FILE = _RUNNERS_DIR / "requirements-mesa-frames.txt"


def test_runner_cmd_has_mesa_frames_entry():
    """RUNNER_CMD must register a mesa_frames entry."""
    assert "mesa_frames" in RUNNER_CMD, (
        "RUNNER_CMD does not contain a 'mesa_frames' key"
    )


def test_mesa_frames_runner_cmd_points_to_venv_python():
    """The mesa_frames RUNNER_CMD must point to .venv-mf/bin/python."""
    cmd = RUNNER_CMD["mesa_frames"]
    assert isinstance(cmd, list), "RUNNER_CMD['mesa_frames'] must be a list"
    assert len(cmd) >= 1, "RUNNER_CMD['mesa_frames'] must be non-empty"
    python_path = cmd[0]
    assert python_path.endswith(".venv-mf/bin/python"), (
        f"Expected path ending with '.venv-mf/bin/python', got: {python_path!r}"
    )


def test_mesa_frames_runner_cmd_uses_module_flag():
    """The mesa_frames RUNNER_CMD must invoke the runner via -m."""
    cmd = RUNNER_CMD["mesa_frames"]
    assert len(cmd) >= 3, "RUNNER_CMD['mesa_frames'] must have at least 3 elements"
    assert cmd[1] == "-m", f"Expected '-m' as second element, got: {cmd[1]!r}"
    assert cmd[2] == "comparison.runners.mesa_frames.run", (
        f"Expected module path 'comparison.runners.mesa_frames.run', got: {cmd[2]!r}"
    )


def test_requirements_pins_numpy_lt_2():
    """requirements-mesa-frames.txt must pin numpy<2."""
    assert _REQS_FILE.exists(), f"Requirements file not found: {_REQS_FILE}"
    content = _REQS_FILE.read_text()
    assert "numpy<2" in content, (
        f"'numpy<2' not found in {_REQS_FILE}; content:\n{content}"
    )


def test_requirements_pins_mesa_frames_version():
    """requirements-mesa-frames.txt must pin mesa-frames==0.1.0a0."""
    assert _REQS_FILE.exists(), f"Requirements file not found: {_REQS_FILE}"
    content = _REQS_FILE.read_text()
    assert "mesa-frames==0.1.0a0" in content, (
        f"'mesa-frames==0.1.0a0' not found in {_REQS_FILE}; content:\n{content}"
    )


# ---------------------------------------------------------------------------
# Agent construction tests (require mesa-frames venv)
# ---------------------------------------------------------------------------


def _make_test_model(n_firms, n_households, n_banks, seed=0):
    """Build a minimal ModelDF for agent construction tests."""
    mf = pytest.importorskip("mesa_frames")
    import polars as pl  # noqa: F401 - imported for side effect check

    class _TestModel(mf.ModelDF):
        def __init__(self):
            super().__init__(seed=seed)
            self.n_firms = n_firms
            self.n_households = n_households
            self.n_banks = n_banks

        def step(self):
            pass

    return _TestModel()


def _make_params():
    """Return a minimal param dict matching the Mesa port's required keys."""
    return {
        "labor_productivity": 0.5,
        "price_init": 0.5,
        "net_worth_ratio": 6.0,
        "savings_init": 1.0,
        "equity_base_init": 5.0,
    }


def test_firms_agent_set_row_count():
    """Firms AgentSetPolars must have exactly n rows after construction."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Firms

    n_firms = 10
    model = _make_test_model(n_firms=n_firms, n_households=50, n_banks=1)
    params = _make_params()
    rng = model.random
    firms = Firms(model, n_firms, params, rng)
    assert len(firms) == n_firms, f"Expected {n_firms} rows, got {len(firms)}"


def test_firms_agent_set_columns_exist():
    """Firms AgentSetPolars must have all expected columns."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Firms

    expected_cols = {
        "unique_id",
        "production",
        "production_prev",
        "inventory",
        "expected_demand",
        "desired_production",
        "labor_productivity",
        "breakeven_price",
        "price",
        "desired_labor",
        "current_labor",
        "wage_offer",
        "wage_bill",
        "n_vacancies",
        "total_funds",
        "net_worth",
        "credit_demand",
        "projected_fragility",
        "gross_profit",
        "net_profit",
        "retained_profit",
    }
    model = _make_test_model(n_firms=5, n_households=25, n_banks=1)
    params = _make_params()
    firms = Firms(model, 5, params, model.random)
    missing = expected_cols - set(firms.agents.columns)
    assert not missing, f"Missing columns in Firms: {missing}"


def test_firms_agent_set_initial_values():
    """Firms initial values must match the Mesa port's derived formulas."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Firms

    n_firms = 10
    n_households = 50
    params = _make_params()
    lp = params["labor_productivity"]
    price_init = params["price_init"]
    production_init = n_households * lp / n_firms
    wage_offer_init = price_init / 3.0
    net_worth_init = production_init * price_init * params["net_worth_ratio"]

    model = _make_test_model(n_firms=n_firms, n_households=n_households, n_banks=1)
    firms = Firms(model, n_firms, params, model.random)
    df = firms.agents

    assert df["production"].to_list() == [0.0] * n_firms
    assert df["production_prev"].to_list() == [production_init] * n_firms
    assert df["inventory"].to_list() == [0.0] * n_firms
    assert df["expected_demand"].to_list() == [1.0] * n_firms
    assert df["desired_production"].to_list() == [0.0] * n_firms
    assert df["labor_productivity"].to_list() == [lp] * n_firms
    assert df["breakeven_price"].to_list() == [price_init] * n_firms
    assert df["price"].to_list() == [price_init] * n_firms
    assert df["desired_labor"].to_list() == [0] * n_firms
    assert df["current_labor"].to_list() == [0] * n_firms
    assert df["wage_offer"].to_list() == [wage_offer_init] * n_firms
    assert df["wage_bill"].to_list() == [0.0] * n_firms
    assert df["n_vacancies"].to_list() == [0] * n_firms
    assert df["total_funds"].to_list() == [net_worth_init] * n_firms
    assert df["net_worth"].to_list() == [net_worth_init] * n_firms
    assert df["credit_demand"].to_list() == [0.0] * n_firms
    assert df["projected_fragility"].to_list() == [0.0] * n_firms
    assert df["gross_profit"].to_list() == [0.0] * n_firms
    assert df["net_profit"].to_list() == [0.0] * n_firms
    assert df["retained_profit"].to_list() == [0.0] * n_firms


def test_households_agent_set_row_count():
    """Households AgentSetPolars must have exactly n rows after construction."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Households

    n = 50
    model = _make_test_model(n_firms=10, n_households=n, n_banks=1)
    params = _make_params()
    households = Households(model, n, params, model.random)
    assert len(households) == n, f"Expected {n} rows, got {len(households)}"


def test_households_agent_set_columns_exist():
    """Households AgentSetPolars must have all expected columns."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Households

    expected_cols = {
        "unique_id",
        "wage",
        "periods_left",
        "contract_expired",
        "fired",
        "income",
        "savings",
        "income_to_spend",
        "propensity",
        "dividends",
    }
    model = _make_test_model(n_firms=5, n_households=25, n_banks=1)
    params = _make_params()
    households = Households(model, 25, params, model.random)
    missing = expected_cols - set(households.agents.columns)
    assert not missing, f"Missing columns in Households: {missing}"


def test_households_agent_set_initial_values():
    """Households initial values must match the Mesa port."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Households

    n = 25
    params = _make_params()
    savings_init = params["savings_init"]

    model = _make_test_model(n_firms=5, n_households=n, n_banks=1)
    households = Households(model, n, params, model.random)
    df = households.agents

    assert df["wage"].to_list() == [0.0] * n
    assert df["periods_left"].to_list() == [0] * n
    assert df["contract_expired"].to_list() == [False] * n
    assert df["fired"].to_list() == [False] * n
    assert df["income"].to_list() == [0.0] * n
    assert df["savings"].to_list() == [savings_init] * n
    assert df["income_to_spend"].to_list() == [0.0] * n
    assert df["propensity"].to_list() == [0.0] * n
    assert df["dividends"].to_list() == [0.0] * n


def test_banks_agent_set_row_count():
    """Banks AgentSetPolars must have exactly n rows after construction."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Banks

    n = 3
    model = _make_test_model(n_firms=10, n_households=50, n_banks=n)
    params = _make_params()
    banks = Banks(model, n, params, model.random)
    assert len(banks) == n, f"Expected {n} rows, got {len(banks)}"


def test_banks_agent_set_columns_exist():
    """Banks AgentSetPolars must have all expected columns."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Banks

    expected_cols = {
        "unique_id",
        "equity_base",
        "credit_supply",
        "interest_rate",
        "opex_shock",
    }
    model = _make_test_model(n_firms=5, n_households=25, n_banks=1)
    params = _make_params()
    banks = Banks(model, 1, params, model.random)
    missing = expected_cols - set(banks.agents.columns)
    assert not missing, f"Missing columns in Banks: {missing}"


def test_banks_agent_set_initial_values():
    """Banks initial values must match the Mesa port."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Banks

    n = 3
    params = _make_params()
    equity_base_init = params["equity_base_init"]

    model = _make_test_model(n_firms=10, n_households=50, n_banks=n)
    banks = Banks(model, n, params, model.random)
    df = banks.agents

    assert df["equity_base"].to_list() == [equity_base_init] * n
    assert df["credit_supply"].to_list() == [0.0] * n
    assert df["interest_rate"].to_list() == [0.0] * n
    assert df["opex_shock"].to_list() == [0.0] * n


def test_unique_id_allocation_non_overlapping():
    """Firms, Households, Banks must get disjoint unique_ids from the same model."""
    pytest.importorskip("mesa_frames")
    from comparison.runners.mesa_frames.agents import Banks, Firms, Households

    n_f, n_h, n_b = 5, 25, 2
    model = _make_test_model(n_firms=n_f, n_households=n_h, n_banks=n_b)
    params = _make_params()

    firms = Firms(model, n_f, params, model.random)
    households = Households(model, n_h, params, model.random)
    banks = Banks(model, n_b, params, model.random)

    firm_ids = set(firms.agents["unique_id"].to_list())
    household_ids = set(households.agents["unique_id"].to_list())
    bank_ids = set(banks.agents["unique_id"].to_list())

    assert len(firm_ids) == n_f
    assert len(household_ids) == n_h
    assert len(bank_ids) == n_b
    assert firm_ids.isdisjoint(household_ids), "Firms and Households share unique_ids"
    assert firm_ids.isdisjoint(bank_ids), "Firms and Banks share unique_ids"
    assert household_ids.isdisjoint(bank_ids), "Households and Banks share unique_ids"


def test_agent_set_dtypes():
    """Key columns must have the correct Polars dtype."""
    pytest.importorskip("mesa_frames")
    import polars as pl

    from comparison.runners.mesa_frames.agents import Banks, Firms, Households

    model = _make_test_model(n_firms=5, n_households=25, n_banks=1)
    params = _make_params()

    firms = Firms(model, 5, params, model.random)
    households = Households(model, 25, params, model.random)
    banks = Banks(model, 1, params, model.random)

    # unique_id must be Int64 (required by AgentSetPolars.add)
    assert firms.agents["unique_id"].dtype == pl.Int64
    assert households.agents["unique_id"].dtype == pl.Int64
    assert banks.agents["unique_id"].dtype == pl.Int64

    # integer counters
    assert firms.agents["desired_labor"].dtype == pl.Int64
    assert firms.agents["current_labor"].dtype == pl.Int64
    assert firms.agents["n_vacancies"].dtype == pl.Int64
    assert households.agents["periods_left"].dtype == pl.Int64

    # boolean flags
    assert households.agents["contract_expired"].dtype == pl.Boolean
    assert households.agents["fired"].dtype == pl.Boolean

    # key floats
    assert firms.agents["price"].dtype == pl.Float64
    assert firms.agents["net_worth"].dtype == pl.Float64
    assert households.agents["savings"].dtype == pl.Float64
    assert banks.agents["equity_base"].dtype == pl.Float64
