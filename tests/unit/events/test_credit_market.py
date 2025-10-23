"""Unit tests for credit market events.

These tests verify that credit market events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_credit_market.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.credit_market import (
    BanksDecideCreditSupply,
    BanksDecideInterestRate,
    BanksProvideLoans,
    FirmsCalcCreditMetrics,
    FirmsDecideCreditDemand,
    FirmsFireWorkers,
    FirmsPrepareLoanApplications,
    FirmsSendOneLoanApp,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_banks_decide_credit_supply_no_dependencies():
    """BanksDecideCreditSupply has no dependencies."""
    event = BanksDecideCreditSupply()
    assert event.dependencies == ()


def test_banks_decide_interest_rate_dependencies():
    """BanksDecideInterestRate declares correct dependency."""
    event = BanksDecideInterestRate()
    assert "banks_decide_credit_supply" in event.dependencies


def test_firms_decide_credit_demand_dependencies():
    """FirmsDecideCreditDemand declares correct dependency."""
    event = FirmsDecideCreditDemand()
    assert "firms_calc_wage_bill" in event.dependencies


def test_firms_calc_credit_metrics_dependencies():
    """FirmsCalcCreditMetrics declares correct dependency."""
    event = FirmsCalcCreditMetrics()
    assert "firms_decide_credit_demand" in event.dependencies


def test_firms_prepare_loan_applications_dependencies():
    """FirmsPrepareLoanApplications declares correct dependencies."""
    event = FirmsPrepareLoanApplications()
    deps = event.dependencies
    assert "firms_calc_credit_metrics" in deps
    assert "banks_decide_interest_rate" in deps


def test_firms_send_one_loan_app_dependencies():
    """FirmsSendOneLoanApp declares correct dependency."""
    event = FirmsSendOneLoanApp()
    assert "firms_prepare_loan_applications" in event.dependencies


def test_banks_provide_loans_dependencies():
    """BanksProvideLoans declares correct dependency."""
    event = BanksProvideLoans()
    assert "firms_send_one_loan_app" in event.dependencies


def test_firms_fire_workers_dependencies():
    """FirmsFireWorkers declares correct dependency."""
    event = FirmsFireWorkers()
    assert "banks_provide_loans" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_banks_decide_credit_supply_executes():
    """BanksDecideCreditSupply executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = BanksDecideCreditSupply()
    event.execute(sim)  # Should not crash


def test_banks_decide_interest_rate_executes():
    """BanksDecideInterestRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = BanksDecideInterestRate()
    event.execute(sim)  # Should not crash


def test_firms_decide_credit_demand_executes():
    """FirmsDecideCreditDemand executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsDecideCreditDemand()
    event.execute(sim)  # Should not crash


def test_firms_calc_credit_metrics_executes():
    """FirmsCalcCreditMetrics executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsCalcCreditMetrics()
    event.execute(sim)  # Should not crash


def test_firms_prepare_loan_applications_executes():
    """FirmsPrepareLoanApplications executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsPrepareLoanApplications()
    event.execute(sim)  # Should not crash


def test_firms_send_one_loan_app_executes():
    """FirmsSendOneLoanApp executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsSendOneLoanApp()
    event.execute(sim)  # Should not crash


def test_banks_provide_loans_executes():
    """BanksProvideLoans executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = BanksProvideLoans()
    event.execute(sim)  # Should not crash


def test_firms_fire_workers_executes():
    """FirmsFireWorkers executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsFireWorkers()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_credit_market_event_chain():
    """Test credit market events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        BanksDecideCreditSupply(),
        BanksDecideInterestRate(),
        FirmsDecideCreditDemand(),
        FirmsCalcCreditMetrics(),
    ]

    for event in events:
        event.execute(sim)

    # Verify state mutations occurred
    assert (sim.lend.credit_supply > 0).all()
    assert (sim.lend.interest_rate > 0).all()
