"""Golden master tests comparing pipeline vs legacy implementation."""

import numpy as np
import pytest

from bamengine.simulation import Simulation


def test_pipeline_matches_legacy_single_step():
    """Pipeline step() produces identical results to legacy step()."""
    # Create two identical simulations
    sim_pipeline = Simulation.init(n_firms=10, n_households=50, seed=42)
    sim_legacy = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Run one step
    sim_pipeline.step()
    sim_legacy._step_legacy()

    # Compare all role states
    assert np.allclose(sim_pipeline.prod.price, sim_legacy.prod.price)
    assert np.allclose(sim_pipeline.prod.inventory, sim_legacy.prod.inventory)
    assert np.allclose(sim_pipeline.prod.production, sim_legacy.prod.production)
    assert np.allclose(sim_pipeline.emp.wage_offer, sim_legacy.emp.wage_offer)
    assert np.allclose(sim_pipeline.emp.current_labor, sim_legacy.emp.current_labor)
    assert np.allclose(sim_pipeline.wrk.wage, sim_legacy.wrk.wage)
    assert np.array_equal(sim_pipeline.wrk.employed, sim_legacy.wrk.employed)
    assert np.array_equal(sim_pipeline.wrk.employer, sim_legacy.wrk.employer)
    assert np.allclose(sim_pipeline.bor.net_worth, sim_legacy.bor.net_worth)
    assert np.allclose(sim_pipeline.lend.equity_base, sim_legacy.lend.equity_base)
    assert np.allclose(sim_pipeline.con.savings, sim_legacy.con.savings)

    # Compare economy state
    assert sim_pipeline.ec.avg_mkt_price == sim_legacy.ec.avg_mkt_price
    assert sim_pipeline.t == sim_legacy.t == 1


def test_pipeline_matches_legacy_multi_step():
    """Pipeline produces identical results over 10 steps."""
    sim_pipeline = Simulation.init(n_firms=20, n_households=100, seed=123)
    sim_legacy = Simulation.init(n_firms=20, n_households=100, seed=123)

    # Run 10 steps
    for _ in range(10):
        sim_pipeline.step()
        sim_legacy._step_legacy()

    # Compare final states
    assert np.allclose(sim_pipeline.prod.price, sim_legacy.prod.price)
    assert np.allclose(sim_pipeline.prod.inventory, sim_legacy.prod.inventory)
    assert np.allclose(sim_pipeline.bor.net_worth, sim_legacy.bor.net_worth)
    assert np.allclose(sim_pipeline.con.savings, sim_legacy.con.savings)
    assert np.array_equal(sim_pipeline.wrk.employed, sim_legacy.wrk.employed)

    # Compare time-series
    assert np.allclose(
        sim_pipeline.ec.avg_mkt_price_history,
        sim_legacy.ec.avg_mkt_price_history,
    )
    assert np.allclose(
        sim_pipeline.ec.unemp_rate_history,
        sim_legacy.ec.unemp_rate_history,
    )

    # Verify both reached same time period
    assert sim_pipeline.t == sim_legacy.t == 10


def test_pipeline_determinism():
    """Pipeline produces deterministic results with same seed."""
    sim1 = Simulation.init(n_firms=10, n_households=50, seed=999)
    sim2 = Simulation.init(n_firms=10, n_households=50, seed=999)

    sim1.step()
    sim2.step()

    assert np.allclose(sim1.prod.price, sim2.prod.price)
    assert np.allclose(sim1.con.savings, sim2.con.savings)
    assert np.array_equal(sim1.wrk.employed, sim2.wrk.employed)


def test_pipeline_matches_legacy_with_different_seed():
    """Pipeline matches legacy with different random seed."""
    sim_pipeline = Simulation.init(n_firms=15, n_households=75, seed=777)
    sim_legacy = Simulation.init(n_firms=15, n_households=75, seed=777)

    # Run 5 steps
    for _ in range(5):
        sim_pipeline.step()
        sim_legacy._step_legacy()

    # Should match exactly
    assert np.allclose(sim_pipeline.prod.price, sim_legacy.prod.price)
    assert np.allclose(sim_pipeline.bor.net_worth, sim_legacy.bor.net_worth)
    assert sim_pipeline.t == sim_legacy.t == 5


def test_pipeline_matches_legacy_long_run():
    """Pipeline matches legacy over 100 periods."""
    sim_pipeline = Simulation.init(n_firms=10, n_households=50, seed=42)
    sim_legacy = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Run 100 steps
    for _ in range(100):
        sim_pipeline.step()
        sim_legacy._step_legacy()

        # Check simulation didn't terminate early
        if sim_pipeline.ec.destroyed or sim_legacy.ec.destroyed:
            # Both should be destroyed or neither
            assert sim_pipeline.ec.destroyed == sim_legacy.ec.destroyed
            break

    # Compare final states
    assert np.allclose(sim_pipeline.prod.price, sim_legacy.prod.price)
    assert np.allclose(sim_pipeline.bor.net_worth, sim_legacy.bor.net_worth)
    assert np.allclose(sim_pipeline.con.savings, sim_legacy.con.savings)

    # Time should match
    assert sim_pipeline.t == sim_legacy.t


def test_pipeline_run_method():
    """Simulation.run() uses pipeline correctly."""
    sim = Simulation.init(n_firms=5, n_households=20, seed=42)

    # Run 20 periods
    sim.run(20)

    # Should have advanced 20 periods
    assert sim.t == 20

    # Verify pipeline executed (check some state changed)
    assert not np.all(sim.prod.production == 4.0)  # Initial value


def test_pipeline_matches_legacy_empty_loanbook():
    """Pipeline handles empty LoanBook same as legacy."""
    sim_pipeline = Simulation.init(n_firms=5, n_households=20, seed=100)
    sim_legacy = Simulation.init(n_firms=5, n_households=20, seed=100)

    # First step should have no loans
    assert sim_pipeline.lb.size == 0
    assert sim_legacy.lb.size == 0

    sim_pipeline.step()
    sim_legacy._step_legacy()

    # States should match
    assert np.allclose(sim_pipeline.bor.net_worth, sim_legacy.bor.net_worth)
    assert sim_pipeline.lb.size == sim_legacy.lb.size
