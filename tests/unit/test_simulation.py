"""Tests for simulation.py input validation and getter methods."""

import numpy as np
import pytest

import bamengine.events  # noqa: F401 - register events
from bamengine.simulation import Simulation


class TestInputValidation:
    """Test input validation for Simulation.init()."""

    def test_wrong_shaped_price_init_2d(self):
        """Reject 2D array for price_init."""
        with pytest.raises(ValueError, match="price_init must be length-10 1-D array"):
            Simulation.init(
                n_firms=10,
                n_households=50,
                price_init=np.array([[1, 2], [3, 4]]),  # 2D array
                seed=42,
            )

    def test_wrong_shaped_price_init_wrong_length(self):
        """Reject 1D array with wrong length for price_init."""
        with pytest.raises(ValueError, match="price_init must be length-10 1-D array"):
            Simulation.init(
                n_firms=10,
                n_households=50,
                price_init=np.array([1.0, 2.0, 3.0]),  # Length 3, not 10
                seed=42,
            )

    def test_wrong_shaped_net_worth_init(self):
        """Reject wrong-shaped array for net_worth_init."""
        with pytest.raises(
            ValueError, match="net_worth_init must be length-10 1-D array"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                net_worth_init=np.array([[100], [200]]),  # 2D array
                seed=42,
            )

    def test_wrong_shaped_savings_init(self):
        """Reject wrong-shaped array for savings_init."""
        with pytest.raises(
            ValueError, match="savings_init must be length-50 1-D array"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                savings_init=np.ones((10, 5)),  # 2D array
                seed=42,
            )

    def test_wrong_length_equity_base_init(self):
        """Reject wrong-length array for equity_base_init."""
        with pytest.raises(
            ValueError, match="equity_base_init must be length-10 1-D array"
        ):
            Simulation.init(
                n_firms=10,
                n_households=50,
                n_banks=10,
                equity_base_init=np.array([100.0, 200.0]),  # Length 2, not 10
                seed=42,
            )

    def test_scalar_values_accepted(self):
        """Scalar values should be accepted and broadcast."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            price_init=1.0,  # Scalar
            net_worth_init=100.0,  # Scalar
            seed=42,
        )
        # Should initialize without error
        assert sim.prod.price.shape == (10,)
        assert sim.bor.net_worth.shape == (10,)

    def test_valid_array_values_accepted(self):
        """Valid 1D arrays of correct length should be accepted."""
        price_array = np.array([1.0, 1.5, 2.0, 1.2, 1.8, 1.3, 1.7, 1.4, 1.6, 1.1])
        net_worth_array = np.array(
            [100.0, 110.0, 95.0, 105.0, 102.0, 98.0, 103.0, 101.0, 99.0, 97.0]
        )

        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            price_init=price_array,  # Valid 1D array, length 10
            net_worth_init=net_worth_array,  # Valid 1D array, length 10
            seed=42,
        )
        # Arrays should be used directly
        np.testing.assert_array_equal(sim.prod.price, price_array)
        np.testing.assert_array_equal(sim.bor.net_worth, net_worth_array)


class TestSimulationControl:
    """Test simulation control flow (step, run, termination)."""

    def test_step_with_destroyed_simulation(self):
        """step() should return early if simulation is destroyed."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Manually mark simulation as destroyed
        sim.ec.destroyed = True
        initial_t = sim.t

        # step() should return early without executing pipeline
        sim.step()

        # Time should not advance
        assert sim.t == initial_t


class TestGetterMethods:
    """Test getter methods for roles, events, and relationships."""

    def test_get_event_nonexistent(self):
        """get_event() raises KeyError for non-existent event."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        with pytest.raises(KeyError, match="Event 'nonexistent_event' not found"):
            sim.get_event("nonexistent_event")

    def test_get_event_shows_available_events(self):
        """get_event() error message includes available events."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        with pytest.raises(KeyError, match="Available events"):
            sim.get_event("nonexistent_event")

    def test_get_event_valid(self):
        """get_event() returns event for valid name."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        event = sim.get_event("firms_decide_desired_production")
        assert event.name == "firms_decide_desired_production"

    def test_get_relationship_nonexistent(self):
        """get_relationship() raises KeyError for non-existent relationship."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        with pytest.raises(KeyError, match="Relationship 'NonExistent' not found"):
            sim.get_relationship("NonExistent")

    def test_get_relationship_shows_available(self):
        """get_relationship() error message includes available relationships."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        with pytest.raises(KeyError, match="Available relationships: \\['LoanBook'\\]"):
            sim.get_relationship("InvalidRelationship")

    def test_get_relationship_valid(self):
        """get_relationship() returns relationship for valid name."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        lb = sim.get_relationship("LoanBook")  # Case-insensitive
        assert lb is sim.lb

    def test_get_relationship_case_insensitive(self):
        """get_relationship() is case-insensitive."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        lb1 = sim.get_relationship("LoanBook")
        lb2 = sim.get_relationship("loanbook")
        lb3 = sim.get_relationship("LOANBOOK")

        assert lb1 is lb2 is lb3 is sim.lb

    def test_get_fallthrough_error(self):
        """get() raises ValueError when name not found anywhere."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        with pytest.raises(ValueError, match="'nonexistent' not found in simulation"):
            sim.get("nonexistent")

    def test_get_finds_role(self):
        """get() finds roles."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        prod = sim.get("Producer")
        assert prod is sim.prod

    def test_get_finds_event(self):
        """get() finds events when role not found."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        event = sim.get("firms_decide_desired_production")
        assert event.name == "firms_decide_desired_production"

    def test_get_finds_relationship(self):
        """get() finds relationships when role and event not found."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Note: get() tries role first, then event, then gives up
        # It doesn't try relationships, so this test documents current behavior
        with pytest.raises(ValueError, match="'LoanBook' not found in simulation"):
            sim.get("LoanBook")
