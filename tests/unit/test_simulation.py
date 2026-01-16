"""Tests for simulation.py input validation and getter methods."""

import numpy as np
import pytest

import bamengine.events  # noqa: F401 - register events
from bamengine import role
from bamengine.simulation import Simulation
from bamengine.typing import Float1D


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

    def test_get_role_builtin_roles_still_work(self):
        """get_role() returns built-in roles correctly (regression test)."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Verify all built-in roles still work
        assert sim.get_role("Producer") is sim.prod
        assert sim.get_role("Worker") is sim.wrk
        assert sim.get_role("Employer") is sim.emp
        assert sim.get_role("Borrower") is sim.bor
        assert sim.get_role("Lender") is sim.lend
        assert sim.get_role("Consumer") is sim.con

    def test_get_role_custom_role_attached_to_simulation(self) -> None:
        """get_role() finds custom role instance attached to simulation."""

        @role
        class TestCustomMetrics:
            """Custom metrics role for testing."""

            value: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Attach custom role instance to simulation via _role_instances
        custom_instance = TestCustomMetrics(value=np.zeros(10))
        sim._role_instances["TestCustomMetrics"] = custom_instance

        # get_role() should find the attached instance
        retrieved = sim.get_role("TestCustomMetrics")
        assert retrieved is custom_instance

    def test_get_role_registered_but_not_attached_raises_error(self) -> None:
        """get_role() raises KeyError for registered role with no instance."""

        @role
        class TestOrphanRole:
            """Role registered but not attached to simulation."""

            data: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Role is registered but not attached to simulation
        with pytest.raises(KeyError, match="Role 'TestOrphanRole' not found"):
            sim.get_role("TestOrphanRole")


class TestUseRole:
    """Test use_role() method for attaching custom roles."""

    def test_use_role_creates_instance(self) -> None:
        """use_role() creates a new role instance when not already attached."""

        @role
        class CustomMetrics:
            """Custom metrics role for testing."""

            score: Float1D
            count: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Use the role - should create instance
        custom = sim.use_role(CustomMetrics)

        # Instance should be created and attached
        assert custom is not None
        assert "CustomMetrics" in sim._role_instances
        assert sim._role_instances["CustomMetrics"] is custom

    def test_use_role_returns_existing(self) -> None:
        """use_role() returns existing instance if already attached."""

        @role
        class ExistingRole:
            """Role that already exists."""

            value: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # First call creates instance
        first = sim.use_role(ExistingRole)

        # Second call returns same instance
        second = sim.use_role(ExistingRole)

        assert first is second

    def test_use_role_initializes_zeros(self) -> None:
        """use_role() initializes all array fields with zeros."""

        @role
        class ZeroInitRole:
            """Role with multiple fields to verify zero init."""

            prices: Float1D
            quantities: Float1D
            flags: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        instance = sim.use_role(ZeroInitRole)

        # All fields should be zero-initialized arrays of correct size
        assert instance.prices.shape == (10,)  # n_firms
        assert instance.quantities.shape == (10,)
        assert instance.flags.shape == (10,)
        np.testing.assert_array_equal(instance.prices, np.zeros(10))
        np.testing.assert_array_equal(instance.quantities, np.zeros(10))
        np.testing.assert_array_equal(instance.flags, np.zeros(10))

    def test_use_role_accessible_via_get_role(self) -> None:
        """Roles attached via use_role() are accessible via get_role()."""

        @role
        class AccessibleRole:
            """Role that should be accessible via get_role."""

            data: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Attach via use_role
        instance = sim.use_role(AccessibleRole)

        # Should be retrievable via get_role
        retrieved = sim.get_role("AccessibleRole")
        assert retrieved is instance

    def test_use_role_with_named_role(self) -> None:
        """use_role() respects the role's explicit name attribute."""

        @role(name="MyCustomName")
        class RoleWithName:
            """Role with explicit name."""

            value: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        instance = sim.use_role(RoleWithName)

        # Should be stored under explicit name
        assert "MyCustomName" in sim._role_instances
        assert sim.get_role("MyCustomName") is instance

    def test_role_instances_contains_builtin_roles(self) -> None:
        """_role_instances contains all built-in roles after init."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # All built-in roles should be in _role_instances
        assert "Producer" in sim._role_instances
        assert "Worker" in sim._role_instances
        assert "Employer" in sim._role_instances
        assert "Borrower" in sim._role_instances
        assert "Lender" in sim._role_instances
        assert "Consumer" in sim._role_instances

        # And they should be the same as the direct attributes
        assert sim._role_instances["Producer"] is sim.prod
        assert sim._role_instances["Worker"] is sim.wrk
        assert sim._role_instances["Employer"] is sim.emp
        assert sim._role_instances["Borrower"] is sim.bor
        assert sim._role_instances["Lender"] is sim.lend
        assert sim._role_instances["Consumer"] is sim.con

    def test_custom_roles_property_backward_compatible(self) -> None:
        """_custom_roles property aliases _role_instances for backward compat."""

        @role
        class BackwardCompatRole:
            """Role for backward compat test."""

            data: Float1D

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Use _custom_roles (old API)
        custom_instance = BackwardCompatRole(data=np.zeros(10))
        sim._custom_roles["BackwardCompatRole"] = custom_instance

        # Should be accessible via _role_instances too
        assert sim._role_instances["BackwardCompatRole"] is custom_instance

        # And vice versa
        sim._role_instances["AnotherRole"] = "test"
        assert sim._custom_roles["AnotherRole"] == "test"


class TestExtraParams:
    """Test extra_params feature for extension parameters."""

    def test_extra_params_stored(self) -> None:
        """Extra kwargs are stored in extra_params dict."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            sigma_min=0.0,
            sigma_max=0.15,
            custom_param=42,
        )
        assert sim.extra_params["sigma_min"] == 0.0
        assert sim.extra_params["sigma_max"] == 0.15
        assert sim.extra_params["custom_param"] == 42

    def test_extra_params_attribute_access(self) -> None:
        """Extra params accessible as attributes via __getattr__."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            sigma_min=0.05,
            sigma_decay=-2.0,
        )
        assert sim.sigma_min == 0.05
        assert sim.sigma_decay == -2.0

    def test_extra_params_missing_raises_attribute_error(self) -> None:
        """Missing extra param raises AttributeError."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        with pytest.raises(
            AttributeError, match="has no attribute 'nonexistent_param'"
        ):
            _ = sim.nonexistent_param

    def test_extra_params_empty_by_default(self) -> None:
        """extra_params is empty dict when no extra kwargs provided."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        assert sim.extra_params == {}

    def test_extra_params_does_not_shadow_builtin_attrs(self) -> None:
        """extra_params doesn't interfere with built-in attributes."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            my_custom_param="test_value",
        )
        # Built-in attributes still work
        assert sim.n_firms == 10
        assert sim.n_households == 50
        assert sim.prod is not None
        # Custom param also works
        assert sim.my_custom_param == "test_value"

    def test_extra_params_private_attr_not_intercepted(self) -> None:
        """Private attributes are not intercepted by __getattr__."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        # Accessing non-existent private attr should raise AttributeError
        with pytest.raises(AttributeError, match="has no attribute '_nonexistent'"):
            _ = sim._nonexistent

    def test_extra_params_various_types(self) -> None:
        """Extra params can store various types (float, int, str, list, dict)."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            float_param=3.14,
            int_param=42,
            str_param="hello",
            list_param=[1, 2, 3],
            dict_param={"a": 1, "b": 2},
        )
        assert sim.float_param == 3.14
        assert sim.int_param == 42
        assert sim.str_param == "hello"
        assert sim.list_param == [1, 2, 3]
        assert sim.dict_param == {"a": 1, "b": 2}


class TestConfigPropertyAccessors:
    """Test property accessors for configuration parameters."""

    def test_labor_productivity_property(self) -> None:
        """Test labor_productivity property accessor."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            labor_productivity=2.5,
        )
        assert sim.labor_productivity == 2.5

    def test_config_property_accessors(self) -> None:
        """Test various config property accessors."""
        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            max_M=4,
            max_H=3,
            max_Z=5,
            theta=12,
            beta=0.9,
            r_bar=0.03,
            v=0.1,
        )
        assert sim.max_M == 4
        assert sim.max_H == 3
        assert sim.max_Z == 5
        assert sim.theta == 12
        assert sim.beta == 0.9
        assert sim.r_bar == 0.03
        assert sim.v == 0.1


class TestLoggingConfiguration:
    """Test logging configuration options."""

    def test_log_file_creation(self, tmp_path) -> None:
        """Test creating simulation with log file configuration."""
        import logging as std_logging

        log_file = tmp_path / "simulation.log"

        sim = Simulation.init(
            n_firms=10,
            n_households=50,
            seed=42,
            logging={"default_level": "DEBUG", "log_file": str(log_file)},
        )

        # Run a step to generate log output
        sim.step()

        # Log file should be created
        assert log_file.exists()

        # Clean up file handler to release the file
        bam_logger = std_logging.getLogger("bamengine")
        for handler in bam_logger.handlers[:]:
            if isinstance(handler, std_logging.FileHandler):
                handler.close()
                bam_logger.removeHandler(handler)


class TestEventPipelineErrors:
    """Test error handling for event access."""

    def test_get_event_registered_but_not_in_pipeline(self) -> None:
        """get_event() raises helpful error for event in registry but not pipeline."""
        from bamengine.core import Pipeline

        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # Create a minimal pipeline that doesn't include a known registered event
        sim.pipeline = Pipeline(events=[])

        # Should raise with helpful message about pipeline
        with pytest.raises(KeyError, match="registered but not in current pipeline"):
            sim.get_event("adjust_minimum_wage")
