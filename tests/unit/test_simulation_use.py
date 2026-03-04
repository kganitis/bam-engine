"""Tests for Simulation.use() and use_relationship()."""

import bamengine.events  # noqa: F401 - register events
from bamengine import Extension, Float, Int, event, role
from bamengine.simulation import Simulation


@role
class _TestFirmRole:
    x: Float


@role
class _TestHouseholdRole:
    y: Int


@role
class _TestBankRole:
    z: Float


@event(after="firms_pay_dividends")
class _TestEvent:
    def execute(self, sim: Simulation) -> None:
        pass


class TestSimulationUse:
    """Test sim.use() with Extension bundles."""

    def test_use_applies_firm_role(self) -> None:
        """sim.use() registers firm-level roles with correct array size."""
        ext = Extension(
            roles={_TestFirmRole: "firms"},
            events=[],
            relationships=[],
            config_dict={},
        )
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        sim.use(ext)
        r = sim.get_role("_TestFirmRole")
        assert r.x.shape == (10,)

    def test_use_applies_household_role(self) -> None:
        """sim.use() registers household-level roles with correct array size."""
        ext = Extension(
            roles={_TestHouseholdRole: "households"},
            events=[],
            relationships=[],
            config_dict={},
        )
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        sim.use(ext)
        r = sim.get_role("_TestHouseholdRole")
        assert r.y.shape == (50,)

    def test_use_applies_bank_role(self) -> None:
        """sim.use() registers bank-level roles with correct array size."""
        ext = Extension(
            roles={_TestBankRole: "banks"},
            events=[],
            relationships=[],
            config_dict={},
        )
        sim = Simulation.init(n_firms=10, n_households=50, n_banks=5, seed=42)
        sim.use(ext)
        r = sim.get_role("_TestBankRole")
        assert r.z.shape == (5,)

    def test_use_applies_events(self) -> None:
        """sim.use() inserts events into the pipeline."""
        ext = Extension(
            roles={},
            events=[_TestEvent],
            relationships=[],
            config_dict={},
        )
        sim = Simulation.init(seed=42)
        sim.use(ext)
        # Event should now be in the pipeline
        pipeline_names = [e.name for e in sim.pipeline.events]
        assert "__test_event" in pipeline_names

    def test_use_applies_config(self) -> None:
        """sim.use() sets extra_params from config_dict."""
        ext = Extension(
            roles={},
            events=[],
            relationships=[],
            config_dict={"my_param": 42.0},
        )
        sim = Simulation.init(seed=42)
        sim.use(ext)
        assert sim.extra_params["my_param"] == 42.0

    def test_use_config_does_not_overwrite_init_kwargs(self) -> None:
        """sim.use() config_dict doesn't overwrite kwargs from init."""
        ext = Extension(
            roles={},
            events=[],
            relationships=[],
            config_dict={"sigma_min": 999.0},
        )
        sim = Simulation.init(sigma_min=0.05, seed=42)
        sim.use(ext)
        assert sim.extra_params["sigma_min"] == 0.05

    def test_multiple_use_calls(self) -> None:
        """Multiple sim.use() calls coexist."""
        ext1 = Extension(
            roles={_TestFirmRole: "firms"},
            events=[],
            relationships=[],
            config_dict={"param_a": 1.0},
        )
        ext2 = Extension(
            roles={_TestHouseholdRole: "households"},
            events=[],
            relationships=[],
            config_dict={"param_b": 2.0},
        )
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        sim.use(ext1)
        sim.use(ext2)
        assert sim.get_role("_TestFirmRole").x.shape == (10,)
        assert sim.get_role("_TestHouseholdRole").y.shape == (50,)
        assert sim.extra_params["param_a"] == 1.0
        assert sim.extra_params["param_b"] == 2.0

    def test_use_coexists_with_manual_calls(self) -> None:
        """sim.use() works alongside manual use_role()/use_events()."""
        ext = Extension(
            roles={_TestFirmRole: "firms"},
            events=[],
            relationships=[],
            config_dict={},
        )
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        sim.use_role(_TestHouseholdRole, n_agents=50)
        sim.use(ext)
        # Both roles should be accessible
        assert sim.get_role("_TestFirmRole").x.shape == (10,)
        assert sim.get_role("_TestHouseholdRole").y.shape == (50,)

    def test_use_empty_extension(self) -> None:
        """sim.use() with empty Extension is a no-op."""
        ext = Extension()
        sim = Simulation.init(seed=42)
        sim.use(ext)
        # Should not raise; simulation still works
        sim.step()


class TestUseRelationship:
    """Test sim.use_relationship() placeholder."""

    def test_use_relationship_exists(self) -> None:
        """use_relationship() is callable."""
        sim = Simulation.init(seed=42)
        sim.use_relationship(object)  # No-op, should not raise

    def test_use_relationship_returns_none(self) -> None:
        """use_relationship() returns None."""
        sim = Simulation.init(seed=42)
        result = sim.use_relationship(object)
        assert result is None
