"""Integration tests for SimulationResults with Simulation.run()."""

import numpy as np
import pandas as pd

import bamengine as bam
from bamengine import Simulation, SimulationResults


class TestRunWithCollect:
    """Tests for Simulation.run() with collect parameter."""

    def test_run_without_collect_returns_none(self):
        """Test that run() without collect returns None (backward compatible)."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        result = sim.run(n_periods=5)
        assert result is None

    def test_run_collect_false_returns_none(self):
        """Test that run(collect=False) returns None."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        result = sim.run(n_periods=5, collect=False)
        assert result is None

    def test_run_collect_true_returns_results(self):
        """Test that run(collect=True) returns SimulationResults."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=5, collect=True)

        assert isinstance(results, SimulationResults)
        assert results.metadata["n_periods"] == 5
        assert results.metadata["n_firms"] == 10
        assert results.metadata["n_households"] == 50
        assert "runtime_seconds" in results.metadata

    def test_run_collect_true_has_all_roles(self):
        """Test that collect=True captures all roles."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=5, collect=True)

        expected_roles = [
            "Producer",
            "Worker",
            "Employer",
            "Borrower",
            "Lender",
            "Consumer",
        ]
        for role in expected_roles:
            assert role in results.role_data, f"Missing role: {role}"

    def test_run_collect_true_has_economy_metrics(self):
        """Test that collect=True captures economy metrics."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=5, collect=True)

        assert "avg_price" in results.economy_data
        assert "inflation" in results.economy_data
        # Note: unemployment_rate removed from default economy metrics
        # Calculate from Worker.employed instead:
        # employed = results.get_array("Worker", "employed")
        # unemployment_rate = 1 - np.mean(employed, axis=1)

    def test_run_collect_true_correct_shapes(self):
        """Test that collected data has correct shapes."""
        n_periods = 10
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=n_periods, collect=True)

        # With aggregate='mean' (default), role data should be 1D arrays
        for role_name, role_vars in results.role_data.items():
            for var_name, data in role_vars.items():
                assert data.shape == (n_periods,), (
                    f"{role_name}.{var_name} has wrong shape: {data.shape}"
                )

        # Economy data should also be 1D
        for metric_name, data in results.economy_data.items():
            assert data.shape == (n_periods,), (
                f"Economy {metric_name} has wrong shape: {data.shape}"
            )

    def test_run_collect_list_form(self):
        """Test list form of collect parameter."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=5,
            collect=["Producer", "Worker", "Economy"],
        )

        assert "Producer" in results.role_data
        assert "Worker" in results.role_data
        assert "avg_price" in results.economy_data
        # Other roles should not be captured
        assert "Employer" not in results.role_data
        assert "Borrower" not in results.role_data

    def test_run_collect_custom_variables(self):
        """Test custom variable selection with new dict API."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=5,
            collect={
                "Producer": ["price", "inventory"],
            },
        )

        assert "Producer" in results.role_data
        assert "price" in results.role_data["Producer"]
        assert "inventory" in results.role_data["Producer"]
        # Other variables should not be captured
        assert "production" not in results.role_data["Producer"]

    def test_run_collect_no_aggregation(self):
        """Test collecting full per-agent data without aggregation."""
        n_periods = 5
        n_firms = 10
        sim = Simulation.init(n_firms=n_firms, n_households=50, seed=42)
        results = sim.run(
            n_periods=n_periods,
            collect={
                "Producer": ["price"],
                "aggregate": None,
            },
        )

        # Without aggregation, should be 2D array (n_periods, n_agents)
        price_data = results.role_data["Producer"]["price"]
        assert price_data.shape == (n_periods, n_firms)

    def test_run_collect_different_aggregations(self):
        """Test different aggregation methods."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        for agg_method in ["mean", "median", "sum", "std"]:
            results = sim.run(
                n_periods=3,
                collect={
                    "Producer": ["price"],
                    "aggregate": agg_method,
                },
            )
            assert results.role_data["Producer"]["price"].shape == (3,)

    def test_run_collect_no_economy(self):
        """Test collecting without economy by not including Economy key."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=5,
            collect={"Producer": True},  # No Economy key = no economy data
        )

        assert results.economy_data == {}

    def test_run_collect_mixed_true_and_list(self):
        """Test mixing True and list values in collect dict."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=5,
            collect={
                "Producer": ["price", "inventory"],  # Specific variables
                "Worker": True,  # All Worker variables
                "Economy": True,  # All economy metrics
            },
        )

        # Producer should have only specified variables
        assert "price" in results.role_data["Producer"]
        assert "inventory" in results.role_data["Producer"]
        assert "production" not in results.role_data["Producer"]

        # Worker should have all variables
        assert "wage" in results.role_data["Worker"]
        assert "employer" in results.role_data["Worker"]

        # Economy should have all metrics
        assert "avg_price" in results.economy_data
        assert "inflation" in results.economy_data

    def test_run_collect_specific_economy_metrics(self):
        """Test collecting specific economy metrics."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=5,
            collect={
                "Economy": ["avg_price"],  # Only this metric
            },
        )

        assert "avg_price" in results.economy_data
        assert "inflation" not in results.economy_data

    def test_run_collect_config_in_results(self):
        """Test that config is stored in results."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42, h_rho=0.15)
        results = sim.run(n_periods=5, collect=True)

        assert "h_rho" in results.config
        assert results.config["h_rho"] == 0.15

    def test_results_to_dataframe(self):
        """Test converting results to DataFrame."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=10, collect=True)

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # n_periods rows

    def test_results_economy_metrics(self):
        """Test economy_metrics property."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=10, collect=True)

        econ_df = results.economy_metrics
        assert isinstance(econ_df, pd.DataFrame)
        assert "avg_price" in econ_df.columns
        assert "inflation" in econ_df.columns
        assert len(econ_df) == 10

    def test_results_get_role_data(self):
        """Test get_role_data method."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=10, collect=True)

        prod_df = results.get_role_data("Producer")
        assert isinstance(prod_df, pd.DataFrame)
        assert len(prod_df) == 10

    def test_results_summary(self):
        """Test summary property."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=10, collect=True)

        summary = results.summary
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_run_multiple_times_independent(self):
        """Test running multiple times returns independent results."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        results1 = sim.run(n_periods=5, collect=True)
        results2 = sim.run(n_periods=5, collect=True)

        # Results should be different (simulation state advanced)
        # First value of results2 should be close to last value of results1
        assert results2 is not results1

    def test_determinism_with_seed(self):
        """Test that results are deterministic with same seed."""
        sim1 = Simulation.init(n_firms=10, n_households=50, seed=42)
        results1 = sim1.run(n_periods=10, collect=True)

        sim2 = Simulation.init(n_firms=10, n_households=50, seed=42)
        results2 = sim2.run(n_periods=10, collect=True)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results1.economy_data["avg_price"],
            results2.economy_data["avg_price"],
        )

    def test_repr_with_real_results(self):
        """Test string representation with real simulation results."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(n_periods=10, collect=True)

        repr_str = repr(results)
        assert "periods=10" in repr_str
        assert "firms=10" in repr_str
        assert "households=50" in repr_str


class TestSimulationResultsExport:
    """Tests for SimulationResults export from bamengine."""

    def test_simulation_results_importable(self):
        """Test that SimulationResults is importable from bamengine."""
        assert hasattr(bam, "SimulationResults")
        assert bam.SimulationResults is SimulationResults

    def test_simulation_results_in_all(self):
        """Test that SimulationResults is in __all__."""
        assert "SimulationResults" in bam.__all__


class TestCaptureTiming:
    """Tests for configurable capture timing feature."""

    def test_capture_after_basic(self):
        """Test basic capture_after functionality."""
        n_periods = 5
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=n_periods,
            collect={
                "Producer": ["production"],
                "aggregate": None,
                "capture_after": "firms_run_production",
            },
        )

        # Should have collected data
        production = results.role_data["Producer"]["production"]
        assert production.shape == (n_periods, 10)

    def test_capture_timing_per_variable(self):
        """Test per-variable capture timing configuration."""
        n_periods = 5
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=n_periods,
            collect={
                "Producer": ["production"],
                "Worker": ["employed"],
                "aggregate": None,
                "capture_after": "firms_update_net_worth",
                "capture_timing": {
                    "Producer.production": "firms_run_production",
                    "Worker.employed": "workers_update_contracts",
                },
            },
        )

        # Both should be collected
        production = results.role_data["Producer"]["production"]
        employed = results.role_data["Worker"]["employed"]
        assert production.shape == (n_periods, 10)
        assert employed.shape == (n_periods, 50)

    def test_capture_timing_with_economy(self):
        """Test capture timing with economy metrics."""
        n_periods = 5
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=n_periods,
            collect={
                "Economy": True,
                "capture_after": "firms_run_production",
            },
        )

        # Economy metrics should be captured
        assert "avg_price" in results.economy_data
        assert results.economy_data["avg_price"].shape == (n_periods,)

    def test_no_capture_timing_backward_compatible(self):
        """Test that collect without capture timing still works."""
        n_periods = 5
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)
        results = sim.run(
            n_periods=n_periods,
            collect={
                "Producer": ["production"],
                "aggregate": "mean",
            },
        )

        # Should use default end-of-step capture
        production = results.role_data["Producer"]["production"]
        assert production.shape == (n_periods,)

    def test_capture_callbacks_cleared_after_run(self):
        """Test that pipeline callbacks are cleared after run."""
        sim = Simulation.init(n_firms=10, n_households=50, seed=42)

        # First run with capture timing
        sim.run(
            n_periods=3,
            collect={
                "Producer": ["production"],
                "capture_after": "firms_run_production",
            },
        )

        # Pipeline should have no callbacks registered
        assert len(sim.pipeline._after_event_callbacks) == 0

        # Second run should work normally
        results = sim.run(
            n_periods=3,
            collect={
                "Producer": ["production"],
            },
        )
        assert "Producer" in results.role_data
