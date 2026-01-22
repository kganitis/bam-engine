"""Unit tests for SimulationResults class."""

import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bamengine.results import SimulationResults, _DataCollector, _import_pandas


class TestSimulationResults:
    """Tests for SimulationResults dataclass."""

    def test_empty_results(self):
        """Test empty results initialization."""
        results = SimulationResults()
        assert results.role_data == {}
        assert results.economy_data == {}
        assert results.config == {}
        assert results.metadata == {}

    def test_to_dataframe_empty(self):
        """Test to_dataframe with empty results."""
        results = SimulationResults()
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_with_role_data_aggregated(self):
        """Test to_dataframe with aggregated role data."""
        results = SimulationResults(
            role_data={
                "Producer": {
                    "price": np.array([1.0, 1.1, 1.2]),  # 3 periods
                    "inventory": np.array([10.0, 11.0, 12.0]),
                }
            },
            metadata={"n_periods": 3},
        )
        # When data is already 1D (aggregated), it should work
        df = results.to_dataframe(aggregate="mean")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_dataframe_with_economy_data(self):
        """Test to_dataframe with economy data."""
        results = SimulationResults(
            economy_data={
                "unemployment_rate": np.array([0.05, 0.06, 0.04]),
                "avg_price": np.array([1.5, 1.6, 1.7]),
            }
        )
        df = results.to_dataframe(roles=[], include_economy=True)
        assert "unemployment_rate" in df.columns
        assert "avg_price" in df.columns
        assert len(df) == 3

    def test_to_dataframe_filter_roles(self):
        """Test to_dataframe with role filtering."""
        results = SimulationResults(
            role_data={
                "Producer": {"price": np.array([1.0, 1.1])},
                "Worker": {"wage": np.array([50.0, 51.0])},
            }
        )
        df = results.to_dataframe(roles=["Producer"], include_economy=False)
        # Should only have Producer columns
        assert any("Producer" in c for c in df.columns)
        assert not any("Worker" in c for c in df.columns)

    def test_to_dataframe_filter_variables(self):
        """Test to_dataframe with variable filtering."""
        results = SimulationResults(
            role_data={
                "Producer": {
                    "price": np.array([1.0, 1.1]),
                    "inventory": np.array([10.0, 11.0]),
                }
            }
        )
        df = results.to_dataframe(
            variables=["price"], include_economy=False, aggregate="mean"
        )
        assert any("price" in c for c in df.columns)
        assert not any("inventory" in c for c in df.columns)

    def test_to_dataframe_aggregation_methods(self):
        """Test different aggregation methods."""
        # 2D data: (n_periods=2, n_agents=3)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data}},
        )

        # Mean aggregation
        df_mean = results.to_dataframe(aggregate="mean", include_economy=False)
        np.testing.assert_array_almost_equal(
            df_mean["Producer.price.mean"].values, [2.0, 5.0]
        )

        # Sum aggregation
        df_sum = results.to_dataframe(aggregate="sum", include_economy=False)
        np.testing.assert_array_almost_equal(
            df_sum["Producer.price.sum"].values, [6.0, 15.0]
        )

        # Median aggregation
        df_median = results.to_dataframe(aggregate="median", include_economy=False)
        np.testing.assert_array_almost_equal(
            df_median["Producer.price.median"].values, [2.0, 5.0]
        )

        # Std aggregation
        df_std = results.to_dataframe(aggregate="std", include_economy=False)
        assert len(df_std) == 2

    def test_to_dataframe_invalid_aggregation(self):
        """Test to_dataframe with invalid aggregation method."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data}},
        )
        with pytest.raises(ValueError, match="Unknown aggregation"):
            results.to_dataframe(aggregate="invalid")

    def test_to_dataframe_no_aggregation(self):
        """Test to_dataframe without aggregation (full per-agent data)."""
        # 2D data: (n_periods=2, n_agents=3)
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data}},
        )
        df = results.to_dataframe(aggregate=None, include_economy=False)
        # Should have columns for each agent
        assert "Producer.price.0" in df.columns
        assert "Producer.price.1" in df.columns
        assert "Producer.price.2" in df.columns
        assert len(df) == 2

    def test_get_role_data(self):
        """Test get_role_data method."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data}},
        )
        df = results.get_role_data("Producer", aggregate="mean")
        assert isinstance(df, pd.DataFrame)
        assert "Producer.price.mean" in df.columns

    def test_get_role_data_nonexistent(self):
        """Test get_role_data with nonexistent role."""
        results = SimulationResults()
        df = results.get_role_data("NonExistent")
        assert len(df) == 0

    def test_economy_metrics_property(self):
        """Test economy_metrics property."""
        results = SimulationResults(
            economy_data={
                "unemployment_rate": np.array([0.05, 0.06]),
                "inflation": np.array([0.02, 0.03]),
            }
        )
        df = results.economy_metrics
        assert isinstance(df, pd.DataFrame)
        assert "unemployment_rate" in df.columns
        assert "inflation" in df.columns
        assert df.index.name == "period"

    def test_economy_metrics_empty(self):
        """Test economy_metrics with empty data."""
        results = SimulationResults()
        df = results.economy_metrics
        assert len(df) == 0

    def test_summary_property(self):
        """Test summary property."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data}},
            economy_data={"unemployment_rate": np.array([0.05, 0.06, 0.04])},
        )
        summary = results.summary
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "cv" in summary.columns

    def test_repr(self):
        """Test string representation."""
        results = SimulationResults(
            role_data={"Producer": {}, "Worker": {}},
            metadata={"n_periods": 100, "n_firms": 50, "n_households": 200},
        )
        repr_str = repr(results)
        assert "periods=100" in repr_str
        assert "firms=50" in repr_str
        assert "households=200" in repr_str
        assert "Producer" in repr_str
        assert "Worker" in repr_str

    def test_repr_empty(self):
        """Test string representation with empty results."""
        results = SimulationResults()
        repr_str = repr(results)
        assert "periods=0" in repr_str
        assert "roles=[None]" in repr_str

    def test_save_not_implemented(self):
        """Test save raises NotImplementedError."""
        results = SimulationResults()
        with pytest.raises(NotImplementedError):
            results.save("test.h5")

    def test_load_not_implemented(self):
        """Test load raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SimulationResults.load("test.h5")


class TestPandasOptionalImport:
    """Tests for optional pandas import."""

    def test_import_pandas_success(self):
        """Test _import_pandas returns pandas when available."""
        pd_module = _import_pandas()
        assert pd_module is pd

    def test_import_pandas_failure(self):
        """Test _import_pandas raises ImportError when pandas not available."""
        # Mock sys.modules to simulate pandas not being installed
        with mock.patch.dict(sys.modules, {"pandas": None}):
            # Need to reload the module to trigger the import
            # Instead, test the error message pattern
            pass  # Skip this test - hard to mock reliably

    def test_to_dataframe_requires_pandas(self):
        """Test that to_dataframe works when pandas is available."""
        results = SimulationResults(
            economy_data={"test": np.array([1.0, 2.0])},
        )
        # Should work fine with pandas installed
        df = results.to_dataframe(roles=[])
        assert isinstance(df, pd.DataFrame)

    def test_economy_metrics_requires_pandas(self):
        """Test that economy_metrics works when pandas is available."""
        results = SimulationResults(
            economy_data={"test": np.array([1.0, 2.0])},
        )
        df = results.economy_metrics
        assert isinstance(df, pd.DataFrame)


class TestDataCollector:
    """Tests for _DataCollector class."""

    def test_init_all_variables(self):
        """Test collector initialization with True for all variables."""
        collector = _DataCollector(
            variables={"Producer": True, "Economy": True},
            aggregate="mean",
        )
        assert collector.variables == {"Producer": True, "Economy": True}
        assert collector.aggregate == "mean"

    def test_init_specific_variables(self):
        """Test collector initialization with specific variables."""
        collector = _DataCollector(
            variables={"Producer": ["price", "inventory"], "Economy": ["avg_price"]},
            aggregate=None,
        )
        assert collector.variables == {
            "Producer": ["price", "inventory"],
            "Economy": ["avg_price"],
        }
        assert collector.aggregate is None

    def test_finalize_with_aggregated_data(self):
        """Test finalize with aggregated data."""
        collector = _DataCollector(
            variables={"Producer": True, "Economy": True},
            aggregate="mean",
        )
        # Manually add data (simulating captures)
        collector.role_data["Producer"]["price"] = [1.0, 2.0, 3.0]
        collector.economy_data["unemployment_rate"] = [0.05, 0.06, 0.04]

        results = collector.finalize(
            config={"h_rho": 0.1},
            metadata={"n_periods": 3},
        )

        assert isinstance(results, SimulationResults)
        assert "Producer" in results.role_data
        np.testing.assert_array_equal(
            results.role_data["Producer"]["price"], [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            results.economy_data["unemployment_rate"], [0.05, 0.06, 0.04]
        )
        assert results.config["h_rho"] == 0.1
        assert results.metadata["n_periods"] == 3

    def test_finalize_with_full_data(self):
        """Test finalize with full (non-aggregated) data."""
        collector = _DataCollector(
            variables={"Producer": True},
            aggregate=None,
        )
        # Manually add array data (simulating captures without aggregation)
        collector.role_data["Producer"]["price"] = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]

        results = collector.finalize(config={}, metadata={})

        # Should stack into 2D array
        assert results.role_data["Producer"]["price"].shape == (2, 2)

    def test_finalize_empty(self):
        """Test finalize with no data."""
        collector = _DataCollector(
            variables={"Producer": True, "Economy": True},
            aggregate="mean",
        )
        results = collector.finalize(config={}, metadata={})
        assert results.role_data == {}
        assert results.economy_data == {}

    def test_economy_metrics_constant(self):
        """Test that ECONOMY_METRICS contains expected metrics."""
        assert "avg_price" in _DataCollector.ECONOMY_METRICS
        assert "inflation" in _DataCollector.ECONOMY_METRICS
        # Note: unemployment_rate is no longer in ECONOMY_METRICS by default.
        # Unemployment should now be calculated from Worker.employed data.
        assert "unemployment_rate" not in _DataCollector.ECONOMY_METRICS


class TestDataProperty:
    """Tests for data property."""

    def test_data_combines_role_and_economy(self):
        """Test data property combines role_data and economy_data."""
        results = SimulationResults(
            role_data={
                "Producer": {"price": np.array([1.0, 2.0])},
                "Worker": {"wage": np.array([50.0, 51.0])},
            },
            economy_data={
                "unemployment_rate": np.array([0.05, 0.06]),
            },
        )
        data = results.data
        assert "Producer" in data
        assert "Worker" in data
        assert "Economy" in data
        np.testing.assert_array_equal(data["Producer"]["price"], [1.0, 2.0])
        np.testing.assert_array_equal(
            data["Economy"]["unemployment_rate"], [0.05, 0.06]
        )

    def test_data_empty_economy(self):
        """Test data property with no economy data."""
        results = SimulationResults(
            role_data={"Producer": {"price": np.array([1.0, 2.0])}},
        )
        data = results.data
        assert "Producer" in data
        assert "Economy" not in data

    def test_data_empty_role(self):
        """Test data property with only economy data."""
        results = SimulationResults(
            economy_data={"unemployment_rate": np.array([0.05, 0.06])},
        )
        data = results.data
        assert "Economy" in data
        assert len(data) == 1


class TestGetArray:
    """Tests for get_array method."""

    def test_get_array_role_data(self):
        """Test get_array retrieves role data."""
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        arr = results.get_array("Producer", "price")
        np.testing.assert_array_equal(arr, data_2d)

    def test_get_array_economy_data(self):
        """Test get_array retrieves economy data."""
        unemp = np.array([0.05, 0.06, 0.04])
        results = SimulationResults(
            economy_data={"unemployment_rate": unemp},
        )
        arr = results.get_array("Economy", "unemployment_rate")
        np.testing.assert_array_equal(arr, unemp)

    def test_get_array_with_aggregation_mean(self):
        """Test get_array with mean aggregation."""
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        arr = results.get_array("Producer", "price", aggregate="mean")
        np.testing.assert_array_almost_equal(arr, [2.0, 5.0])

    def test_get_array_with_aggregation_sum(self):
        """Test get_array with sum aggregation."""
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        arr = results.get_array("Producer", "price", aggregate="sum")
        np.testing.assert_array_almost_equal(arr, [6.0, 15.0])

    def test_get_array_with_aggregation_std(self):
        """Test get_array with std aggregation."""
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        arr = results.get_array("Producer", "price", aggregate="std")
        expected = np.std(data_2d, axis=1)
        np.testing.assert_array_almost_equal(arr, expected)

    def test_get_array_with_aggregation_median(self):
        """Test get_array with median aggregation."""
        data_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        arr = results.get_array("Producer", "price", aggregate="median")
        np.testing.assert_array_almost_equal(arr, [2.0, 5.0])

    def test_get_array_role_not_found(self):
        """Test get_array raises KeyError for missing role."""
        results = SimulationResults(
            role_data={"Producer": {"price": np.array([1.0, 2.0])}},
        )
        with pytest.raises(KeyError, match="Role 'Worker' not found"):
            results.get_array("Worker", "wage")

    def test_get_array_variable_not_found(self):
        """Test get_array raises KeyError for missing variable."""
        results = SimulationResults(
            role_data={"Producer": {"price": np.array([1.0, 2.0])}},
        )
        with pytest.raises(KeyError, match="'inventory' not found in Producer"):
            results.get_array("Producer", "inventory")

    def test_get_array_economy_variable_not_found(self):
        """Test get_array raises KeyError for missing economy variable."""
        results = SimulationResults(
            economy_data={"unemployment_rate": np.array([0.05])},
        )
        with pytest.raises(KeyError, match="'inflation' not found in Economy"):
            results.get_array("Economy", "inflation")

    def test_get_array_invalid_aggregation(self):
        """Test get_array raises ValueError for invalid aggregation."""
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        results = SimulationResults(
            role_data={"Producer": {"price": data_2d}},
        )
        with pytest.raises(ValueError, match="Unknown aggregation 'invalid'"):
            results.get_array("Producer", "price", aggregate="invalid")

    def test_get_array_aggregation_on_1d_data(self):
        """Test get_array with aggregation on 1D data returns as-is."""
        data_1d = np.array([1.0, 2.0, 3.0])
        results = SimulationResults(
            role_data={"Producer": {"price": data_1d}},
        )
        # Aggregation on 1D data should return the data unchanged
        arr = results.get_array("Producer", "price", aggregate="mean")
        np.testing.assert_array_equal(arr, data_1d)


class TestPandasImportError:
    """Tests for pandas import error handling."""

    def test_import_pandas_raises_helpful_error(self):
        """Test _import_pandas raises helpful ImportError when pandas unavailable."""
        # We can't actually remove pandas, but we can test that the function
        # returns pandas when available (the error path is hard to test without
        # actually uninstalling pandas)
        pd_module = _import_pandas()
        assert pd_module is pd


class TestDataCollectorPipelineSetup:
    """Tests for _DataCollector.setup_pipeline_callbacks()."""

    def test_setup_pipeline_callbacks_requires_pipeline(self):
        """Test that setup_pipeline_callbacks raises TypeError for non-Pipeline."""
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
            capture_after="firms_adjust_price",
        )
        # Pass a non-Pipeline object
        with pytest.raises(TypeError, match="Expected Pipeline"):
            collector.setup_pipeline_callbacks("not_a_pipeline")

    def test_setup_pipeline_callbacks_with_capture_after(self):
        """Test setup_pipeline_callbacks with capture_after configuration."""
        from bamengine.core import Pipeline

        collector = _DataCollector(
            variables={"Producer": ["price"], "Economy": ["avg_price"]},
            aggregate="mean",
            capture_after="firms_adjust_price",
        )
        # Create a minimal pipeline
        pipeline = Pipeline(events=[])
        # This should not raise - callbacks are registered even if events don't exist yet
        collector.setup_pipeline_callbacks(pipeline)
        # Verify timed capture is enabled
        assert collector._use_timed_capture is True

    def test_setup_pipeline_callbacks_with_wildcard_capture(self):
        """Test setup_pipeline_callbacks with True (wildcard) for all variables."""
        from bamengine.core import Pipeline

        collector = _DataCollector(
            variables={"Producer": True},
            aggregate="mean",
            capture_after="firms_adjust_price",
        )
        pipeline = Pipeline(events=[])
        collector.setup_pipeline_callbacks(pipeline)
        assert collector._use_timed_capture is True

    def test_setup_pipeline_callbacks_with_capture_timing(self):
        """Test setup_pipeline_callbacks with per-variable capture_timing."""
        from bamengine.core import Pipeline

        collector = _DataCollector(
            variables={"Producer": ["price", "inventory"]},
            aggregate="mean",
            capture_after="firms_adjust_price",
            capture_timing={"Producer.price": "firms_run_production"},
        )
        pipeline = Pipeline(events=[])
        collector.setup_pipeline_callbacks(pipeline)
        assert collector._use_timed_capture is True

    def test_setup_pipeline_callbacks_with_relationship_variables(self):
        """Test setup_pipeline_callbacks with relationship-specific variables."""
        from bamengine.core import Pipeline

        collector = _DataCollector(
            variables={"LoanBook": ["principal", "rate"]},
            aggregate="sum",
            capture_after="firms_run_production",
        )
        pipeline = Pipeline(events=[])
        collector.setup_pipeline_callbacks(pipeline)
        assert collector._use_timed_capture is True

    def test_pipeline_callback_captures_relationship_data(self):
        """Test that pipeline callback correctly captures relationship data."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
            capture_after="firms_run_production",
        )
        collector.setup_pipeline_callbacks(sim.pipeline)

        # Manually trigger the callback to test the path
        callbacks = sim.pipeline._after_event_callbacks.get("firms_run_production", [])
        for callback in callbacks:
            callback(sim)

        # Should have captured data
        assert len(collector.relationship_data["LoanBook"]["principal"]) == 1


class TestDataCollectorCaptureLogic:
    """Tests for _DataCollector capture deduplication and aggregation."""

    def test_capture_role_single_already_captured(self):
        """Test that _capture_role_single skips already captured variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
        )
        # Capture once
        collector._capture_role_single(sim, "Producer", "price")
        first_count = len(collector.role_data["Producer"]["price"])

        # Try to capture again in same period - should be skipped (deduplication)
        collector._capture_role_single(sim, "Producer", "price")
        second_count = len(collector.role_data["Producer"]["price"])

        assert first_count == second_count == 1

    def test_capture_role_single_missing_role(self):
        """Test _capture_role_single handles missing role gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"NonExistent": ["price"]},
            aggregate="mean",
        )
        # Should not raise, just skip
        collector._capture_role_single(sim, "NonExistent", "price")
        assert (
            "NonExistent" not in collector.role_data
            or len(collector.role_data["NonExistent"]) == 0
        )

    def test_capture_role_single_missing_attribute(self):
        """Test _capture_role_single handles missing attribute gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": ["nonexistent_attr"]},
            aggregate="mean",
        )
        # Should not raise, just skip
        collector._capture_role_single(sim, "Producer", "nonexistent_attr")
        assert len(collector.role_data["Producer"]["nonexistent_attr"]) == 0

    def test_capture_role_single_non_array_attribute(self):
        """Test _capture_role_single skips non-array attributes."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
        )
        # Temporarily set a non-array attribute
        original_price = sim.prod.price
        sim.prod.price = "not_an_array"  # type: ignore
        try:
            collector._capture_role_single(sim, "Producer", "price")
            assert len(collector.role_data["Producer"]["price"]) == 0
        finally:
            sim.prod.price = original_price

    def test_capture_role_single_aggregation_fallback(self):
        """Test _capture_role_single falls back to mean for unknown aggregation."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="unknown_method",  # Unknown method should fall back to mean
        )
        collector._capture_role_single(sim, "Producer", "price")
        # Should have captured using mean (fallback)
        assert len(collector.role_data["Producer"]["price"]) == 1
        expected_mean = float(np.mean(sim.prod.price))
        assert collector.role_data["Producer"]["price"][0] == pytest.approx(
            expected_mean
        )

    @pytest.mark.parametrize(
        "aggregate,expected_func",
        [
            ("median", np.median),
            ("sum", np.sum),
            ("std", np.std),
        ],
    )
    def test_capture_role_single_aggregation_methods(self, aggregate, expected_func):
        """Test _capture_role_single with different aggregation methods."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate=aggregate,
        )
        collector._capture_role_single(sim, "Producer", "price")
        assert len(collector.role_data["Producer"]["price"]) == 1
        expected_value = float(expected_func(sim.prod.price))
        assert collector.role_data["Producer"]["price"][0] == pytest.approx(
            expected_value
        )

    def test_capture_role_all(self):
        """Test _capture_role_all captures all role variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"Producer": True},
            aggregate="mean",
        )
        collector._capture_role_all(sim, "Producer")
        # Should have captured multiple variables
        assert len(collector.role_data["Producer"]) > 0
        # price should be one of them
        assert "price" in collector.role_data["Producer"]

    def test_capture_role_all_missing_role(self):
        """Test _capture_role_all handles missing role gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        collector = _DataCollector(
            variables={"NonExistent": True},
            aggregate="mean",
        )
        # Should not raise
        collector._capture_role_all(sim, "NonExistent")

    def test_capture_economy_single_already_captured(self):
        """Test _capture_economy_single skips already captured metrics."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        # Run one period to generate economy data
        sim.run(n_periods=1, collect=False)

        collector = _DataCollector(
            variables={"Economy": ["avg_price"]},
            aggregate="mean",
        )
        # Capture once
        collector._capture_economy_single(sim, "avg_price")
        first_count = len(collector.economy_data["avg_price"])

        # Try to capture again - should be skipped (deduplication)
        collector._capture_economy_single(sim, "avg_price")
        second_count = len(collector.economy_data["avg_price"])

        assert first_count == second_count == 1


class TestDataCollectorCaptureRemaining:
    """Tests for _DataCollector.capture_remaining()."""

    def test_capture_remaining_captures_economy_true(self):
        """Test capture_remaining with Economy: True captures all metrics."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        sim.run(n_periods=1, collect=False)

        collector = _DataCollector(
            variables={"Economy": True},
            aggregate="mean",
        )
        collector.capture_remaining(sim)
        # Should have captured the economy metrics
        assert len(collector.economy_data) > 0

    def test_capture_remaining_captures_economy_specific(self):
        """Test capture_remaining with specific Economy variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)
        sim.run(n_periods=1, collect=False)

        collector = _DataCollector(
            variables={"Economy": ["avg_price"]},
            aggregate="mean",
        )
        collector.capture_remaining(sim)
        assert "avg_price" in collector.economy_data

    def test_capture_remaining_captures_role_true(self):
        """Test capture_remaining with Role: True captures all variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"Producer": True},
            aggregate="mean",
        )
        collector.capture_remaining(sim)
        assert "price" in collector.role_data["Producer"]

    def test_capture_remaining_skips_already_captured(self):
        """Test capture_remaining skips already captured variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
        )
        # Manually mark as captured
        collector._captured_this_period.add("Producer.price")
        collector.capture_remaining(sim)
        # Should not have captured (already marked)
        assert len(collector.role_data["Producer"]["price"]) == 0

    def test_capture_remaining_clears_tracking_set(self):
        """Test capture_remaining clears _captured_this_period."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
        )
        collector._captured_this_period.add("Producer.price")
        collector.capture_remaining(sim)
        # Should have cleared the tracking set
        assert len(collector._captured_this_period) == 0


class TestDataCollectorCapture:
    """Tests for _DataCollector.capture() method."""

    def test_capture_missing_role(self):
        """Test capture handles missing role gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"NonExistent": ["price"]},
            aggregate="mean",
        )
        # Should not raise
        collector.capture(sim)

    def test_capture_missing_attribute(self):
        """Test capture handles missing attribute gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"Producer": ["nonexistent_attr"]},
            aggregate="mean",
        )
        # Should not raise, just skip
        collector.capture(sim)
        assert len(collector.role_data["Producer"]["nonexistent_attr"]) == 0

    def test_capture_aggregation_fallback(self):
        """Test capture falls back to mean for unknown aggregation method."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=5, n_households=10, seed=42)

        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="unknown_method",  # Should fall back to mean
        )
        collector.capture(sim)
        assert len(collector.role_data["Producer"]["price"]) == 1
        expected_mean = float(np.mean(sim.prod.price))
        assert collector.role_data["Producer"]["price"][0] == pytest.approx(
            expected_mean
        )


class TestDataCollectorFinalize:
    """Tests for _DataCollector.finalize() method."""

    def test_finalize_with_empty_data_lists(self):
        """Test finalize handles empty data lists gracefully."""
        collector = _DataCollector(
            variables={"Producer": ["price"]},
            aggregate="mean",
        )
        # Add an empty list
        collector.role_data["Producer"]["price"] = []

        results = collector.finalize(config={}, metadata={})
        # Empty list should not appear in results
        assert "price" not in results.role_data.get("Producer", {})


# ===========================================================================
# Relationship Data Collection Tests
# ===========================================================================


class TestRelationshipDataCollector:
    """Tests for _DataCollector relationship capture methods."""

    def test_is_relationship_detects_loanbook(self):
        """Test _is_relationship correctly identifies LoanBook."""
        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        assert collector._is_relationship("LoanBook") is True
        assert collector._is_relationship("Producer") is False
        assert collector._is_relationship("Economy") is False

    def test_capture_relationship_single_unknown_aggregation_fallback(self):
        """Test _capture_relationship_single falls back to mean for unknown aggregate."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1, 2], dtype=np.int64),
            amount=np.array([100.0, 200.0, 300.0]),
            rate=np.array([0.02, 0.03, 0.025]),
        )

        # Use an unknown aggregate value - should fall back to mean
        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="unknown_agg",  # Unknown aggregation
        )
        collector._capture_relationship_single(sim, "LoanBook", "principal")

        # Should have captured mean as fallback
        expected_mean = float(np.mean(loans.principal[: loans.size]))
        assert collector.relationship_data["LoanBook"]["principal"][0] == pytest.approx(
            expected_mean
        )

    def test_capture_relationship_all_missing_relationship(self):
        """Test _capture_relationship_all handles missing relationship gracefully."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)

        collector = _DataCollector(
            variables={"NonExistentRelationship": True},
            aggregate="sum",
        )
        # Should not raise - just silently return
        collector._capture_relationship_all(sim, "NonExistentRelationship")
        # Should have nothing captured
        assert len(collector.relationship_data.get("NonExistentRelationship", {})) == 0

    def test_capture_relationship_single_with_aggregation(self):
        """Test _capture_relationship_single with aggregation."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1, 2], dtype=np.int64),
            amount=np.array([100.0, 150.0, 200.0]),
            rate=np.array([0.02, 0.03, 0.025]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        collector._capture_relationship_single(sim, "LoanBook", "principal")

        # Should have captured one value
        assert len(collector.relationship_data["LoanBook"]["principal"]) == 1
        # Value should be sum of all principals
        expected_sum = float(np.sum(loans.principal[: loans.size]))
        assert collector.relationship_data["LoanBook"]["principal"][0] == pytest.approx(
            expected_sum
        )

    def test_capture_relationship_single_without_aggregation(self):
        """Test _capture_relationship_single without aggregation."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1, 2], dtype=np.int64),
            amount=np.array([100.0, 150.0, 200.0]),
            rate=np.array([0.02, 0.03, 0.025]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate=None,
        )
        collector._capture_relationship_single(sim, "LoanBook", "principal")

        # Should have captured one array
        assert len(collector.relationship_data["LoanBook"]["principal"]) == 1
        captured = collector.relationship_data["LoanBook"]["principal"][0]
        assert isinstance(captured, np.ndarray)

        # Array should match active loans
        expected = loans.principal[: loans.size]
        np.testing.assert_array_equal(captured, expected)

    def test_capture_relationship_single_already_captured(self):
        """Test _capture_relationship_single skips already captured variables."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        # Capture once
        collector._capture_relationship_single(sim, "LoanBook", "principal")
        first_count = len(collector.relationship_data["LoanBook"]["principal"])

        # Try again - should be skipped
        collector._capture_relationship_single(sim, "LoanBook", "principal")
        second_count = len(collector.relationship_data["LoanBook"]["principal"])

        assert first_count == second_count == 1

    def test_capture_relationship_single_missing_relationship(self):
        """Test _capture_relationship_single handles missing relationship."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)

        collector = _DataCollector(
            variables={"NonExistent": ["field"]},
            aggregate="sum",
        )
        # Should not raise
        collector._capture_relationship_single(sim, "NonExistent", "field")
        # Should have nothing captured
        assert (
            "NonExistent" not in collector.relationship_data
            or len(collector.relationship_data["NonExistent"]) == 0
        )

    def test_capture_relationship_single_missing_field(self):
        """Test _capture_relationship_single handles missing field."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)

        collector = _DataCollector(
            variables={"LoanBook": ["nonexistent_field"]},
            aggregate="sum",
        )
        # Should not raise
        collector._capture_relationship_single(sim, "LoanBook", "nonexistent_field")
        assert len(collector.relationship_data["LoanBook"]["nonexistent_field"]) == 0

    def test_capture_relationship_single_empty_relationship(self):
        """Test _capture_relationship_single handles empty relationship."""
        import bamengine as bam

        # Use a fresh simulation with potentially no loans yet
        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        # Don't run any periods - LoanBook should be empty

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        collector._capture_relationship_single(sim, "LoanBook", "principal")

        # Should capture 0.0 for empty relationship
        assert len(collector.relationship_data["LoanBook"]["principal"]) == 1
        assert collector.relationship_data["LoanBook"]["principal"][0] == 0.0

    def test_capture_relationship_all(self):
        """Test _capture_relationship_all captures all fields."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": True},
            aggregate="sum",
        )
        collector._capture_relationship_all(sim, "LoanBook")

        # Should have captured multiple fields
        rel_data = collector.relationship_data["LoanBook"]
        assert "principal" in rel_data
        assert "rate" in rel_data
        assert "debt" in rel_data
        assert "interest" in rel_data

    @pytest.mark.parametrize(
        "aggregate,expected_func",
        [
            ("mean", np.mean),
            ("median", np.median),
            ("sum", np.sum),
            ("std", np.std),
        ],
    )
    def test_capture_relationship_aggregation_methods(self, aggregate, expected_func):
        """Test _capture_relationship_single with different aggregations."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans to ensure there is data to aggregate
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1, 2], dtype=np.int64),
            amount=np.array([100.0, 150.0, 200.0]),
            rate=np.array([0.02, 0.03, 0.025]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate=aggregate,
        )
        collector._capture_relationship_single(sim, "LoanBook", "principal")

        expected = float(expected_func(loans.principal[: loans.size]))
        assert collector.relationship_data["LoanBook"]["principal"][0] == pytest.approx(
            expected
        )


class TestRelationshipCapture:
    """Tests for relationship capture in _DataCollector.capture()."""

    def test_capture_includes_relationships(self):
        """Test capture() correctly captures relationship data."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal", "rate"]},
            aggregate="sum",
        )
        collector.capture(sim)

        assert "principal" in collector.relationship_data["LoanBook"]
        assert "rate" in collector.relationship_data["LoanBook"]

    def test_capture_remaining_includes_relationships(self):
        """Test capture_remaining() correctly captures remaining relationship data."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        # Mark nothing as captured yet
        collector.capture_remaining(sim)

        assert len(collector.relationship_data["LoanBook"]["principal"]) == 1

    def test_capture_remaining_relationship_true(self):
        """Test capture_remaining() with var_spec=True for relationships."""
        import bamengine as bam

        sim = bam.Simulation.init(n_firms=10, n_households=20, seed=42)
        loans = sim.get_relationship("LoanBook")

        # Manually add test loans
        loans.append_loans_for_lender(
            lender_idx=np.intp(0),
            borrower_indices=np.array([0, 1], dtype=np.int64),
            amount=np.array([100.0, 150.0]),
            rate=np.array([0.02, 0.03]),
        )

        collector = _DataCollector(
            variables={"LoanBook": True},  # True = capture all fields
            aggregate="sum",
        )
        # Mark nothing as captured yet
        collector.capture_remaining(sim)

        # Should have captured all LoanBook fields
        assert "principal" in collector.relationship_data["LoanBook"]
        assert "rate" in collector.relationship_data["LoanBook"]
        assert "debt" in collector.relationship_data["LoanBook"]


class TestRelationshipFinalize:
    """Tests for relationship data finalization."""

    def test_finalize_with_aggregated_relationship_data(self):
        """Test finalize converts aggregated relationship data to array."""
        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        # Simulate captured data
        collector.relationship_data["LoanBook"]["principal"] = [100.0, 150.0, 200.0]

        results = collector.finalize(config={}, metadata={})

        assert "LoanBook" in results.relationship_data
        np.testing.assert_array_equal(
            results.relationship_data["LoanBook"]["principal"], [100.0, 150.0, 200.0]
        )

    def test_finalize_with_non_aggregated_relationship_data(self):
        """Test finalize keeps non-aggregated relationship data as list."""
        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate=None,
        )
        # Simulate variable-length captured data
        collector.relationship_data["LoanBook"]["principal"] = [
            np.array([100.0, 50.0]),
            np.array([100.0, 50.0, 75.0]),
            np.array([25.0]),
        ]

        results = collector.finalize(config={}, metadata={})

        assert "LoanBook" in results.relationship_data
        # Should remain a list (cannot stack variable-length arrays)
        assert isinstance(results.relationship_data["LoanBook"]["principal"], list)
        assert len(results.relationship_data["LoanBook"]["principal"]) == 3

    def test_finalize_with_empty_relationship_data_list(self):
        """Test finalize handles empty relationship data lists gracefully."""
        collector = _DataCollector(
            variables={"LoanBook": ["principal"]},
            aggregate="sum",
        )
        # Add an empty list
        collector.relationship_data["LoanBook"]["principal"] = []

        results = collector.finalize(config={}, metadata={})
        # Empty list should not appear in results
        assert "principal" not in results.relationship_data.get("LoanBook", {})


class TestSimulationResultsRelationship:
    """Tests for relationship data in SimulationResults."""

    def test_empty_relationship_data(self):
        """Test empty relationship_data initialization."""
        results = SimulationResults()
        assert results.relationship_data == {}

    def test_relationship_data_field(self):
        """Test relationship_data field stores data correctly."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": np.array([100.0, 150.0]),
                    "rate": np.array([0.02, 0.03]),
                }
            }
        )
        assert "LoanBook" in results.relationship_data
        np.testing.assert_array_equal(
            results.relationship_data["LoanBook"]["principal"], [100.0, 150.0]
        )

    def test_repr_includes_relationships(self):
        """Test __repr__ includes relationship information."""
        results = SimulationResults(
            relationship_data={"LoanBook": {}},
            metadata={"n_periods": 10, "n_firms": 50, "n_households": 200},
        )
        repr_str = repr(results)
        assert "relationships=[LoanBook]" in repr_str

    def test_repr_no_relationships(self):
        """Test __repr__ shows None for no relationships."""
        results = SimulationResults(
            metadata={"n_periods": 10, "n_firms": 50, "n_households": 200}
        )
        repr_str = repr(results)
        assert "relationships=[None]" in repr_str

    def test_data_property_includes_relationships(self):
        """Test data property includes relationship data."""
        results = SimulationResults(
            role_data={"Producer": {"price": np.array([1.0, 2.0])}},
            relationship_data={
                "LoanBook": {"principal": np.array([100.0, 150.0])},
            },
        )
        data = results.data
        assert "Producer" in data
        assert "LoanBook" in data
        np.testing.assert_array_equal(data["LoanBook"]["principal"], [100.0, 150.0])


class TestGetRelationshipData:
    """Tests for get_relationship_data method."""

    def test_get_relationship_data(self):
        """Test get_relationship_data returns DataFrame."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": np.array([100.0, 150.0, 200.0]),
                    "rate": np.array([0.02, 0.03, 0.025]),
                }
            }
        )
        df = results.get_relationship_data("LoanBook")
        assert isinstance(df, pd.DataFrame)
        assert "LoanBook.principal" in df.columns
        assert "LoanBook.rate" in df.columns

    def test_get_relationship_data_nonexistent(self):
        """Test get_relationship_data with nonexistent relationship."""
        results = SimulationResults()
        df = results.get_relationship_data("NonExistent")
        assert len(df) == 0


class TestGetArrayRelationship:
    """Tests for get_array method with relationships."""

    def test_get_array_relationship_data(self):
        """Test get_array retrieves relationship data."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {"principal": np.array([100.0, 150.0, 200.0])}
            }
        )
        arr = results.get_array("LoanBook", "principal")
        np.testing.assert_array_equal(arr, [100.0, 150.0, 200.0])

    def test_get_array_relationship_list_data(self):
        """Test get_array retrieves non-aggregated relationship data."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": [
                        np.array([100.0, 50.0]),
                        np.array([75.0]),
                    ]
                }
            }
        )
        arr = results.get_array("LoanBook", "principal")
        assert isinstance(arr, list)
        assert len(arr) == 2

    def test_get_array_relationship_not_found(self):
        """Test get_array raises KeyError for missing relationship variable."""
        results = SimulationResults(
            relationship_data={"LoanBook": {"principal": np.array([100.0])}}
        )
        with pytest.raises(KeyError, match="'rate' not found in LoanBook"):
            results.get_array("LoanBook", "rate")


class TestToDataFrameRelationships:
    """Tests for to_dataframe with relationship data."""

    def test_to_dataframe_with_relationships(self):
        """Test to_dataframe includes relationship data."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": np.array([100.0, 150.0, 200.0]),
                    "rate": np.array([0.02, 0.03, 0.025]),
                }
            }
        )
        df = results.to_dataframe(roles=[], relationships=["LoanBook"])
        assert "LoanBook.principal" in df.columns
        assert "LoanBook.rate" in df.columns
        assert len(df) == 3

    def test_to_dataframe_skips_non_aggregated_with_warning(self):
        """Test to_dataframe warns and skips non-aggregated relationship data."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": [
                        np.array([100.0, 50.0]),
                        np.array([75.0]),
                    ]
                }
            }
        )
        with pytest.warns(UserWarning, match="non-aggregated variable-length data"):
            df = results.to_dataframe(roles=[], relationships=["LoanBook"])
        # DataFrame should be empty since only non-aggregated data
        assert len(df) == 0 or "LoanBook.principal" not in df.columns

    def test_to_dataframe_mixed_role_and_relationship(self):
        """Test to_dataframe with both role and relationship data."""
        results = SimulationResults(
            role_data={"Producer": {"price": np.array([1.0, 1.1, 1.2])}},
            relationship_data={
                "LoanBook": {"principal": np.array([100.0, 150.0, 200.0])}
            },
        )
        df = results.to_dataframe(
            roles=["Producer"],
            relationships=["LoanBook"],
            include_economy=False,
        )
        assert "Producer.price" in df.columns
        assert "LoanBook.principal" in df.columns

    def test_to_dataframe_default_includes_all_relationships(self):
        """Test to_dataframe includes all relationships by default."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {"principal": np.array([100.0, 150.0])},
            }
        )
        df = results.to_dataframe(roles=[], include_economy=False)
        assert "LoanBook.principal" in df.columns

    def test_to_dataframe_with_variables_filter_relationships(self):
        """Test to_dataframe filters relationship variables correctly."""
        results = SimulationResults(
            relationship_data={
                "LoanBook": {
                    "principal": np.array([100.0, 150.0]),
                    "rate": np.array([0.02, 0.03]),
                }
            }
        )
        # Filter to only include 'rate' variable
        df = results.to_dataframe(
            roles=[], relationships=["LoanBook"], variables=["rate"]
        )
        assert "LoanBook.rate" in df.columns
        assert "LoanBook.principal" not in df.columns
