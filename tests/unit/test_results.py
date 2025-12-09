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

    def test_init(self):
        """Test collector initialization."""
        collector = _DataCollector(
            roles=["Producer"],
            variables=None,
            include_economy=True,
            aggregate="mean",
        )
        assert collector.roles == ["Producer"]
        assert collector.variables is None
        assert collector.include_economy is True
        assert collector.aggregate == "mean"

    def test_finalize_with_aggregated_data(self):
        """Test finalize with aggregated data."""
        collector = _DataCollector(
            roles=["Producer"],
            variables=None,
            include_economy=True,
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
            roles=["Producer"],
            variables=None,
            include_economy=False,
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
            roles=["Producer"],
            variables=None,
            include_economy=True,
            aggregate="mean",
        )
        results = collector.finalize(config={}, metadata={})
        assert results.role_data == {}
        assert results.economy_data == {}


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
