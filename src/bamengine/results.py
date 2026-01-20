"""
Simulation results container for BAM Engine.

This module provides the SimulationResults class that encapsulates
simulation output data and provides convenient methods for data access
and export to pandas DataFrames.

Note: pandas is an optional dependency. It is only required when using
DataFrame export methods (to_dataframe, get_role_data, economy_metrics, summary).
Install with: pip install bamengine[pandas] or pip install pandas
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame

    from bamengine.simulation import Simulation


def _import_pandas() -> Any:
    """
    Lazily import pandas with helpful error message if not installed.

    Returns
    -------
    module
        The pandas module.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd

        return pd
    except ImportError:  # pragma: no cover
        raise ImportError(
            "pandas is required for DataFrame export methods. "
            "Install it with: pip install pandas"
        ) from None


class _DataCollector:
    """
    Internal helper to collect data during simulation.

    This class captures per-period snapshots of role and economy data
    during simulation execution. It's used by Simulation.run() when
    collect=True or collect={...} is specified.

    Parameters
    ----------
    variables : dict
        Mapping of role/component name to variables to capture.
        Keys are role names (e.g., 'Producer', 'Worker') or 'Economy'.
        Values are either:
        - list[str]: specific variables to capture
        - True: capture all variables for that role/component
    aggregate : str or None, default='mean'
        Aggregation method ('mean', 'median', 'sum', 'std') or None for full data.
    capture_after : str or None, default=None
        Default event name after which to capture data. If None, captures
        at end of period (after all events).
    capture_timing : dict or None, default=None
        Per-variable capture timing overrides. Maps "RoleName.var_name" to
        event name. Variables not in this dict use capture_after default.

    Examples
    --------
    Collect all variables from Producer and Worker, economy metrics:

    >>> collector = _DataCollector(
    ...     variables={"Producer": True, "Worker": True, "Economy": True},
    ...     aggregate="mean",
    ... )

    Collect specific variables with custom capture timing:

    >>> collector = _DataCollector(
    ...     variables={"Producer": ["production"], "Worker": ["employed", "wage"]},
    ...     aggregate=None,
    ...     capture_after="firms_update_net_worth",  # Default capture event
    ...     capture_timing={
    ...         "Producer.production": "firms_run_production",  # Before bankruptcy
    ...         "Worker.wage": "workers_receive_wage",
    ...     },
    ... )
    """

    # Available economy metrics (unemployment_rate removed - calculate from Worker.employed)
    ECONOMY_METRICS = [
        "avg_price",
        "inflation",
    ]

    def __init__(
        self,
        variables: dict[str, list[str] | Literal[True]],
        aggregate: str | None = "mean",
        capture_after: str | None = None,
        capture_timing: dict[str, str] | None = None,
    ) -> None:
        self.variables = variables
        self.aggregate = aggregate
        self.capture_after = capture_after
        self.capture_timing = capture_timing or {}
        # Storage: role_data[role_name][var_name] = list of arrays/scalars
        self.role_data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.economy_data: dict[str, list[float]] = defaultdict(list)
        # Track which variables have been captured this period
        self._captured_this_period: set[str] = set()
        # Flag to indicate if timed capture is active
        self._use_timed_capture = bool(capture_after or capture_timing)

    def setup_pipeline_callbacks(self, pipeline: Any) -> None:
        """
        Register capture callbacks with the pipeline for timed data capture.

        This method groups variables by their capture event and registers
        callbacks that will fire after each relevant event during pipeline
        execution.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to register callbacks with.

        Notes
        -----
        This method should be called before starting the simulation run.
        The callbacks will capture data at the appropriate events, and
        `capture_remaining()` should be called at end-of-period to capture
        any variables that weren't captured by callbacks.
        """
        from bamengine.core import Pipeline

        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Expected Pipeline, got {type(pipeline)}")

        # Group variables by their capture event
        event_to_vars: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for role_name, var_spec in self.variables.items():
            if role_name == "Economy":
                # Economy uses capture_after for all metrics
                if self.capture_after:
                    if var_spec is True:
                        vars_to_capture = self.ECONOMY_METRICS
                    else:
                        vars_to_capture = var_spec
                    for var_name in vars_to_capture:
                        event_to_vars[self.capture_after].append(("Economy", var_name))
            else:
                # Role data
                try:
                    # Can't check variables until we have sim, so skip validation
                    if var_spec is True:
                        # Will capture all at runtime
                        if self.capture_after:
                            event_to_vars[self.capture_after].append((role_name, "*"))
                    else:
                        for var_name in var_spec:
                            key = f"{role_name}.{var_name}"
                            event = self.capture_timing.get(key, self.capture_after)
                            if event:
                                event_to_vars[event].append((role_name, var_name))
                except Exception:  # pragma: no cover
                    pass  # Will capture at end-of-period

        # Register callbacks for each event
        for event_name, vars_list in event_to_vars.items():
            # Create callback with closure over vars_list
            def make_callback(
                vars_to_capture: list[tuple[str, str]],
            ) -> Callable[[Simulation], None]:
                def callback(sim: Simulation) -> None:
                    for role_name, var_name in vars_to_capture:
                        if var_name == "*":  # pragma: no cover
                            # Capture all variables from this role
                            self._capture_role_all(sim, role_name)
                        elif role_name == "Economy":
                            self._capture_economy_single(sim, var_name)
                        else:
                            self._capture_role_single(sim, role_name, var_name)

                return callback

            pipeline.register_after_event(event_name, make_callback(vars_list))

    def _capture_role_single(
        self, sim: Simulation, role_name: str, var_name: str
    ) -> None:
        """Capture a single variable from a role."""
        key = f"{role_name}.{var_name}"
        if key in self._captured_this_period:
            return  # Already captured

        try:
            role = sim.get_role(role_name)
        except KeyError:
            return

        if not hasattr(role, var_name):
            return

        data = getattr(role, var_name)
        if not isinstance(data, np.ndarray):
            return

        # Apply aggregation if requested
        if self.aggregate:
            if self.aggregate == "mean":
                value = float(np.mean(data))
            elif self.aggregate == "median":
                value = float(np.median(data))
            elif self.aggregate == "sum":
                value = float(np.sum(data))
            elif self.aggregate == "std":
                value = float(np.std(data))
            else:
                value = float(np.mean(data))  # fallback
            self.role_data[role_name][var_name].append(value)
        else:
            # Store full array (copy to avoid mutation issues)
            self.role_data[role_name][var_name].append(data.copy())

        self._captured_this_period.add(key)

    def _capture_role_all(self, sim: Simulation, role_name: str) -> None:
        """Capture all variables from a role."""
        try:
            role = sim.get_role(role_name)
        except KeyError:
            return

        var_names = [f for f in role.__dataclass_fields__ if not f.startswith("_")]
        for var_name in var_names:
            self._capture_role_single(sim, role_name, var_name)

    def _capture_economy_single(self, sim: Simulation, metric_name: str) -> None:
        """Capture a single economy metric."""
        key = f"Economy.{metric_name}"
        if key in self._captured_this_period:
            return  # Already captured

        ec = sim.ec

        metric_sources = {
            "avg_price": ec.avg_mkt_price_history,
            "inflation": ec.inflation_history,
        }

        if metric_name in metric_sources:
            history = metric_sources[metric_name]
            if len(history) > 0:
                self.economy_data[metric_name].append(float(history[-1]))
                self._captured_this_period.add(key)

    def capture_remaining(self, sim: Simulation) -> None:
        """
        Capture any variables not yet captured this period.

        This is called at the end of each period to capture variables that
        weren't captured by timed callbacks (either because they have no
        capture_timing specified, or timed capture is not being used).

        After capturing, resets the captured tracking set for the next period.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to capture data from.
        """
        for name, var_spec in self.variables.items():
            if name == "Economy":
                # Capture remaining economy metrics
                if var_spec is True:
                    metrics = self.ECONOMY_METRICS
                else:
                    metrics = var_spec
                for metric in metrics:
                    key = f"Economy.{metric}"
                    if key not in self._captured_this_period:
                        self._capture_economy_single(sim, metric)
            else:
                # Capture remaining role variables
                if var_spec is True:
                    self._capture_role_all(sim, name)
                else:
                    for var_name in var_spec:
                        key = f"{name}.{var_name}"
                        if key not in self._captured_this_period:  # pragma: no cover
                            self._capture_role_single(sim, name, var_name)

        # Reset for next period
        self._captured_this_period.clear()

    def capture(self, sim: Simulation) -> None:
        """
        Capture one period of data from simulation.

        This is the original capture method for non-timed capture (when
        capture_after and capture_timing are not specified). All data is
        captured at the same point (end of period).

        For timed capture (when capture_after or capture_timing are specified),
        use `setup_pipeline_callbacks()` before the run and `capture_remaining()`
        at the end of each period instead.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to capture data from.
        """
        for name, var_spec in self.variables.items():
            if name == "Economy":
                # Handle Economy as a pseudo-role
                self._capture_economy(sim, var_spec)
            else:
                # Handle regular roles
                self._capture_role(sim, name, var_spec)

    def _capture_role(
        self, sim: Simulation, role_name: str, var_spec: list[str] | Literal[True]
    ) -> None:
        """Capture data from a single role."""
        try:
            role = sim.get_role(role_name)
        except KeyError:
            return

        # Determine which variables to capture
        if var_spec is True:
            # Capture all public fields (those not starting with underscore)
            var_names = [f for f in role.__dataclass_fields__ if not f.startswith("_")]
        else:
            var_names = var_spec

        for var_name in var_names:
            if not hasattr(role, var_name):
                continue

            data = getattr(role, var_name)
            if not isinstance(data, np.ndarray):
                continue

            # Apply aggregation if requested
            if self.aggregate:
                if self.aggregate == "mean":
                    value = float(np.mean(data))
                elif self.aggregate == "median":
                    value = float(np.median(data))
                elif self.aggregate == "sum":
                    value = float(np.sum(data))
                elif self.aggregate == "std":
                    value = float(np.std(data))
                else:
                    value = float(np.mean(data))  # fallback
                self.role_data[role_name][var_name].append(value)
            else:
                # Store full array (copy to avoid mutation issues)
                self.role_data[role_name][var_name].append(data.copy())

    def _capture_economy(
        self, sim: Simulation, var_spec: list[str] | Literal[True]
    ) -> None:
        """Capture economy metrics."""
        ec = sim.ec

        # Determine which metrics to capture
        if var_spec is True:
            metrics_to_capture = self.ECONOMY_METRICS
        else:
            metrics_to_capture = var_spec

        # Map metric names to history arrays
        metric_sources = {
            "avg_price": ec.avg_mkt_price_history,
            "unemployment_rate": ec.unemp_rate_history,
            "inflation": ec.inflation_history,
        }

        for metric_name in metrics_to_capture:
            if metric_name in metric_sources:
                history = metric_sources[metric_name]
                if len(history) > 0:
                    self.economy_data[metric_name].append(float(history[-1]))

    def finalize(
        self, config: dict[str, Any], metadata: dict[str, Any]
    ) -> SimulationResults:
        """
        Convert collected data to SimulationResults.

        Parameters
        ----------
        config : dict
            Simulation configuration parameters.
        metadata : dict
            Run metadata (n_periods, seed, runtime, etc.).

        Returns
        -------
        SimulationResults
            Results container with collected data as NumPy arrays.
        """
        # Convert role data lists to arrays
        final_role_data: dict[str, dict[str, NDArray[Any]]] = {}
        for role_name, role_vars in self.role_data.items():
            final_role_data[role_name] = {}
            for var_name, data_list in role_vars.items():
                if not data_list:
                    continue
                if self.aggregate:
                    # List of scalars -> 1D array
                    final_role_data[role_name][var_name] = np.array(data_list)
                else:
                    # List of arrays -> 2D array (n_periods, n_agents)
                    final_role_data[role_name][var_name] = np.stack(data_list, axis=0)

        # Convert economy data lists to arrays
        final_economy_data: dict[str, NDArray[Any]] = {}
        for metric_name, data_list in self.economy_data.items():
            if data_list:
                final_economy_data[metric_name] = np.array(data_list)

        return SimulationResults(
            role_data=final_role_data,
            economy_data=final_economy_data,
            config=config,
            metadata=metadata,
        )


@dataclass
class SimulationResults:
    """
    Container for simulation results with convenient data access methods.

    This class is returned by Simulation.run() and provides structured
    access to simulation data, including time series of role states,
    economy-wide metrics, and metadata about the simulation run.

    Attributes
    ----------
    role_data : dict
        Time series data for each role, keyed by role name.
        Each value is a dict of arrays with shape (n_periods, n_agents).
    economy_data : dict
        Time series of economy-wide metrics with shape (n_periods,).
    config : dict
        Configuration parameters used for this simulation.
    metadata : dict
        Run metadata (seed, runtime, n_periods, etc.).

    Examples
    --------
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> results = sim.run(n_periods=100)
    >>> # Get all data as DataFrame
    >>> df = results.to_dataframe()
    >>> # Get specific role data
    >>> prod_df = results.get_role_data("Producer")
    >>> # Access economy metrics directly
    >>> unemployment = results.economy_data["unemployment_rate"]
    """

    role_data: dict[str, dict[str, NDArray[Any]]] = field(default_factory=dict)
    economy_data: dict[str, NDArray[Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(
        self,
        roles: list[str] | None = None,
        variables: list[str] | None = None,
        include_economy: bool = True,
        aggregate: str | None = None,
    ) -> DataFrame:
        """
        Export results to a pandas DataFrame.

        Parameters
        ----------
        roles : list of str, optional
            Specific roles to include. If None, includes all roles.
        variables : list of str, optional
            Specific variables to include. If None, includes all variables.
        include_economy : bool, default=True
            Whether to include economy-wide metrics.
        aggregate : {'mean', 'median', 'sum', 'std'}, optional
            How to aggregate agent-level data. If None, returns all agents.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results. Index is period number.
            Columns depend on parameters and aggregation method.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        # Get everything
        >>> df = results.to_dataframe()

        # Get only Producer price and inventory, averaged
        >>> df = results.to_dataframe(
        ...     roles=["Producer"], variables=["price", "inventory"], aggregate="mean"
        ... )

        # Get only economy metrics
        >>> df = results.to_dataframe(include_economy=True, roles=[])
        """
        pd = _import_pandas()
        dfs = []

        # Add role data
        if roles is None:
            roles = list(self.role_data.keys())

        for role_name in roles:
            if role_name not in self.role_data:
                continue

            role_dict = self.role_data[role_name]

            for var_name, data in role_dict.items():
                if variables and var_name not in variables:
                    continue

                # Handle both 1D (already aggregated) and 2D (per-agent) data
                if data.ndim == 1:
                    # Data is already 1D (aggregated during collection)
                    df = pd.DataFrame({f"{role_name}.{var_name}": data})
                    dfs.append(df)
                elif aggregate:
                    # 2D data, aggregate across agents (axis=1)
                    if aggregate == "mean":
                        agg_data = np.mean(data, axis=1)
                    elif aggregate == "median":
                        agg_data = np.median(data, axis=1)
                    elif aggregate == "sum":
                        agg_data = np.sum(data, axis=1)
                    elif aggregate == "std":
                        agg_data = np.std(data, axis=1)
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregate}")

                    df = pd.DataFrame({f"{role_name}.{var_name}.{aggregate}": agg_data})
                    dfs.append(df)
                else:
                    # 2D data, return all agents
                    _n_periods, n_agents = data.shape
                    columns = {
                        f"{role_name}.{var_name}.{i}": data[:, i]
                        for i in range(n_agents)
                    }
                    df = pd.DataFrame(columns)
                    dfs.append(df)

        # Add economy data
        if include_economy and self.economy_data:
            econ_df = pd.DataFrame(self.economy_data)
            dfs.append(econ_df)

        # Combine all DataFrames
        if not dfs:
            return cast("DataFrame", pd.DataFrame())

        result = pd.concat(dfs, axis=1)
        result.index.name = "period"
        return cast("DataFrame", result)

    def get_role_data(self, role_name: str, aggregate: str | None = None) -> DataFrame:
        """
        Get data for a specific role as a DataFrame.

        Parameters
        ----------
        role_name : str
            Name of the role (e.g., 'Producer', 'Worker').
        aggregate : {'mean', 'median', 'sum', 'std'}, optional
            How to aggregate across agents.

        Returns
        -------
        pd.DataFrame
            DataFrame with the role's time series data.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> prod_df = results.get_role_data("Producer")
        >>> prod_mean = results.get_role_data("Producer", aggregate="mean")
        """
        return self.to_dataframe(
            roles=[role_name], include_economy=False, aggregate=aggregate
        )

    @property
    def economy_metrics(self) -> DataFrame:
        """
        Get economy-wide metrics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with economy time series (unemployment rate, GDP, etc.).

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> econ_df = results.economy_metrics
        >>> econ_df[["unemployment_rate", "avg_price"]].plot()
        """
        pd = _import_pandas()
        if not self.economy_data:
            return cast("DataFrame", pd.DataFrame())

        df = pd.DataFrame(self.economy_data)
        df.index.name = "period"
        return cast("DataFrame", df)

    @property
    def data(self) -> dict[str, dict[str, NDArray[Any]]]:
        """
        Unified access to all data (roles + economy).

        Economy data is accessible under the "Economy" key.

        Returns
        -------
        dict
            Combined role and economy data. Keys are role names
            (plus "Economy" for economy metrics).

        Examples
        --------
        >>> results.data["Producer"]["price"]
        >>> results.data["Economy"]["unemployment_rate"]
        """
        combined: dict[str, dict[str, NDArray[Any]]] = dict(self.role_data)
        if self.economy_data:
            combined["Economy"] = self.economy_data
        return combined

    def get_array(
        self,
        role_name: str,
        variable_name: str,
        aggregate: str | None = None,
    ) -> NDArray[Any]:
        """
        Get a variable as a numpy array directly.

        This provides a convenient way to access simulation data without
        needing to navigate nested dictionaries.

        Parameters
        ----------
        role_name : str
            Role name ("Producer", "Worker", "Economy", etc.)
        variable_name : str
            Variable name ("price", "unemployment_rate", etc.)
        aggregate : {'mean', 'sum', 'std', 'median'}, optional
            Aggregation method for 2D data. If provided, reduces
            (n_periods, n_agents) to (n_periods,).

        Returns
        -------
        NDArray
            1D array (n_periods,) or 2D array (n_periods, n_agents).

        Raises
        ------
        KeyError
            If role or variable not found.

        Examples
        --------
        >>> productivity = results.get_array("Producer", "labor_productivity")
        >>> avg_prod = results.get_array(
        ...     "Producer", "labor_productivity", aggregate="mean"
        ... )
        >>> unemployment = results.get_array("Economy", "unemployment_rate")
        """
        # Handle Economy data specially
        if role_name == "Economy":
            if variable_name not in self.economy_data:
                available = list(self.economy_data.keys())
                raise KeyError(
                    f"'{variable_name}' not found in Economy. Available: {available}"
                )
            return self.economy_data[variable_name]

        # Handle role data
        if role_name not in self.role_data:
            available = list(self.role_data.keys())
            raise KeyError(f"Role '{role_name}' not found. Available: {available}")

        role_dict = self.role_data[role_name]
        if variable_name not in role_dict:
            available = list(role_dict.keys())
            raise KeyError(
                f"'{variable_name}' not found in {role_name}. Available: {available}"
            )

        data = role_dict[variable_name]

        # Apply aggregation if requested and data is 2D
        if aggregate and data.ndim == 2:
            AggFunc = Callable[[NDArray[Any], int], NDArray[Any]]
            agg_funcs: dict[str, AggFunc] = {
                "mean": np.mean,
                "sum": np.sum,
                "std": np.std,
                "median": np.median,
            }
            if aggregate not in agg_funcs:
                raise ValueError(
                    f"Unknown aggregation '{aggregate}'. "
                    f"Use one of: {list(agg_funcs.keys())}"
                )
            return agg_funcs[aggregate](data, 1)

        return data

    @property
    def summary(self) -> DataFrame:
        """
        Get summary statistics for key metrics.

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, std, min, max) for key variables.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> print(results.summary)
        """
        # Get aggregated data (this will call _import_pandas via to_dataframe)
        df = self.to_dataframe(aggregate="mean")

        # Compute summary statistics
        summary = df.describe().T

        # Add additional statistics if useful
        summary["cv"] = summary["std"] / summary["mean"]  # Coefficient of variation

        return summary

    def save(self, filepath: str) -> None:
        """
        Save results to disk (HDF5 or pickle format).

        Parameters
        ----------
        filepath : str
            Path to save file. Use .h5 for HDF5, .pkl for pickle.

        Examples
        --------
        >>> results.save("results.h5")
        >>> results.save("results.pkl")
        """
        # Implementation would use pandas HDFStore or pickle
        # This is a placeholder for the interface
        raise NotImplementedError("Save functionality not yet implemented")

    @classmethod
    def load(cls, filepath: str) -> SimulationResults:
        """
        Load results from disk.

        Parameters
        ----------
        filepath : str
            Path to saved results file.

        Returns
        -------
        SimulationResults
            Loaded results object.

        Examples
        --------
        >>> results = SimulationResults.load("results.h5")
        """
        # Implementation would use pandas HDFStore or pickle
        # This is a placeholder for the interface
        raise NotImplementedError("Load functionality not yet implemented")

    def __repr__(self) -> str:
        """String representation showing summary information."""
        n_periods = self.metadata.get("n_periods", 0)
        n_firms = self.metadata.get("n_firms", 0)
        n_households = self.metadata.get("n_households", 0)

        roles_str = ", ".join(self.role_data.keys()) if self.role_data else "None"

        return (
            f"SimulationResults("
            f"periods={n_periods}, "
            f"firms={n_firms}, "
            f"households={n_households}, "
            f"roles=[{roles_str}])"
        )
