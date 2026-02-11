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
    aggregate : str or None, default=None
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
        "n_firm_bankruptcies",
        "n_bank_bankruptcies",
    ]

    def __init__(
        self,
        variables: dict[str, list[str] | Literal[True]],
        aggregate: str | None = None,
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
        # Storage for relationship data: rel_data[rel_name][var_name] = list
        self.relationship_data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Track which variables have been captured this period
        self._captured_this_period: set[str] = set()
        # Flag to indicate if timed capture is active
        self._use_timed_capture = bool(capture_after or capture_timing)
        # Cache for relationship names (populated on first use)
        self._relationship_names: set[str] | None = None

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
        # Each entry is (name, var_name, is_relationship)
        event_to_vars: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)

        for name, var_spec in self.variables.items():
            is_rel = self._is_relationship(name)

            if name == "Economy":
                # Economy metrics: check capture_timing first, then capture_after
                if var_spec is True:
                    vars_to_capture = self.ECONOMY_METRICS
                else:
                    vars_to_capture = var_spec
                for var_name in vars_to_capture:
                    key = f"Economy.{var_name}"
                    event = self.capture_timing.get(key, self.capture_after)
                    if event:
                        event_to_vars[event].append(("Economy", var_name, False))
            else:
                # Role or Relationship data
                try:
                    # Can't check variables until we have sim, so skip validation
                    if var_spec is True:
                        # Will capture all at runtime
                        if self.capture_after:
                            event_to_vars[self.capture_after].append(
                                (name, "*", is_rel)
                            )
                    else:
                        for var_name in var_spec:
                            key = f"{name}.{var_name}"
                            event = self.capture_timing.get(key, self.capture_after)
                            if event:
                                event_to_vars[event].append((name, var_name, is_rel))
                except Exception:  # pragma: no cover
                    pass  # Will capture at end-of-period

        # Register callbacks for each event
        for event_name, vars_list in event_to_vars.items():
            # Create callback with closure over vars_list
            def make_callback(
                vars_to_capture: list[tuple[str, str, bool]],
            ) -> Callable[[Simulation], None]:
                def callback(sim: Simulation) -> None:
                    for name, var_name, is_rel in vars_to_capture:
                        if var_name == "*":  # pragma: no cover
                            # Capture all variables from this role/relationship
                            if is_rel:
                                self._capture_relationship_all(sim, name)
                            else:
                                self._capture_role_all(sim, name)
                        elif name == "Economy":
                            self._capture_economy_single(sim, var_name)
                        elif is_rel:
                            self._capture_relationship_single(sim, name, var_name)
                        else:
                            self._capture_role_single(sim, name, var_name)

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

        # History-based metrics (take last value from history array)
        history_sources = {
            "avg_price": ec.avg_mkt_price_history,
            "inflation": ec.inflation_history,
        }

        if metric_name in history_sources:
            history = history_sources[metric_name]
            if len(history) > 0:
                self.economy_data[metric_name].append(float(history[-1]))
                self._captured_this_period.add(key)
            return

        # Transient metrics (capture current value, not from history)
        if metric_name == "n_firm_bankruptcies":
            self.economy_data[metric_name].append(len(ec.exiting_firms))
            self._captured_this_period.add(key)
        elif metric_name == "n_bank_bankruptcies":
            self.economy_data[metric_name].append(len(ec.exiting_banks))
            self._captured_this_period.add(key)

    def _is_relationship(self, name: str) -> bool:
        """Check if a name refers to a registered relationship."""
        if self._relationship_names is None:
            from bamengine.core.registry import list_relationships

            self._relationship_names = set(list_relationships())
        return name in self._relationship_names

    def _capture_relationship_single(
        self, sim: Simulation, rel_name: str, field_name: str
    ) -> None:
        """Capture a single field from a relationship."""
        key = f"{rel_name}.{field_name}"
        if key in self._captured_this_period:
            return  # Already captured

        try:
            rel = sim.get_relationship(rel_name)
        except KeyError:
            return

        if not hasattr(rel, field_name):
            return

        data = getattr(rel, field_name)
        if not isinstance(data, np.ndarray):
            return

        # Slice to only valid entries (up to rel.size)
        valid_data = data[: rel.size]

        # Apply aggregation if requested
        if self.aggregate:
            if len(valid_data) == 0:
                # Empty relationship, store NaN or 0
                value = 0.0
            elif self.aggregate == "mean":
                value = float(np.mean(valid_data))
            elif self.aggregate == "median":
                value = float(np.median(valid_data))
            elif self.aggregate == "sum":
                value = float(np.sum(valid_data))
            elif self.aggregate == "std":
                value = float(np.std(valid_data))
            else:
                value = float(np.mean(valid_data))  # fallback
            self.relationship_data[rel_name][field_name].append(value)
        else:
            # Store full array (copy to avoid mutation issues)
            self.relationship_data[rel_name][field_name].append(valid_data.copy())

        self._captured_this_period.add(key)

    def _capture_relationship_all(self, sim: Simulation, rel_name: str) -> None:
        """Capture all fields from a relationship."""
        try:
            rel = sim.get_relationship(rel_name)
        except KeyError:
            return

        # Get edge-specific fields (exclude base fields)
        base_fields = {"source_ids", "target_ids", "size", "capacity"}
        fields = getattr(rel, "__dataclass_fields__", {})
        field_names = [
            f for f in fields if f not in base_fields and not f.startswith("_")
        ]

        for field_name in field_names:
            self._capture_relationship_single(sim, rel_name, field_name)

    def _capture_relationship(
        self, sim: Simulation, rel_name: str, var_spec: list[str] | Literal[True]
    ) -> None:
        """Capture data from a relationship."""
        if var_spec is True:
            self._capture_relationship_all(sim, rel_name)
        else:
            for field_name in var_spec:
                self._capture_relationship_single(sim, rel_name, field_name)

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
            elif self._is_relationship(name):
                # Capture remaining relationship fields
                if var_spec is True:
                    self._capture_relationship_all(sim, name)
                else:
                    for field_name in var_spec:
                        key = f"{name}.{field_name}"
                        if key not in self._captured_this_period:
                            self._capture_relationship_single(sim, name, field_name)
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
            elif self._is_relationship(name):
                # Handle relationships
                self._capture_relationship(sim, name, var_spec)
            else:
                # Handle regular roles
                self._capture_role(sim, name, var_spec)

        # Clear tracking for next period
        self._captured_this_period.clear()

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

        # History-based metrics (take last value from history array)
        history_sources = {
            "avg_price": ec.avg_mkt_price_history,
            "unemployment_rate": ec.unemp_rate_history,
            "inflation": ec.inflation_history,
        }

        for metric_name in metrics_to_capture:
            if metric_name in history_sources:
                history = history_sources[metric_name]
                if len(history) > 0:
                    self.economy_data[metric_name].append(float(history[-1]))
            elif metric_name == "n_firm_bankruptcies":
                self.economy_data[metric_name].append(len(ec.exiting_firms))
            elif metric_name == "n_bank_bankruptcies":
                self.economy_data[metric_name].append(len(ec.exiting_banks))

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

        # Convert relationship data lists to arrays or keep as list
        final_relationship_data: dict[
            str, dict[str, NDArray[Any] | list[NDArray[Any]]]
        ] = {}
        for rel_name, rel_vars in self.relationship_data.items():
            final_relationship_data[rel_name] = {}
            for field_name, data_list in rel_vars.items():
                if not data_list:
                    continue
                if self.aggregate:
                    # List of scalars -> 1D array
                    final_relationship_data[rel_name][field_name] = np.array(data_list)
                else:
                    # List of variable-length arrays -> keep as list
                    # (cannot stack into 2D because edge counts vary per period)
                    final_relationship_data[rel_name][field_name] = data_list

        return SimulationResults(
            role_data=final_role_data,
            economy_data=final_economy_data,
            relationship_data=final_relationship_data,
            config=config,
            metadata=metadata,
        )


@dataclass
class SimulationResults:
    """
    Container for simulation results with convenient data access methods.

    This class is returned by Simulation.run() and provides structured
    access to simulation data, including time series of role states,
    economy-wide metrics, relationship edge data, and metadata about the
    simulation run.

    Attributes
    ----------
    role_data : dict
        Time series data for each role, keyed by role name.
        Each value is a dict of arrays with shape (n_periods, n_agents).
    economy_data : dict
        Time series of economy-wide metrics with shape (n_periods,).
    relationship_data : dict
        Time series data for each relationship, keyed by relationship name.
        Each value is a dict of arrays. When aggregated, arrays have shape
        (n_periods,). When not aggregated, values are lists of variable-length
        arrays (one per period).
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
    >>> # Access relationship data (when collected)
    >>> results = sim.run(n_periods=100, collect={"LoanBook": True, "aggregate": "sum"})
    >>> total_principal = results.relationship_data["LoanBook"]["principal"]
    """

    role_data: dict[str, dict[str, NDArray[Any]]] = field(default_factory=dict)
    economy_data: dict[str, NDArray[Any]] = field(default_factory=dict)
    relationship_data: dict[str, dict[str, NDArray[Any] | list[NDArray[Any]]]] = field(
        default_factory=dict
    )
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(
        self,
        roles: list[str] | None = None,
        variables: list[str] | None = None,
        include_economy: bool = True,
        aggregate: str | None = None,
        relationships: list[str] | None = None,
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
        relationships : list of str, optional
            Specific relationships to include. If None, includes all relationships
            with aggregated data. Relationships with non-aggregated data (list of
            arrays) are skipped with a warning.

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

        # Include relationship data
        >>> df = results.to_dataframe(relationships=["LoanBook"])
        """
        pd = _import_pandas()
        import warnings

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

        # Add relationship data
        if relationships is None:
            relationships = list(self.relationship_data.keys())

        for rel_name in relationships:
            if rel_name not in self.relationship_data:
                continue

            rel_dict = self.relationship_data[rel_name]

            for var_name, rel_data in rel_dict.items():
                if variables and var_name not in variables:
                    continue

                # Check if data is a list (non-aggregated variable-length arrays)
                if isinstance(rel_data, list):
                    warnings.warn(
                        f"Relationship '{rel_name}.{var_name}' has non-aggregated "
                        f"variable-length data and cannot be included in DataFrame. "
                        f"Access it directly via results.relationship_data['{rel_name}']"
                        f"['{var_name}'] or use aggregation during collection.",
                        stacklevel=2,
                    )
                    continue

                # Data is already 1D (aggregated during collection)
                df = pd.DataFrame({f"{rel_name}.{var_name}": rel_data})
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

    def get_relationship_data(
        self, rel_name: str, aggregate: str | None = None
    ) -> DataFrame:
        """
        Get data for a specific relationship as a DataFrame.

        Parameters
        ----------
        rel_name : str
            Name of the relationship (e.g., 'LoanBook').
        aggregate : {'mean', 'median', 'sum', 'std'}, optional
            How to aggregate (only used if data needs re-aggregation).

        Returns
        -------
        pd.DataFrame
            DataFrame with the relationship's time series data.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Notes
        -----
        If the relationship data was collected without aggregation
        (variable-length arrays per period), this method will issue a
        warning and return an empty DataFrame. Use
        ``results.relationship_data[rel_name]`` directly for such data.

        Examples
        --------
        >>> loans_df = results.get_relationship_data("LoanBook")
        """
        return self.to_dataframe(
            roles=[],
            relationships=[rel_name],
            include_economy=False,
            aggregate=aggregate,
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
    def data(self) -> dict[str, dict[str, NDArray[Any] | list[NDArray[Any]]]]:
        """
        Unified access to all data (roles + economy + relationships).

        Economy data is accessible under the "Economy" key.
        Relationship data is merged with role data (relationships have
        unique names so no conflicts).

        Returns
        -------
        dict
            Combined role, economy, and relationship data. Keys are role names,
            relationship names, and "Economy" for economy metrics.

        Examples
        --------
        >>> results.data["Producer"]["price"]
        >>> results.data["Economy"]["unemployment_rate"]
        >>> results.data["LoanBook"]["principal"]  # if collected
        """
        combined: dict[str, dict[str, NDArray[Any] | list[NDArray[Any]]]] = {}
        # Add role data (NDArray values are compatible with the union type)
        for role_name, role_dict in self.role_data.items():
            combined[role_name] = cast(
                dict[str, NDArray[Any] | list[NDArray[Any]]], role_dict
            )
        if self.economy_data:
            combined["Economy"] = cast(
                dict[str, NDArray[Any] | list[NDArray[Any]]], self.economy_data
            )
        # Add relationship data (already has the right type)
        for rel_name, rel_dict in self.relationship_data.items():
            combined[rel_name] = rel_dict
        return combined

    def get_array(
        self,
        name: str,
        variable_name: str,
        aggregate: str | None = None,
    ) -> NDArray[Any] | list[NDArray[Any]]:
        """
        Get a variable as a numpy array directly.

        This provides a convenient way to access simulation data without
        needing to navigate nested dictionaries.

        Parameters
        ----------
        name : str
            Role, relationship, or "Economy" name (e.g., "Producer",
            "LoanBook", "Economy").
        variable_name : str
            Variable name ("price", "principal", "unemployment_rate", etc.)
        aggregate : {'mean', 'sum', 'std', 'median'}, optional
            Aggregation method for 2D data. If provided, reduces
            (n_periods, n_agents) to (n_periods,).

        Returns
        -------
        NDArray or list[NDArray]
            1D array (n_periods,), 2D array (n_periods, n_agents), or
            list of arrays for non-aggregated relationship data.

        Raises
        ------
        KeyError
            If role/relationship or variable not found.

        Examples
        --------
        >>> productivity = results.get_array("Producer", "labor_productivity")
        >>> avg_prod = results.get_array(
        ...     "Producer", "labor_productivity", aggregate="mean"
        ... )
        >>> unemployment = results.get_array("Economy", "unemployment_rate")
        >>> total_principal = results.get_array("LoanBook", "principal")
        """
        # Handle Economy data specially
        if name == "Economy":
            if variable_name not in self.economy_data:
                available = list(self.economy_data.keys())
                raise KeyError(
                    f"'{variable_name}' not found in Economy. Available: {available}"
                )
            return self.economy_data[variable_name]

        # Check role data first
        if name in self.role_data:
            role_dict = self.role_data[name]
            if variable_name not in role_dict:
                available = list(role_dict.keys())
                raise KeyError(
                    f"'{variable_name}' not found in {name}. Available: {available}"
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

        # Check relationship data
        if name in self.relationship_data:
            rel_dict = self.relationship_data[name]
            if variable_name not in rel_dict:
                available = list(rel_dict.keys())
                raise KeyError(
                    f"'{variable_name}' not found in {name}. Available: {available}"
                )

            rel_data = rel_dict[variable_name]
            # Relationship data is either 1D (aggregated) or list of arrays
            # No additional aggregation is applied here (already done during collection)
            return rel_data

        # Not found in role_data or relationship_data
        available_roles = list(self.role_data.keys())
        available_rels = list(self.relationship_data.keys())
        # For backwards compatibility, use "Role" in error message
        raise KeyError(
            f"Role '{name}' not found. Available: {available_roles + available_rels}"
        )

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
        rels_str = (
            ", ".join(self.relationship_data.keys())
            if self.relationship_data
            else "None"
        )

        return (
            f"SimulationResults("
            f"periods={n_periods}, "
            f"firms={n_firms}, "
            f"households={n_households}, "
            f"roles=[{roles_str}], "
            f"relationships=[{rels_str}])"
        )
