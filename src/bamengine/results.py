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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore[import-untyped]

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
    except ImportError:
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
    roles : list of str
        Role names to capture (e.g., ['Producer', 'Worker']).
    variables : dict or None
        Mapping of role name to list of variables to capture.
        If None, captures all variables for each role.
    include_economy : bool
        Whether to capture economy-wide metrics.
    aggregate : str or None
        Aggregation method ('mean', 'median', 'sum', 'std') or None for full data.
    """

    def __init__(
        self,
        roles: list[str],
        variables: dict[str, list[str]] | None,
        include_economy: bool,
        aggregate: str | None,
    ) -> None:
        self.roles = roles
        self.variables = variables
        self.include_economy = include_economy
        self.aggregate = aggregate
        # Storage: role_data[role_name][var_name] = list of arrays/scalars
        self.role_data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.economy_data: dict[str, list[float]] = defaultdict(list)

    def capture(self, sim: Simulation) -> None:
        """
        Capture one period of data from simulation.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to capture data from.
        """
        # Capture role data
        for role_name in self.roles:
            try:
                role = sim.get_role(role_name)
            except KeyError:
                continue

            # Get variables to capture for this role
            if self.variables and role_name in self.variables:
                var_names = self.variables[role_name]
            else:
                # Capture all public fields (those not starting with underscore)
                var_names = [
                    f for f in role.__dataclass_fields__ if not f.startswith("_")
                ]

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

        # Capture economy metrics
        if self.include_economy:
            ec = sim.ec
            # Capture the latest values from history arrays
            if len(ec.avg_mkt_price_history) > 0:
                self.economy_data["avg_price"].append(
                    float(ec.avg_mkt_price_history[-1])
                )
            if len(ec.unemp_rate_history) > 0:
                self.economy_data["unemployment_rate"].append(
                    float(ec.unemp_rate_history[-1])
                )
            if len(ec.inflation_history) > 0:
                self.economy_data["inflation"].append(float(ec.inflation_history[-1]))

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
    ) -> pd.DataFrame:
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
            return pd.DataFrame()

        result = pd.concat(dfs, axis=1)
        result.index.name = "period"
        return result

    def get_role_data(
        self, role_name: str, aggregate: str | None = None
    ) -> pd.DataFrame:
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
    def economy_metrics(self) -> pd.DataFrame:
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
            return pd.DataFrame()

        df = pd.DataFrame(self.economy_data)
        df.index.name = "period"
        return df

    @property
    def summary(self) -> pd.DataFrame:
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
