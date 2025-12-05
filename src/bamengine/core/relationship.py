"""
Relationship system for managing many-to-many relationships between roles.

This module provides a base class for defining relationships between roles,
storing edges (connections) between agents with edge-specific data. Uses
COO (Coordinate List) sparse format for efficient storage and querying.

Design Notes
------------
- Relationships store edges between source and target roles
- Edge data stored in parallel NumPy arrays (COO format)
- Supports many-to-many, one-to-many, and many-to-one cardinality
- Auto-registration via __init_subclass__ hook
- Query methods use vectorized NumPy operations

Examples
--------
Define a loan relationship using @relationship decorator:

>>> from bamengine import relationship, get_role, Float
>>>
>>> @relationship(source=get_role("Borrower"), target=get_role("Lender"))
... class LoanBook:
...     principal: Float
...     rate: Float
...     debt: Float

Access relationship in simulation:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
>>> loans = sim.get_relationship("LoanBook")
>>> # Query loans from specific borrower
>>> borrower_id = 5
>>> mask = loans.source_ids[: loans.size] == borrower_id
>>> borrower_loans = loans.principal[: loans.size][mask]

Traditional syntax:

>>> from dataclasses import dataclass
>>> from bamengine.core import Relationship
>>> from bamengine.typing import Float, Int
>>>
>>> @dataclass(slots=True)
... class Employment(Relationship):
...     source_role = get_role("Borrower")
...     target_role = get_role("Lender")
...     wage: Float
...     contract_duration: Int

See Also
--------
:class:`~bamengine.core.Role` : Base class for component state
:func:`bamengine.relationship` : Simplified @relationship decorator
:class:`~bamengine.relationships.LoanBook` : Concrete relationship between Borrower and Lender
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

import numpy as np

from bamengine.typing import Float1D, Idx1D

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.core.role import Role

T = TypeVar("T")


@dataclass(slots=True)
class Relationship(
    ABC
):  # Some abstract methods/edge cases not covered - tested via LoanBook
    """
    Base class for defining relationships between roles.

    Relationships store edges (connections) between agents in different roles,
    along with edge-specific data. Internally uses COO (Coordinate List) sparse
    format for efficient storage and querying.

    Example
    -------
    Define a loan relationship between borrowers and lenders::

        @relationship(source=Borrower, target=Lender, cardinality="many-to-many")
        class LoanBook:
            principal: Float1D
            rate: Float1D
            interest: Float1D
            debt: Float1D

    Parameters
    ----------
    source_ids : Idx1D
        Array of source agent IDs
    target_ids : Idx1D
        Array of target agent IDs
    size : int
        Current number of active edges
    capacity : int
        Maximum number of edges that can be stored

    Notes
    -----
    - Edges are stored in COO format with parallel arrays
    - Empty slots are filled with sentinels (-1 for indices)
    - Subclasses define edge-specific data as additional fields
    - Query methods use NumPy operations for vectorized performance
    - The __init_subclass__ hook automatically registers relationships
    """

    # Metadata (set by __init_subclass__)
    name: ClassVar[str | None] = None
    source_role: ClassVar[type[Role] | None] = None
    target_role: ClassVar[type[Role] | None] = None
    # noinspection PyClassVar
    cardinality: ClassVar[Literal["many-to-many", "one-to-many", "many-to-one"]] = (
        "many-to-many"
    )

    # COO format arrays (always present)
    source_ids: Idx1D  # Source entity IDs
    target_ids: Idx1D  # Target entity IDs
    size: int  # Current number of active edges
    capacity: int  # Maximum number of edges

    def __init_subclass__(
        cls,
        name: str | None = None,
        source: type[Role] | None = None,
        target: type[Role] | None = None,
        cardinality: Literal[
            "many-to-many", "one-to-many", "many-to-one"
        ] = "many-to-many",
        **kwargs: Any,
    ) -> None:
        """
        Auto-register Relationship subclasses in the global registry.

        This hook is called automatically when a class inherits from Relationship.

        Parameters
        ----------
        name : str, optional
            Custom name for the relationship. If not provided, uses the class name.
        source : type[Role], optional
            Source role class for the relationship
        target : type[Role], optional
            Target role class for the relationship
        cardinality : {"many-to-many", "one-to-many", "many-to-one"},
            optional, default "many-to-many",
            Cardinality constraint for the relationship
        **kwargs
            Additional keyword arguments passed to parent __init_subclass__.
        """
        super(Relationship, cls).__init_subclass__(**kwargs)

        # Use custom name if provided, otherwise preserve existing name or use cls name
        # This handles the case where @dataclass(slots=True) creates a new class
        # and triggers __init_subclass__ a second time without the custom name
        if name is not None:
            cls.name = name
        elif cls.name is None:
            cls.name = cls.__name__

        # Set relationship metadata
        if source is not None:
            cls.source_role = source
        if target is not None:
            cls.target_role = target
        if cardinality != "many-to-many" or cls.cardinality == "many-to-many":
            cls.cardinality = cardinality

        # Auto-register in global registry
        from bamengine.core.registry import _RELATIONSHIP_REGISTRY

        _RELATIONSHIP_REGISTRY[cls.name] = cls

    def __repr__(self) -> str:
        """Provide informative repr showing relationship metadata and state."""
        fields = getattr(self, "__dataclass_fields__", {})
        # Count only the edge data fields (exclude the base fields)
        base_fields = {"source_ids", "target_ids", "size", "capacity"}
        edge_fields = [f for f in fields if f not in base_fields]

        relationship_name = self.name or self.__class__.__name__
        source_name = self.source_role.__name__ if self.source_role else "?"
        target_name = self.target_role.__name__ if self.target_role else "?"

        return (
            f"{relationship_name}("
            f"{source_name}->{target_name}, "
            f"cardinality={self.cardinality}, "
            f"edges={self.size}/{self.capacity}, "
            f"fields={len(edge_fields)}"
            ")"
        )

    def query_sources(self, source_id: int) -> Idx1D:
        """
        Get indices of all edges originating from a source.

        Parameters
        ----------
        source_id : int
            Source agent ID to query

        Returns
        -------
        Idx1D
            Array of edge indices where source_ids == source_id
        """
        return np.where(self.source_ids[: self.size] == source_id)[0]

    def query_targets(self, target_id: int) -> Idx1D:
        """
        Get indices of all edges pointing to a target.

        Parameters
        ----------
        target_id : int
            Target agent ID to query

        Returns
        -------
        Idx1D
            Array of edge indices where target_ids == target_id
        """
        return np.where(self.target_ids[: self.size] == target_id)[0]

    def aggregate_by_source(
        self,
        component: np.ndarray,
        *,
        func: Literal["sum", "mean", "count"] = "sum",
        n_sources: int | None = None,
        out: Float1D | None = None,
    ) -> Float1D:
        """
        Aggregate component values grouped by source.

        Parameters
        ----------
        component : np.ndarray
            Array of component values to aggregate (length = size)
        func : {"sum", "mean", "count"}, default "sum"
            Aggregation function
        n_sources : int, optional
            Number of source agents (for output array size).
            If None, inferred from max source_id + 1.
        out : Float1D, optional
            Pre-allocated output array

        Returns
        -------
        Float1D
            Aggregated values per source (length = n_sources)
        """
        if n_sources is None:
            active_sources = self.source_ids[: self.size]
            n_sources = int(active_sources.max()) + 1 if active_sources.size > 0 else 0

        if out is None:
            out = np.zeros(n_sources, dtype=np.float64)
        else:
            out[:] = 0.0

        if self.size == 0:
            return out

        active_sources = self.source_ids[: self.size]
        active_component = component[: self.size]

        if func == "sum":
            np.add.at(out, active_sources, active_component)
        elif func == "mean":
            # Sum values
            np.add.at(out, active_sources, active_component)
            # Count edges per source
            counts = np.bincount(active_sources, minlength=n_sources)
            # Divide by counts (avoid division by zero)
            mask = counts > 0
            out[mask] /= counts[mask]
        elif func == "count":
            counts = np.bincount(active_sources, minlength=n_sources)
            out[:] = counts
        else:
            raise ValueError(f"Unknown aggregation function: {func}")

        return out

    def aggregate_by_target(
        self,
        component: np.ndarray,
        *,
        func: Literal["sum", "mean", "count"] = "sum",
        n_targets: int | None = None,
        out: Float1D | None = None,
    ) -> Float1D:
        """
        Aggregate component values grouped by target.

        Parameters
        ----------
        component : np.ndarray
            Array of component values to aggregate (length = size)
        func : {"sum", "mean", "count"}, default "sum"
            Aggregation function
        n_targets : int, optional
            Number of target agents (for output array size).
            If None, inferred from max target_id + 1.
        out : Float1D, optional
            Pre-allocated output array

        Returns
        -------
        Float1D
            Aggregated values per target (length = n_targets)
        """
        if n_targets is None:
            active_targets = self.target_ids[: self.size]
            n_targets = int(active_targets.max()) + 1 if active_targets.size > 0 else 0

        if out is None:
            out = np.zeros(n_targets, dtype=np.float64)
        else:
            out[:] = 0.0

        if self.size == 0:
            return out

        active_targets = self.target_ids[: self.size]
        active_component = component[: self.size]

        if func == "sum":
            np.add.at(out, active_targets, active_component)
        elif func == "mean":
            # Sum values
            np.add.at(out, active_targets, active_component)
            # Count edges per target
            counts = np.bincount(active_targets, minlength=n_targets)
            # Divide by counts (avoid division by zero)
            mask = counts > 0
            out[mask] /= counts[mask]
        elif func == "count":
            counts = np.bincount(active_targets, minlength=n_targets)
            out[:] = counts
        else:
            raise ValueError(f"Unknown aggregation function: {func}")

        return out

    def drop_rows(self, mask: np.ndarray) -> int:
        """
        Remove edges matching a boolean mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array (length = size) indicating which edges to remove

        Returns
        -------
        int
            Number of edges removed
        """
        if self.size == 0:
            return 0

        mask_active = mask[: self.size]
        n_drop = int(np.sum(mask_active))

        if n_drop == 0:
            return 0

        # Invert mask to get edges to keep
        keep_mask = ~mask_active
        n_keep = self.size - n_drop

        # Compact arrays by keeping only non-dropped edges
        self.source_ids[:n_keep] = self.source_ids[: self.size][keep_mask]
        self.target_ids[:n_keep] = self.target_ids[: self.size][keep_mask]

        # Update any edge-specific component arrays (must be handled by subclass)
        # Subclasses should override this method to compact their own arrays

        # Update size
        self.size = n_keep

        return n_drop

    def purge_sources(self, source_ids: Idx1D) -> int:
        """
        Remove all edges originating from given source IDs.

        Parameters
        ----------
        source_ids : Idx1D
            Array of source IDs to purge

        Returns
        -------
        int
            Number of edges removed
        """
        if self.size == 0 or source_ids.size == 0:
            return 0

        # Create mask for edges to drop
        drop_mask = np.isin(self.source_ids[: self.size], source_ids)
        return self.drop_rows(drop_mask)

    def purge_targets(self, target_ids: Idx1D) -> int:
        """
        Remove all edges pointing to given target IDs.

        Parameters
        ----------
        target_ids : Idx1D
            Array of target IDs to purge

        Returns
        -------
        int
            Number of edges removed
        """
        if self.size == 0 or target_ids.size == 0:
            return 0

        # Create mask for edges to drop
        drop_mask = np.isin(self.target_ids[: self.size], target_ids)
        return self.drop_rows(drop_mask)

    def append_edges(
        self,
        source_ids: Idx1D,
        target_ids: Idx1D,
        **component_arrays: Any,
    ) -> None:
        """
        Append new edges with given source/target IDs and component values.

        Parameters
        ----------
        source_ids : Idx1D
            Array of source IDs for new edges
        target_ids : Idx1D
            Array of target IDs for new edges
        **component_arrays
            Component arrays (must match subclass fields)

        Raises
        ------
        ValueError
            If arrays have mismatched lengths or exceed capacity
        """
        n_new = source_ids.size

        if n_new == 0:
            return

        if source_ids.size != target_ids.size:
            raise ValueError("source_ids and target_ids must have same length")

        if self.size + n_new > self.capacity:
            raise ValueError(
                f"Cannot append {n_new} edges: would exceed capacity "
                f"({self.size} + {n_new} > {self.capacity})"
            )

        # Append IDs
        new_start = self.size
        new_end = self.size + n_new
        self.source_ids[new_start:new_end] = source_ids
        self.target_ids[new_start:new_end] = target_ids

        # Subclasses must override to append their component arrays

        # Update size
        self.size = new_end
