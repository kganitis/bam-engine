"""
Relationship system for managing many-to-many relationships between roles.

This module provides a base class and decorator for defining relationships
between roles.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, TypeVar

import numpy as np

from bamengine.typing import Float1D, Idx1D

if TYPE_CHECKING:
    from bamengine.core.role import Role

_EPS = 1.0e-9

T = TypeVar("T")

# Global relationship registry
_RELATIONSHIP_REGISTRY: dict[str, type[Relationship]] = {}


@dataclass(slots=True)
class Relationship(ABC):
    """
    Base class for many-to-many relationships between roles.

    Relationships store edges (connections) between agents in different roles,
    along with edge-specific data. Internally uses COO (Coordinate List) sparse
    format for efficient storage and querying.

    Example
    -------
    Define a loan relationship between borrowers and lenders::

        @relationship(source=Borrower, target=Lender, cardinality="many-to-many")
        @dataclass(slots=True)
        class LoanBook(Relationship):
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
    """

    # Metadata (set by subclass or decorator)
    _source_role: ClassVar[type[Role] | None] = None
    _target_role: ClassVar[type[Role] | None] = None
    _cardinality: ClassVar[Literal["many-to-many", "one-to-many", "many-to-one"]] = (
        "many-to-many"
    )
    _relationship_name: ClassVar[str] = ""

    # COO format arrays (always present)
    source_ids: Idx1D  # Source entity IDs
    target_ids: Idx1D  # Target entity IDs
    size: int  # Current number of active edges
    capacity: int  # Maximum number of edges

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


def relationship(
    cls: type[T] | None = None,
    *,
    source: type[Role] | None = None,
    target: type[Role] | None = None,
    cardinality: Literal["many-to-many", "one-to-many", "many-to-one"] = "many-to-many",
    name: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator to define a Relationship with automatic inheritance and registration.

    This decorator dramatically simplifies Relationship definition by:
    1. Making the class inherit from Relationship (if not already)
    2. Applying @dataclass(slots=True)
    3. Setting source/target roles as class variables
    4. Setting cardinality
    5. Registering the relationship in the global registry

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    source : type[Role], optional
        Source role type (e.g., Borrower)
    target : type[Role], optional
        Target role type (e.g., Lender)
    cardinality : {"many-to-many", "one-to-many", "many-to-one"}, default "many-to-many"
        Relationship cardinality
    name : str, optional
        Custom name for the relationship. If None, uses the class name.
    **dataclass_kwargs : Any
        Additional keyword arguments to pass to @dataclass.
        By default, slots=True is set.

    Returns
    -------
    type | Callable
        The decorated class or a decorator function

    Examples
    --------
    Simplest usage (no inheritance needed)::

        @relationship(source=Borrower, target=Lender)
        class LoanBook:
            principal: Float1D
            rate: Float1D
            interest: Float1D
            debt: Float1D

    With custom name and cardinality::

        @relationship(
            source=Worker,
            target=Employer,
            cardinality="many-to-many",
            name="MultiJobEmployment"
        )
        class Employment:
            wage: Float1D
            contract_duration: Int1D

    Traditional usage (still works)::

        @relationship(source=Borrower, target=Lender)
        class LoanBook(Relationship):
            principal: Float1D
            rate: Float1D

    Notes
    -----
    - No need to inherit from Relationship explicitly (decorator adds it)
    - No need for @dataclass(slots=True) (decorator applies it)
    - Registration happens automatically
    - slots=True is set by default for memory efficiency
    - Source and target roles can be None (useful to avoid circular imports)
    """
    # Set default slots=True unless explicitly overridden
    dataclass_kwargs.setdefault("slots", True)

    def decorator(cls: type[T]) -> type[T]:
        # Check if cls already inherits from Relationship
        if not issubclass(cls, Relationship):
            # Dynamically create a new class that inherits from Relationship
            # Copy annotations and methods from the original class
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__annotations__": getattr(cls, "__annotations__", {}),
            }
            # Copy methods and class attributes
            for attr_name in dir(cls):
                if not attr_name.startswith("__"):
                    namespace[attr_name] = getattr(cls, attr_name)

            cls = type(cls.__name__, (Relationship,), namespace)

        # Set metadata as class variables
        cls._source_role = source  # type: ignore[attr-defined]
        cls._target_role = target  # type: ignore[attr-defined]
        cls._cardinality = cardinality  # type: ignore[attr-defined]

        # Determine relationship name
        relationship_name = name if name is not None else cls.__name__
        cls._relationship_name = relationship_name  # type: ignore[attr-defined]

        # Apply dataclass decorator
        # Check if class already has __slots__ (from Relationship base)
        # If it does, don't apply slots=True again to avoid conflicts
        if hasattr(cls, "__slots__"):
            # Remove slots from dataclass_kwargs to avoid conflict
            dc_kwargs = {k: v for k, v in dataclass_kwargs.items() if k != "slots"}
            cls = dataclass(**dc_kwargs)(cls)
        else:
            cls = dataclass(**dataclass_kwargs)(cls)

        # Register in global registry
        if relationship_name in _RELATIONSHIP_REGISTRY:
            raise ValueError(
                f"Relationship '{relationship_name}' is already registered. "
                f"Use a different name or unregister the existing one first."
            )

        _RELATIONSHIP_REGISTRY[relationship_name] = cls  # type: ignore[assignment]

        return cls

    # Support both @relationship and @relationship() syntax
    if cls is None:
        # Called with arguments: @relationship(source=..., target=...)
        return decorator
    else:
        # Called without arguments: @relationship (not typical for relationships)
        return decorator(cls)


def get_relationship(name: str) -> type[Relationship]:
    """
    Retrieve a registered relationship class by name.

    Parameters
    ----------
    name : str
        Name of the relationship

    Returns
    -------
    type[Relationship]
        Relationship class

    Raises
    ------
    KeyError
        If relationship is not registered
    """
    try:
        return _RELATIONSHIP_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Relationship '{name}' not found in registry. "
            f"Available relationships: {list(_RELATIONSHIP_REGISTRY.keys())}"
        )


def list_relationships() -> list[str]:
    """
    List all registered relationships.

    Returns
    -------
    list[str]
        List of registered relationship names
    """
    return list(_RELATIONSHIP_REGISTRY.keys())


def unregister_relationship(name: str) -> None:
    """
    Remove a relationship from the registry.

    Parameters
    ----------
    name : str
        Name of the relationship to unregister

    Raises
    ------
    KeyError
        If relationship is not registered
    """
    try:
        del _RELATIONSHIP_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Relationship '{name}' not found in registry. "
            f"Available relationships: {list(_RELATIONSHIP_REGISTRY.keys())}"
        )
