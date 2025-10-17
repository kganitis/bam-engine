"""Event (System) base class definition."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters (except first)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Event(ABC):
    """
    Base class for all events (systems) in the BAM-ECS architecture.

    An Event encapsulates a piece of simulation logic that operates on roles
    (components). Events are executed by the Pipeline in a dependency-aware
    order each simulation period.

    Each Event class should:
    - Define a unique `name` class attribute
    - Implement the `execute()` method with simulation logic
    - Optionally specify `dependencies` (events that must run first)
    - Optionally override `log_level` for per-event logging control

    Examples
    --------
    >>> class FirmsDecidePrice(Event):
    ...     name = "firms_decide_price"
    ...
    ...     def execute(self, sim: Simulation) -> None:
    ...         # Access roles
    ...         prod = sim.get_role(Producer)
    ...
    ...         # Call existing system function
    ...         from bamengine.systems.planning import firms_adjust_price
    ...         firms_adjust_price(
    ...             prod,
    ...             p_avg=sim.economy.avg_mkt_price,
    ...             h_eta=sim.config.h_eta,
    ...             rng=sim.rng,
    ...         )
    ...
    ...     @property
    ...     def dependencies(self) -> tuple[str, ...]:
    ...         return ("firms_calc_breakeven_price",)

    Notes
    -----
    - Events should be stateless (no instance variables beyond config)
    - All state mutations happen in roles, not in the event itself
    - Events are registered using the `@event()` decorator
    """

    # Unique identifier for this event (auto-generated from class name if not set)
    name: ClassVar[str] = ""  # Default empty, will be set by __init_subclass__

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-generate snake_case name from class name if not explicitly set."""
        super().__init_subclass__(**kwargs)
        # If name is not defined on the subclass or is empty, generate from class name
        if not hasattr(cls, "name") or not cls.name:
            cls.name = _camel_to_snake(cls.__name__)

    @abstractmethod
    def execute(self, sim: Simulation) -> None:
        """
        Execute this event's logic.

        This method receives the full Simulation instance and should:
        1. Retrieve needed roles via `sim.get_role()`
        2. Call existing system functions (during migration)
        3. Mutate role state in-place

        Parameters
        ----------
        sim : Simulation
            The simulation instance containing all state and configuration.

        Returns
        -------
        None
            All mutations are in-place on role state.
        """
        pass

    @property
    def dependencies(self) -> tuple[str, ...]:
        # noinspection PyTypeChecker, PyShadowingNames
        """
        Names of events that must run before this one.

        These are intrinsic dependencies (required for correctness).
        Additional ordering can be specified via YAML config.

        Returns
        -------
        tuple[str, ...]
            Event names this event depends on.

        Examples
        --------
        >>> @property
        >>> def dependencies(self) -> tuple[str, ...]:
        ...     return "firms_calc_breakeven_price", "update_avg_mkt_price"
        """
        return ()

    @property
    def log_level(self) -> int:
        """
        Default logging level for this event.

        Can be overridden via YAML configuration on a per-event basis.

        Returns
        -------
        int
            Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        return logging.INFO

    def __repr__(self) -> str:
        """Provide informative repr."""
        deps = self.dependencies
        dep_str = f", deps={len(deps)}" if deps else ""
        return f"{self.__class__.__name__}(name={self.name!r}{dep_str})"
