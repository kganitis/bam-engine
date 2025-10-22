"""Event (System) base class definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Any

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters (except first)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass(slots=True)
class Event(ABC):
    """
    Base class for all events (systems) in the BAM-ECS architecture.

    An Event encapsulates economic logic that operates on roles and mutates
    simulation state in-place. Events are executed by the Pipeline in a
    specific order determined by dependencies.

    Design Guidelines
    -----------------
    - Inherit from Event and implement `execute()` method
    - Use `name` class variable for unique identification
    - Declare dependencies via `dependencies` property (optional)
    - Events receive full Simulation instance for maximum flexibility

    Notes
    -----
    Events are registered automatically via __init_subclass__ hook.
    """

    # Class variable for event name (set by subclass)
    name: ClassVar[str | None] = None

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        """
        Auto-register Event subclasses in the global registry.

        This hook is called automatically when a class inherits from Event.

        Parameters
        ----------
        name : str, optional
            Custom name for the event. If not provided, uses the class name.
        **kwargs
            Additional keyword arguments passed to parent __init_subclass__.
        """
        super(Event, cls).__init_subclass__(**kwargs)

        # Use custom name if provided, otherwise preserve existing name or use cls name
        # This handles the case where @dataclass(slots=True) creates a new class
        # and triggers __init_subclass__ a second time without the custom name
        if name is not None:
            cls.name = name
        elif cls.name is None:
            cls.name = _camel_to_snake(cls.__name__)

        # Auto-register in global registry
        from bamengine.core.registry import _EVENT_REGISTRY

        _EVENT_REGISTRY[cls.name] = cls

    @abstractmethod
    def execute(self, sim: Simulation) -> None:
        """
        Execute the event's logic.

        Mutates simulation state in-place.

        Parameters
        ----------
        sim : Simulation
            The simulation instance containing all state and configuration.

        Returns
        -------
        None
            All mutations are in-place.
        """
        pass

    @property
    def dependencies(self) -> tuple[str, ...]:
        """
        Events that must run before this one.

        Override this property in subclasses to declare dependencies.

        Returns
        -------
        tuple[str, ...]
            Tuple of event names that must execute before this event.
        """
        return ()

    def __repr__(self) -> str:
        """Provide informative repr."""
        deps = self.dependencies
        dep_str = f", deps={len(deps)}" if deps else ""
        return f"{self.__class__.__name__}(name={self.name!r}{dep_str})"
