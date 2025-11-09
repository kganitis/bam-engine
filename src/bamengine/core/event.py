"""Event (System) base class definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
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
    simulation state in-place. Events are executed by the Pipeline in the
    exact order specified.

    Design Guidelines
    -----------------
    - Inherit from Event and implement `execute()` method
    - Use `name` class variable for unique identification
    - Events receive full Simulation instance for maximum flexibility

    Notes
    -----
    Events are registered automatically via __init_subclass__ hook.
    The order of event execution is critical and must be explicitly
    defined in the pipeline configuration.
    """

    # Class variable for event name (set by subclass)
    name: ClassVar[str] = ""

    def __init_subclass__(cls, name: str = "", **kwargs: Any) -> None:
        """
        Auto-register Event subclasses in the global registry.

        This hook is called automatically when a class inherits from Event.

        Parameters
        ----------
        name : str, optional
            Custom name for the event.
            If not provided, uses the class name converted to snake_case.
        **kwargs
            Additional keyword arguments passed to parent __init_subclass__.
        """
        super(Event, cls).__init_subclass__(**kwargs)

        # Use custom name if provided, otherwise preserve existing name
        # or use cls name converted to snake_case
        # This handles the case where @dataclass(slots=True) creates a new class
        # and triggers __init_subclass__ a second time without the custom name
        if name != "":
            cls.name = name
        elif cls.name == "":
            cls.name = _camel_to_snake(cls.__name__)

        # Auto-register in global registry
        from bamengine.core.registry import _EVENT_REGISTRY

        _EVENT_REGISTRY[cls.name] = cls

    def get_logger(self) -> logging.Logger:
        """
        Get logger for this event with per-event log level applied.

        Returns
        -------
        logging.Logger
            Logger instance with event-specific configuration.

        Notes
        -----
        Logger name format: 'bamengine.events.{event_name}'
        Per-event log levels can be configured via defaults.yml or kwargs:

        logging:
          events:
            firms_adjust_price: DEBUG
            workers_send_one_round: WARNING
        """
        logger_name = f"bamengine.events.{self.name}"
        return logging.getLogger(logger_name)

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

        Notes
        -----
        Use self.get_logger() to get a logger with per-event log level:

        logger = self.get_logger()
        logger.info("Starting execution")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Expensive debug info: %s", compute_stats())
        """
        pass

    def __repr__(self) -> str:
        """Provide informative repr."""
        return f"{self.__class__.__name__}(name={self.name!r})"
