"""
Event (System) base class definition.

This module defines the Event base class, which encapsulates economic logic
that operates on roles and mutates simulation state. Events are the "systems"
in the BAM-ECS architecture.

Design Notes
------------
- All Events auto-register via __init_subclass__ hook
- Event names are automatically converted from CamelCase to snake_case
- Events execute in explicit order defined by Pipeline (no automatic sorting)
- Events receive full Simulation instance for maximum flexibility

Auto-Registration
-----------------
When a class inherits from Event, __init_subclass__ automatically:

1. Converts class name to snake_case for event name
2. Registers the event class in the global _EVENT_REGISTRY
3. Makes the event retrievable via get_event(name)

This eliminates manual registration boilerplate and ensures all
events are discoverable at runtime.

See Also
--------
:class:`~bamengine.core.Role` : Base class for roles (components) in BAM-ECS
:class:`~bamengine.core.Pipeline` : Manages event execution order
:mod:`bamengine.core.registry` : Global registries for events and roles
:func:`bamengine.event` : Simplified decorator for defining events
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from bamengine import logging

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


def _camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.

    Parameters
    ----------
    name : str
        CamelCase string to convert.

    Returns
    -------
    str
        snake_case version of the input string.

    Examples
    --------
    >>> _camel_to_snake("FirmsDecideDesiredProduction")
    'firms_decide_desired_production'
    >>> _camel_to_snake("WorkersSendOneRound")
    'workers_send_one_round'
    """
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

    Class Attributes
    ----------------
    name : str
        Event name in snake_case, automatically derived from class name.

    Design Guidelines
    -----------------
    - Inherit from Event and implement `execute()` method
    - Use `name` class variable for unique identification
    - Events receive full Simulation instance for maximum flexibility
    - Use self.get_logger() for event-specific logging

    Examples
    --------
    Define an event using traditional syntax:

    >>> from dataclasses import dataclass
    >>> from bamengine.core import Event
    >>> from bamengine.simulation import Simulation
    >>>
    >>> @dataclass(slots=True)
    ... class MyCustomEvent(Event):
    ...     def execute(self, sim: Simulation) -> None:
    ...         logger = self.get_logger()
    ...         logger.info("Executing custom logic")
    ...         # Mutate simulation state here
    >>> # Event is auto-registered as 'my_custom_event'

    Define an event using the @event decorator (simplified):

    >>> from bamengine import event
    >>>
    >>> @event
    ... class AnotherEvent:
    ...     def execute(self, sim):
    ...         pass  # Implementation here
    >>> # Event is auto-registered as 'another_event'

    Execute an event:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_adjust_price")
    >>> event.execute(sim)

    Per-event logging configuration:

    >>> logger = event.get_logger()
    >>> logger.info("Starting execution")
    >>> if logger.isEnabledFor(logging.DEBUG):
    ...     logger.debug("Expensive debug info: %s", compute_stats())

    Notes
    -----
    Events are registered automatically via __init_subclass__ hook.
    The order of event execution is critical and must be explicitly
    defined in the pipeline configuration (no automatic dependency sorting).

    See Also
    --------
    :class:`~bamengine.core.Role` : Base class for roles (components) in BAM-ECS
    :class:`~bamengine.core.Pipeline` : Manages event execution order
    :mod:`bamengine.core.registry` : Global registries for events and roles
    :mod:`bamengine.logging` : Logging utilities for BAM-ECS
    :func:`bamengine.event` : Simplified @event decorator
    """

    # Class variable for event name (set by subclass)
    name: ClassVar[str] = ""

    def __init_subclass__(cls, name: str = "", **kwargs: Any) -> None:
        """
        Auto-register Event subclasses in the global registry.

        This hook is called automatically when a class inherits from Event.
        It handles event registration and automatic name conversion.

        Parameters
        ----------
        name : str, optional
            Custom name for the event.
            If not provided, uses the class name converted to snake_case.
        **kwargs : Any
            Additional keyword arguments passed to parent __init_subclass__.

        Notes
        -----
        This method is called twice when using @dataclass(slots=True):
        once during class definition and once when dataclass creates the
        final class. The name preservation logic handles this correctly.

        Examples
        --------
        Normal usage (automatic snake_case conversion):

        >>> @dataclass(slots=True)
        ... class FirmsAdjustPrice(Event):
        ...     def execute(self, sim):
        ...         pass
        >>> # Registered as 'firms_adjust_price'

        Custom name:

        >>> @dataclass(slots=True)
        ... class MyEvent(Event, name="custom_event_name"):
        ...     def execute(self, sim):
        ...         pass
        >>> # Registered as 'custom_event_name'
        """
        super(Event, cls).__init_subclass__(**kwargs)

        # Use custom name if provided, otherwise preserve existing name
        # or use cls name converted to snake_case
        if name != "":
            cls.name = name
        elif cls.name == "":
            cls.name = _camel_to_snake(cls.__name__)

        # Auto-register in global registry
        from bamengine.core.registry import _EVENT_REGISTRY

        _EVENT_REGISTRY[cls.name] = cls

    def get_logger(self) -> logging.BamLogger:
        """
        Get logger for this event with per-event log level applied.

        Returns
        -------
        logging.BamLogger
            Logger instance with event-specific configuration.

        Examples
        --------
        Use logger in event execute method:

        >>> class MyEvent(Event):
        ...     def execute(self, sim):
        ...         logger = self.get_logger()
        ...         logger.info("Starting execution")
        ...         # Expensive computation
        ...         if logger.isEnabledFor(logging.DEBUG):
        ...             logger.debug("Details: %s", expensive_stats())

        Configure per-event log levels:

        >>> # In config YAML or kwargs
        >>> config = {
        ...     "logging": {
        ...         "events": {
        ...             "firms_adjust_price": "DEBUG",
        ...             "workers_send_one_round": "WARNING",
        ...         }
        ...     }
        ... }

        Notes
        -----
        Logger name format: 'bamengine.events.{event_name}'
        Per-event log levels can be configured via config/defaults.yml or kwargs.
        Use isEnabledFor() to avoid expensive computations when logging
        is disabled.
        """
        logger_name = f"bamengine.events.{self.name}"
        return logging.getLogger(logger_name)

    @abstractmethod
    def execute(self, sim: Simulation) -> None:
        """
        Execute the event's logic.

        Mutates simulation state in-place. This method must be implemented
        by all Event subclasses.

        Parameters
        ----------
        sim : Simulation
            The simulation instance containing all state and configuration.

        Returns
        -------
        None
            All mutations are in-place.

        Examples
        --------
        Implement execute in a custom event:

        >>> from bamengine import event, ops
        >>>
        >>> @event
        ... class CustomPricingEvent:
        ...     def execute(self, sim):
        ...         prod = sim.get_role("Producer")
        ...         # Apply 10% markup to all prices
        ...         new_prices = ops.multiply(prod.price, 1.1)
        ...         ops.assign(prod.price, new_prices)

        Access configuration and RNG:

        >>> @event
        ... class StochasticEvent:
        ...     def execute(self, sim):
        ...         shock = sim.config.h_rho
        ...         random_values = sim.rng.uniform(0, shock, size=sim.config.n_firms)
        ...         # Use random_values in calculations

        Notes
        -----
        The execute method receives full Simulation access, including:
        - All roles: sim.get_role("RoleName") or sim.prod, sim.wrk, etc.
        - Configuration: sim.config
        - RNG: sim.rng
        - Economy state: sim.ec
        """
        pass  # pragma: no cover - abstract method, overridden by subclasses

    def __repr__(self) -> str:
        """
        Provide informative repr showing event name.

        Returns
        -------
        str
            String representation in format "EventClassName(name='event_name')".

        Examples
        --------
        >>> from bamengine.events import FirmsAdjustPrice
        >>> event = FirmsAdjustPrice()
        >>> repr(event)
        "FirmsAdjustPrice(name='firms_adjust_price')"
        """
        return f"{self.__class__.__name__}(name={self.name!r})"
