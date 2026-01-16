"""
Event Pipeline with explicit execution order.

The Pipeline manages event execution in a fixed, user-defined order.
Unlike traditional ECS systems with dependency-based topological sorting,
BAM Engine uses explicit ordering to ensure deterministic, reproducible
simulation behavior.

Design Philosophy
-----------------
Explicit ordering over dependency resolution:

- Users specify exact execution sequence via YAML
- No automatic reordering or optimization
- Guarantees pipeline matches legacy implementation
- Makes execution trace obvious for debugging

Pipeline YAML Format
--------------------
events:
  - event_name                    # Single execution
  - event_name x N                # Repeat N times
  - event1 <-> event2 x N         # Interleave N times

Parameter substitution:
  - event_{i}                     # Substitute {i} with parameter value

Examples
--------
Load and execute the default pipeline:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, seed=42)
>>> pipeline = create_default_pipeline(max_M=5, max_H=3, max_Z=2)
>>> pipeline.execute(sim)

Load a custom pipeline from YAML:

>>> from bamengine.core import Pipeline
>>> pipeline = Pipeline.from_yaml("my_custom_pipeline.yml", max_M=5, max_H=3, max_Z=2)

Modify an existing pipeline:

>>> # Insert custom event after standard event
>>> pipeline.insert_after("firms_adjust_price", "my_custom_pricing")
>>>
>>> # Remove an event
>>> pipeline.remove("workers_send_one_round_0")
>>>
>>> # Replace an event with custom implementation
>>> pipeline.replace("firms_decide_desired_production", "my_production_rule")

See Also
--------
:class:`~bamengine.core.Event` : Base class for all events
:func:`create_default_pipeline` : Factory for canonical BAM pipeline
:meth:`bamengine.config.ConfigValidator.validate_pipeline_yaml` : Pipeline validation
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from bamengine.core.event import Event
from bamengine.core.registry import get_event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class RepeatedEvent:
    """
    Wrapper for events that execute multiple times per period.

    Used for market rounds where agents interact over multiple rounds
    (e.g., job applications, loan applications, shopping rounds).

    Attributes
    ----------
    event : Event
        The event to repeat.
    n_repeats : int
        Number of times to execute the event.

    Examples
    --------
    Wrap an event to repeat it multiple times:

    >>> from bamengine.core.registry import get_event
    >>> sim = Simulation.init(n_firms=100, seed=42)
    >>> event_cls = get_event("workers_send_one_round")
    >>> event = event_cls()
    >>> repeated = RepeatedEvent(event, n_repeats=5)
    >>> repeated.execute(sim)  # Executes 5 times

    See Also
    --------
    Pipeline.from_event_list : Build pipeline with repeated events
    """

    event: Event
    n_repeats: int

    def execute(self, sim: Simulation) -> None:
        """
        Execute the event n_repeats times.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to operate on.

        Returns
        -------
        None
            All mutations are in-place.
        """
        for _ in range(self.n_repeats):
            self.event.execute(sim)

    @property
    def name(self) -> str:
        """
        Return the name of the underlying event.

        Returns
        -------
        str
            Event name in snake_case.
        """
        return self.event.name


@dataclass(slots=True)
class Pipeline:
    """
    Event execution pipeline with explicit ordering.

    The Pipeline manages event execution in a fixed, user-defined order.
    Events are executed sequentially in the exact order specified, with
    no automatic dependency resolution or reordering.

    Attributes
    ----------
    events : list[Event]
        Ordered list of event instances to execute.
    _event_map : dict[str, Event]
        Internal mapping from event names to instances for quick lookup.

    Examples
    --------
    Create pipeline from event list:

    >>> from bamengine.core import Pipeline
    >>> sim = Simulation.init(n_firms=100, seed=42)
    >>> pipeline = Pipeline.from_event_list(
    ...     [
    ...         "firms_decide_desired_production",
    ...         "firms_adjust_price",
    ...         "workers_send_one_round_0",
    ...     ]
    ... )
    >>> pipeline.execute(sim)

    Load pipeline from YAML:

    >>> pipeline = Pipeline.from_yaml("custom_pipeline.yml", max_M=5, max_H=3, max_Z=2)

    Modify pipeline after creation:

    >>> pipeline.insert_after("firms_adjust_price", "my_custom_event")
    >>> pipeline.remove("workers_send_one_round_0")
    >>> pipeline.replace("firms_decide_desired_production", "my_production_rule")

    Notes
    -----
    The order of events is critical for correct simulation behavior.
    Users are responsible for ensuring the order is logically correct.

    See Also
    --------
    Pipeline.from_event_list : Build pipeline from event name list
    Pipeline.from_yaml : Build pipeline from YAML configuration
    create_default_pipeline : Factory for canonical BAM pipeline
    """

    events: list[Event] = field(default_factory=list)
    _event_map: dict[str, Event] = field(default_factory=dict, init=False, repr=False)
    _after_event_callbacks: dict[str, list[Callable[[Simulation], None]]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Build internal event mapping and initialize callbacks."""
        self._event_map = {event.name: event for event in self.events}
        # Ensure _after_event_callbacks is a defaultdict (in case of copy/pickle)
        if not isinstance(self._after_event_callbacks, defaultdict):  # pragma: no cover
            self._after_event_callbacks = defaultdict(list, self._after_event_callbacks)

    @classmethod
    def from_event_list(
        cls,
        event_names: list[str],
        *,
        repeats: dict[str, int] | None = None,
        apply_hooks: bool = True,
    ) -> Pipeline:
        """
        Build pipeline from ordered list of event names.

        Events are executed in the exact order provided. Users are
        responsible for ensuring the order is logically correct.

        Parameters
        ----------
        event_names : list[str]
            Event names in desired execution order.
        repeats : dict[str, int], optional
            Events that should repeat multiple times.
            Format: {event_name: n_repeats}
        apply_hooks : bool, default True
            Whether to automatically apply registered event hooks.
            Hooks are registered via the ``@event(after=..., before=..., replace=...)``
            decorator. Set to False to skip hook application (useful for testing).

        Returns
        -------
        Pipeline
            Pipeline with events in the order specified, plus any hooked events.

        Raises
        ------
        ValueError
            If event name not found in registry.

        Notes
        -----
        The order of events is critical for correct simulation behavior.
        Use the default pipeline as a reference for the required ordering.

        When ``apply_hooks=True`` (default), any events registered with pipeline
        hooks via ``@event(after=..., before=..., replace=...)`` are automatically
        inserted into the pipeline at their specified positions.
        """
        repeats = repeats or {}

        # Instantiate events (wrap repeated ones)
        event_instances = []
        for name in event_names:
            event_cls = get_event(name)
            event = event_cls()

            # Wrap in RepeatedEvent if specified
            if name in repeats:
                event = cast(
                    Event, cast(object, RepeatedEvent(event, n_repeats=repeats[name]))
                )

            event_instances.append(event)

        pipeline = cls(events=event_instances)

        # Apply registered hooks
        if apply_hooks:
            pipeline._apply_hooks()

        return pipeline

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        *,
        apply_hooks: bool = True,
        **params: int,
    ) -> Pipeline:
        """
        Build pipeline from YAML configuration file.

        The YAML file should have an 'events' key with a list of event
        specifications. Supports special syntax:
        - 'event_name' - single event
        - 'event_name x N' - repeat event N times
        - 'event1 <-> event2 x N' - interleave two events N times

        Parameters can be substituted using {param_name} syntax.

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML configuration file.
        apply_hooks : bool, default True
            Whether to automatically apply registered event hooks.
            Hooks are registered via the ``@event(after=..., before=..., replace=...)``
            decorator. Set to False to skip hook application (useful for testing).
        **params : int
            Parameters to substitute in the YAML (e.g., max_M=5, max_H=3, max_Z=2).

        Returns
        -------
        Pipeline
            Pipeline with events parsed from YAML, plus any hooked events.

        Raises
        ------
        ValueError
            If YAML format is invalid or event not found in registry.

        Examples
        --------
        >>> pipeline = Pipeline.from_yaml("my_pipeline.yml", max_M=5, max_H=3, max_Z=2)
        """
        # Read YAML file
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        if "events" not in config:
            raise ValueError(f"YAML file must have 'events' key: {yaml_path}")

        event_specs = config["events"]
        event_names = []

        # Parse each event specification
        for spec in event_specs:
            # Substitute parameters
            for param_name, param_value in params.items():
                spec = spec.replace(f"{{{param_name}}}", str(param_value))

            # Parse the spec
            event_names.extend(cls._parse_event_spec(spec))

        return cls.from_event_list(event_names, apply_hooks=apply_hooks)

    @staticmethod
    def _parse_event_spec(spec: str) -> list[str]:
        """
        Parse event specification string into list of event names.

        Supports:
        - 'event_name' -> ['event_name']
        - 'event_name x 3' -> ['event_name', 'event_name', 'event_name']
        - 'event1 <-> event2 x 3' -> ['event1', 'event2',
                                      'event1', 'event2',
                                      'event1', 'event2']
        """
        spec = spec.strip()

        # Pattern 1: Interleaved events (event1 <-> event2 x N)
        interleaved_pattern = r"^(.+?)\s*<->\s*(.+?)\s+x\s+(\d+)$"
        match = re.match(interleaved_pattern, spec)
        if match:
            event1 = match.group(1).strip()
            event2 = match.group(2).strip()
            count = int(match.group(3))
            result = []
            for _ in range(count):
                result.append(event1)
                result.append(event2)
            return result

        # Pattern 2: Repeated event (event_name x N)
        repeated_pattern = r"^(.+?)\s+x\s+(\d+)$"
        match = re.match(repeated_pattern, spec)
        if match:
            event_name = match.group(1).strip()
            count = int(match.group(2))
            return [event_name] * count

        # Pattern 3: Single event (event_name)
        return [spec]

    def execute(self, sim: Simulation) -> None:
        """
        Execute all events in pipeline order, firing callbacks after each event.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to operate on.

        Returns
        -------
        None
            All mutations are in-place.

        Notes
        -----
        Callbacks registered via ``register_after_event()`` are fired after
        each event completes. This is used for data capture timing in
        SimulationResults.
        """
        for event in self.events:
            event.execute(sim)
            # Fire registered callbacks for this event (for data capture timing)
            for callback in self._after_event_callbacks.get(event.name, []):
                callback(sim)

    def register_after_event(
        self, event_name: str, callback: Callable[[Simulation], None]
    ) -> None:
        """
        Register a callback to run after a specific event.

        This is used by SimulationResults for configurable data capture timing.
        Callbacks are fired in the order they were registered.

        Parameters
        ----------
        event_name : str
            Name of the event after which to run the callback.
        callback : Callable[[Simulation], None]
            Function to call after the event executes.
            Receives the Simulation instance as its only argument.

        Notes
        -----
        This callback system is separate from the event hook system
        (``@event(after=..., before=..., replace=...)``). Event hooks are
        for inserting new events into the pipeline at initialization time.
        This callback system is for running arbitrary code (like data capture)
        after events during execution.

        Examples
        --------
        >>> def capture_production(sim):
        ...     print(f"Production: {sim.prod.production.sum()}")
        >>> pipeline.register_after_event("firms_run_production", capture_production)
        >>> pipeline.execute(sim)  # Prints production after firms_run_production
        """
        self._after_event_callbacks[event_name].append(callback)

    def clear_callbacks(self) -> None:
        """
        Clear all registered after-event callbacks.

        This should be called after a simulation run to ensure the pipeline
        can be reused without accumulating callbacks from previous runs.

        Examples
        --------
        >>> pipeline.register_after_event("firms_run_production", my_callback)
        >>> pipeline.execute(sim)
        >>> pipeline.clear_callbacks()  # Remove all callbacks for reuse
        """
        self._after_event_callbacks.clear()

    def insert_after(self, after: str, events: Event | str | list[Event | str]) -> None:
        """
        Insert event(s) after specified event.

        Parameters
        ----------
        after : str
            Event name to insert after.
        events : Event | str | list[Event | str]
            Event instance, event name, or list of events to insert.
            If a list is provided, events are inserted in order.

        Raises
        ------
        ValueError
            If 'after' event not found in pipeline.

        Examples
        --------
        Insert a single event:

        >>> pipeline.insert_after("firms_pay_dividends", "custom_event")

        Insert multiple events:

        >>> pipeline.insert_after(
        ...     "firms_pay_dividends",
        ...     [
        ...         "event_a",
        ...         "event_b",
        ...         "event_c",
        ...     ],
        ... )
        """
        if after not in self._event_map:
            raise ValueError(f"Event '{after}' not found in pipeline")

        # Convert single event to list
        event_list = events if isinstance(events, list) else [events]

        # Find insertion point
        idx = self.events.index(self._event_map[after])

        # Insert events in reverse order to maintain list order
        for event in reversed(event_list):
            # Instantiate if name provided
            if isinstance(event, str):
                event_cls = get_event(event)
                event = event_cls()

            self.events.insert(idx + 1, event)
            self._event_map[event.name] = event

    def insert_before(
        self, before: str, events: Event | str | list[Event | str]
    ) -> None:
        """
        Insert event(s) before specified event.

        Parameters
        ----------
        before : str
            Event name to insert before.
        events : Event | str | list[Event | str]
            Event instance, event name, or list of events to insert.
            If a list is provided, events are inserted in order.

        Raises
        ------
        ValueError
            If 'before' event not found in pipeline.

        Examples
        --------
        Insert a single event:

        >>> pipeline.insert_before("firms_adjust_price", "pre_pricing_check")

        Insert multiple events:

        >>> pipeline.insert_before(
        ...     "firms_adjust_price",
        ...     [
        ...         "event_a",
        ...         "event_b",
        ...         "event_c",
        ...     ],
        ... )
        """
        if before not in self._event_map:
            raise ValueError(f"Event '{before}' not found in pipeline")

        # Convert single event to list
        event_list = events if isinstance(events, list) else [events]

        # Find insertion point
        idx = self.events.index(self._event_map[before])

        # Insert events in reverse order to maintain list order
        for event in reversed(event_list):
            # Instantiate if name provided
            if isinstance(event, str):
                event_cls = get_event(event)
                event = event_cls()

            self.events.insert(idx, event)
            self._event_map[event.name] = event

    def remove(self, event_name: str) -> None:
        """
        Remove event from pipeline.

        Parameters
        ----------
        event_name : str
            Name of event to remove.

        Raises
        ------
        ValueError
            If event not found in pipeline.
        """
        if event_name not in self._event_map:
            raise ValueError(f"Event '{event_name}' not found in pipeline")

        event = self._event_map[event_name]
        self.events.remove(event)
        del self._event_map[event_name]

    def replace(self, old_name: str, new_event: Event | str) -> None:
        """
        Replace event with another event.

        Parameters
        ----------
        old_name : str
            Name of event to replace.
        new_event : Event | str
            New event instance or event name.

        Raises
        ------
        ValueError
            If old event not found in pipeline.
        """
        if old_name not in self._event_map:
            raise ValueError(f"Event '{old_name}' not found in pipeline")

        # Instantiate if name provided
        if isinstance(new_event, str):
            event_cls = get_event(new_event)
            new_event = event_cls()

        # Replace in list
        idx = self.events.index(self._event_map[old_name])
        self.events[idx] = new_event

        # Update mapping
        del self._event_map[old_name]
        self._event_map[new_event.name] = new_event

    def _apply_hooks(self) -> None:
        """
        Apply all registered hooks to the pipeline.

        Hooks are applied in registration order for deterministic behavior.
        Multiple events targeting the same hook point are inserted in
        registration order (first registered = closest to target).

        This method is called automatically by ``from_event_list()`` and
        ``from_yaml()`` unless ``apply_hooks=False`` is specified.

        Notes
        -----
        - Replace hooks are applied first
        - Then before/after hooks are applied
        - If target event not in pipeline, hook is silently skipped
        - If event already in pipeline, hook is silently skipped
        """
        from bamengine.core.registry import get_event_hooks

        hooks = get_event_hooks()

        # Apply replace hooks first (order matters less for replace)
        for event_name, hook_spec in hooks.items():
            if hook_spec.get("replace"):
                target = hook_spec["replace"]
                # Skip if target not in pipeline or event already in pipeline
                if target in self._event_map and event_name not in self._event_map:
                    self.replace(target, event_name)

        # Apply after/before hooks in registration order
        # Track insertion points so multiple hooks targeting the same event
        # are inserted in registration order (first registered = closest to target)
        after_insertion_points: dict[str, str] = {}
        before_insertion_points: dict[str, str] = {}

        for event_name, hook_spec in hooks.items():
            # Skip if event already in pipeline (from replace or previous hook)
            if event_name in self._event_map:
                continue

            if hook_spec.get("after"):
                target = hook_spec["after"]
                if target in self._event_map:
                    # Use tracked insertion point if available, else use target
                    insertion_point = after_insertion_points.get(target, target)
                    self.insert_after(insertion_point, event_name)
                    # Update insertion point for next hook targeting same target
                    after_insertion_points[target] = event_name

            elif hook_spec.get("before"):
                target = hook_spec["before"]
                if target in self._event_map:
                    # Use tracked insertion point if available, else use target
                    insertion_point = before_insertion_points.get(target, target)
                    self.insert_before(insertion_point, event_name)
                    # Update insertion point for next hook targeting same target
                    before_insertion_points[target] = event_name

    def __len__(self) -> int:
        """Return number of events in pipeline."""
        return len(self.events)

    def __repr__(self) -> str:
        """Provide informative repr."""
        return f"Pipeline(n_events={len(self.events)})"


def create_default_pipeline(max_M: int, max_H: int, max_Z: int) -> Pipeline:
    """
    Create default BAM simulation event pipeline.

    Loads the pipeline from config/default_pipeline.yml and substitutes
    market round parameters (max_M, max_H, max_Z).

    Parameters
    ----------
    max_M : int
        Number of job application rounds.
    max_H : int
        Number of loan application rounds.
    max_Z : int
        Number of shopping rounds.

    Returns
    -------
    Pipeline
        Default BAM pipeline with all events in correct order.

    Notes
    -----
    This function creates the "canonical" BAM pipeline. Users can modify
    it using insert_after(), remove(), replace() methods, or create their
    own pipeline from a custom YAML file using Pipeline.from_yaml().

    Market rounds are explicitly interleaved: send-hire-send-hire pattern
    for labor market, same for credit market and goods market.
    """
    # Locate the default pipeline YAML file
    try:
        # Python 3.9+: importlib.resources.files() returns a Traversable.
        # Use as_file() to obtain a real filesystem Path (mypy-compatible).
        traversable = resources.files("bamengine") / "config" / "default_pipeline.yml"
        with resources.as_file(traversable) as yaml_fs_path:
            return Pipeline.from_yaml(
                Path(yaml_fs_path),
                max_M=max_M,
                max_H=max_H,
                max_Z=max_Z,
            )
    except (
        AttributeError
    ):  # pragma: no cover - Python 3.8 fallback, not tested on 3.13+
        # Fallback for Python 3.8 where resources.files() is unavailable.
        import bamengine

        yaml_path = Path(bamengine.__file__).parent / "config" / "default_pipeline.yml"
        return Pipeline.from_yaml(yaml_path, max_M=max_M, max_H=max_H, max_Z=max_Z)
