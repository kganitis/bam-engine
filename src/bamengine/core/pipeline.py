"""Event Pipeline with explicit execution order."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from typing import cast
from bamengine.core.event import Event
from bamengine.core.registry import get_event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class RepeatedEvent:
    """
    Wrapper for events that execute multiple times per period.

    Used for market rounds where agents interact over multiple rounds
    (e.g., job applications, loan applications, shopping).

    Attributes
    ----------
    event : Event
        The event to repeat.
    n_repeats : int
        Number of times to execute the event.
    """

    event: Event
    n_repeats: int

    def execute(self, sim: Simulation) -> None:
        """Execute the event n_repeats times."""
        for _ in range(self.n_repeats):
            self.event.execute(sim)

    @property
    def name(self) -> str:
        """Return the name of the underlying event."""
        return self.event.name


@dataclass(slots=True)
class Pipeline:
    """
    Dependency-aware, topologically sorted event execution pipeline.

    The Pipeline manages event execution order by resolving dependencies
    declared in Event classes and additional before/after constraints
    from configuration.

    Attributes
    ----------
    events : list[Event]
        Ordered list of event instances to execute.
    _event_map : dict[str, Event]
        Internal mapping from event names to instances for quick lookup.

    See Also
    --------
    Pipeline.from_event_list : Build pipeline from event name list
    """

    events: list[Event] = field(default_factory=list)
    _event_map: dict[str, Event] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build internal event mapping."""
        self._event_map = {event.name: event for event in self.events}

    @classmethod
    def from_event_list(
        cls,
        event_names: list[str],
        *,
        repeats: dict[str, int] | None = None,
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

        Returns
        -------
        Pipeline
            Pipeline with events in the order specified.

        Raises
        ------
        ValueError
            If event name not found in registry.

        Notes
        -----
        The order of events is critical for correct simulation behavior.
        Use the default pipeline as a reference for the required ordering.
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

        return cls(events=event_instances)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
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
        **params : int
            Parameters to substitute in the YAML (e.g., max_M=5, max_H=3, max_Z=2).

        Returns
        -------
        Pipeline
            Pipeline with events parsed from YAML.

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

        return cls.from_event_list(event_names)

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
        Execute all events in pipeline order.

        Parameters
        ----------
        sim : Simulation
            Simulation instance to operate on.

        Returns
        -------
        None
            All mutations are in-place.
        """
        for event in self.events:
            event.execute(sim)

    def insert_after(self, after: str, event: Event | str) -> None:
        """
        Insert event after specified event.

        Parameters
        ----------
        after : str
            Event name to insert after.
        event : Event | str
            Event instance or event name to insert.

        Raises
        ------
        ValueError
            If 'after' event not found in pipeline.
        """
        if after not in self._event_map:
            raise ValueError(f"Event '{after}' not found in pipeline")

        # Instantiate if name provided
        if isinstance(event, str):
            event_cls = get_event(event)
            event = event_cls()

        # Find insertion point
        idx = self.events.index(self._event_map[after])
        self.events.insert(idx + 1, event)
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

    def __len__(self) -> int:
        """Return number of events in pipeline."""
        return len(self.events)

    def __repr__(self) -> str:
        """Provide informative repr."""
        return f"Pipeline(n_events={len(self.events)})"
