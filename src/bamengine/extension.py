"""
Extension bundle for BAM Engine.

An :class:`Extension` bundles related roles, events, relationships, and
configuration parameters into a single object that can be applied to a
simulation with one call:

>>> sim = bam.Simulation.init(seed=42)
>>> sim.use(MY_EXTENSION)

This replaces the manual three-step pattern::

    sim.use_role(SomeRole, n_agents=sim.n_firms)
    sim.use_events(*SOME_EVENTS)
    sim.use_config(SOME_CONFIG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Extension:
    """Immutable bundle of extension components.

    Parameters
    ----------
    roles : dict[type, str]
        Mapping of role class to agent type (``"firms"``, ``"households"``,
        or ``"banks"``). The simulation resolves agent count from the type.
    events : list[type]
        Event classes with pipeline hook metadata (``@event(after=...)``).
    relationships : list[type]
        Relationship classes (reserved for future use).
    config_dict : dict[str, Any]
        Default extension parameters (merged with "don't overwrite" semantics).

    Examples
    --------
    >>> from bamengine import Extension
    >>> MY_EXT = Extension(
    ...     roles={MyRole: "firms"},
    ...     events=[MyEvent],
    ...     relationships=[],
    ...     config_dict={"my_param": 0.5},
    ... )
    >>> sim.use(MY_EXT)
    """

    roles: dict[type, str] = field(default_factory=dict)
    events: list[type] = field(default_factory=list)
    relationships: list[type] = field(default_factory=list)
    config_dict: dict[str, Any] = field(default_factory=dict)
