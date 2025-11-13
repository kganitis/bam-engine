"""
Registry system for roles, events and relationships.

Provides global lookup for all registered roles, events, and relationships
in the BAM Engine. Registration happens automatically via __init_subclass__
hooks, eliminating manual registration boilerplate.

Design Notes
------------
- Thread-safe read-only access (registries populated at import time)
- No dynamic registration after initialization
- Clear error messages with suggestions for misspellings
- List functions for discovery (list_roles, list_events, list_relationships)

Usage Pattern
-------------
Automatic registration (happens at import):

>>> from dataclasses import dataclass
>>> from bamengine.core import Role
>>> from bamengine import Float
>>>
>>> @dataclass(slots=True)
... class MyRole(Role):
...     field: Float
>>> # MyRole is now automatically registered!

Retrieval:

>>> from bamengine.core.registry import get_role
>>> role_cls = get_role("MyRole")
>>> import numpy as np
>>> instance = role_cls(field=np.array([1.0, 2.0, 3.0]))

Discovery:

>>> from bamengine.core.registry import list_roles
>>> all_roles = list_roles()
>>> print(all_roles)
['Borrower', 'Consumer', 'Employer', 'Lender', 'MyRole', 'Producer', 'Worker']

Error Handling
--------------
If a role/event is not found, registry functions raise KeyError with
helpful message listing all available options:

>>> get_role("Producter")  # Typo in name
Traceback (most recent call last):
    ...
KeyError: Role 'Producter' not found in registry.
Available roles: Borrower, Consumer, Employer, Lender, Producer, Worker

See Also
--------
Role : Base class with __init_subclass__ registration
Event : Base class with __init_subclass__ registration
Relationship : Base class with __init_subclass__ registration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.core.event import Event
    from bamengine.core.relationship import Relationship
    from bamengine.core.role import Role

# Type variables for generic decorator typing
R = TypeVar("R", bound="Role")
E = TypeVar("E", bound="Event")
L = TypeVar("L", bound="Relationship")

# Global registry storage
_ROLE_REGISTRY: dict[str, type[Role]] = {}
_EVENT_REGISTRY: dict[str, type[Event]] = {}
_RELATIONSHIP_REGISTRY: dict[str, type[Relationship]] = {}


def get_role(name: str) -> type[Role]:
    """
    Retrieve a role class from the registry by name.

    Parameters
    ----------
    name : str
        Name of the role to retrieve (case-sensitive).

    Returns
    -------
    type[Role]
        The registered role class.

    Raises
    ------
    KeyError
        If the role name is not found in the registry. Error message
        includes list of all available roles.

    Examples
    --------
    Retrieve a role class and create instance:

    >>> from bamengine.core.registry import get_role
    >>> import numpy as np
    >>> Producer = get_role("Producer")
    >>> prod = Producer(
    ...     price=np.array([1.0, 1.2]),
    ...     production=np.array([100.0, 120.0]),
    ...     inventory=np.array([0.0, 10.0]),
    ...     labor_productivity=np.array([2.0, 2.0])
    ... )

    Use in simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> Producer = get_role("Producer")
    >>> assert isinstance(sim.prod, Producer)

    Handle missing role:

    >>> try:
    ...     get_role("NonExistent")
    ... except KeyError as e:
    ...     print(e)
    Role 'NonExistent' not found in registry. Available roles: ...

    See Also
    --------
    list_roles : Get list of all registered role names
    get_event : Retrieve event class from registry
    """
    if name not in _ROLE_REGISTRY:
        available = ", ".join(sorted(_ROLE_REGISTRY.keys()))
        raise KeyError(
            f"Role '{name}' not found in registry. " f"Available roles: {available}"
        )
    return _ROLE_REGISTRY[name]


def get_event(name: str) -> type[Event]:
    """
    Retrieve an event class from the registry by name.

    Parameters
    ----------
    name : str
        Name of the event to retrieve (snake_case, case-sensitive).

    Returns
    -------
    type[Event]
        The registered event class.

    Raises
    ------
    KeyError
        If the event name is not found in the registry. Error message
        includes list of all available events.

    Examples
    --------
    Retrieve and execute an event:

    >>> from bamengine.core.registry import get_event
    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> FirmsAdjustPrice = get_event("firms_adjust_price")
    >>> event_instance = FirmsAdjustPrice()
    >>> event_instance.execute(sim)

    Check event availability:

    >>> from bamengine.core.registry import list_events
    >>> 'firms_adjust_price' in list_events()
    True

    See Also
    --------
    list_events : Get list of all registered event names
    get_role : Retrieve role class from registry
    """
    if name not in _EVENT_REGISTRY:
        available = ", ".join(sorted(_EVENT_REGISTRY.keys()))
        raise KeyError(
            f"Event '{name}' not found in registry. " f"Available events: {available}"
        )
    return _EVENT_REGISTRY[name]


def get_relationship(name: str) -> type[Relationship]:
    """
    Retrieve a relationship class from the registry by name.

    Parameters
    ----------
    name : str
        Name of the relationship to retrieve.

    Returns
    -------
    type[Relationship]
        The registered relationship class

    Raises
    ------
    KeyError
        If the relationship name is not found in the registry.
    """
    if name not in _RELATIONSHIP_REGISTRY:
        available = ", ".join(sorted(_RELATIONSHIP_REGISTRY.keys()))
        raise KeyError(
            f"Relationship '{name}' not found in registry. "
            f"Available relationships: {available}"
        )
    return _RELATIONSHIP_REGISTRY[name]


def list_roles() -> list[str]:
    """
    Return sorted list of all registered role names.

    Returns
    -------
    list[str]
        Sorted list of role names.

    Examples
    --------
    >>> from bamengine.core.registry import list_roles
    >>> roles = list_roles()
    >>> 'Producer' in roles
    True
    >>> 'Worker' in roles
    True
    """
    return sorted(_ROLE_REGISTRY.keys())


def list_events() -> list[str]:
    """
    Return sorted list of all registered event names.

    Returns
    -------
    list[str]
        Sorted list of event names in snake_case.

    Examples
    --------
    >>> from bamengine.core.registry import list_events
    >>> events = list_events()
    >>> len(events)  # Should be 39 BAM events + any custom
    39
    >>> 'firms_adjust_price' in events
    True
    """
    return sorted(_EVENT_REGISTRY.keys())


def list_relationships() -> list[str]:
    """
    Return sorted list of all registered relationship names.

    Returns
    -------
    list[str]
        Sorted list of relationship names.

    Examples
    --------
    >>> from bamengine.core.registry import list_relationships
    >>> rels = list_relationships()
    >>> 'LoanBook' in rels
    True
    """
    return sorted(_RELATIONSHIP_REGISTRY.keys())


def clear_registry() -> None:
    """
    Clear all registrations (useful for testing).

    WARNING: This is a destructive operation. Only use in test teardown.
    """
    _ROLE_REGISTRY.clear()
    _EVENT_REGISTRY.clear()
    _RELATIONSHIP_REGISTRY.clear()
