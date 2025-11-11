"""Registry system for roles, events and relationships."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from bamengine.core.event import Event
    from bamengine.core.role import Role
    from bamengine.core.relationship import Relationship

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
        Name of the role to retrieve.

    Returns
    -------
    type[Role]
        The registered role class.

    Raises
    ------
    KeyError
        If the role name is not found in the registry.
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
        Name of the event to retrieve.

    Returns
    -------
    type[Event]
        The registered event class.

    Raises
    ------
    KeyError
        If the event name is not found in the registry.
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
    """Return sorted list of all registered role names."""
    return sorted(_ROLE_REGISTRY.keys())


def list_events() -> list[str]:
    """Return sorted list of all registered event names."""
    return sorted(_EVENT_REGISTRY.keys())


def list_relationships() -> list[str]:
    """Return sorted list of all registered relationship names."""
    return sorted(_RELATIONSHIP_REGISTRY.keys())


def clear_registry() -> None:
    """
    Clear all registrations (useful for testing).

    WARNING: This is a destructive operation. Only use in test teardown.
    """
    _ROLE_REGISTRY.clear()
    _EVENT_REGISTRY.clear()
    _RELATIONSHIP_REGISTRY.clear()
