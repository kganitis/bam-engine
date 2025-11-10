"""Tests for registry system."""

from dataclasses import dataclass

# noinspection PyPackageRequirements
import pytest

from bamengine import relationship
from bamengine.core import (Role, Event, Relationship,
                            get_role, get_event, get_relationship)
from bamengine.core.registry import (clear_registry,
                                     list_events, list_roles, list_relationships)
from bamengine.typing import Float1D


@pytest.fixture
def clean_registry():
    """
    Save registry state, clear it for the test, then restore it.

    This fixture should be explicitly requested by tests that need isolation
    from real BAM components. Tests should use this fixture if they create
    synthetic roles/events for testing registry mechanics.

    DO NOT use autouse=True, as it would interfere with integration tests
    that rely on real components being registered.
    """
    # noinspection PyProtectedMember
    from bamengine.core.registry import _ROLE_REGISTRY, _EVENT_REGISTRY

    # Save current state
    saved_roles = dict(_ROLE_REGISTRY)
    saved_events = dict(_EVENT_REGISTRY)

    # Clear for test
    clear_registry()

    yield

    # Restore original state
    _ROLE_REGISTRY.clear()
    _ROLE_REGISTRY.update(saved_roles)
    _EVENT_REGISTRY.clear()
    _EVENT_REGISTRY.update(saved_events)


# Test roles for relationship tests
@dataclass(slots=True)
class SourceRole(Role):
    pass


@dataclass(slots=True)
class TargetRole(Role):
    pass


def test_role_auto_registration(clean_registry):
    """Test that roles are auto-registered via __init_subclass__ (no decorator)."""

    @dataclass(slots=True)
    class MyRole(Role):
        values: Float1D

    retrieved = get_role("MyRole")
    assert retrieved is MyRole
    assert hasattr(MyRole, "__dataclass_fields__")  # Auto-dataclass
    assert MyRole.name == "MyRole"  # Auto-named


def test_role_custom_name(clean_registry):
    """Test that roles can have custom names during class definition."""

    @dataclass(slots=True)
    class MyRole(Role, name="custom_role_name"):
        values: Float1D

    # Should be registered under custom name
    retrieved = get_role("custom_role_name")
    assert retrieved is MyRole
    assert MyRole.name == "custom_role_name"
    assert MyRole.__name__ == "MyRole"  # Class name unchanged


def test_event_auto_registration(clean_registry):
    """Test that events are auto-registered via __init_subclass__ (no decorator)."""

    class MyEvent(Event):
        def execute(self, sim):
            pass

    retrieved = get_event("my_event")
    assert retrieved is MyEvent


def test_event_custom_name(clean_registry):
    """Test that events can have custom names during class definition."""

    @dataclass(slots=True)
    class MyEvent(Event, name="custom_event_name"):
        def execute(self, sim):
            pass

    # Should be registered under custom name
    retrieved = get_event("custom_event_name")
    assert retrieved is MyEvent
    assert MyEvent.name == "custom_event_name"
    assert MyEvent.__name__ == "MyEvent"  # Class name unchanged


def test_relationship_auto_registration(clean_registry):
    """
    Test that relationships are auto-registered
    via __init_subclass__ (no decorator)."""

    @dataclass(slots=True)
    class MyRelationship(Relationship):
        data: Float1D

    retrieved = get_relationship("MyRelationship")
    assert retrieved is MyRelationship
    assert "data" in MyRelationship.__dataclass_fields__


def test_relationship_custom_name(clean_registry):
    """
    Test that relationships can have custom names
    during class definition.
    """

    @dataclass(slots=True)
    class MyRelationship(Relationship, name="custom_relationship_name"):
        data: Float1D

    # Should be registered under custom name
    retrieved = get_relationship("custom_relationship_name")
    assert retrieved is MyRelationship
    assert MyRelationship.name == "custom_relationship_name"
    assert MyRelationship.__name__ == "MyRelationship"  # Class name unchanged


# noinspection PyUnusedLocal
def test_last_registration_wins_role(clean_registry):
    """Test that last role registration overwrites previous (auto-registration)."""

    # Note: In practice, you wouldn't do this. But __init_subclass__ allows it.
    # The second class replaces the first in the registry.

    @dataclass(slots=True)
    class FirstRole(Role, name="MyRole"):
        values: Float1D

    # This replaces the previous MyRole
    @dataclass(slots=True)
    class SecondRole(Role, name="MyRole"):
        data: Float1D

    retrieved = get_role("MyRole")
    assert retrieved.__name__ == "SecondRole"
    # noinspection PyUnresolvedReferences
    assert "data" in retrieved.__dataclass_fields__


# noinspection PyUnusedLocal
def test_last_registration_wins_event(clean_registry):
    """Test that last event registration overwrites previous (auto-registration)."""

    @dataclass(slots=True)
    class FirstEvent(Event, name="my_event"):
        def execute(self, sim):
            pass

    # This replaces the previous MyEvent
    @dataclass(slots=True)
    class SecondEvent(Event, name="my_event"):
        def execute(self, sim):
            pass

    retrieved = get_event("my_event")
    assert retrieved.__name__ == "SecondEvent"
    assert "execute" in dir(retrieved)


def test_last_registration_wins_relationship(clean_registry):
    """
    Test that last relationship registration
    overwrites previous (auto-registration).
    """

    @relationship(source=SourceRole, target=TargetRole, name="my_relationship")
    class FirstRelationship:
        x: Float1D

    # This replaces the previous MyRelationship
    @relationship(source=SourceRole, target=TargetRole, name="my_relationship")
    class SecondRelationship:
        y: Float1D

    retrieved = get_relationship("my_relationship")
    assert retrieved.__name__ == "SecondRelationship"
    assert "y" in retrieved.__dataclass_fields__
    assert "x" not in retrieved.__dataclass_fields__


def test_get_role_not_found(clean_registry):
    """Test clear error when role not found."""
    with pytest.raises(KeyError, match="Role 'NonExistent' not found"):
        get_role("NonExistent")


def test_get_event_not_found(clean_registry):
    """Test clear error when event not found."""
    with pytest.raises(KeyError, match="Event 'NonExistent' not found"):
        get_event("NonExistent")


def test_get_relationship_not_found(clean_registry):
    """Test clear error when relationship not found."""
    with pytest.raises(KeyError, match="Relationship 'NonExistent' not found"):
        get_relationship("NonExistent")


# noinspection PyUnusedLocal
def test_list_roles(clean_registry):
    """Test listing all registered roles."""

    @dataclass(slots=True)
    class RoleA(Role):
        x: Float1D

    @dataclass(slots=True)
    class RoleB(Role):
        y: Float1D

    roles = list_roles()
    assert "RoleA" in roles
    assert "RoleB" in roles
    assert roles == sorted(roles)  # Should be sorted


# noinspection PyUnusedLocal
def test_list_events(clean_registry):
    """Test listing all registered events (auto-registration)."""

    @dataclass(slots=True)
    class EventB(Event):
        def execute(self, sim):
            pass

    @dataclass(slots=True)
    class EventA(Event):
        def execute(self, sim):
            pass

    events = list_events()
    assert "event_a" in events
    assert "event_b" in events
    assert events == sorted(events)  # Should be sorted


def test_list_relationships(clean_registry):
    """Test listing all registered relationships (auto-registration)."""

    @relationship(source=SourceRole, target=TargetRole, name="rel_b")
    class RelationA:
        x: Float1D

    @relationship(source=SourceRole, target=TargetRole, name="rel_a")
    class RelationB:
        y: Float1D

    relationships = list_relationships()
    assert "rel_a" in relationships
    assert "rel_b" in relationships
    assert relationships == sorted(relationships)  # Should be sorted
