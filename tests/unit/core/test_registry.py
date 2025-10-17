"""Tests for registry system."""

from dataclasses import dataclass

# noinspection PyPackageRequirements
import pytest

from bamengine.core import Role, Event, role, event, get_role, get_event
from bamengine.core.registry import clear_registry, list_events, list_roles
from bamengine.typing import Float1D


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_role_registration_with_name():
    """Test registering a role with explicit name."""

    @role("TestRole")
    @dataclass(slots=True)
    class MyRole(Role):
        values: Float1D

    retrieved = get_role("TestRole")
    assert retrieved is MyRole


def test_role_registration_without_name():
    """Test registering a role using class name."""

    @role()
    @dataclass(slots=True)
    class MyRole(Role):
        values: Float1D

    retrieved = get_role("MyRole")
    assert retrieved is MyRole


def test_event_registration_with_name():
    """Test registering an event with explicit name."""

    @event("test_event")
    class MyEvent(Event):
        name = "test_event"

        def execute(self, sim):
            pass

    retrieved = get_event("test_event")
    assert retrieved is MyEvent


def test_event_registration_without_name():
    """Test registering an event using name attribute."""

    @event()
    class MyEvent(Event):
        name = "my_event"

        def execute(self, sim):
            pass

    retrieved = get_event("my_event")
    assert retrieved is MyEvent


# noinspection PyUnusedLocal
def test_last_registration_wins_role():
    """Test that last role registration overwrites previous."""

    @role("MyRole")
    @dataclass(slots=True)
    class FirstRole(Role):
        values: Float1D

    @role("MyRole")
    @dataclass(slots=True)
    class SecondRole(Role):
        data: Float1D

    retrieved = get_role("MyRole")
    assert retrieved is SecondRole


# noinspection PyUnusedLocal
def test_last_registration_wins_event():
    """Test that last event registration overwrites previous."""

    @event("my_event")
    class FirstEvent(Event):
        name = "my_event"

        def execute(self, sim):
            pass

    @event("my_event")
    class SecondEvent(Event):
        name = "my_event"

        def execute(self, sim):
            pass

    retrieved = get_event("my_event")
    assert retrieved is SecondEvent


def test_get_role_not_found():
    """Test clear error when role not found."""
    with pytest.raises(KeyError, match="Role 'NonExistent' not found"):
        get_role("NonExistent")


def test_get_event_not_found():
    """Test clear error when event not found."""
    with pytest.raises(KeyError, match="Event 'NonExistent' not found"):
        get_event("NonExistent")


# noinspection PyUnusedLocal
def test_list_roles():
    """Test listing all registered roles."""

    @role()
    @dataclass(slots=True)
    class RoleA(Role):
        x: Float1D

    @role()
    @dataclass(slots=True)
    class RoleB(Role):
        y: Float1D

    roles = list_roles()
    assert "RoleA" in roles
    assert "RoleB" in roles
    assert roles == sorted(roles)  # Should be sorted


# noinspection PyUnusedLocal
def test_list_events():
    """Test listing all registered events."""

    @event()
    class EventA(Event):
        name = "event_a"

        def execute(self, sim):
            pass

    @event()
    class EventB(Event):
        name = "event_b"

        def execute(self, sim):
            pass

    events = list_events()
    assert "event_a" in events
    assert "event_b" in events
    assert events == sorted(events)  # Should be sorted
