"""Tests for Event base class."""

import logging
from unittest.mock import MagicMock

# noinspection PyPackageRequirements
import pytest

from bamengine.core import Event


class TestEvent(Event):
    """Concrete event for testing."""

    name = "test_event"

    def execute(self, sim):
        """Test implementation."""
        pass


class TestEventWithDeps(Event):
    """Event with dependencies."""

    name = "test_event_with_deps"

    def execute(self, sim):
        pass

    @property
    def dependencies(self) -> tuple[str, ...]:
        return "event_a", "event_b"


def test_event_creation():
    """Test creating a concrete event."""
    event = TestEvent()

    assert event.name == "test_event"
    assert event.dependencies == ()
    assert event.log_level == logging.INFO


def test_event_execute():
    """Test event execute method."""
    event = TestEvent()
    sim = MagicMock()

    # Should not raise
    event.execute(sim)


def test_event_dependencies():
    """Test event with dependencies."""
    event = TestEventWithDeps()

    assert event.dependencies == ("event_a", "event_b")


def test_event_repr():
    """Test event string representation."""
    event = TestEvent()
    repr_str = repr(event)

    assert "test_event" in repr_str


def test_event_abstract():
    """Test that Event cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # noinspection PyAbstractClass
        Event()
