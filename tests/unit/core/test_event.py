"""Unit tests for Event base class."""

from dataclasses import dataclass

from bamengine import Simulation
from bamengine.core.event import Event


@dataclass(slots=True)
class DummyEvent(Event):
    """Concrete role for testing."""

    def execute(self, sim: Simulation):
        pass


def test_event_slots():
    """Test that event uses slots (no __dict__)."""
    event = DummyEvent()
    assert not hasattr(event, "__dict__")


def test_event_repr():
    """Event repr shows name."""

    class MyEvent(Event):
        def execute(self, sim):
            pass

    event = MyEvent()
    assert repr(event) == "MyEvent(name='my_event')"
