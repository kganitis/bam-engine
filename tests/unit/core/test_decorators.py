"""Tests for Role and Event decorators."""

import numpy as np
import pytest

from bamengine.core import Event, Role, event, role
from bamengine.core.registry import get_event, get_role
from bamengine.typing import Float1D


class TestRoleDecorator:
    """Test the @role decorator."""

    def test_basic_role_decorator_without_parens(self):
        """Test @role without parentheses (no inheritance)."""

        @role
        class TestRole1:  # ← No (Role) inheritance
            """Test role."""

            value: Float1D

        # Should be a dataclass
        assert hasattr(TestRole1, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRole1, "__slots__")

        # Should be registered
        registered_class = get_role("TestRole1")
        assert registered_class is TestRole1

        # Should be instantiable
        arr = np.array([1.0, 2.0, 3.0])
        instance = TestRole1(value=arr)
        assert np.array_equal(instance.value, arr)

    def test_basic_role_decorator_with_parens(self):
        """Test @role() with parentheses (no inheritance)."""

        @role()
        class TestRole2:  # ← No (Role) inheritance
            """Test role."""

            value: Float1D

        # Should be a dataclass
        assert hasattr(TestRole2, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRole2, "__slots__")

        # Should be registered
        registered_class = get_role("TestRole2")
        assert registered_class is TestRole2

        # Should inherit from Role
        assert issubclass(TestRole2, Role)

    def test_role_decorator_with_custom_name(self):
        """Test @role with custom name parameter (no inheritance)."""

        @role(name="CustomRoleName")
        class TestRole3:  # ← No (Role) inheritance
            """Test role."""

            value: Float1D

        # Should be registered with custom name
        registered_class = get_role("CustomRoleName")
        assert registered_class is TestRole3

        # Should have the custom name
        assert TestRole3.name == "CustomRoleName"

    def test_role_decorator_with_dataclass_kwargs(self):
        """Test @role with additional dataclass kwargs (repr, no inheritance)."""

        @role(repr=False)
        class TestRole4:  # ← No (Role) inheritance
            """Test role with repr=False."""

            value: Float1D

        # Should not have default repr
        arr = np.array([1.0, 2.0, 3.0])
        instance = TestRole4(value=arr)

        # repr should not contain field values
        repr_str = repr(instance)
        assert "TestRole4" in repr_str or "fields=" in repr_str

    def test_role_decorator_slots_default(self):
        """Test that slots=True is set by default (no inheritance)."""

        @role
        class TestRole5:  # ← No (Role) inheritance
            """Test role."""

            value: Float1D

        # Should have __slots__
        assert hasattr(TestRole5, "__slots__")

        # Should not have __dict__ (slots prevents it)
        arr = np.array([1.0])
        instance = TestRole5(value=arr)
        assert not hasattr(instance, "__dict__")

    def test_role_decorator_can_override_slots(self):
        """Test that slots can be explicitly overridden (no inheritance)."""

        @role(slots=False)
        class TestRole6:  # ← No (Role) inheritance
            """Test role without slots."""

            value: Float1D

        # Should have __dict__ (no slots)
        arr = np.array([1.0])
        instance = TestRole6(value=arr)
        assert hasattr(instance, "__dict__")

    def test_role_decorator_with_explicit_inheritance_still_works(self):
        """Test that explicit inheritance from Role still works."""

        @role
        class TestRole7(Role):
            """Test role with explicit inheritance."""

            value: Float1D

        # Should be a dataclass
        assert hasattr(TestRole7, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRole7, "__slots__")

        # Should be registered
        registered_class = get_role("TestRole7")
        assert registered_class is TestRole7

        # Should inherit from Role
        assert issubclass(TestRole7, Role)

        # Should be instantiable
        arr = np.array([1.0, 2.0, 3.0])
        instance = TestRole7(value=arr)
        assert np.array_equal(instance.value, arr)


class TestEventDecorator:
    """Test the @event decorator."""

    def test_basic_event_decorator_without_parens(self):
        """Test @event without parentheses (no inheritance)."""

        @event
        class TestEvent1:  # ← No (Event) inheritance
            """Test event."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should be a dataclass
        assert hasattr(TestEvent1, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestEvent1, "__slots__")

        # Should be registered with snake_case name
        registered_class = get_event("test_event1")
        assert registered_class is TestEvent1

        # Should be instantiable
        instance = TestEvent1()
        assert isinstance(instance, Event)

        # Should inherit from Event
        assert issubclass(TestEvent1, Event)

    def test_basic_event_decorator_with_parens(self):
        """Test @event() with parentheses (no inheritance)."""

        @event()
        class TestEvent2:  # ← No (Event) inheritance
            """Test event."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should be a dataclass
        assert hasattr(TestEvent2, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestEvent2, "__slots__")

        # Should be registered with snake_case name
        registered_class = get_event("test_event2")
        assert registered_class is TestEvent2

        # Should inherit from Event
        assert issubclass(TestEvent2, Event)

    def test_event_decorator_with_custom_name(self):
        """Test @event with custom name parameter (no inheritance)."""

        @event(name="CustomEventName")
        class TestEvent3:  # ← No (Event) inheritance
            """Test event."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should be registered with custom name
        registered_class = get_event("CustomEventName")
        assert registered_class is TestEvent3

        # Should have the custom name
        assert TestEvent3.name == "CustomEventName"

    def test_event_decorator_with_dataclass_kwargs(self):
        """Test @event with additional dataclass kwargs (repr, no inheritance)."""

        @event(repr=False)
        class TestEvent4:  # ← No (Event) inheritance
            """Test event with repr=False."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should not have default repr
        instance = TestEvent4()

        # repr should not contain field values (custom repr from Event base class)
        repr_str = repr(instance)
        assert "TestEvent4" in repr_str or "test_event4" in repr_str

    def test_event_decorator_slots_default(self):
        """Test that slots=True is set by default (no inheritance)."""

        @event
        class TestEvent5:  # ← No (Event) inheritance
            """Test event."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should have __slots__
        assert hasattr(TestEvent5, "__slots__")

        # Should not have __dict__ (slots prevents it)
        instance = TestEvent5()
        assert not hasattr(instance, "__dict__")

    def test_event_decorator_can_override_slots(self):
        """Test that slots can be explicitly overridden (no inheritance)."""

        @event(slots=False)
        class TestEvent6:  # ← No (Event) inheritance
            """Test event without slots."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should have __dict__ (no slots)
        instance = TestEvent6()
        assert hasattr(instance, "__dict__")

    def test_event_decorator_with_explicit_inheritance_still_works(self):
        """Test that explicit inheritance from Event still works."""

        @event
        class TestEvent8(Event):
            """Test event with explicit inheritance."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Should be a dataclass
        assert hasattr(TestEvent8, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestEvent8, "__slots__")

        # Should be registered
        registered_class = get_event("test_event8")
        assert registered_class is TestEvent8

        # Should inherit from Event
        assert issubclass(TestEvent8, Event)

        # Should be instantiable
        instance = TestEvent8()
        assert isinstance(instance, Event)

    def test_event_decorator_no_dependencies_parameter(self):
        """Test that @event does NOT accept dependencies parameter."""
        # This test ensures we've correctly removed the dependencies parameter

        # Attempting to use dependencies should raise TypeError
        with pytest.raises(TypeError):

            @event(dependencies=["SomeEvent"])  # type: ignore[call-arg]
            class TestEvent7(Event):
                """Test event."""

                def execute(self, sim) -> None:
                    """Execute the event."""
                    pass


class TestDecoratorComparison:
    """Test that decorators produce equivalent results to manual approach."""

    def test_role_decorator_equivalent_to_manual(self):
        """Test that @role produces the same result as manual @dataclass(slots=True)."""
        from dataclasses import dataclass

        @role
        class DecoratedRole(Role):
            """Role using @role decorator."""

            value: Float1D

        @dataclass(slots=True)
        class ManualRole(Role):
            """Role using manual @dataclass(slots=True)."""

            value: Float1D

        # Both should be dataclasses
        assert hasattr(DecoratedRole, "__dataclass_fields__")
        assert hasattr(ManualRole, "__dataclass_fields__")

        # Both should have slots
        assert hasattr(DecoratedRole, "__slots__")
        assert hasattr(ManualRole, "__slots__")

        # Both should be instantiable the same way
        arr = np.array([1.0, 2.0, 3.0])
        decorated_instance = DecoratedRole(value=arr)
        manual_instance = ManualRole(value=arr)

        assert np.array_equal(decorated_instance.value, manual_instance.value)

    def test_event_decorator_equivalent_to_manual(self):
        """Test @event produces same result as manual @dataclass(slots=True)."""
        from dataclasses import dataclass

        @event
        class DecoratedEvent(Event):
            """Event using @event decorator."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        @dataclass(slots=True)
        class ManualEvent(Event):
            """Event using manual @dataclass(slots=True)."""

            def execute(self, sim) -> None:
                """Execute the event."""
                pass

        # Both should be dataclasses
        assert hasattr(DecoratedEvent, "__dataclass_fields__")
        assert hasattr(ManualEvent, "__dataclass_fields__")

        # Both should have slots
        assert hasattr(DecoratedEvent, "__slots__")
        assert hasattr(ManualEvent, "__slots__")

        # Both should be instantiable
        decorated_instance = DecoratedEvent()
        manual_instance = ManualEvent()

        assert isinstance(decorated_instance, Event)
        assert isinstance(manual_instance, Event)
