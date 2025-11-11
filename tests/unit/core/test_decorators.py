"""Tests for Role and Event decorators."""

import numpy as np

from bamengine import relationship
from bamengine.core import Event, Role, event, role, get_relationship, Relationship
from bamengine.core.registry import get_event, get_role
from bamengine.typing import Float1D

# TODO Resolve ALL "Expected type 'type[Role] | None', got 'type[SourceRole]' instead"
#  and "Expected type 'type[Role] | None', got 'type[TargetRole]' instead" in the file


# noinspection PyUnresolvedReferences
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


# noinspection PyUnresolvedReferences
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


# Test roles for relationship tests
@role
class SourceRole:
    """Source role for relationship tests."""

    value: Float1D


@role
class TargetRole:
    """Target role for relationship tests."""

    value: Float1D


# noinspection PyUnresolvedReferences
class TestRelationshipDecorator:
    """Test the @relationship decorator."""

    def test_basic_relationship_decorator_without_parens(self):
        """Test @relationship without parentheses (no inheritance)."""

        @relationship(source=SourceRole, target=TargetRole)
        class TestRelationship1:  # ← No (Relationship) inheritance
            """Test relationship."""

            weight: Float1D

        # Should be a dataclass
        assert hasattr(TestRelationship1, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRelationship1, "__slots__")

        # Should be registered with class name
        registered_class = get_relationship("TestRelationship1")
        assert registered_class is TestRelationship1

        # Should inherit from Relationship
        assert issubclass(TestRelationship1, Relationship)

    def test_basic_relationship_decorator_with_parens(self):
        """Test @relationship() with parentheses (no inheritance)."""

        @relationship(source=SourceRole, target=TargetRole)
        class TestRelationship2:  # ← No (Relationship) inheritance
            """Test relationship."""

            weight: Float1D

        # Should be a dataclass
        assert hasattr(TestRelationship2, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRelationship2, "__slots__")

        # Should be registered with class name
        registered_class = get_relationship("TestRelationship2")
        assert registered_class is TestRelationship2

        # Should inherit from Relationship
        assert issubclass(TestRelationship2, Relationship)

    def test_relationship_decorator_with_custom_name(self):
        """Test @relationship with custom name parameter (no inheritance)."""

        @relationship(
            source=SourceRole,
            target=TargetRole,
            name="CustomRelationshipName",
        )
        class TestRelationship3:  # ← No (Relationship) inheritance
            """Test relationship."""

            weight: Float1D

        # Should be registered with custom name
        registered_class = get_relationship("CustomRelationshipName")
        assert registered_class is TestRelationship3

        # Should have the custom name
        assert TestRelationship3.name == "CustomRelationshipName"

    def test_relationship_decorator_with_dataclass_kwargs(self):
        """Test @relationship with additional dataclass kwargs (repr, no inheritance)."""

        @relationship(source=SourceRole, target=TargetRole, repr=False)
        class TestRelationship4:  # ← No (Relationship) inheritance
            """Test relationship with repr=False."""

            weight: Float1D

        # Should not have default repr
        arr = np.array([1.0, 2.0, 3.0])
        instance = TestRelationship4(
            source_ids=np.array([0, 1, 2]),
            target_ids=np.array([3, 4, 5]),
            size=3,
            capacity=128,
            weight=arr
        )

        # repr should not contain field values
        repr_str = repr(instance)
        assert "TestRelationship4" in repr_str or "fields=" in repr_str

    def test_relationship_decorator_with_slots_default(self):
        """Test that slots=True is set by default (no inheritance)."""

        @relationship(source=SourceRole, target=TargetRole)
        class TestRelationship5:  # ← No (Relationship) inheritance
            """Test relationship."""

            weight: Float1D

        # Should have __slots__
        assert hasattr(TestRelationship5, "__slots__")

        # Should not have __dict__ (slots prevents it)
        arr = np.array([1.0])
        instance = TestRelationship5(
            source_ids=np.array([0]),
            target_ids=np.array([1]),
            size=1,
            capacity=128,
            weight=arr
        )
        assert not hasattr(instance, "__dict__")

    def test_relationship_decorator_can_override_slots(self):
        """Test that slots can be explicitly overridden (no inheritance)."""

        @relationship(source=SourceRole, target=TargetRole, slots=False)
        class TestRelationship6:  # ← No (Relationship) inheritance
            """Test relationship without slots."""

            weight: Float1D

        # Should have __dict__ (no slots)
        arr = np.array([1.0])
        instance = TestRelationship6(
            source_ids=np.array([0]),
            target_ids=np.array([1]),
            size=1,
            capacity=128,
            weight=arr
        )
        assert hasattr(instance, "__dict__")

    def test_relationship_decorator_with_explicit_inheritance_still_works(self):
        """Test that explicit inheritance from Relationship still works."""

        @relationship(source=SourceRole, target=TargetRole)
        class TestRelationship7(Relationship):
            """Test relationship with explicit inheritance."""

            weight: Float1D

        # Should be a dataclass
        assert hasattr(TestRelationship7, "__dataclass_fields__")

        # Should have slots
        assert hasattr(TestRelationship7, "__slots__")

        # Should be registered
        registered_class = get_relationship("TestRelationship7")
        assert registered_class is TestRelationship7

        # Should inherit from Relationship
        assert issubclass(TestRelationship7, Relationship)

        # Should be instantiable
        arr = np.array([1.0, 2.0, 3.0])
        instance = TestRelationship7(
            source_ids=np.array([0, 1, 2]),
            target_ids=np.array([3, 4, 5]),
            size=3,
            capacity=128,
            weight=arr
        )
        assert np.array_equal(instance.weight, arr)

    def test_relationship_decorator_basic(self):
        """Test basic @relationship decorator usage."""

        @relationship(source=SourceRole, target=TargetRole)
        class TestRelation:
            weight: Float1D

        assert TestRelation.source_role is SourceRole
        assert TestRelation.target_role is TargetRole
        assert TestRelation.cardinality == "many-to-many"
        assert TestRelation.name == "TestRelation"

    def test_relationship_decorator_custom_name(self):
        """Test @relationship decorator with custom name."""

        @relationship(source=SourceRole, target=TargetRole, name="custom_relation")
        class TestRelation:
            weight: Float1D

        assert TestRelation.name == "custom_relation"
        retrieved = get_relationship("custom_relation")
        assert retrieved is TestRelation

    def test_relationship_decorator_cardinality(self):
        """Test @relationship decorator with different cardinalities."""

        @relationship(source=SourceRole, target=TargetRole, cardinality="one-to-many")
        class TestRelation:
            weight: Float1D

        assert TestRelation.cardinality == "one-to-many"

    def test_relationship_decorator_registration(self):
        """Test that @relationship automatically registers the relationship."""

        @relationship(source=SourceRole, target=TargetRole, name="test_rel")
        class TestRelation:
            weight: Float1D

        retrieved = get_relationship("test_rel")
        assert retrieved is TestRelation

    def test_relationship_decorator_auto_inheritance(self):
        """Test that @relationship decorator automatically adds Relationship inheritance."""

        @relationship(source=SourceRole, target=TargetRole)
        class AutoInherited:
            weight: Float1D

        # Should automatically inherit from Relationship
        assert issubclass(AutoInherited, Relationship)
        # Should be a dataclass
        assert hasattr(AutoInherited, "__dataclass_fields__")
        # Should have slots
        assert hasattr(AutoInherited, "__slots__")


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

    def test_relationship_decorator_equivalent_to_manual(self):
        """Test @relationship produces same result as manual @dataclass(slots=True)."""
        from dataclasses import dataclass

        @relationship(source=SourceRole, target=TargetRole)
        class DecoratedRelationship:
            """Relationship using @relationship decorator."""

            weight: Float1D

        @dataclass(slots=True)
        class ManualRelationship(Relationship):
            """Relationship using manual @dataclass(slots=True)."""

            weight: Float1D

            source_role = SourceRole
            target_role = TargetRole
            cardinality = "many-to-many"
            name = "ManualRelationship"

        # Both should be dataclasses
        assert hasattr(DecoratedRelationship, "__dataclass_fields__")
        assert hasattr(ManualRelationship, "__dataclass_fields__")

        # Both should have slots
        assert hasattr(DecoratedRelationship, "__slots__")
        assert hasattr(ManualRelationship, "__slots__")

        # Both should be instantiable the same way
        arr = np.array([1.0, 2.0, 3.0])
        decorated_instance = DecoratedRelationship(
            source_ids=np.array([0, 1, 2]),
            target_ids=np.array([3, 4, 5]),
            size=3,
            capacity=128,
            weight=arr
        )
        manual_instance = ManualRelationship(
            source_ids=np.array([0, 1, 2]),
            target_ids=np.array([3, 4, 5]),
            size=3,
            capacity=128,
            weight=arr
        )

        # Both should inherit from Relationship
        assert isinstance(decorated_instance, Relationship)
        assert isinstance(manual_instance, Relationship)
