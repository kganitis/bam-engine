# src/bamengine/core/decorators.py
"""
Decorators for simplified Role and Event definition.

This module provides decorators that dramatically simplify the syntax for
defining Roles and Events. They automatically apply @dataclass(slots=True),
handle inheritance from Role/Event, and manage registration.

Usage
-----
Instead of:
    from dataclasses import dataclass
    from bamengine import Role, Float

    @dataclass(slots=True)
    class Producer(Role):
        price: Float
        production: Float

You can write:
    from bamengine import role, Float

    @role
    class Producer:
        price: Float
        production: Float

Or with custom name:
    @role(name="MyProducer")
    class Producer:
        price: Float
        production: Float

The decorator handles:
- Making the class a dataclass with slots
- Making it inherit from Role/Event (if not already)
- Auto-registration via __init_subclass__
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")


def role(
    cls: type[T] | None = None,
    *,
    name: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorator to define a Role with automatic inheritance and dataclass.

    This decorator dramatically simplifies Role definition by:
    1. Making the class inherit from Role (if not already)
    2. Applying @dataclass(slots=True)
    3. Handling registration automatically

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    name : str | None
        Optional custom name for the role. If None, uses the class name.
    **dataclass_kwargs : Any
        Additional keyword arguments to pass to @dataclass.
        By default, slots=True is set.

    Returns
    -------
    type | Callable
        The decorated class or a decorator function

    Examples
    --------
    Simplest usage (no inheritance needed):
        @role
        class Producer:
            price: Float
            production: Float

    With custom name:
        @role(name="MyProducer")
        class Producer:
            price: Float
            production: Float

    Traditional usage (still works):
        @role
        class Producer(Role):
            price: Float
            production: Float

    Notes
    -----
    - No need to inherit from Role explicitly (decorator adds it)
    - No need for @dataclass(slots=True) (decorator applies it)
    - Registration happens automatically via Role.__init_subclass__
    - slots=True is set by default for memory efficiency
    - frozen=True is not supported (Role base class is not frozen)
    """
    # Import here to avoid circular imports
    from bamengine.core.role import Role

    # Set default slots=True unless explicitly overridden
    dataclass_kwargs.setdefault("slots", True)

    def decorator(cls: type[T]) -> type[T]:
        # Check if cls already inherits from Role
        if not issubclass(cls, Role):
            # Dynamically create a new class that inherits ONLY from Role
            # Copy annotations and methods from the original class
            # This ensures slots work properly (no multiple inheritance issues)
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__annotations__": getattr(cls, "__annotations__", {}),
            }
            # Copy methods and class attributes (but not __dict__, __weakref__, etc.)
            for attr_name in dir(cls):
                if not attr_name.startswith("__"):
                    namespace[attr_name] = getattr(cls, attr_name)

            cls = type(cls.__name__, (Role,), namespace)

        # Set custom name BEFORE applying dataclass
        # This ensures __init_subclass__ sees the correct name
        if name is not None:
            cls.name = name  # type: ignore[attr-defined]

        # Apply dataclass decorator
        cls = dataclass(**dataclass_kwargs)(cls)

        return cls

    # Support both @role and @role() syntax
    if cls is None:
        # Called with arguments: @role(name="...")
        return decorator
    else:
        # Called without arguments: @role
        return decorator(cls)


def event(
    cls: type[T] | None = None,
    *,
    name: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorator to define an Event with automatic inheritance and dataclass.

    This decorator dramatically simplifies Event definition by:
    1. Making the class inherit from Event (if not already)
    2. Applying @dataclass(slots=True)
    3. Handling registration automatically

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    name : str | None
        Optional custom name for the event. If None, uses class name (snake_case).
    **dataclass_kwargs : Any
        Additional keyword arguments to pass to @dataclass.
        By default, slots=True is set.

    Returns
    -------
    type | Callable
        The decorated class or a decorator function

    Examples
    --------
    Simplest usage (no inheritance needed):
        @event
        class Planning:
            def execute(self, sim: Simulation) -> None:
                # implementation

    With custom name:
        @event(name="my_planning")
        class Planning:
            def execute(self, sim: Simulation) -> None:
                # implementation

    Traditional usage (still works):
        @event
        class Planning(Event):
            def execute(self, sim: Simulation) -> None:
                # implementation

    Notes
    -----
    - No need to inherit from Event explicitly (decorator adds it)
    - No need for @dataclass(slots=True) (decorator applies it)
    - Registration happens automatically via Event.__init_subclass__
    - slots=True is set by default for memory efficiency
    - frozen=True is not supported (Event base class is not frozen)
    - Dependencies are NOT supported (they've been removed from the Event system)
    """
    # Import here to avoid circular imports
    from bamengine.core.event import Event

    # Set default slots=True unless explicitly overridden
    dataclass_kwargs.setdefault("slots", True)

    def decorator(cls: type[T]) -> type[T]:
        # Check if cls already inherits from Event
        if not issubclass(cls, Event):
            # Dynamically create a new class that inherits ONLY from Event
            # Copy annotations and methods from the original class
            # This ensures slots work properly (no multiple inheritance issues)
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__annotations__": getattr(cls, "__annotations__", {}),
            }
            # Copy methods and class attributes (but not __dict__, __weakref__, etc.)
            for attr_name in dir(cls):
                if not attr_name.startswith("__"):
                    namespace[attr_name] = getattr(cls, attr_name)

            cls = type(cls.__name__, (Event,), namespace)

        # Set custom name BEFORE applying dataclass
        # This ensures __init_subclass__ sees the correct name
        if name is not None:
            cls.name = name  # type: ignore[attr-defined]

        # Apply dataclass decorator
        cls = dataclass(**dataclass_kwargs)(cls)

        return cls

    # Support both @event and @event() syntax
    if cls is None:
        # Called with arguments: @event(name="...")
        return decorator
    else:
        # Called without arguments: @event
        return decorator(cls)
