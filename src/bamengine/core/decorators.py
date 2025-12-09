"""
Decorators for simplified Role, Event and Relationship definition.

This module provides decorators that simplify the syntax for defining
Roles, Events and Relationships. They automatically apply @dataclass(slots=True),
handle inheritance from Role/Event/Relationship, and manage registration.

Design Notes
------------
The decorators handle three key tasks:

1. Making the class a dataclass with slots
2. Making it inherit from Role/Event/Relationship (if not already)
3. Auto-registration via __init_subclass__

Examples
--------
Role decorator (simplest syntax):

>>> from bamengine import role, Float
>>>
>>> @role
... class Producer:
...     price: Float
...     production: Float
>>> # Automatically inherits from Role, is a dataclass, and is registered!

Role with custom name:

>>> @role(name="MyProducer")
... class Producer:
...     price: Float
...     production: Float

Event decorator:

>>> from bamengine import event
>>>
>>> @event
... class CustomPricingEvent:
...     def execute(self, sim):
...         prod = sim.get_role("Producer")
...         # Apply custom pricing logic

Relationship decorator:

>>> from bamengine import relationship, get_role, Float
>>>
>>> @relationship(source=get_role("Borrower"), target=get_role("Lender"))
... class LoanBook:
...     principal: Float
...     rate: Float
...     debt: Float

Traditional syntax (still works):

>>> from dataclasses import dataclass
>>> from bamengine.core import Role
>>> from bamengine import Float
>>>
>>> @dataclass(slots=True)
... class Producer(Role):
...     price: Float
...     production: Float

See Also
--------
:class:`~bamengine.core.Role` : Base class for roles (components)
:class:`~bamengine.core.Event` : Base class for events (systems)
:class:`~bamengine.core.Relationship` : Base class for relationships
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.core import Role

if TYPE_CHECKING:  # pragma: no cover
    pass

T = TypeVar("T")


def role(
    cls: type[T] | None = None,
    *,
    name: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator to define a Role with automatic inheritance and dataclass.

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

    >>> from bamengine.typing import Float
    >>> @role
    ... class Producer:
    ...     price: Float
    ...     production: Float

    With custom name:

    >>> @role(name="MyProducer")
    ... class Producer:
    ...     price: Float
    ...     production: Float
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
                "__doc__": cls.__doc__,  # Preserve docstring for Sphinx
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
    after: str | None = None,
    before: str | None = None,
    replace: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator to define an Event with automatic inheritance and dataclass.

    This decorator dramatically simplifies Event definition by:
    1. Making the class inherit from Event (if not already)
    2. Applying @dataclass(slots=True)
    3. Handling registration automatically
    4. Optionally registering pipeline hooks for automatic positioning

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    name : str | None
        Optional custom name for the event. If None, uses class name (snake_case).
    after : str | None
        Insert this event immediately after the target event in the pipeline.
        Hooks are applied automatically when pipelines are created.
    before : str | None
        Insert this event immediately before the target event in the pipeline.
    replace : str | None
        Replace the target event with this event in the pipeline.
    **dataclass_kwargs : Any
        Additional keyword arguments to pass to @dataclass.
        By default, slots=True is set.

    Returns
    -------
    type | Callable
        The decorated class or a decorator function

    Raises
    ------
    ValueError
        If more than one of ``after``, ``before``, or ``replace`` is specified.

    Examples
    --------
    Simplest usage (no inheritance needed):

    >>> from bamengine import Simulation
    >>> @event
    ... class Planning:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # implementation

    With custom name:

    >>> @event(name="my_planning")
    ... class Planning:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # implementation

    With pipeline hook (inserted after another event):

    >>> @event(after="firms_pay_dividends")
    ... class MyCustomEvent:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # This event auto-inserts after firms_pay_dividends

    With pipeline hook (inserted before another event):

    >>> @event(before="firms_adjust_price")
    ... class PrePricingCheck:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # This event auto-inserts before firms_adjust_price

    With pipeline hook (replaces another event):

    >>> @event(replace="firms_decide_desired_production")
    ... class CustomProductionRule:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # This event replaces the original

    Notes
    -----
    Pipeline hooks are applied automatically when ``Pipeline.from_yaml()``
    or ``Pipeline.from_event_list()`` is called. Events must be imported
    before ``Simulation.init()`` for hooks to take effect.

    Multiple events can target the same hook point. They are inserted in
    registration order (first registered = closest to target event).

    See Also
    --------
    :class:`~bamengine.core.Pipeline` : Pipeline that applies hooks
    :func:`~bamengine.core.registry.register_event_hook` : Low-level hook registration
    """
    # Import here to avoid circular imports
    from bamengine.core.event import Event
    from bamengine.core.registry import register_event_hook

    # Validate hook parameters: at most one hook type allowed
    hooks_specified = sum(x is not None for x in [after, before, replace])
    if hooks_specified > 1:
        raise ValueError(
            "Only one of 'after', 'before', or 'replace' can be specified. "
            f"Got: after={after!r}, before={before!r}, replace={replace!r}"
        )

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
                "__doc__": cls.__doc__,  # Preserve docstring for Sphinx
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

        # Register pipeline hook if specified
        # cls.name is now set (either custom or auto-generated from __init_subclass__)
        if hooks_specified > 0:
            register_event_hook(
                cls.name,  # type: ignore[attr-defined]
                after=after,
                before=before,
                replace=replace,
            )

        return cls

    # Support both @event and @event() syntax
    if cls is None:
        # Called with arguments: @event(name="...")
        return decorator
    else:
        # Called without arguments: @event
        return decorator(cls)


def relationship(
    cls: type[T] | None = None,
    *,
    source: type[Role] | None = None,
    target: type[Role] | None = None,
    cardinality: Literal["many-to-many", "one-to-many", "many-to-one"] = "many-to-many",
    name: str | None = None,
    **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator to define a Relationship with automatic inheritance and registration.

    This decorator dramatically simplifies Relationship definition by:
    1. Making the class inherit from Relationship (if not already)
    2. Applying @dataclass(slots=True)
    3. Setting source/target roles as class variables
    4. Setting cardinality
    5. Registering the relationship in the global registry

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    source : type[Role], optional
        Source role type (e.g., Borrower)
    target : type[Role], optional
        Target role type (e.g., Lender)
    cardinality : {"many-to-many", "one-to-many", "many-to-one"}, default "many-to-many"
        Relationship cardinality
    name : str, optional
        Custom name for the relationship. If None, uses the class name.
    **dataclass_kwargs : Any
        Additional keyword arguments to pass to @dataclass.
        By default, slots=True is set.

    Returns
    -------
    type | Callable
        The decorated class or a decorator function

    Examples
    --------
    Simplest usage:

    >>> from bamengine import get_role
    >>> from bamengine.typing import Float, Int
    >>> @relationship(source=get_role("Borrower"), target=get_role("Lender"))
    ... class LoanBook:
    ...     principal: Float
    ...     rate: Float
    ...     interest: Float
    ...     debt: Float

    With custom name and cardinality:

    >>> @relationship(
    ...     source=get_role("Worker"),
    ...     target=get_role("Employer"),
    ...     cardinality="many-to-many",
    ...     name="MultiJobEmployment",
    ... )
    ... class Employment:
    ...     wage: Float
    ...     contract_duration: Int
    """
    # Import here to avoid circular imports
    from bamengine.core import Relationship

    # Set default slots=True unless explicitly overridden
    dataclass_kwargs.setdefault("slots", True)

    def decorator(cls: type[T]) -> type[T]:
        # Check if cls already inherits from Relationship
        if not issubclass(cls, Relationship):
            # Dynamically create a new class that inherits from Relationship
            # Copy annotations and methods from the original class
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,  # Preserve docstring for Sphinx
                "__annotations__": getattr(cls, "__annotations__", {}),
            }
            # Copy methods and class attributes
            for attr_name in dir(cls):
                if not attr_name.startswith("__"):
                    namespace[attr_name] = getattr(cls, attr_name)

            cls = type(cls.__name__, (Relationship,), namespace)

        # Set metadata as class variables
        cls.source_role = source  # type: ignore[attr-defined]
        cls.target_role = target  # type: ignore[attr-defined]
        cls.cardinality = cardinality  # type: ignore[attr-defined]

        # Set custom name BEFORE applying dataclass
        # This ensures __init_subclass__ sees the correct name
        if name is not None:
            cls.name = name  # type: ignore[attr-defined]

        # Apply dataclass decorator
        cls = dataclass(**dataclass_kwargs)(cls)

        return cls

    # Support both @relationship and @relationship() syntax
    if cls is None:
        # Called with arguments: @relationship(source=..., target=...)
        return decorator
    else:
        # Called without arguments: @relationship (not typical for relationships)
        return decorator(cls)
