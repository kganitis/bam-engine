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

T = TypeVar("T")


def _inject_base_class(cls: type, base_cls: type) -> type:
    """Create a new class inheriting from *base_cls*, copying *cls*'s namespace.

    Used by the ``role``, ``event`` and ``relationship`` decorators when the
    decorated class does not already inherit from the required base.  The copy
    ensures ``@dataclass(slots=True)`` works without multiple-inheritance issues.
    """
    namespace = {
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
        "__annotations__": getattr(cls, "__annotations__", {}),
    }
    for attr_name in dir(cls):
        if not attr_name.startswith("__"):
            namespace[attr_name] = getattr(cls, attr_name)
    return type(cls.__name__, (base_cls,), namespace)


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
        if not issubclass(cls, Role):
            cls = _inject_base_class(cls, Role)

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
    4. Optionally storing pipeline hook metadata for explicit positioning

    Parameters
    ----------
    cls : type | None
        The class to decorate (provided automatically when used without parens)
    name : str | None
        Optional custom name for the event. If None, uses class name (snake_case).
    after : str | None
        Insert this event immediately after the target event in the pipeline.
        Hooks are applied explicitly via ``sim.use_events()`` or ``pipeline.apply_hooks()``.
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
    ...         pass  # Applied via sim.use_events(MyCustomEvent)

    With pipeline hook (inserted before another event):

    >>> @event(before="firms_adjust_price")
    ... class PrePricingCheck:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # Applied via sim.use_events(PrePricingCheck)

    With pipeline hook (replaces another event):

    >>> @event(replace="firms_decide_desired_production")
    ... class CustomProductionRule:
    ...     def execute(self, sim: Simulation) -> None:
    ...         pass  # This event replaces the original

    Notes
    -----
    Pipeline hooks are stored as class attributes (``_hook_after``,
    ``_hook_before``, ``_hook_replace``) and applied explicitly via
    ``sim.use_events()`` or ``pipeline.apply_hooks()``.

    Multiple events can target the same hook point. They are inserted in
    the order passed to ``apply_hooks()`` (first = closest to target event).

    See Also
    --------
    :class:`~bamengine.core.Pipeline` : Pipeline that applies hooks
    :meth:`~bamengine.simulation.Simulation.use_events` : Apply hooks to simulation
    """
    # Import here to avoid circular imports
    from bamengine.core.event import Event

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
        if not issubclass(cls, Event):
            cls = _inject_base_class(cls, Event)

        # Set custom name BEFORE applying dataclass
        # This ensures __init_subclass__ sees the correct name
        if name is not None:
            cls.name = name  # type: ignore[attr-defined]

        # Apply dataclass decorator
        cls = dataclass(**dataclass_kwargs)(cls)

        # Store pipeline hook metadata on the class itself
        # cls.name is now set (either custom or auto-generated from __init_subclass__)
        if hooks_specified > 0:
            cls._hook_after = after  # type: ignore[attr-defined]
            cls._hook_before = before  # type: ignore[attr-defined]
            cls._hook_replace = replace  # type: ignore[attr-defined]

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
        if not issubclass(cls, Relationship):
            cls = _inject_base_class(cls, Relationship)

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
