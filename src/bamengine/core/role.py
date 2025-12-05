"""
Role (Component) base class definition.

This module defines the Role base class, the fundamental building block
of the BAM-ECS architecture. Roles are dataclasses containing NumPy arrays
that represent specific aspects of agent behavior.

Design Notes
------------
- All Roles auto-register via __init_subclass__ hook
- Each Role has a name (ClassVar) set automatically
- Roles should be immutable containers; use system functions for mutations
- NumPy array fields enable vectorized operations across all agents

Auto-Registration
-----------------
When a class inherits from Role, __init_subclass__ automatically:

1. Sets the role name (cls.name) to class name if not provided
2. Registers the role class in the global _ROLE_REGISTRY
3. Makes the role retrievable via get_role(name)

This eliminates manual registration boilerplate and ensures all
roles are discoverable at runtime.

See Also
--------
:class:`~bamengine.core.Event` : Base class for events (systems) in BAM-ECS
:mod:`bamengine.core.registry` : Global role and event lookup system
:func:`bamengine.role` : Simplified decorator for defining roles
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(slots=True)
class Role(ABC):
    """
    Base class for all roles (components) in the BAM-ECS architecture.

    A Role is a dataclass containing NumPy arrays representing state variables
    for a specific aspect of agent behavior (e.g., Producer, Worker, Lender).
    Each array index corresponds to an agent ID.

    Class Attributes
    ----------------
    name : str
        Role name, automatically set to class name if not provided.

    Design Guidelines
    -----------------
    - All state variables should be NumPy arrays (Float, Int, Bool types)
    - Scratch buffers (optional fields) can be added for performance
    - Avoid methods that mutate state; use system functions instead
    - Use @role decorator for simplified role definition

    Examples
    --------
    Define a role using traditional syntax:

    >>> from dataclasses import dataclass
    >>> from bamengine.core import Role
    >>> from bamengine import Float
    >>> import numpy as np
    >>>
    >>> @dataclass(slots=True)
    ... class Producer(Role):
    ...     price: Float
    ...     production: Float
    >>> # Producer is now auto-registered!

    Define a role using the @role decorator (simplified):

    >>> from bamengine import role, Float
    >>>
    >>> @role
    ... class MyRole:
    ...     value: Float
    >>> # MyRole is now auto-registered!

    Access registered roles:

    >>> from bamengine.core.registry import get_role
    >>> prod_cls = get_role("Producer")
    >>> instance = prod_cls(
    ...     price=np.array([1.0, 1.2]), production=np.array([100.0, 120.0])
    ... )

    Notes
    -----
    The __init_subclass__ hook automatically registers roles in the global
    registry and sets the role name. This eliminates manual registration
    boilerplate.

    For example, if there are 100 firms, `Producer.price` would be a 1D NumPy
    array of length 100, where index i corresponds to firm i.

    See Also
    --------
    :class:`~bamengine.core.Event` : Base class for events (systems) in BAM-ECS
    :func:`bamengine.role` : Simplified @role decorator
    :func:`bamengine.core.registry.get_role` : Retrieve role by name
    """

    # Class variable to store role name (set automatically by __init_subclass__)
    name: ClassVar[str | None] = None

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        """
        Auto-register Role subclasses in the global registry.

        This hook is called automatically when a class inherits from Role.
        It handles role registration and name assignment.

        Parameters
        ----------
        name : str, optional
            Custom name for the role. If not provided, uses the class name.
        **kwargs : Any
            Additional keyword arguments passed to parent __init_subclass__.

        Notes
        -----
        This method is called twice when using @dataclass(slots=True):
        once during class definition and once when dataclass creates the
        final class. The name preservation logic handles this correctly.

        Examples
        --------
        Normal usage (automatic):

        >>> from bamengine.typing import Float
        >>> from dataclasses import dataclass
        >>> @dataclass(slots=True)
        ... class MyRole(Role):
        ...     value: Float

        Custom name:

        >>> @dataclass(slots=True)
        ... class MyRole(Role, name="CustomName"):
        ...     value: Float
        """
        super(Role, cls).__init_subclass__(**kwargs)

        # Use custom name if provided, otherwise preserve existing name or use cls name
        # This handles the case where @dataclass(slots=True) creates a new class
        # and triggers __init_subclass__ a second time without the custom name
        if name is not None:
            cls.name = name
        elif cls.name is None:
            cls.name = cls.__name__

        # Auto-register in global registry
        from bamengine.core.registry import _ROLE_REGISTRY

        _ROLE_REGISTRY[cls.name] = cls

    def __repr__(self) -> str:
        """
        Provide informative repr showing role name and field count.

        Returns
        -------
        str
            String representation in format "RoleName(fields=N)".

        Examples
        --------
        >>> from bamengine.roles import Producer
        >>> import numpy as np
        >>> prod = Producer(
        ...     price=np.array([1.0]),
        ...     production=np.array([100.0]),
        ...     inventory=np.array([0.0]),
        ...     labor_productivity=np.array([2.0]),
        ... )
        >>> repr(prod)
        'Producer(fields=4)'
        """
        fields = getattr(self, "__dataclass_fields__", {})
        role_name = self.name or self.__class__.__name__
        return f"{role_name}(fields={len(fields)})"
