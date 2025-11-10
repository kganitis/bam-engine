"""Role (Component) base class definition."""

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

    Each array index corresponds to an agent ID. For example, if there are
    100 firms, `Producer.price` would be a 1D NumPy array of length 100.

    Design Guidelines
    -----------------
    - All state variables should be NumPy arrays (Float1D, Int1D, Bool1D)
    - Scratch buffers (optional fields) can be added for performance
    - Avoid methods that mutate state; use system functions instead
    - Use @role decorator to define and register new roles

    Notes
    -----
    The __init_subclass__ hook automatically:
    - Registers roles in the global registry
    - Sets name to the Role class name
    """

    # Class variable to store role name (set automatically by __init_subclass__)
    name: ClassVar[str | None] = None

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        """
        Auto-register Role subclasses in the global registry.

        This hook is called automatically when a class inherits from Role.

        Parameters
        ----------
        name : str, optional
            Custom name for the role. If not provided, uses the class name.
        **kwargs
            Additional keyword arguments passed to parent __init_subclass__.
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
        """Provide informative repr showing role name and field count."""
        fields = getattr(self, "__dataclass_fields__", {})
        role_name = self.name or self.__class__.__name__
        return f"{role_name}(fields={len(fields)})"
