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
    - Use `slots=True` for memory efficiency
    - All state variables should be NumPy arrays (Float1D, Int1D, Bool1D)
    - Scratch buffers (optional fields) can be added for performance
    - Avoid methods that mutate state; use system functions instead

    Examples
    --------
    >>> from bamengine.typing import Float1D
    >>> import numpy as np
    >>>
    >>> @dataclass(slots=True)
    >>> class Producer(Role):
    ...     '''Producer role for firms.'''
    ...     price: Float1D
    ...     production: Float1D
    ...     inventory: Float1D
    >>>
    >>> # Create role for 10 firms
    >>> prod = Producer(
    ...     price=np.ones(10) * 1.5,
    ...     production=np.ones(10) * 4.0,
    ...     inventory=np.zeros(10),
    ... )

    Notes
    -----
    This is an abstract base class. All concrete roles should inherit from it
    and be registered using the `@role()` decorator.
    """

    # Class variable to store role name (set by decorator or auto-generated)
    _role_name: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-set _role_name to class name if not explicitly set by decorator."""
        # Call parent __init_subclass__ if it exists
        try:
            super().__init_subclass__(**kwargs)
        except TypeError:
            # Handle edge case with dataclass decorator
            pass

        # Only set if it hasn't been set yet (avoid overwriting decorator's value)
        if not hasattr(cls, "_role_name") or cls._role_name is None:
            cls._role_name = cls.__name__

    def __repr__(self) -> str:
        """Provide informative repr showing role name and field count."""
        fields = getattr(self, "__dataclass_fields__", {})
        role_name = self._role_name or self.__class__.__name__
        return f"{role_name}(fields={len(fields)})"
