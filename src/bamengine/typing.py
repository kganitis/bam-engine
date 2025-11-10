"""
Type aliases for BAM Engine.

Provides both internal types (Float1D, Int1D, etc.) and user-friendly
type aliases (Float, Int, etc.) for defining custom roles.

Examples
--------
Define a custom role using user-friendly types:

>>> from dataclasses import dataclass
>>> from bamengine import Role
>>> from bamengine.typing import Float, Int, Bool, Agent
>>>
>>> @dataclass(slots=True)
>>> class Inventory(Role):
...     goods_on_hand: Float
...     reorder_point: Float
...     supplier_id: Agent
...     days_until_delivery: Int
...     needs_reorder: Bool
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# === Internal Type Aliases (precise numpy types) ===

Float1D: TypeAlias = NDArray[np.float64]
Int1D: TypeAlias = NDArray[np.int64]
Bool1D: TypeAlias = NDArray[np.bool_]
Idx1D: TypeAlias = NDArray[np.intp]

Float2D: TypeAlias = NDArray[np.float64]
Int2D: TypeAlias = NDArray[np.int64]
Idx2D: TypeAlias = NDArray[np.intp]

# === Legacy Aliases (backward compatibility) ===

FloatA = Float1D
IntA = Int1D
BoolA = Bool1D
IdxA = Idx1D

# === User-Friendly Type Aliases ===

Float = Float1D
"""Array of floating-point values (prices, quantities, rates, etc.)."""

Int = Int1D
"""Array of integer values (counts, periods, etc.)."""

Bool = Bool1D
"""Array of boolean values (flags, conditions, etc.)."""

Agent = Idx1D
"""Array of agent IDs (integer indices, -1 for unassigned)."""

__all__ = [
    # User-friendly (recommended for custom roles)
    "Float",
    "Int",
    "Bool",
    "Agent",
    # Internal (used in bamengine code)
    "Float1D",
    "Int1D",
    "Bool1D",
    "Idx1D",
    "Float2D",
    "Int2D",
    "Idx2D",
    # Legacy (backward compatibility)
    "FloatA",
    "IntA",
    "BoolA",
    "IdxA",
]
