"""
Type aliases for BAM Engine.

Provides both internal types (Float1D, Int1D, etc.) used in bamengine code
and user-friendly type aliases (Float, Int, etc.) for defining custom roles.

Design Notes
------------
- **Internal types** (Float1D, Int1D, Bool1D, Idx1D): Used in bamengine roles
- **User-friendly types** (Float, Int, Bool, Agent): Recommended for custom roles
- All types are actually NDArray with specific dtypes for type safety
- AgentId type uses np.intp for platform-independent integer indexing

Type Mapping
------------
- `Float` → `Float1D` → `NDArray[np.float64]` (prices, quantities, rates)
- `Int` → `Int1D` → `NDArray[np.int64]` (counts, periods, durations)
- `Bool` → `Bool1D` → `NDArray[np.bool_]` (flags, conditions, masks)
- `Agent` → `Idx1D` → `NDArray[np.intp]` (agent IDs, -1 for unassigned)

Examples
--------
Define a custom role using user-friendly types:

>>> from bamengine import role, Float, Int, Bool, Agent
>>>
>>> @role
... class Inventory:
...     goods_on_hand: Float
...     reorder_point: Float
...     supplier_id: Agent
...     days_until_delivery: Int
...     needs_reorder: Bool

Use in bamengine internal code (Float1D for precision):

>>> from dataclasses import dataclass
>>> from bamengine.core import Role
>>> from bamengine.typing import Float1D, Int1D, Bool1D
>>>
>>> @dataclass(slots=True)
... class Producer(Role):
...     price: Float1D
...     production: Float1D
...     inventory: Float1D
...     labor_productivity: Float1D

See Also
--------
bamengine.role : Role base class for defining components
bamengine.ops : Operations on these array types
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

"""Array of floating-point values (prices, quantities, rates, etc.)."""
Float = Float1D

"""Array of integer values (counts, periods, etc.)."""
Int = Int1D

"""Array of boolean values (flags, conditions, etc.)."""
Bool = Bool1D

"""Array of agent IDs (integer indices, -1 for unassigned)."""
Agent = Idx1D

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
