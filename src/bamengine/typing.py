# src/bamengine/typing.py
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Float1D: TypeAlias = NDArray[np.float64]
Int1D: TypeAlias = NDArray[np.int64]
Bool1D: TypeAlias = NDArray[np.bool_]
Idx1D: TypeAlias = NDArray[np.intp]

Float2D: TypeAlias = NDArray[np.float64]
Int2D: TypeAlias = NDArray[np.int64]
Idx2D: TypeAlias = NDArray[np.intp]

FloatA = Float1D
IntA = Int1D
BoolA = Bool1D
IdxA = Idx1D

__all__ = [
    "FloatA",
    "IntA",
    "BoolA",
    "IdxA",
    "Float1D",
    "Int1D",
    "Bool1D",
    "Idx1D",
    "Float2D",
    "Int2D",
    "Idx2D",
]
