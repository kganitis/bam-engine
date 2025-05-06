from typing import Any, TypeAlias

import numpy as np

FloatA: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
IntA: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]
BoolA: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
IdxA: TypeAlias = np.ndarray[Any, np.dtype[np.intp]]

__all__ = ["FloatA", "IntA", "BoolA", "IdxA"]
