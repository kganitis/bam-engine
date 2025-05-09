from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Float1D: TypeAlias = NDArray[np.float64]
Int1D: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]
Bool1D: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
Idx1D: TypeAlias = np.ndarray[Any, np.dtype[np.intp]]

FloatA = Float1D
IntA = Int1D
BoolA = Bool1D
IdxA = Idx1D

__all__ = ["FloatA", "IntA", "BoolA", "IdxA", "Float1D", "Int1D", "Bool1D", "Idx1D"]

# ------------------------------------------------------------------ #
# Back‑compat shim – will disappear in v0.2                          #
# ------------------------------------------------------------------ #
from numpy.typing import NDArray as _NDArray  # noqa: E402

NDArray = _NDArray  # type: ignore  # pragma: no cover
