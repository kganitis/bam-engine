from typing import Any, TypeAlias

import numpy as np

FloatA: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
IntA: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]
BoolA: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
IdxA: TypeAlias = np.ndarray[Any, np.dtype[np.intp]]

__all__ = ["FloatA", "IntA", "BoolA", "IdxA"]

# ------------------------------------------------------------------ #
# Back‑compat shim – will disappear in v0.2                          #
# ------------------------------------------------------------------ #
from numpy.typing import NDArray as _NDArray  # noqa: E402

NDArray = _NDArray  # type: ignore  # pragma: no cover
