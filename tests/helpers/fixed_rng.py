from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

_IntArray = NDArray[np.int64]


# Deterministic stub RNG for unit-tests
class FixedRNG:
    """
    Tiny deterministic stand-in for `numpy.random.Generator`.

    Only the subset of the NumPy API required by the production code is
    implemented:

    * `integers`
    * `choice`

    The sequence of pseudo-random numbers is taken from the *fixed* array
    supplied at construction, so results are completely reproducible.
    """

    _buffer: NDArray[np.int64]
    _cursor: int

    def __init__(self, data: NDArray[np.int64]) -> None:
        if data.ndim == 0:
            data = data.reshape(1)
        self._buffer = data.copy()
        self._cursor = 0

    # helpers
    def _next_ints(self, n: int) -> NDArray[np.int64]:
        """Return *n* ints from the internal buffer (wrap-around safe)."""
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if self._cursor + n > self._buffer.size:
            raise RuntimeError("FixedRNG exhausted – enlarge the seed vector")
        out = self._buffer[self._cursor: self._cursor + n]
        self._cursor += n
        return out

    # public API subset
    # noinspection PyTypeHints
    def integers(  # noqa: D401  (matching numpy.signature)
        self,
        low: int | np.int64,
        high: int | None = None,
        size: int | Tuple[int, ...] | None = None,
        dtype: type[np.int64] | np.dtype[np.int64] = np.int64,
        endpoint: bool = False,  # kept for signature parity
    ) -> NDArray[np.int64]:
        """
        Deterministic stand-in for `Generator.integers`.

        *Ignores* the `low`/`high` range – values come straight from the
        fixed buffer, cast to the requested dtype and reshaped to *size*.
        """
        if size is None:
            size = 1
        n = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        out = self._next_ints(n).astype(dtype, copy=False)
        return out.reshape(size)

    def choice(
        self,
        a: Sequence[int] | _IntArray,
        size: int | Tuple[int, ...] | None = None,
        replace: bool = True,
        p: Any | None = None,
        axis: int = 0,
    ) -> _IntArray:
        pool: _IntArray = np.asarray(a, dtype=np.int64)
        if pool.size == 0:
            raise ValueError("choice() called with an empty population")

        if size is None:
            size = 1
        n = int(np.prod(size)) if isinstance(size, tuple) else int(size)

        if not replace and n > pool.size:
            raise ValueError(
                "Cannot take a larger sample than population " "when 'replace=False'"
            )

        # deterministic index stream
        idx_stream = self._next_ints(n if replace else pool.size) % pool.size

        if replace:
            idx = idx_stream  # may repeat indices
        else:
            # take the *first* n distinct indices in original order
            idx_stream_flat = np.asarray(idx_stream, dtype=np.int64).ravel()

            seen: set[int] = set()
            uniq_idx: list[int] = []

            for j in idx_stream_flat:
                j_int = int(j.item())  # robust scalar extract
                if j_int not in seen:
                    seen.add(j_int)
                    uniq_idx.append(j_int)
                    if len(uniq_idx) == n:
                        break

            #  fill-up if duplicates left us short
            if len(uniq_idx) < n:  # ← NEW
                # append the smallest still-unused indices
                for extra in range(pool.size):
                    if extra not in seen:
                        uniq_idx.append(extra)
                        if len(uniq_idx) == n:
                            break

            idx = np.asarray(uniq_idx, dtype=np.int64)

        picked: _IntArray = pool[idx]
        return picked.reshape(size)
