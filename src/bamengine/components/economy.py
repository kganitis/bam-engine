# src/bamengine/components/economy.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamengine.typing import Float1D, Idx1D, Int1D, Bool1D


@dataclass(slots=True)
class Economy:
    """
    Pure *state* container for economy-wide parameters and transient lists.
    """

    # ── policy / structural scalars ──────────────────────────────────────
    avg_mkt_price: float
    min_wage: float
    min_wage_rev_period: int
    r_bar: float  # base interest-rate
    v: float  # capital-adequacy coefficient

    # ── time-series ──────────────────────────────────────────────────────
    avg_mkt_price_history: Float1D  # shape  (t+1,)
    unemp_rate_history: Float1D  # shape  (t+1,)
    inflation_history: Float1D  # shape  (t+1,)

    # ── transient exit lists (flushed each Entry event) ──────────────────
    exiting_firms: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))
    exiting_banks: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))

    # End simulation flag
    destroyed: bool = False


# TODO Move LoanBook and its tests to a separate module
#  - also consider moving the methods into systems
@dataclass(slots=True)
class LoanBook:
    # noinspection PyUnresolvedReferences
    """
    Edge-list ledger for storing and managing *active* loans.

    This structure maintains a sparse representation of loan contracts
    using a **Coordinate List (COO) format**. It efficiently tracks lending
    relationships without the need for a dense `(N_borrowers × N_lenders)` matrix,
    reducing memory consumption and enhancing vectorized operations.

    The `LoanBook` is designed to grow automatically, with amortized O(1) complexity
    for append operations, avoiding per-step allocations during hot loops.

    Attributes
    ----------
    borrower : Int1D
        Array of indices (`int64`) representing borrowers.
        Size: `M`, where `M` is the number of active loans.
    lender : Int1D
        Array of indices (`int64`) representing lenders.
        Size: `M`.
    principal : Float1D
        Array of principal amounts (`float64`) for each loan.
        This is the original loan amount at the time of signing.
    rate : Float1D
        Array of contractual interest rates (`float64`) for each loan.
    interest : Float1D
        Cached interest amounts (`float64`) calculated as `rate * principal`.
        This enables O(1) aggregation without recomputation.
    debt : Float1D
        Cached total debt amounts (`float64`) calculated as `principal * (1 + rate)`.
    capacity : int, optional
        The current physical storage capacity of the ledger (default is 128).
        This represents the allocated space, not the number of active rows.
    size : int, optional
        The number of currently active loans in the ledger (default is 0).

    Notes
    -----
    The `LoanBook` is structured as a *sparse edge-list*, where only active
    loan relationships are recorded. The edge list grows as new loans are issued
    and is only resized when capacity is exhausted, ensuring optimal memory usage.

    The six columns are 1-D NumPy arrays of equal length `M`, where `M` is the
    number of active loans. Operations such as aggregation and updates are
    efficiently vectorized. For example, to sum all debt per borrower:

        >>> borrower_debt = np.zeros(N)
        >>> np.add.at(borrower_debt, lb.borrower, lb.debt)

    Advantages of the edge-list design:
    - **Sparse Representation:** Memory usage scales with the number of active loans,
    not the full firm-bank matrix.
    - **Vectorized Operations:** Supports fast aggregations using `np.bincount` and
    `np.ufunc.at`.
    - **Append-Only Write Pattern:** New loans are appended in constant time,
    avoiding cache-thrashing.
    - **Dynamic Resize:** Automatically doubles capacity when full,
    ensuring amortized O(1) append complexity.
    """

    borrower: Int1D = field(default_factory=lambda: np.empty(0, np.int64))
    lender: Int1D = field(default_factory=lambda: np.empty(0, np.int64))
    principal: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    rate: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    interest: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    debt: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    capacity: int = 128  # current physical length
    size: int = 0  # number of *filled* rows

    def _ensure_capacity(self, extra: int) -> None:
        needed = self.size + extra
        if needed <= self.capacity:
            new_cap = self.capacity
        else:
            new_cap = max(self.capacity * 2, needed, 128)

        for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
            arr = getattr(self, name)
            if arr.size != new_cap:  # only when really needed
                new_arr = np.resize(arr, new_cap)
                setattr(self, name, new_arr)

        self.capacity = new_cap
        # sanity
        assert all(
            getattr(self, n).size == new_cap
            for n in ("borrower", "lender", "principal", "rate", "interest", "debt")
        )

    # ------------------------------------------------------------------ #
    #   API                                                              #
    # ------------------------------------------------------------------ #
    def debt_per_borrower(self, n_borrowers: int) -> Float1D:
        out = np.zeros(n_borrowers, dtype=np.float64)
        np.add.at(out, self.borrower[: self.size], self.debt[: self.size])
        return out

    def interest_per_borrower(self, n_borrowers: int) -> Float1D:
        out = np.zeros(n_borrowers, dtype=np.float64)
        np.add.at(out, self.borrower[: self.size], self.interest[: self.size])
        return out

    def append_loans_for_lender(
        self,
        lender_idx: np.intp,
        borrower_indices: Idx1D,
        amount: Float1D,
        rate: Float1D,
    ) -> None:
        self._ensure_capacity(amount.size)
        start, stop = self.size, self.size + amount.size

        self.borrower[start:stop] = borrower_indices
        self.lender[start:stop] = lender_idx  # ← scalar broadcast
        self.principal[start:stop] = amount
        self.rate[start:stop] = rate
        self.interest[start:stop] = amount * rate
        self.debt[start:stop] = amount * (1.0 + rate)
        self.size = stop

    def drop_rows(self, rows_mask: Bool1D) -> int:
        """
        Hard-delete the rows where *rows_mask* is True and compact the edge list
        **in-place**.

        Parameters
        ----------
        rows_mask : 1-D bool[>= self.size]
            Mask over the *current* part of every column.
            `True` → row will be removed.

        Returns
        -------
        removed: int
            How many rows were deleted.
        """
        if self.size == 0 or not rows_mask.any():
            return 0  # nothing to do

        self._ensure_capacity(0)  # no growth, only normalisation

        keep = ~rows_mask[: self.size]  # rows to keep
        new_size = int(keep.sum())

        if new_size < self.size:  # only touch memory when shrinking
            for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
                col = getattr(self, name)
                col[:new_size] = col[: self.size][keep]

            removed = self.size - new_size
            self.size = new_size
            return removed

        return 0

    def purge_borrowers(self, borrower_ids: Idx1D) -> int:
        """
        Remove every loan whose *borrower* is in *borrower_ids*.

        Returns
        -------
        int
            Number of rows removed.
        """
        mask = np.isin(self.borrower[: self.size], borrower_ids)
        return self.drop_rows(mask)

    def purge_lenders(self, lender_ids: Idx1D) -> int:
        """
        Delete every loan whose *lender* is in *lender_ids*.

        Returns
        -------
        int
            Number of rows removed.
        """
        mask = np.isin(self.lender[: self.size], lender_ids)
        return self.drop_rows(mask)
