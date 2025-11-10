from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamengine.core import relationship
from bamengine.typing import Bool1D, Float1D, Idx1D, Int1D


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

    # Termination flag
    destroyed: bool = False


# Avoid circular imports by importing roles here (after Economy definition)
# This allows LoanBook to reference Borrower and Lender types
def _get_borrower_role() -> type:
    """Lazy import to avoid circular dependency."""
    from bamengine.roles.borrower import Borrower

    return Borrower


def _get_lender_role() -> type:
    """Lazy import to avoid circular dependency."""
    from bamengine.roles.lender import Lender

    return Lender


# Use @relationship decorator to define LoanBook as a Relationship between
# Borrower (source) and Lender (target) roles
# Note: We use lazy imports above to avoid circular import issues
@relationship(  # type: ignore[operator]
    source=_get_borrower_role(),
    target=_get_lender_role(),
    cardinality="many-to-many",
    name="LoanBook",
)
class LoanBook:
    # noinspection PyUnresolvedReferences
    """
    Edge-list ledger for storing and managing *active* loans.

    This is a Relationship between Borrower (source) and Lender (target) roles,
    maintaining a sparse representation of loan contracts using **Coordinate List
    (COO) format**. It efficiently tracks lending relationships without the need
    for a dense `(N_borrowers × N_lenders)` matrix, reducing memory consumption
    and enhancing vectorized operations.

    Inherits from Relationship base class, which provides:
    - source_ids (borrower indices)
    - target_ids (lender indices)
    - size (number of active loans)
    - capacity (allocated storage)

    Edge Components (per loan)
    ---------------------------
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

    Notes
    -----
    The `LoanBook` is structured as a *sparse edge-list*, where only active
    loan relationships are recorded. The edge list grows as new loans are issued
    and is only resized when capacity is exhausted, ensuring optimal memory usage.

    Operations such as aggregation and updates are efficiently vectorized.
    For example, to sum all debt per borrower:

        >>> borrower_debt = lb.aggregate_by_source(lb.debt, func="sum", n_sources=N)

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

    # Edge-specific components (loan data per edge)
    principal: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    rate: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    interest: Float1D = field(default_factory=lambda: np.empty(0, np.float64))
    debt: Float1D = field(default_factory=lambda: np.empty(0, np.float64))

    # Default values for base class fields (from Relationship)
    # These must come after edge components due to dataclass field ordering
    source_ids: Idx1D = field(default_factory=lambda: np.empty(0, np.int64))
    target_ids: Idx1D = field(default_factory=lambda: np.empty(0, np.int64))
    size: int = 0
    capacity: int = 128

    # Backward compatibility aliases for existing code
    @property
    def borrower(self) -> Int1D:
        """Alias for source_ids (borrower indices)."""
        return self.source_ids

    @borrower.setter
    def borrower(self, value: Int1D) -> None:
        """Alias setter for source_ids."""
        self.source_ids = value

    @property
    def lender(self) -> Int1D:
        """Alias for target_ids (lender indices)."""
        return self.target_ids

    @lender.setter
    def lender(self, value: Int1D) -> None:
        """Alias setter for target_ids."""
        self.target_ids = value

    def _ensure_capacity(self, extra: int) -> None:
        """
        Ensure capacity for additional edges, resizing arrays if needed.

        Parameters
        ----------
        extra : int
            Number of additional edges to accommodate
        """
        needed = self.size + extra
        if needed <= self.capacity:
            new_cap = self.capacity
        else:
            new_cap = max(self.capacity * 2, needed, 128)

        # Resize base class arrays (source_ids, target_ids)
        for name in ("source_ids", "target_ids"):
            arr = getattr(self, name)
            if arr.size != new_cap:  # only when really needed
                new_arr = np.resize(arr, new_cap)
                setattr(self, name, new_arr)

        # Resize edge-specific component arrays
        for name in ("principal", "rate", "interest", "debt"):
            arr = getattr(self, name)
            if arr.size != new_cap:  # only when really needed
                new_arr = np.resize(arr, new_cap)
                setattr(self, name, new_arr)

        self.capacity = new_cap
        # sanity check
        assert all(
            getattr(self, n).size == new_cap
            for n in (
                "source_ids",
                "target_ids",
                "principal",
                "rate",
                "interest",
                "debt",
            )
        )

    # ------------------------------------------------------------------ #
    #   API (using Relationship base methods)                           #
    # ------------------------------------------------------------------ #
    def debt_per_borrower(self, n_borrowers: int) -> Float1D:
        """
        Aggregate total debt per borrower.

        Parameters
        ----------
        n_borrowers : int
            Number of borrowers in the simulation

        Returns
        -------
        Float1D
            Array of total debt per borrower
        """
        return self.aggregate_by_source(self.debt, func="sum", n_sources=n_borrowers)  # type: ignore[no-any-return, attr-defined]

    def interest_per_borrower(self, n_borrowers: int) -> Float1D:
        """
        Aggregate total interest per borrower.

        Parameters
        ----------
        n_borrowers : int
            Number of borrowers in the simulation

        Returns
        -------
        Float1D
            Array of total interest per borrower
        """
        return self.aggregate_by_source(  # type: ignore[no-any-return, attr-defined]
            self.interest, func="sum", n_sources=n_borrowers
        )

    def append_loans_for_lender(
        self,
        lender_idx: np.intp,
        borrower_indices: Idx1D,
        amount: Float1D,
        rate: Float1D,
    ) -> None:
        """
        Append new loans from a specific lender to multiple borrowers.

        Parameters
        ----------
        lender_idx : np.intp
            Index of the lender providing loans
        borrower_indices : Idx1D
            Indices of borrowers receiving loans
        amount : Float1D
            Principal amounts for each loan
        rate : Float1D
            Interest rates for each loan
        """
        self._ensure_capacity(amount.size)
        start, stop = self.size, self.size + amount.size

        # Use base class fields (source_ids, target_ids)
        self.source_ids[start:stop] = borrower_indices
        self.target_ids[start:stop] = lender_idx  # ← scalar broadcast

        # Set edge-specific components
        self.principal[start:stop] = amount
        self.rate[start:stop] = rate
        self.interest[start:stop] = amount * rate
        self.debt[start:stop] = amount * (1.0 + rate)
        self.size = stop

    def drop_rows(self, rows_mask: Bool1D) -> int:
        """
        Hard-delete the rows where *rows_mask* is True and compact the edge list
        **in-place**.

        Overrides Relationship.drop_rows() to also compact loan-specific
        component arrays.

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
            # Compact base class arrays (source_ids, target_ids)
            self.source_ids[:new_size] = self.source_ids[: self.size][keep]
            self.target_ids[:new_size] = self.target_ids[: self.size][keep]

            # Compact edge-specific component arrays
            for name in ("principal", "rate", "interest", "debt"):
                col = getattr(self, name)
                col[:new_size] = col[: self.size][keep]

            removed = self.size - new_size
            self.size = new_size
            return removed

        return 0

    def purge_borrowers(self, borrower_ids: Idx1D) -> int:
        """
        Remove every loan whose *borrower* is in *borrower_ids*.

        Uses Relationship.purge_sources() internally.

        Parameters
        ----------
        borrower_ids : Idx1D
            Array of borrower indices to purge

        Returns
        -------
        int
            Number of rows removed.
        """
        return self.purge_sources(borrower_ids)  # type: ignore[no-any-return, attr-defined]

    def purge_lenders(self, lender_ids: Idx1D) -> int:
        """
        Delete every loan whose *lender* is in *lender_ids*.

        Uses Relationship.purge_targets() internally.

        Parameters
        ----------
        lender_ids : Idx1D
            Array of lender indices to purge

        Returns
        -------
        int
            Number of rows removed.
        """
        return self.purge_targets(lender_ids)  # type: ignore[no-any-return, attr-defined]
