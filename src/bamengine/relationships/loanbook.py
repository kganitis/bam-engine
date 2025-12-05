"""
LoanBook relationship for tracking loans between borrowers and lenders.

This module implements the LoanBook relationship, a many-to-many connection
between Borrower (firms) and Lender (banks) roles. It uses COO (Coordinate List)
sparse format to efficiently store active loan contracts.

Design Philosophy
-----------------
The LoanBook uses a sparse edge-list representation rather than a dense
(n_firms × n_banks) matrix. This design choice provides:

1. **Memory efficiency**: O(active_loans) instead of O(n_firms × n_banks)
2. **Cache-friendly**: Sequential access patterns for vectorized operations
3. **Dynamic growth**: Amortized O(1) append via doubling strategy
4. **Fast aggregation**: Vectorized sums using np.bincount and np.add.at

COO Format Structure
--------------------
The LoanBook stores five parallel arrays:

- source_ids (borrower IDs)
- target_ids (lender IDs)
- principal (loan amounts)
- rate (interest rates)
- interest (cached: rate × principal)
- debt (cached: principal × (1 + rate))

Only the first `size` entries in each array are valid. The remaining entries
up to `capacity` are pre-allocated but unused.

Examples
--------
Create empty LoanBook:

>>> from bamengine.relationships import LoanBook
>>> loans = LoanBook()
>>> loans.size
0
>>> loans.capacity
128

Append loans:

>>> import numpy as np
>>> loans.append_loans_for_lender(
...     lender_idx=0,
...     borrower_indices=np.array([1, 2, 3]),
...     amount=np.array([100.0, 150.0, 200.0]),
...     rate=np.array([0.02, 0.03, 0.02]),
... )
>>> loans.size
3

Aggregate debt by borrower:

>>> debt_per_borrower = loans.debt_per_borrower(n_borrowers=10)
>>> debt_per_borrower.shape
(10,)

Purge loans from bankrupt firms:

>>> bankrupt_firms = np.array([1, 5, 7])
>>> removed = loans.purge_borrowers(bankrupt_firms)
>>> removed  # doctest: +SKIP
1

See Also
--------
:class:`bamengine.core.Relationship` : Base class with query methods
:class:`~bamengine.roles.Borrower` : Source role (firms)
:class:`~bamengine.roles.Lender` : Target role (banks)
"""

from __future__ import annotations

from dataclasses import field

import numpy as np

from bamengine.core import get_role, relationship
from bamengine.typing import Bool1D, Float1D, Idx1D, Int1D


# Use @relationship decorator to define LoanBook as a Relationship between
# Borrower (source) and Lender (target) roles
# Note: We use lazy imports above to avoid circular import issues
@relationship(
    source=get_role("Borrower"),
    target=get_role("Lender"),
    cardinality="many-to-many",
    name="LoanBook",
)
class LoanBook:
    # noinspection PyUnresolvedReferences
    """
    Sparse edge-list ledger for managing active loan contracts.

    LoanBook is a Relationship between Borrower (source/firms) and Lender
    (target/banks), storing loan contracts in COO (Coordinate List) sparse
    format. This avoids the memory overhead of a dense (n_firms × n_banks)
    matrix while enabling efficient vectorized operations.

    Inherits from Relationship base class, which provides:

    - source_ids (borrower indices)
    - target_ids (lender indices)
    - size (number of active loans)
    - capacity (allocated storage)
    - Query methods (query_sources, query_targets)
    - Aggregation methods (aggregate_by_source, aggregate_by_target)
    - Deletion methods (drop_rows, purge_sources, purge_targets)

    Parameters
    ----------
    principal : Float1D
        Loan principal amounts (original loan amounts at signing).
    rate : Float1D
        Contractual interest rates for each loan.
    interest : Float1D
        Cached interest amounts (rate × principal), enables O(1) aggregation.
    debt : Float1D
        Cached total debt (principal × (1 + rate)), enables O(1) aggregation.
    source_ids : Idx1D
        Borrower (firm) IDs for each loan.
    target_ids : Idx1D
        Lender (bank) IDs for each loan.
    size : int
        Number of active loans (valid entries in arrays).
    capacity : int
        Allocated array size (grows via doubling when exceeded).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> loans = sim.loans
    >>> loans.size  # doctest: +SKIP
    45

    Append new loans from bank 0 to firms 1, 2, 3:

    >>> import numpy as np
    >>> loans.append_loans_for_lender(
    ...     lender_idx=0,
    ...     borrower_indices=np.array([1, 2, 3]),
    ...     amount=np.array([100.0, 150.0, 200.0]),
    ...     rate=np.array([0.02, 0.03, 0.02]),
    ... )

    Query loans for specific borrower:

    >>> borrower_id = 5
    >>> loan_mask = loans.source_ids[: loans.size] == borrower_id
    >>> borrower_loans = loans.principal[: loans.size][loan_mask]
    >>> borrower_loans.sum()  # doctest: +SKIP
    250.0

    Aggregate total debt per borrower:

    >>> total_debt = loans.debt_per_borrower(n_borrowers=100)
    >>> total_debt.shape
    (100,)
    >>> total_debt[5]  # doctest: +SKIP
    255.0

    Aggregate total interest per borrower:

    >>> total_interest = loans.interest_per_borrower(n_borrowers=100)
    >>> total_interest[5]  # doctest: +SKIP
    5.0

    Purge loans from bankrupt firms:

    >>> bankrupt = np.array([1, 7, 12])
    >>> removed = loans.purge_borrowers(bankrupt)
    >>> removed  # doctest: +SKIP
    3

    Notes
    -----
    **Memory Efficiency**: For 100 firms, 10 banks, and 50 active loans:

    - Dense matrix: 100 × 10 × 4 fields × 8 bytes = 32,000 bytes
    - Sparse COO: 50 × 6 arrays × 8 bytes = 2,400 bytes (~13x smaller)

    **Performance**: Aggregation operations use vectorized NumPy primitives:

    - debt_per_borrower: O(size) using np.add.at
    - purge_borrowers: O(size) using np.isin and boolean indexing
    - append_loans: O(1) amortized via doubling strategy

    **Backward Compatibility**: The borrower and lender properties provide
    aliases for source_ids and target_ids to maintain compatibility with
    existing code that predates the Relationship abstraction.

    See Also
    --------
    :class:`bamengine.core.Relationship` : Base class with query/aggregation
    :class:`~bamengine.roles.Borrower` : Source role (firms seeking credit)
    :class:`~bamengine.roles.Lender` : Target role (banks providing credit)
    :mod:`bamengine.events._internal.credit_market` : Loan creation logic
    :mod:`bamengine.events._internal.revenue` : Debt repayment logic
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
        """
        Alias for source_ids (borrower firm indices).

        Provides backward compatibility with code written before the
        Relationship abstraction. New code should use source_ids directly.

        Returns
        -------
        Int1D
            Array of borrower indices (same as source_ids).

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> loans = LoanBook()
        >>> loans.borrower is loans.source_ids
        True
        """
        return self.source_ids

    @borrower.setter
    def borrower(self, value: Int1D) -> None:
        """
        Setter for borrower alias (updates source_ids).

        Parameters
        ----------
        value : Int1D
            New borrower indices array.
        """
        self.source_ids = value

    @property
    def lender(self) -> Int1D:
        """
        Alias for target_ids (lender bank indices).

        Provides backward compatibility with code written before the
        Relationship abstraction. New code should use target_ids directly.

        Returns
        -------
        Int1D
            Array of lender indices (same as target_ids).

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> loans = LoanBook()
        >>> loans.lender is loans.target_ids
        True
        """
        return self.target_ids

    @lender.setter
    def lender(self, value: Int1D) -> None:
        """
        Setter for lender alias (updates target_ids).

        Parameters
        ----------
        value : Int1D
            New lender indices array.
        """
        self.target_ids = value

    def _ensure_capacity(self, extra: int) -> None:
        """
        Ensure capacity for additional edges, resizing arrays if needed.

        Uses a doubling strategy: when capacity is exceeded, doubles the
        current capacity or sets to max(128, needed), whichever is larger.
        This provides amortized O(1) append complexity.

        Parameters
        ----------
        extra : int
            Number of additional edges to accommodate.

        Notes
        -----
        This is an internal method called by append_loans_for_lender and
        drop_rows. It resizes all six parallel arrays (source_ids, target_ids,
        principal, rate, interest, debt) to maintain consistency.

        The doubling strategy ensures:

        - O(1) amortized append time
        - O(log n) total resize operations for n appends
        - Predictable memory allocation pattern

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> loans = LoanBook()
        >>> loans.capacity
        128
        >>> loans._ensure_capacity(150)  # Need 150 total slots
        >>> loans.capacity
        256
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
        Aggregate total debt per borrower using vectorized summation.

        Uses the inherited aggregate_by_source method from Relationship base
        class, which employs np.add.at for efficient aggregation.

        Parameters
        ----------
        n_borrowers : int
            Number of borrowers in the simulation (typically n_firms).

        Returns
        -------
        Float1D
            Array of shape (n_borrowers,) containing total debt per borrower.
            Borrowers with no loans have debt = 0.0.

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 1, 3]),
        ...     amount=np.array([100.0, 50.0, 200.0]),
        ...     rate=np.array([0.02, 0.03, 0.02]),
        ... )
        >>> debt = loans.debt_per_borrower(n_borrowers=5)
        >>> debt.shape
        (5,)
        >>> debt[1]  # doctest: +SKIP
        154.5
        >>> debt[3]  # doctest: +SKIP
        204.0

        See Also
        --------
        interest_per_borrower : Aggregate interest per borrower
        bamengine.core.relationship.Relationship.aggregate_by_source : Base method
        """
        return self.aggregate_by_source(self.debt, func="sum", n_sources=n_borrowers)  # type: ignore[no-any-return, attr-defined]

    def interest_per_borrower(self, n_borrowers: int) -> Float1D:
        """
        Aggregate total interest per borrower using vectorized summation.

        Uses the inherited aggregate_by_source method from Relationship base
        class, which employs np.add.at for efficient aggregation.

        Parameters
        ----------
        n_borrowers : int
            Number of borrowers in the simulation (typically n_firms).

        Returns
        -------
        Float1D
            Array of shape (n_borrowers,) containing total interest per borrower.
            Borrowers with no loans have interest = 0.0.

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 1, 3]),
        ...     amount=np.array([100.0, 50.0, 200.0]),
        ...     rate=np.array([0.02, 0.03, 0.02]),
        ... )
        >>> interest = loans.interest_per_borrower(n_borrowers=5)
        >>> interest.shape
        (5,)
        >>> interest[1]  # doctest: +SKIP
        3.5
        >>> interest[3]  # doctest: +SKIP
        4.0

        See Also
        --------
        debt_per_borrower : Aggregate debt per borrower
        bamengine.core.relationship.Relationship.aggregate_by_source : Base method
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

        Automatically resizes arrays if needed using doubling strategy.
        Caches interest and debt for O(1) aggregation later.

        Parameters
        ----------
        lender_idx : np.intp
            Index of the lender providing loans (bank ID).
        borrower_indices : Idx1D
            Indices of borrowers receiving loans (firm IDs).
        amount : Float1D
            Principal amounts for each loan.
        rate : Float1D
            Interest rates for each loan.

        Notes
        -----
        This method:

        1. Ensures capacity via _ensure_capacity (may trigger resize)
        2. Appends source_ids (borrowers), target_ids (lender)
        3. Appends principal and rate
        4. Caches interest = amount × rate
        5. Caches debt = amount × (1 + rate)
        6. Updates size counter

        The lender_idx is broadcast to all new loan entries (scalar expansion).

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 2, 3]),
        ...     amount=np.array([100.0, 150.0, 200.0]),
        ...     rate=np.array([0.02, 0.03, 0.02]),
        ... )
        >>> loans.size
        3
        >>> loans.source_ids[:3]
        array([1, 2, 3])
        >>> loans.target_ids[:3]
        array([0, 0, 0])
        >>> loans.principal[:3]
        array([100., 150., 200.])

        See Also
        --------
        _ensure_capacity : Internal resize method
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
        Remove loans matching a boolean mask and compact arrays in-place.

        Overrides Relationship.drop_rows() to also compact loan-specific
        component arrays (principal, rate, interest, debt).

        Parameters
        ----------
        rows_mask : Bool1D
            Boolean mask over active loans (length >= size).
            True → loan will be removed, False → loan kept.

        Returns
        -------
        int
            Number of loans removed.

        Notes
        -----
        This method:

        1. Inverts mask to get loans to keep
        2. Compacts all six arrays (source_ids, target_ids, principal, rate, interest, debt)
        3. Updates size counter
        4. Returns number of removed loans

        The compaction is done in-place using boolean indexing, which is
        cache-friendly and avoids temporary array allocations.

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 2, 3, 4]),
        ...     amount=np.array([100.0, 150.0, 200.0, 250.0]),
        ...     rate=np.array([0.02, 0.03, 0.02, 0.03]),
        ... )
        >>> loans.size
        4
        >>> # Remove loans with principal > 150
        >>> mask = loans.principal[: loans.size] > 150.0
        >>> removed = loans.drop_rows(mask)
        >>> removed
        2
        >>> loans.size
        2

        See Also
        --------
        purge_borrowers : Remove loans by borrower IDs
        purge_lenders : Remove loans by lender IDs
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

        return 0  # pragma: no cover - edge case: no loans to purge

    def purge_borrowers(self, borrower_ids: Idx1D) -> int:
        """
        Remove all loans from specified borrowers (firms).

        Uses inherited purge_sources() method from Relationship base class,
        which internally uses np.isin for efficient matching and drop_rows
        for compaction.

        Parameters
        ----------
        borrower_ids : Idx1D
            Array of borrower (firm) indices to purge.

        Returns
        -------
        int
            Number of loans removed.

        Notes
        -----
        This is typically called during bankruptcy resolution to remove all
        loans from insolvent firms.

        Time complexity: O(size) for np.isin + O(size) for compaction = O(size).

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 2, 3, 4, 5]),
        ...     amount=np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
        ...     rate=np.array([0.02, 0.03, 0.02, 0.03, 0.02]),
        ... )
        >>> loans.size
        5
        >>> # Purge bankrupt firms 2 and 4
        >>> removed = loans.purge_borrowers(np.array([2, 4]))
        >>> removed
        2
        >>> loans.size
        3
        >>> np.sort(loans.source_ids[: loans.size])
        array([1, 3, 5])

        See Also
        --------
        purge_lenders : Remove loans by lender IDs
        drop_rows : Remove loans by boolean mask
        bamengine.core.relationship.Relationship.purge_sources : Base method
        """
        return self.purge_sources(borrower_ids)  # type: ignore[no-any-return, attr-defined]

    def purge_lenders(self, lender_ids: Idx1D) -> int:
        """
        Remove all loans from specified lenders (banks).

        Uses inherited purge_targets() method from Relationship base class,
        which internally uses np.isin for efficient matching and drop_rows
        for compaction.

        Parameters
        ----------
        lender_ids : Idx1D
            Array of lender (bank) indices to purge.

        Returns
        -------
        int
            Number of loans removed.

        Notes
        -----
        This is typically called during bank bankruptcy resolution to remove
        all loans from insolvent banks.

        Time complexity: O(size) for np.isin + O(size) for compaction = O(size).

        Examples
        --------
        >>> from bamengine.relationships import LoanBook
        >>> import numpy as np
        >>> loans = LoanBook()
        >>> # Add loans from two different banks
        >>> loans.append_loans_for_lender(
        ...     lender_idx=0,
        ...     borrower_indices=np.array([1, 2]),
        ...     amount=np.array([100.0, 150.0]),
        ...     rate=np.array([0.02, 0.03]),
        ... )
        >>> loans.append_loans_for_lender(
        ...     lender_idx=1,
        ...     borrower_indices=np.array([3, 4]),
        ...     amount=np.array([200.0, 250.0]),
        ...     rate=np.array([0.02, 0.03]),
        ... )
        >>> loans.size
        4
        >>> # Purge bankrupt bank 0
        >>> removed = loans.purge_lenders(np.array([0]))
        >>> removed
        2
        >>> loans.size
        2
        >>> loans.target_ids[: loans.size]
        array([1, 1])

        See Also
        --------
        purge_borrowers : Remove loans by borrower IDs
        drop_rows : Remove loans by boolean mask
        bamengine.core.relationship.Relationship.purge_targets : Base method
        """
        return self.purge_targets(lender_ids)  # type: ignore[no-any-return, attr-defined]
