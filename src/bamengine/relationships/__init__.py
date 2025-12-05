"""
Relationship classes for many-to-many connections between roles.

This package defines relationship classes that represent connections (edges)
between agents in different roles. Relationships use COO (Coordinate List)
sparse format for efficient storage and querying of many-to-many connections.

Relationship Design
-------------------
- **Sparse edge-list format**: Only active connections stored (not dense matrix)
- **COO format**: Parallel arrays (source_ids, target_ids, edge_data)
- **Vectorized operations**: Fast aggregations using NumPy
- **Auto-registration**: All relationships inherit from Relationship base class
- **Dynamic resize**: Arrays grow as needed (amortized O(1) append)

Available Relationships
-----------------------
LoanBook : Relationship
    Many-to-many loan relationship between Borrower (firms) and Lender (banks).
    Tracks principal, rate, interest, and debt for each loan.

Examples
--------
Access LoanBook from simulation:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
>>> loans = sim.get_relationship("loanbook")
>>> loans.size  # doctest: +SKIP
45
>>> loans.principal[: loans.size].sum()  # doctest: +SKIP
1250.0

Query loans by borrower:

>>> borrower_id = 5
>>> import numpy as np
>>> loan_indices = np.where(loans.source_ids[: loans.size] == borrower_id)[0]
>>> borrower_debt = loans.debt[loan_indices].sum()
>>> borrower_debt  # doctest: +SKIP
150.0

Aggregate debt per borrower:

>>> total_debt_per_firm = loans.debt_per_borrower(n_borrowers=100)
>>> total_debt_per_firm.shape
(100,)

See Also
--------
:class:`bamengine.core.Relationship` : Base class for all relationships
:class:`~bamengine.relationships.LoanBook` : Loan relationship implementation
"""

from .loanbook import LoanBook

__all__ = [
    "LoanBook",
]
