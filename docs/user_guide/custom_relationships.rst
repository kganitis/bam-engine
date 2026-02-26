Custom Relationships
====================

Relationships represent many-to-many connections between agents with per-edge
data. They are stored in a sparse COO (Coordinate) format for memory efficiency,
since most agent pairs are not connected at any given time.


Quick Example
-------------

.. code-block:: python

   from bamengine import relationship, get_role
   from bamengine.typing import Float, Int


   @relationship(source=get_role("Worker"), target=get_role("Employer"))
   class Employment:
       """Employment relationship between workers and firms."""

       wage: Float
       contract_duration: Int
       start_period: Int


The ``@relationship`` Decorator
-------------------------------

The decorator defines a relationship type with source/target roles and
per-edge data fields:

.. code-block:: python

   @relationship(
       source=get_role("Borrower"),  # Source agent role
       target=get_role("Lender"),  # Target agent role
       cardinality="many-to-many",  # Default
       name="CustomLoanBook",  # Optional name override
   )
   class CustomLoanBook:
       principal: Float
       rate: Float

Parameters:

- **source**: The role type on the "from" side of each edge
- **target**: The role type on the "to" side of each edge
- **cardinality**: ``"many-to-many"`` (default), ``"one-to-many"``, or
  ``"many-to-one"``
- **name**: Custom name for registry lookup (defaults to class name)

The decorator registers the relationship in the global registry and converts
edge data fields into parallel arrays.


COO Sparse Format
-----------------

Relationships store edges as parallel arrays in COO (Coordinate) format:

.. code-block:: text

   Index:      0       1       2       3       4
   source_ids: [3,     7,      3,      12,     7    ]   ← borrower firm IDs
   target_ids: [0,     2,      1,      0,      0    ]   ← lender bank IDs
   principal:  [10.5,  25.0,   8.3,    15.0,   12.0 ]   ← per-edge data
   rate:       [0.03,  0.02,   0.04,   0.03,   0.02 ]   ← per-edge data

Key attributes:

- ``source_ids`` — array of source agent IDs
- ``target_ids`` — array of target agent IDs
- ``size`` — number of active edges (valid entries in the arrays)
- ``capacity`` — allocated array length (grows dynamically via doubling)

Only indices ``0`` through ``size - 1`` contain valid data.


Adding and Removing Edges
-------------------------

**Adding edges:**

.. code-block:: python

   import numpy as np

   # Append edges with source/target IDs and component data
   rel.append_edges(
       source_ids=np.array([0, 1, 2]),
       target_ids=np.array([5, 5, 3]),
       principal=np.array([10.0, 20.0, 15.0]),
       rate=np.array([0.02, 0.03, 0.025]),
   )

The ``LoanBook`` has a convenience method for appending loans from a single
lender:

.. code-block:: python

   loans.append_loans_for_lender(
       lender_idx=np.intp(0),
       borrower_indices=np.array([3, 7]),
       amount=np.array([10.5, 25.0]),
       rate=np.array([0.03, 0.02]),
   )

**Removing edges:**

.. code-block:: python

   # Remove by boolean mask (True = remove)
   mask = rel.principal < 1.0  # Remove tiny loans
   rel.drop_rows(mask[: rel.size])

   # Remove all edges from specific sources
   rel.purge_sources(np.array([3, 12]))  # Remove all loans from firms 3 and 12

   # Remove all edges to specific targets
   rel.purge_targets(np.array([0]))  # Remove all loans to bank 0

   # Clear all edges
   rel.size = 0


Querying Edges
--------------

Find edges by source or target agent:

.. code-block:: python

   # Get indices of all loans from firm 7
   edge_indices = loans.query_sources(7)
   firm_7_rates = loans.rate[edge_indices]

   # Get indices of all loans to bank 0
   edge_indices = loans.query_targets(0)
   bank_0_principal = loans.principal[edge_indices]


Aggregating Edge Data
---------------------

Compute per-agent aggregates from edge data:

.. code-block:: python

   # Total debt per borrower (firm)
   debt_by_firm = loans.aggregate_by_source(
       "debt",
       func="sum",
       n_sources=sim.n_firms,
   )

   # Total lending per lender (bank)
   lending_by_bank = loans.aggregate_by_target(
       "principal",
       func="sum",
       n_targets=sim.n_banks,
   )

   # Average interest rate per bank
   avg_rate_by_bank = loans.aggregate_by_target(
       "rate",
       func="mean",
       n_targets=sim.n_banks,
   )

   # Number of loans per firm
   loan_count_by_firm = loans.aggregate_by_source(
       "principal",
       func="count",
       n_sources=sim.n_firms,
   )

Available aggregation functions: ``"sum"``, ``"mean"``, ``"count"``,
``"min"``, ``"max"``.


Built-in Relationship: LoanBook
--------------------------------

The :class:`~bamengine.relationships.loanbook.LoanBook` relationship tracks
loans between firms (borrowers) and banks (lenders):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``principal``
     - Original loan amount at signing
   * - ``rate``
     - Contractual interest rate
   * - ``interest``
     - Cached interest amount (``rate * principal``)
   * - ``debt``
     - Cached total debt (``principal * (1 + rate)``)
   * - ``source_ids``
     - Borrower (firm) IDs (also accessible as ``.borrower``)
   * - ``target_ids``
     - Lender (bank) IDs (also accessible as ``.lender``)

Convenience methods:

.. code-block:: python

   loans = sim.lb  # shortcut for sim.get_relationship("LoanBook")

   loans.debt_per_borrower(n_borrowers=sim.n_firms)  # Total debt per firm
   loans.interest_per_borrower(n_borrowers=sim.n_firms)  # Total interest per firm
   loans.principal_per_borrower(n_borrowers=sim.n_firms)  # Total principal per firm


Tips
----

- **Edge indices are NOT stable**: Adding or removing edges may shift indices.
  Don't store edge indices across operations.
- **Use ``size``, not ``len()``**: Only ``relationship.size`` entries are valid.
  Arrays may have extra capacity beyond ``size``.
- **COO is efficient for sparse, dynamic graphs**: The format excels when most
  agent pairs are unconnected and the set of connections changes each period.
- **Aggregation is vectorized**: ``aggregate_by_source`` and
  ``aggregate_by_target`` use ``np.bincount`` internally, so they're fast even
  with thousands of edges.


.. seealso::

   - :doc:`data_collection` for collecting relationship data during runs
   - :doc:`custom_events` for using relationships in event implementations
   - :class:`~bamengine.relationships.loanbook.LoanBook` API reference
