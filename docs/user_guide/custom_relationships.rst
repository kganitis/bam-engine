Custom Relationships
====================

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for custom relationship usage patterns.

Relationships represent many-to-many connections between agents with edge-specific data.

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

Built-in Relationships
----------------------

BAM Engine includes the ``LoanBook`` relationship for tracking loans between
borrowers (firms) and lenders (banks):

.. code-block:: python

   loans = sim.get_relationship("LoanBook")

   # Query loans
   debt_per_firm = loans.debt_per_borrower(n_borrowers=sim.config.n_firms)

Topics to be covered:

* Relationship decorator syntax
* COO sparse format internals
* Adding and removing edges
* Querying edges by source/target
* Aggregating edge data
* Built-in LoanBook relationship
