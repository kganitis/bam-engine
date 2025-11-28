Operations Module
=================

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for operations module usage patterns.

The ``ops`` module provides NumPy-like operations with safe defaults, allowing
users to write custom events without importing NumPy directly.

Quick Example
-------------

.. code-block:: python

   from bamengine import ops

   # Safe division (handles zeros)
   unit_cost = ops.divide(wages, productivity)

   # Conditional assignment
   new_prices = ops.where(demand > supply, price * 1.1, price * 0.9)

   # In-place assignment
   ops.assign(prod.price, new_prices)

Available Operations
--------------------

**Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``

**Comparisons**: ``equal``, ``not_equal``, ``less``, ``greater``, ``less_equal``, ``greater_equal``

**Logical**: ``logical_and``, ``logical_or``, ``logical_not``

**Conditional**: ``where``, ``select``

**Element-wise**: ``maximum``, ``minimum``, ``clip``

**Aggregation**: ``sum``, ``mean``, ``any``, ``all``

**Array Creation**: ``zeros``, ``ones``, ``full``, ``empty``

**Utilities**: ``unique``, ``bincount``, ``isin``, ``argsort``, ``sort``

**Assignment**: ``assign``

**Random**: ``uniform`` (requires RNG)

Topics to be covered:

* Complete operation reference
* Safe division behavior
* In-place operations with ``out=``
* Random number generation
* When to use ops vs NumPy directly
