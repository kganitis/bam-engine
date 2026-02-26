Operations Module
=================

The :mod:`~bamengine.ops` module provides NumPy-like operations with safe defaults
for writing custom events. It handles common pitfalls (division by zero, in-place
mutation semantics) and provides a consistent API that works on agent arrays
without requiring direct NumPy imports.

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


Why Use ops?
------------

The ``ops`` module exists for several reasons:

- **Safe division**: ``ops.divide(a, b)`` returns 0 when ``b`` is zero, avoiding
  ``ZeroDivisionError`` and ``RuntimeWarning`` from NumPy
- **In-place semantics**: ``ops.assign(target, value)`` ensures role arrays are
  updated in place, which is required for changes to be visible across events
- **Consistent API**: All operations work uniformly on scalars and arrays
- **Self-documenting**: Using ``ops.divide`` signals intent more clearly than
  raw ``a / b`` in economic model code

The ``ops`` functions are thin wrappers around NumPy — there is no performance
overhead compared to using NumPy directly.


Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Category
     - Functions
     - Notes
   * - Arithmetic
     - ``add``, ``subtract``, ``multiply``, ``divide``
     - ``divide`` uses epsilon guard (default: 1e-10)
   * - Assignment
     - ``assign``
     - Required for in-place role updates
   * - Comparisons
     - ``equal``, ``not_equal``, ``less``, ``greater``, ``less_equal``, ``greater_equal``
     - Return boolean arrays
   * - Logical
     - ``logical_and``, ``logical_or``, ``logical_not``
     - Combine boolean masks
   * - Conditional
     - ``where``, ``select``
     - Vectorized if-then-else
   * - Element-wise
     - ``maximum``, ``minimum``, ``clip``
     - Per-element bounds
   * - Mathematical
     - ``log``, ``exp``
     - Natural logarithm and exponential
   * - Aggregation
     - ``sum``, ``mean``, ``std``, ``min``, ``max``, ``any``, ``all``
     - Reduce across agents
   * - Array Creation
     - ``zeros``, ``ones``, ``full``, ``empty``, ``arange``, ``asarray``, ``array``
     - Create agent-sized arrays
   * - Utilities
     - ``unique``, ``bincount``, ``isin``, ``argsort``, ``sort``
     - Sorting and set operations
   * - Random
     - ``uniform``
     - Requires an RNG argument


Core Operations
---------------

Arithmetic
~~~~~~~~~~

Standard element-wise arithmetic on agent arrays:

.. code-block:: python

   from bamengine import ops

   revenue = ops.multiply(price, quantity_sold)
   gross_profit = ops.subtract(revenue, wage_bill)
   markup = ops.divide(price, unit_cost)  # safe: returns 0 when unit_cost is 0

All arithmetic functions accept an optional ``out=`` parameter for in-place
computation:

.. code-block:: python

   # Write result directly into an existing array
   ops.multiply(price, 1.1, out=price)

**Safe division.** ``ops.divide(a, b)`` adds a tiny epsilon (1e-10) to the
denominator to prevent division by zero. This is essential in economic models
where denominators like production or net worth can legitimately be zero:

.. code-block:: python

   # Without ops: RuntimeWarning when productivity is 0
   unit_cost = wage_bill / productivity

   # With ops: safely returns 0 for zero-productivity firms
   unit_cost = ops.divide(wage_bill, productivity)

Assignment
~~~~~~~~~~

``ops.assign`` copies values into an existing array **in place**. This is the
correct way to update role fields:

.. code-block:: python

   # Correct: updates the array that prod.price points to
   ops.assign(prod.price, new_prices)

   # Wrong: rebinds the local variable, role is unchanged
   prod.price = new_prices

.. warning::

   Direct assignment (``role.field = value``) creates a new array instead of
   updating the existing one. Other events that hold a reference to the original
   array will not see the change. Always use ``ops.assign()``.

Conditional
~~~~~~~~~~~

``ops.where`` is the vectorized if-then-else — the workhorse for conditional
logic in events:

.. code-block:: python

   # Increase price if sold out, decrease if overstocked
   new_price = ops.where(
       inventory == 0,  # condition
       price * (1 + shock),  # value when True
       price * (1 - shock),  # value when False
   )
   ops.assign(prod.price, new_price)

Aggregation
~~~~~~~~~~~

Reduce operations across agent arrays:

.. code-block:: python

   total_output = ops.sum(prod.production)
   avg_price = ops.mean(prod.price)
   any_bankrupt = ops.any(bor.net_worth < 0)

   # With mask: aggregate only employed workers
   avg_employed_wage = ops.mean(wrk.wage, where=wrk.employed)

Array Creation
~~~~~~~~~~~~~~

Create arrays sized to the agent population:

.. code-block:: python

   # Zero-initialized array for n_firms agents
   scratch = ops.zeros(sim.n_firms)

   # Constant-filled array
   base_rate = ops.full(sim.n_banks, 0.02)

Random
~~~~~~

``ops.uniform`` draws from a uniform distribution, but requires an explicit
RNG argument to maintain determinism:

.. code-block:: python

   # Draw production shocks for all firms
   shock = ops.uniform(sim.rng, 0, sim.h_rho, size=sim.n_firms)


Common Patterns
---------------

These patterns appear frequently in BAM Engine events:

**Safe denominator pattern.** When dividing by a value that may be zero,
guard the denominator with ``ops.where``:

.. code-block:: python

   # Avoid division by zero for zero-NW firms
   safe_nw = ops.where(bor.net_worth > 0, bor.net_worth, 1.0)
   leverage = ops.where(bor.net_worth > 0, ops.divide(debt, safe_nw), max_leverage)

**Masked update.** Update only agents that meet a condition:

.. code-block:: python

   # Only raise wages for firms with vacancies
   ops.assign(
       emp.wage_offer,
       ops.where(emp.n_vacancies > 0, emp.wage_offer * (1 + shock), emp.wage_offer),
   )

**Aggregate then broadcast.** Compute an economy-wide statistic and use it
per-agent:

.. code-block:: python

   avg_savings = ops.mean(con.savings)
   relative_savings = ops.divide(con.savings, avg_savings)


When to Use ops vs. NumPy Directly
-----------------------------------

Use ``ops`` for:

- **Role mutations**: Any write to a role field must use ``ops.assign()``
- **Division**: Use ``ops.divide()`` whenever the denominator might be zero
- **Conditional logic**: ``ops.where()`` is clearer than nested ``np.where``

NumPy directly is fine for:

- **Local computation**: Intermediate calculations that don't mutate roles
- **Advanced operations**: Functions not in ``ops`` (e.g., ``np.cumsum``,
  ``np.histogram``, ``np.linalg``)
- **Indexing**: Fancy indexing and slicing work the same on role arrays

.. code-block:: python

   import numpy as np
   from bamengine import ops

   # NumPy fine for local computation
   log_prices = np.log(prod.price)
   sorted_idx = np.argsort(prod.price)

   # ops required for role mutation
   ops.assign(prod.price, np.exp(log_prices * 1.01))


.. seealso::

   - :doc:`custom_events` for using ops in event implementations
   - :mod:`bamengine.ops` API reference for full function signatures
