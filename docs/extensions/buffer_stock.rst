Buffer-Stock Consumption Extension
===================================

*Based on Section 3.9.4 of Delli Gatti et al. (2011)*

The buffer-stock extension replaces the baseline consumption rule with a
**target savings-to-income ratio** mechanism. Households aim to maintain savings
equal to :math:`h` times their income. When savings fall below the target, they
cut spending; when savings exceed it, they spend more freely.


Quick Start
-----------

.. code-block:: python

   import bamengine as bam
   from extensions.buffer_stock import (
       BufferStock,
       BUFFER_STOCK_EVENTS,
       BUFFER_STOCK_CONFIG,
   )

   sim = bam.Simulation.init(seed=42)
   sim.use_role(BufferStock, n_agents=sim.n_households)
   sim.use_events(*BUFFER_STOCK_EVENTS)
   sim.use_config(BUFFER_STOCK_CONFIG)

   results = sim.run(n_periods=1000, collect=True)

.. note::

   The BufferStock role is registered at the **household level** since it tracks
   per-household state. Pass ``n_agents=sim.n_households`` to ``use_role()``.


Role Fields
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``prev_income``
     - Previous period's income (used to detect income changes)
   * - ``propensity``
     - Buffer-stock MPC — may exceed 1.0 when households dissave


Events
------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Event
     - Hook
     - Description
   * - ``ConsumersCalcBufferStockPropensity``
     - replaces ``consumers_calc_propensity``
     - Compute MPC from income changes and savings-target gap
   * - ``ConsumersDecideBufferStockSpending``
     - replaces ``consumers_decide_income_to_spend``
     - Allocate spending using buffer-stock MPC


Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``buffer_stock_h``
     - 2.0
     - Target savings-to-income ratio. Higher values mean more precautionary
       saving.

**Expected behavior:** More realistic wealth distribution, precautionary saving
by low-income households, and counter-cyclical consumption smoothing.


API Reference
-------------

.. automodule:: extensions.buffer_stock
   :members:
   :undoc-members:

.. automodule:: extensions.buffer_stock.role
   :members:
   :undoc-members:

.. automodule:: extensions.buffer_stock.events
   :members:
   :undoc-members:


.. seealso::

   - :doc:`/user_guide/extensions` for writing custom extensions
   - Buffer-stock validation: ``python -m validation.scenarios.buffer_stock``
   - :doc:`/auto_examples/extensions/example_buffer_stock`
