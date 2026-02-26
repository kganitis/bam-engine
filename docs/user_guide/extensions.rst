Model Extensions
================

BAM Engine includes three built-in extensions that add economic mechanisms
beyond the baseline BAM model. Extensions follow a consistent pattern:
define new roles and events, package them with a config dictionary, and
activate them explicitly after simulation initialization.


The Extension Pattern
---------------------

Every extension exports three components:

1. **Role class(es)** — New agent state variables
2. **Event list** (``*_EVENTS``) — New behavioral rules with pipeline hooks
3. **Config dictionary** (``*_CONFIG``) — Default parameter values

Activation always follows the same three steps:

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

   sim = bam.Simulation.init(seed=42)

   # Step 1: Register the role
   sim.use_role(RnD)

   # Step 2: Apply event hooks to the pipeline
   sim.use_events(*RND_EVENTS)

   # Step 3: Set default extension parameters
   sim.use_config(RND_CONFIG)

.. warning::

   All three steps are required. Forgetting ``use_events()`` means the extension
   events will never execute. Forgetting ``use_config()`` means extension
   parameters will be missing at runtime.


Growth+ (R&D) Extension
------------------------

*Based on Section 3.9.2 of Delli Gatti et al. (2011)*

The Growth+ extension adds **endogenous productivity growth** via R&D investment.
Profitable firms invest a fraction of their profits in R&D, and their labor
productivity grows stochastically based on R&D intensity. This creates persistent
firm-level heterogeneity and aggregate productivity growth.

**The RnD Role**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``sigma``
     - R&D share of profits (0 to ``sigma_max``), decreasing with financial fragility
   * - ``rnd_intensity``
     - Expected productivity gain (:math:`\mu`), derived from sigma and profit
   * - ``productivity_increment``
     - Actual productivity gain drawn from exponential distribution each period
   * - ``fragility``
     - Financial fragility metric (wage bill / net worth)

**Events**

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Event
     - Hook
     - Description
   * - ``FirmsComputeRnDIntensity``
     - after ``firms_validate_debt_commitments``
     - Compute R&D share :math:`\sigma` and intensity :math:`\mu` from profits
       and fragility
   * - ``FirmsApplyProductivityGrowth``
     - after ``firms_compute_rnd_intensity``
     - Draw productivity increment from :math:`\text{Exp}(\mu)` and add to
       labor productivity
   * - ``FirmsDeductRnDExpenditure``
     - after ``firms_apply_productivity_growth``
     - Reduce net profit by :math:`(1 - \sigma)` factor before dividends

**Configuration**

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Description
   * - ``sigma_min``
     - 0.0
     - Minimum R&D share (for highly fragile firms)
   * - ``sigma_max``
     - 0.1
     - Maximum R&D share (for financially healthy firms)
   * - ``sigma_decay``
     - -1.0
     - Decay rate for sigma as a function of fragility

**Complete Setup**

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

   sim = bam.Simulation.init(seed=42)
   sim.use_role(RnD)
   sim.use_events(*RND_EVENTS)
   sim.use_config(RND_CONFIG)

   results = sim.run(n_periods=1000, collect=True)

**Expected behavior:** Rising average labor productivity over time, increasing
firm size heterogeneity, and a positive long-run growth trend in output.


Buffer-Stock Consumption Extension
-----------------------------------

*Based on Section 3.9.4 of Delli Gatti et al. (2011)*

The buffer-stock extension replaces the baseline consumption rule with a
**target savings-to-income ratio** mechanism. Households aim to maintain savings
equal to :math:`h` times their income. When savings fall below the target, they
cut spending; when savings exceed it, they spend more freely.

**The BufferStock Role**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``prev_income``
     - Previous period's income (used to detect income changes)
   * - ``propensity``
     - Buffer-stock MPC — may exceed 1.0 when households dissave

.. note::

   The BufferStock role is registered at the **household level** since it tracks
   per-household state:

   .. code-block:: python

      sim.use_role(BufferStock, n_agents=sim.n_households)

**Events**

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

**Configuration**

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

**Complete Setup**

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

**Expected behavior:** More realistic wealth distribution, precautionary saving
by low-income households, and counter-cyclical consumption smoothing.


Taxation Extension
------------------

The taxation extension adds **profit taxation** to firms. Tax revenue is
removed from the economy (no redistribution) — this is designed for structural
experiments testing the role of automatic stabilizers and entry dynamics.

**Events**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Event
     - Hook
     - Description
   * - ``FirmsTaxProfits``
     - after ``firms_validate_debt_commitments``
     - Deduct ``profit_tax_rate * max(0, net_profit)`` from firm funds

**Configuration**

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``profit_tax_rate``
     - 0.0
     - Tax rate on positive net profit (0.0 = no tax)

**Complete Setup**

.. code-block:: python

   import bamengine as bam
   from extensions.taxation import FirmsTaxProfits, TAXATION_CONFIG

   sim = bam.Simulation.init(seed=42, profit_tax_rate=0.20)
   sim.use_events(FirmsTaxProfits)

   results = sim.run(n_periods=1000, collect=True)

.. note::

   The taxation extension has no role — it operates directly on the existing
   ``Borrower`` role fields (``net_profit`` and ``total_funds``).


Combining Extensions
--------------------

Multiple extensions can be used together. Apply them in order, with R&D events
before taxation (since R&D deducts from net profit before tax):

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG
   from extensions.buffer_stock import (
       BufferStock,
       BUFFER_STOCK_EVENTS,
       BUFFER_STOCK_CONFIG,
   )
   from extensions.taxation import FirmsTaxProfits, TAXATION_CONFIG

   sim = bam.Simulation.init(seed=42)

   # R&D extension (firm-level)
   sim.use_role(RnD)
   sim.use_events(*RND_EVENTS)
   sim.use_config(RND_CONFIG)

   # Buffer-stock extension (household-level)
   sim.use_role(BufferStock, n_agents=sim.n_households)
   sim.use_events(*BUFFER_STOCK_EVENTS)
   sim.use_config(BUFFER_STOCK_CONFIG)

   # Taxation extension (no role needed)
   sim.use_events(FirmsTaxProfits)
   sim.use_config(TAXATION_CONFIG)

   results = sim.run(n_periods=1000, collect=True)


Writing Your Own Extension
--------------------------

Follow the same pattern as the built-in extensions:

1. **Define a role** (if new state variables are needed):

   .. code-block:: python

      from bamengine import role
      from bamengine.typing import Float


      @role
      class MyExtension:
          custom_field: Float

2. **Define events** with pipeline hooks:

   .. code-block:: python

      from bamengine import event, ops


      @event(after="firms_collect_revenue")
      class MyCustomEvent:
          def execute(self, sim):
              ext = sim.get_role("MyExtension")
              ops.assign(ext.custom_field, ...)

3. **Package exports** for clean activation:

   .. code-block:: python

      MY_EVENTS = [MyCustomEvent]
      MY_CONFIG = {"my_param": 0.5}

4. **Activate** in user code:

   .. code-block:: python

      sim.use_role(MyExtension)
      sim.use_events(*MY_EVENTS)
      sim.use_config(MY_CONFIG)

See :doc:`custom_roles`, :doc:`custom_events`, and :doc:`configuration` for
detailed syntax.


.. seealso::

   - :doc:`custom_roles` for defining extension roles
   - :doc:`custom_events` for defining extension events
   - :doc:`pipelines` for understanding hook insertion points
   - :doc:`configuration` for extension parameters (``extra_params``)
   - :doc:`validation` for validating extension behavior
