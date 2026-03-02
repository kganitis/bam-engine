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


Built-in Extensions
--------------------

BAM Engine ships with three extensions:

- **Growth+ (R&D)** — Endogenous productivity growth via R&D investment
  (Section 3.8). Firms invest profits in R&D, producing stochastic
  productivity gains.

- **Buffer-Stock Consumption** — Target savings-to-income ratio mechanism
  (Section 3.9.4). Households adjust spending based on a savings buffer
  target.

- **Taxation** — Profit taxation without redistribution (Section 3.10.2).
  Designed for structural experiments on entry dynamics.

.. seealso::

   Full reference for each extension, including role fields, events,
   configuration parameters, and autodoc API:

   - :doc:`/extensions/rnd` — Growth+ (R&D)
   - :doc:`/extensions/buffer_stock` — Buffer-Stock Consumption
   - :doc:`/extensions/taxation` — Taxation


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
