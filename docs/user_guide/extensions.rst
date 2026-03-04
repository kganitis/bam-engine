Model Extensions
================

BAM Engine includes three built-in extensions that add economic mechanisms
beyond the baseline BAM model. Extensions follow a consistent pattern:
define new roles and events, package them with a config dictionary, and
activate them explicitly after simulation initialization.


The Extension Pattern
---------------------

Every extension provides a pre-built :class:`~bamengine.Extension` bundle that
activates all components in a single call:

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RND

   sim = bam.Simulation.init(seed=42)
   sim.use(RND)  # registers role, events, and config

For finer control, each extension also exports individual components — a role
class, an event list (``*_EVENTS``), and a config dictionary (``*_CONFIG``) —
that can be activated manually:

.. code-block:: python

   from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

   sim = bam.Simulation.init(seed=42)
   sim.use_role(RnD)
   sim.use_events(*RND_EVENTS)
   sim.use_config(RND_CONFIG)

.. warning::

   When using the manual pattern, all three steps are required. Forgetting
   ``use_events()`` means the extension events will never execute. Forgetting
   ``use_config()`` means extension parameters will be missing at runtime.
   Using ``sim.use()`` avoids this problem.


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
   from extensions.rnd import RND
   from extensions.buffer_stock import BUFFER_STOCK
   from extensions.taxation import TAXATION

   sim = bam.Simulation.init(seed=42)
   sim.use(RND)
   sim.use(BUFFER_STOCK)
   sim.use(TAXATION)

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

3. **Bundle into an Extension** for one-call activation:

   .. code-block:: python

      from bamengine import Extension

      MY_EVENTS = [MyCustomEvent]
      MY_CONFIG = {"my_param": 0.5}

      MY_EXT = Extension(
          roles={MyExtension: "firms"},
          events=MY_EVENTS,
          relationships=[],
          config_dict=MY_CONFIG,
      )

4. **Activate** in user code:

   .. code-block:: python

      sim.use(MY_EXT)

See :doc:`custom_roles`, :doc:`custom_events`, and :doc:`configuration` for
detailed syntax.


.. seealso::

   - :doc:`custom_roles` for defining extension roles
   - :doc:`custom_events` for defining extension events
   - :doc:`pipelines` for understanding hook insertion points
   - :doc:`configuration` for extension parameters (``extra_params``)
   - :doc:`validation` for validating extension behavior
