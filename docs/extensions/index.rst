Model Extensions
================

BAM Engine includes three built-in extensions that add economic mechanisms
beyond the baseline BAM model. Extensions follow a consistent three-step
activation pattern.

Quick Start
-----------

Every extension exports three components — a role class, an event list, and
a config dictionary:

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

   results = sim.run(n_periods=1000, collect=True)

.. warning::

   All three steps are required. Forgetting ``use_events()`` means the extension
   events will never execute. Forgetting ``use_config()`` means extension
   parameters will be missing at runtime.


Available Extensions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Extension
     - Book Section
     - Description
   * - :doc:`rnd`
     - Section 3.8
     - Endogenous productivity growth via R&D investment
   * - :doc:`buffer_stock`
     - Section 3.9.4
     - Target savings-to-income ratio consumption rule
   * - :doc:`taxation`
     - Section 3.10.2
     - Profit taxation for structural experiments

Multiple extensions can be combined. See :doc:`/user_guide/extensions` for
the combining pattern and for writing your own extensions.


.. toctree::
   :maxdepth: 2
   :hidden:

   rnd
   buffer_stock
   taxation
