Model Extensions
================

BAM Engine includes three built-in extensions that add economic mechanisms
beyond the baseline BAM model.

Quick Start
-----------

Each extension provides an :class:`~bamengine.Extension` bundle that activates
all its components in a single call:

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RND

   sim = bam.Simulation.init(seed=42)
   sim.use(RND)

   results = sim.run(n_periods=1000, collect=True)

Individual components (roles, events, config) are also available for manual
activation — see :doc:`/user_guide/extensions` for the manual pattern.


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
