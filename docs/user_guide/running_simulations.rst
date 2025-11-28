Running Simulations
===================

.. note::

   This section is under construction. See the :doc:`/quickstart` for a quick
   introduction.

This guide covers how to initialize and run simulations with BAM Engine.

Initialization
--------------

Create a simulation with :meth:`~bamengine.Simulation.init`:

.. code-block:: python

   import bamengine as bam

   # Default configuration
   sim = bam.Simulation.init(seed=42)

   # Custom agent counts
   sim = bam.Simulation.init(
       n_firms=200,
       n_households=1000,
       n_banks=15,
       seed=42
   )

Running Periods
---------------

Run multiple periods with :meth:`~bamengine.Simulation.run`:

.. code-block:: python

   results = sim.run(n_periods=100)

Or step through one period at a time:

.. code-block:: python

   for period in range(100):
       sim.step()
       # Inspect state after each period
       print(f"Period {period}: unemployment = {sim.ec.unemp_rate_history[-1]:.2%}")

Topics to be covered:

* Initialization parameters
* Single-step vs multi-period execution
* Reproducibility with seeds
* Warm-up periods
* Continuing simulations
* Parallel/batch runs
