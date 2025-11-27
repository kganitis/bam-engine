Quickstart
==========

This guide will help you get started with BAM Engine.

Basic Simulation
----------------

.. code-block:: python

   import bamengine as bam

   # Create simulation with default parameters
   sim = bam.Simulation.init(seed=42)

   # Run for 100 periods
   sim.run(n_periods=100)

   # Access results
   print(f"Final unemployment: {sim.ec.unemp_rate_history[-1]:.2%}")

Collecting Results
------------------

.. code-block:: python

   # Run and collect results
   results = sim.run(n_periods=100, collect=True)

   # Export to DataFrame (requires pandas)
   df = results.to_dataframe()

   # Get economy metrics
   econ_df = results.economy_metrics

Configuration
-------------

.. code-block:: python

   # Custom configuration
   sim = bam.Simulation.init(
       n_firms=200,
       n_households=1000,
       n_banks=15,
       seed=42
   )

See the :doc:`Examples </auto_examples/index>` for more detailed usage.
