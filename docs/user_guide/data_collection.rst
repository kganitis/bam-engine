Data Collection
===============

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for data collection patterns.

BAM Engine provides ``SimulationResults`` for collecting simulation data.

Quick Example
-------------

.. code-block:: python

   import bamengine as bam

   # Run simulation and collect results
   sim = bam.Simulation.init(seed=42)
   results = sim.run(n_periods=100)

   # Access economy-wide time series
   unemployment = results.economy["unemployment_rate"]
   inflation = results.economy["inflation_rate"]

   # Export to pandas DataFrame (requires pandas)
   df_economy = results.to_dataframe("economy")
   df_firms = results.to_dataframe("firms")

Topics to be covered:

* SimulationResults class
* Economy metrics
* Role snapshots
* DataFrame export
* Custom data collectors
