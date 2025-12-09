Data Collection
===============

BAM Engine provides ``SimulationResults`` for collecting simulation data
during runs. The ``collect`` parameter in ``sim.run()`` controls what data
is captured and how it's aggregated.

Quick Example
-------------

.. code-block:: python

   import bamengine as bam

   # Run simulation and collect all data (default: aggregated means)
   sim = bam.Simulation.init(seed=42)
   results = sim.run(n_periods=100, collect=True)

   # Access economy-wide time series
   unemployment = results.economy_data["unemployment_rate"]
   inflation = results.economy_data["inflation"]

   # Export to pandas DataFrame (requires pandas)
   df = results.to_dataframe()

Collection Options
------------------

The ``collect`` parameter accepts three forms:

**Boolean** (simplest):

.. code-block:: python

   # Collect all roles and economy with aggregated means
   results = sim.run(n_periods=100, collect=True)

**List** (select roles):

.. code-block:: python

   # Collect specific roles with all their variables
   results = sim.run(
       n_periods=100,
       collect=["Producer", "Worker", "Economy"],
   )

**Dict** (full control):

.. code-block:: python

   # Specify exactly what to collect
   results = sim.run(
       n_periods=100,
       collect={
           "Producer": ["price", "inventory"],  # Specific variables
           "Worker": True,  # All Worker variables
           "Economy": True,  # All economy metrics
           "aggregate": "mean",  # Aggregation method
       },
   )

Collection Settings
-------------------

In dict form, the following keys are recognized:

* **Role names** (e.g., "Producer", "Worker"): Values are either ``True``
  (all variables) or a list of variable names.
* **"Economy"**: Treated as a pseudo-role for economy metrics.
  Available metrics: ``avg_price``, ``unemployment_rate``, ``inflation``.
* **"aggregate"**: How to aggregate across agents. Options:
  ``"mean"`` (default), ``"median"``, ``"sum"``, ``"std"``, or ``None``
  for full per-agent data.

Full Per-Agent Data
-------------------

Set ``aggregate=None`` to collect full arrays instead of aggregated values:

.. code-block:: python

   results = sim.run(
       n_periods=100,
       collect={
           "Producer": ["price"],
           "aggregate": None,  # Full per-agent data
       },
   )

   # Shape: (n_periods, n_firms)
   prices = results.role_data["Producer"]["price"]

Accessing Results
-----------------

``SimulationResults`` provides several ways to access data:

.. code-block:: python

   # Direct access to nested dicts
   results.role_data["Producer"]["price"]
   results.economy_data["unemployment_rate"]

   # Cleaner access via get_array()
   results.get_array("Producer", "price")
   results.get_array("Economy", "unemployment_rate")

   # Unified access via data property
   results.data["Producer"]["price"]
   results.data["Economy"]["unemployment_rate"]

See the :doc:`examples </auto_examples/index>` for more data collection patterns.
