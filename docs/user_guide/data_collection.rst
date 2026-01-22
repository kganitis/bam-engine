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

Relationship Data Collection
----------------------------

Relationships (like ``LoanBook``) can also be collected. Unlike roles,
relationships are **opt-in only** - they are NOT included when using
``collect=True``.

.. code-block:: python

   # Collect LoanBook data along with role data
   results = sim.run(
       n_periods=100,
       collect={
           "Producer": ["price"],
           "LoanBook": ["principal", "rate"],  # Relationship fields
           "Economy": True,
           "aggregate": "sum",  # Sum across all active loans
       },
   )

   # Access relationship data
   total_principal = results.relationship_data["LoanBook"]["principal"]
   avg_rate = results.get_array("LoanBook", "rate")

**Available aggregations for relationships:**

* ``"sum"``: Total across all edges (e.g., total outstanding principal)
* ``"mean"``: Average value across all edges (e.g., average interest rate)
* ``"std"``: Standard deviation across edges
* ``None``: Full edge data (list of variable-length arrays per period)

**Non-aggregated relationship data:**

When ``aggregate=None``, relationship data cannot be stacked into 2D arrays
because edge counts vary per period. Instead, data is stored as a list of
arrays:

.. code-block:: python

   results = sim.run(
       n_periods=50,
       collect={
           "LoanBook": ["principal"],
           "aggregate": None,
       },
   )

   # List of variable-length arrays (one per period)
   principal_per_period = results.relationship_data["LoanBook"]["principal"]
   # principal_per_period[0] might have 5 loans, period 10 might have 12

.. warning::

   Non-aggregated relationship data cannot be included in DataFrame exports
   due to variable lengths. Use ``results.relationship_data`` directly or
   use aggregation during collection.

Accessing Results
-----------------

``SimulationResults`` provides several ways to access data:

.. code-block:: python

   # Direct access to nested dicts
   results.role_data["Producer"]["price"]
   results.economy_data["unemployment_rate"]
   results.relationship_data["LoanBook"]["principal"]  # if collected

   # Cleaner access via get_array()
   results.get_array("Producer", "price")
   results.get_array("Economy", "unemployment_rate")
   results.get_array("LoanBook", "principal")  # if collected

   # Unified access via data property
   results.data["Producer"]["price"]
   results.data["Economy"]["unemployment_rate"]
   results.data["LoanBook"]["principal"]  # if collected

   # Get role/relationship as DataFrame
   prod_df = results.get_role_data("Producer")
   loans_df = results.get_relationship_data("LoanBook")  # if collected

See the :doc:`examples </auto_examples/index>` for more data collection patterns.
