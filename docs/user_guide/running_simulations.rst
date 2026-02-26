Running Simulations
===================

This guide covers how to initialize, run, and inspect simulations with BAM Engine.


Initialization
--------------

Create a simulation with :meth:`~bamengine.Simulation.init`:

.. code-block:: python

   import bamengine as bam

   # Default configuration
   sim = bam.Simulation.init(seed=42)

   # Custom agent counts
   sim = bam.Simulation.init(n_firms=200, n_households=1000, n_banks=15, seed=42)

   # Configuration from a YAML file
   sim = bam.Simulation.init(config="my_config.yml", seed=42)

   # Combining YAML file with keyword overrides (kwargs take priority)
   sim = bam.Simulation.init(config="my_config.yml", n_firms=300, seed=42)

See :doc:`configuration` for the full parameter reference.


Running Periods
---------------

Run multiple periods with :meth:`~bamengine.Simulation.run`:

.. code-block:: python

   # Run 100 periods (no data collection)
   sim.run(n_periods=100)

   # Run and collect data for analysis
   results = sim.run(n_periods=100, collect=True)

See :doc:`data_collection` for details on the ``collect`` parameter.


Single-Step Execution
---------------------

For debugging or custom control loops, step through one period at a time
with :meth:`~bamengine.Simulation.step`:

.. code-block:: python

   for period in range(100):
       sim.step()
       # Inspect state after each period
       print(f"Period {period}: unemployment = {sim.ec.unemp_rate_history[-1]:.2%}")

This is useful when you need to:

- Inspect agent state between periods
- Implement custom stopping conditions
- Log or visualize intermediate results
- Debug custom events


Accessing State During Simulation
---------------------------------

BAM Engine provides shortcut attributes on the simulation object for quick
access to roles, the economy, and relationships:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Shortcut
     - Full Access
     - Description
   * - ``sim.prod``
     - ``sim.get_role("Producer")``
     - Firm production and pricing state
   * - ``sim.emp``
     - ``sim.get_role("Employer")``
     - Firm hiring and wage state
   * - ``sim.bor``
     - ``sim.get_role("Borrower")``
     - Firm financial state (net worth, profit)
   * - ``sim.wrk``
     - ``sim.get_role("Worker")``
     - Household employment state
   * - ``sim.con``
     - ``sim.get_role("Consumer")``
     - Household spending and savings state
   * - ``sim.sh``
     - ``sim.get_role("Shareholder")``
     - Household dividend income
   * - ``sim.lend``
     - ``sim.get_role("Lender")``
     - Bank credit supply and interest rates
   * - ``sim.ec``
     - ``sim.economy``
     - Economy-wide state (prices, unemployment, inflation)
   * - ``sim.lb``
     - ``sim.get_relationship("LoanBook")``
     - Active loans between firms and banks

Example:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)
   sim.run(n_periods=50)

   # Access role data
   avg_price = sim.prod.price.mean()
   total_inventory = sim.prod.inventory.sum()
   unemployment = (~sim.wrk.employed).mean()

   # Access economy state
   min_wage = sim.ec.min_wage
   inflation = sim.ec.inflation_history[-1]

   # Access loan data
   total_lending = sim.lb.principal[: sim.lb.size].sum()

The simulation also exposes frequently-used configuration values as properties:

.. code-block:: python

   sim.n_firms  # Number of firms
   sim.n_households  # Number of households
   sim.n_banks  # Number of banks
   sim.theta  # Contract length
   sim.delta  # Dividend payout ratio
   sim.rng  # NumPy random generator
   sim.t  # Current period number


Stopping Conditions
-------------------

BAM Engine detects economy collapse (all firms bankrupt) and sets a flag:

.. code-block:: python

   sim.run(n_periods=1000)

   if sim.ec.collapsed:
       print(f"Economy collapsed at period {sim.t}")

For custom stopping conditions, use single-step execution:

.. code-block:: python

   for period in range(1000):
       sim.step()

       # Stop if unemployment exceeds threshold
       if sim.ec.unemp_rate_history[-1] > 0.5:
           print(f"High unemployment at period {period}")
           break


.. _performance-and-scaling:

Performance & Scaling
---------------------

BAM Engine uses fully vectorized NumPy operations for high performance.
This section helps you choose appropriate configurations for your research needs.

Runtime Expectations
~~~~~~~~~~~~~~~~~~~~

Expected runtimes on modern hardware (Apple M4 Pro, Python 3.13):

.. list-table::
   :header-rows: 1
   :widths: 20 25 15 15 15

   * - Use Case
     - Configuration
     - 100 periods
     - 1000 periods
     - Throughput
   * - Prototyping
     - 100 firms, 500 households
     - ~0.2s
     - ~2s
     - ~500 periods/s
   * - Development
     - 200 firms, 1000 households
     - ~0.4s
     - ~4s
     - ~240 periods/s
   * - Production
     - 500 firms, 2500 households
     - ~1.3s
     - ~13s
     - ~77 periods/s

.. note::

   Actual performance varies by hardware. These benchmarks provide relative guidance
   for choosing configurations.

.. _choosing-agent-counts:

Choosing Agent Counts
~~~~~~~~~~~~~~~~~~~~~

Select agent counts based on your research phase:

* **Quick prototyping** (100 firms): Fast iteration for testing ideas and debugging
* **Development runs** (200 firms): Balance of speed and statistical stability
* **Production research** (500+ firms): Publication-quality results with robust statistics
* **Large-scale studies** (1000+ firms): For analyzing rare events or detailed distributions

The model's emergent properties (business cycles, unemployment dynamics) are stable
across agent counts, so smaller configurations are valid for development and testing.

Memory Usage
~~~~~~~~~~~~

Approximate memory requirements by configuration:

* 100 firms, 500 households: ~50 MB
* 200 firms, 1000 households: ~80 MB
* 500 firms, 2500 households: ~150 MB
* 1000 firms, 5000 households: ~300 MB

For memory-constrained environments, prefer smaller configurations during development.

Scaling Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

Performance scales sub-linearly with agent count due to NumPy vectorization efficiency.
The scaling exponent is approximately 0.85, meaning:

* Doubling agents increases runtime by ~1.8x (not 2x)
* Time per agent *decreases* with larger simulations

This makes BAM Engine efficient for both small prototypes and large-scale studies.


.. seealso::

   - :doc:`configuration` for the full parameter reference
   - :doc:`data_collection` for collecting and exporting simulation data
   - :doc:`/quickstart` for a minimal working example
   - :doc:`/performance` for profiling and optimization guidance
