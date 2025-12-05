Running Simulations
===================

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

Performance & Scaling
---------------------

BAM Engine uses fully vectorized NumPy operations for high performance.
This section helps you choose appropriate configurations for your research needs.

Runtime Expectations
~~~~~~~~~~~~~~~~~~~~

Expected runtimes on modern hardware (Apple M4 Pro, Python 3.12):

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
     - ~0.4s
     - ~4s
     - ~250 periods/s
   * - Development
     - 200 firms, 1000 households
     - ~0.8s
     - ~8s
     - ~125 periods/s
   * - Production
     - 500 firms, 2500 households
     - ~2s
     - ~20s
     - ~50 periods/s

.. note::

   Actual performance varies by hardware. These benchmarks provide relative guidance
   for choosing configurations.

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

   For profiling and optimization guidance, see :doc:`/performance`.
