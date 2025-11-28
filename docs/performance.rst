Performance
===========

BAM Engine achieves excellent performance through fully vectorized NumPy operations.

Benchmarks
----------

Current benchmark results (Apple M4 Pro, macOS 15.1, Python 3.12):

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 10 15 15 15

   * - Configuration
     - Firms
     - Households
     - Banks
     - 1000 periods
     - 100 periods
     - Throughput
   * - Small
     - 100
     - 500
     - 10
     - 4.1s
     - 0.4s
     - 244 periods/s
   * - Medium
     - 200
     - 1,000
     - 10
     - 8.0s
     - 0.8s
     - 125 periods/s
   * - Large
     - 500
     - 2,500
     - 10
     - 20.5s
     - 2.1s
     - 49 periods/s

Performance scales sub-linearly with agent count due to NumPy vectorization efficiency.

Full ASV Results
----------------

For detailed benchmark history and regression tracking, see the
`ASV Benchmark Dashboard <https://kganitis.github.io/bam-engine/>`_.

The ASV dashboard provides:

* Historical performance tracking across commits
* Regression detection
* Interactive visualization
* Machine-specific baselines

Running Benchmarks Locally
--------------------------

Quick benchmarks
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Full simulation benchmark
   python benchmarks/bench_full_simulation.py

   # Profiling with cProfile
   python benchmarks/profile_simulation.py

ASV benchmarks
~~~~~~~~~~~~~~

For comprehensive benchmarking with historical tracking:

.. code-block:: bash

   cd asv_benchmarks

   # Benchmark current commit
   asv run

   # Check for regressions
   asv continuous HEAD~1 HEAD

   # View results
   asv publish && asv preview

Performance regression tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run performance regression tests
   pytest tests/performance/ -v -m regression

The regression tests ensure performance doesn't degrade beyond 15% of established baselines.

Optimization Notes
------------------

Key performance optimizations in BAM Engine:

* **Vectorization-first**: All agent-level operations use NumPy vectorized operations
* **Pre-allocation**: Fixed-size arrays allocated at initialization
* **In-place operations**: Use ``out=`` parameters to avoid temporary arrays
* **Sparse relationships**: COO format for memory-efficient loan book storage
* **Efficient sorting**: Use ``argpartition`` instead of ``argsort`` when full sort unnecessary

The critical path is the market queuing system (labor, credit, goods markets) which
requires Python loops for the multi-round matching process.
