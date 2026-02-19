Performance & Profiling
=======================

This guide covers performance analysis, profiling, and optimization for developers
and power users extending BAM Engine.

.. note::

   For user-facing performance expectations and configuration guidance, see
   :ref:`user_guide/running_simulations:Performance & Scaling`.

Optimization Philosophy
-----------------------

Before optimizing, follow these principles:

1. **Readability first**: Clear code is maintainable code. Don't sacrifice clarity
   for marginal performance gains.

2. **Algorithm before implementation**: A better algorithm beats micro-optimization.
   Review the literature before investing in code-level optimization.

3. **Profile before optimizing**: Measure actual bottlenecks, not assumptions.
   The critical path is often surprising.

4. **Verify improvements**: Always benchmark before and after changes to confirm
   the optimization works as expected.

Benchmark Results
-----------------

Current benchmarks (Apple M4 Pro, macOS 15.2, Python 3.13):

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 10 15 15

   * - Configuration
     - Firms
     - Households
     - Banks
     - 1000 periods
     - Throughput
   * - Small
     - 100
     - 500
     - 10
     - 2.1s
     - 475 periods/s
   * - Medium
     - 200
     - 1,000
     - 10
     - 4.4s
     - 225 periods/s
   * - Large
     - 500
     - 2,500
     - 10
     - 13.6s
     - 73 periods/s

Performance scales approximately linearly with agent count. While NumPy vectorization
is highly efficient, the per-agent computation cost means doubling the number of
agents roughly doubles simulation time.

For historical benchmark tracking across commits, see the
`ASV Benchmark Dashboard <https://kganitis.github.io/bam-engine/>`_.

Profiling
---------

cProfile
~~~~~~~~

Generate function-level timing breakdown using the built-in profiling script:

.. code-block:: bash

   python benchmarks/profile_simulation.py

This runs a 1000-period simulation and outputs:

* Top 30 functions by cumulative time (including subcalls)
* Top 30 functions by self time (time in function itself)
* Binary profile saved to ``benchmarks/simulation_profile.prof``

Visualize the profile interactively with snakeviz:

.. code-block:: bash

   pip install snakeviz
   snakeviz benchmarks/simulation_profile.prof

IPython %prun
~~~~~~~~~~~~~

For quick profiling in interactive sessions:

.. code-block:: python

   import bamengine as bam
   sim = bam.Simulation.init(seed=42)

   # Overall profile sorted by cumulative time
   %prun -s cumulative sim.run(100)

   # Profile a single step
   %prun -s tottime sim.step()

Line-level Profiling
~~~~~~~~~~~~~~~~~~~~

For detailed line-by-line analysis, use ``line_profiler``:

.. code-block:: bash

   pip install line_profiler

In IPython/Jupyter:

.. code-block:: python

   %load_ext line_profiler

   # Profile the step method
   %lprun -f sim.step sim.run(10)

   # Profile a specific internal function
   from bamengine.events._internal.goods_market import consumers_decide_firms_to_visit
   %lprun -f consumers_decide_firms_to_visit sim.run(10)

Memory Profiling
----------------

Track memory usage with ``memory_profiler``:

.. code-block:: bash

   pip install memory_profiler

In IPython/Jupyter:

.. code-block:: python

   %load_ext memory_profiler

   # Peak memory for a run
   %memit sim.run(100)

   # Line-by-line memory usage
   %mprun -f sim.step sim.run(10)

ASV Benchmarking
----------------

BAM Engine uses `ASV (Airspeed Velocity) <https://asv.readthedocs.io/>`_ for
automated performance tracking with machine-specific baselines.

Running Benchmarks
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd asv_benchmarks

   # Benchmark current commit
   asv run

   # Compare two commits
   asv continuous HEAD~1 HEAD

   # Compare specific benchmark
   asv continuous -b SimulationSuite HEAD~1 HEAD

   # Generate and view HTML report
   asv publish && asv preview

Benchmark Suites
~~~~~~~~~~~~~~~~

The ASV configuration includes seven benchmark suites:

* **SimulationSuite**: Full simulation runs (100/1000 periods) across small/medium/large
* **PipelineSuite**: Single step performance (all events)
* **MemorySuite**: Peak memory during initialization and simulation
* **CriticalEventSuite**: Individual event benchmarks for the critical path (goods/labor/credit markets)
* **InitSuite**: Initialization costs across different scales (100-1000 firms)
* **LoanBookSuite**: Sparse relationship operations (append, aggregate, purge)
* **ScalingSuite**: Performance scaling analysis with agent count (50-400 firms)

Quick Benchmarks (pytest-benchmark)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For fast local benchmarking during development, use the pytest-benchmark tests:

.. code-block:: bash

   # Run all quick benchmarks
   pytest tests/performance/test_quick_benchmarks.py -v

   # With detailed statistics
   pytest tests/performance/test_quick_benchmarks.py -v --benchmark-verbose

   # Save baseline for comparison
   pytest tests/performance/test_quick_benchmarks.py --benchmark-save=baseline

   # Compare against saved baseline
   pytest tests/performance/test_quick_benchmarks.py --benchmark-compare

These benchmarks cover core operations (single step, initialization) and critical
events (goods/labor market operations) with a small configuration for quick feedback.

Regression Testing
~~~~~~~~~~~~~~~~~~

Performance regression tests run automatically as part of the ``pytest`` suite.
Coverage instrumentation is automatically disabled for performance tests (see
``tests/performance/conftest.py``) to avoid measurement distortion from
``sys.settrace`` overhead, which can inflate timings by 40-60%.

.. code-block:: bash

   # Run regression tests (included in full pytest runs)
   pytest -m regression

   # Run only performance tests
   pytest tests/performance/ -v

These tests ensure performance doesn't degrade beyond 15% of established
baselines. Baselines are machine-specific and must be updated manually after
confirmed improvements.

ASV benchmarks complement these tests with cross-commit tracking and
machine-specific baselines, but require separate invocation and do not
integrate into ``pytest``.

Architecture Performance Notes
------------------------------

Key optimizations in BAM Engine's architecture:

Vectorization
~~~~~~~~~~~~~

All agent-level operations use NumPy vectorized operations. Avoid Python loops
over agents:

.. code-block:: python

   from bamengine import ops

   # Good: vectorized (fast)
   ops.assign(role.price, ops.multiply(role.price, 1.1))

   # Bad: Python loop (slow)
   for i in range(len(role.price)):
       role.price[i] *= 1.1

Pre-allocation
~~~~~~~~~~~~~~

Fixed-size arrays are allocated at initialization. Avoid dynamic allocation
during simulation:

.. code-block:: python

   # Good: use pre-allocated scratch buffer
   ops.assign(role.scratch_buffer, computed_values)

   # Bad: allocate new array each step
   role.scratch_buffer = np.zeros(n_agents)

Sparse Relationships
~~~~~~~~~~~~~~~~~~~~

The LoanBook uses COO (Coordinate List) sparse format for memory efficiency:

* Storage: O(active_loans) instead of O(n_firms × n_banks)
* Efficient append: amortized O(1) with capacity doubling
* Vectorized aggregation: ``np.bincount()`` and ``np.add.at()``

Efficient Sorting
~~~~~~~~~~~~~~~~~

Use ``argpartition`` instead of ``argsort`` when only top-k elements are needed:

.. code-block:: python

   from bamengine.utils import select_top_k_indices_sorted

   # Get top 10 indices efficiently
   top_k = select_top_k_indices_sorted(values, k=10)

Critical Path
~~~~~~~~~~~~~

The market queuing system (labor, credit, goods markets) contains the primary
bottlenecks. Most operations are now vectorized, but some sequential matching
remains:

* **Goods market**: ``consumers_decide_firms_to_visit`` is fully vectorized using
  batch random sampling and 2D array operations. ``consumers_shop_sequential``
  requires sequential processing due to inventory state dependencies.
* **Labor market**: ``workers_decide_firms_to_apply``, ``firms_hire_workers``
  use sequential queue processing for multi-round matching.
* **Credit market**: Similar sequential matching for loan applications.

The sequential shopping and hiring loops are inherently O(n) and cannot be
fully parallelized due to state-dependent matching where each transaction
affects subsequent ones.
