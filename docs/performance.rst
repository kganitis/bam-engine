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

Current benchmarks (Apple M4 Pro, macOS 15.1, Python 3.12):

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

The ASV configuration includes three benchmark suites:

* **SimulationSuite**: Full simulation runs (100/1000 periods) across small/medium/large
* **PipelineSuite**: Single step performance (all 39 events)
* **MemorySuite**: Peak memory during initialization and simulation

Regression Testing
~~~~~~~~~~~~~~~~~~

Performance regression tests run in the test suite:

.. code-block:: bash

   pytest tests/performance/ -v -m regression

These tests ensure performance doesn't degrade beyond 15% of established baselines.

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

The market queuing system (labor, credit, goods markets) requires Python loops
for multi-round matching. This is the primary bottleneck, accounting for
approximately 50% of simulation time:

* **Goods market**: ~48% (``consumers_decide_firms_to_visit``, ``consumers_shop_one_round``)
* **Labor market**: ~5% (``workers_decide_firms_to_apply``, ``firms_hire_workers``)
* **Credit market**: ~5%

These loops are inherently sequential and cannot be easily vectorized due to
the state-dependent matching process.
