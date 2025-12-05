ASV Benchmarks
==============

Airspeed Velocity (ASV) benchmarks for tracking performance across commits
with machine-specific baselines.

Benchmark Suites
----------------

**SimulationSuite**: Full simulation runs

- Configurations: small (100 firms), medium (200 firms), large (500 firms)
- Tests: 100 and 1000 period simulations

**PipelineSuite**: Single step performance

- Benchmarks one simulation step (all 39 events)

**MemorySuite**: Peak memory usage

- Tracks memory during initialization and 100-period runs
- Tests scaling from 100 to 500 firms

Quick Start
-----------

Run benchmarks::

    cd asv_benchmarks
    asv run

Compare commits::

    asv compare HEAD~5 HEAD

Generate HTML report::

    asv publish
    asv preview

Configuration
-------------

``asv.conf.json`` defines:

- Project and repository settings
- Python version (3.12)
- Build commands (pip wheel)
- Results storage (``.asv/results/``)

Results
-------

- Raw results: ``.asv/results/``
- HTML reports: ``.asv/html/``

See Also
--------

- ``benchmarks/`` - cProfile-based profiling tools
