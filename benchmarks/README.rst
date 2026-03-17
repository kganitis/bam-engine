Benchmarks
==========

Tools for profiling and stability benchmarking of BAM Engine simulations.

Profiling
---------

cProfile-based profiling for detailed function-level performance analysis.

Files
~~~~~

- ``profile_simulation.py``: Generate cProfile output for 1000 periods
- ``simulation_profile.prof``: Binary profile data for analysis with snakeviz

Usage
~~~~~

Run profiling::

    python benchmarks/profile_simulation.py

Analyze results interactively::

    pip install snakeviz
    snakeviz benchmarks/simulation_profile.prof

When to Use
~~~~~~~~~~~

Use this tool when:

- You need function-level timing breakdown
- You want to identify performance bottlenecks
- You're debugging a regression found by ASV

Use ASV (``asv_benchmarks/``) when:

- You want to track performance across commits
- You need automatic regression detection

Seed Stability Benchmarking
----------------------------

``bench_seed_stability.py`` runs large-scale seed stability tests across all
three validation scenarios (baseline, growth+, buffer-stock) with 1000 seeds
parallelized across 10 workers. Produces JSON result files for the
`bamengine.org stability dashboard <https://bamengine.org/stability/>`_.

Usage
~~~~~

Run against the current working tree::

    PYTHONPATH=src python benchmarks/bench_seed_stability.py

Run a single scenario::

    PYTHONPATH=src python benchmarks/bench_seed_stability.py --scenario baseline

Benchmark historical commits via git worktrees::

    python benchmarks/bench_seed_stability.py --tags v0.5.0..v0.6.2
    python benchmarks/bench_seed_stability.py --commits HEAD~5..HEAD

Preview what would run::

    PYTHONPATH=src python benchmarks/bench_seed_stability.py --dry-run

CLI Options
~~~~~~~~~~~

``--scenario {baseline,growth_plus,buffer_stock}``
    Run a single scenario instead of all three.

``--seeds N``
    Total number of seeds (default: 1000).

``--workers N``
    Number of parallel workers (default: 10).

``--commits SPEC [SPEC ...]``
    Specific commits or ranges (``X..Y``) to benchmark using git worktrees.

``--tags SPEC [SPEC ...]``
    Specific tags or ranges (``vX..vY``) to benchmark using git worktrees.

``--force``
    Allow more than 20 commits without confirmation.

``--dry-run``
    Print what would be run without executing.

Output Format
~~~~~~~~~~~~~

Results are saved as JSON files in ``benchmarks/results/`` (committed to the repository) with
the naming convention ``{scenario}_{commit_short}_{timestamp}.json``. Each file
contains:

- **metadata**: commit hash, tag, version, timestamp, seed/worker counts,
  elapsed time
- **summary**: pass rate, mean/std/min/max score
- **metrics**: per-metric mean, std, pass rate, weight, group
- **failing_seeds**: list of seeds that failed with their failing metric names

Workflow
~~~~~~~~

1. Run the benchmark: ``PYTHONPATH=src python benchmarks/bench_seed_stability.py``
2. Commit the JSON results: ``git add benchmarks/results/*.json && git commit``
3. Push to main — the ``validation-status`` CI workflow checks pass rates automatically
4. Tagged release results are automatically published to bamengine.org by the
   ``validation-status`` CI workflow (clears stale results before copying, so
   this directory is the single source of truth)

See Also
--------

- ``asv_benchmarks/`` - ASV performance tracking
- ``tests/performance/`` - pytest regression tests
