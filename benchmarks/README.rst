Profiling Tools
===============

cProfile-based profiling for detailed function-level performance analysis.

Files
-----

- ``profile_simulation.py``: Generate cProfile output for 1000 periods
- ``simulation_profile.prof``: Binary profile data for analysis with snakeviz

Usage
-----

Run profiling::

    python benchmarks/profile_simulation.py

Analyze results interactively::

    pip install snakeviz
    snakeviz benchmarks/simulation_profile.prof

When to Use
-----------

Use this tool when:

- You need function-level timing breakdown
- You want to identify performance bottlenecks
- You're debugging a regression found by ASV

Use ASV (``asv_benchmarks/``) when:

- You want to track performance across commits
- You need automatic regression detection

See Also
--------

- ``asv_benchmarks/`` - ASV performance tracking
