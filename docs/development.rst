Development
===========

This guide covers setting up a development environment and contributing to BAM Engine.

Setup
-----

Clone the repository and install in development mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/kganitis/bam-engine.git
   cd bam-engine

   # Create and activate a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in editable mode with development dependencies
   pip install -e ".[dev]"

Code Style
----------

BAM Engine uses the following tools for code quality:

* `Ruff <https://github.com/astral-sh/ruff>`_ for formatting and linting
* `Mypy <http://mypy-lang.org/>`_ for type checking

Run all code quality checks:

.. code-block:: bash

   # Format code (auto-fixes)
   ruff format .

   # Lint code (auto-fixes when possible)
   ruff check --fix .

   # Type checking
   mypy src/

   # All checks in one command
   ruff format . && ruff check --fix . && mypy src/

The configuration in ``pyproject.toml`` automatically includes ``src/``, ``tests/``,
``benchmarks/``, ``examples/``, and ``docs/conf.py`` while excluding generated files.

Testing
-------

Run the test suite with pytest:

.. code-block:: bash

   # Run all tests with coverage (requires 99% coverage)
   pytest

   # Run specific test file
   pytest tests/unit/events/internal/test_labor_market.py

   # Run specific test function
   pytest tests/unit/events/internal/test_labor_market.py::test_function_name

   # Run with verbose output
   pytest -v

   # Run quick tests only (skip slow tests)
   pytest -m "not slow and not regression and not invariants"

Test categories:

* **Unit tests** (``tests/unit/``): Test individual components in isolation
* **Integration tests** (``tests/integration/``): Test event chains and full simulations
* **Property-based tests** (``tests/property/``): Hypothesis-driven randomized testing
* **Performance tests** (``tests/performance/``): Benchmark regression testing

Performance & Benchmarking
--------------------------

BAM Engine uses two complementary approaches for performance monitoring:

* **Pytest regression tests** (``tests/performance/test_regression.py``): Automated
  pass/fail checks with hardcoded baselines that run as part of ``pytest``. Coverage
  is automatically disabled for these tests (see ``tests/performance/conftest.py``)
  to avoid measurement distortion from ``sys.settrace`` overhead.
* **ASV benchmarks** (``asv_benchmarks/``): Cross-commit performance tracking with
  machine-specific baselines and historical analysis. Requires separate invocation.

Pytest Regression Tests
~~~~~~~~~~~~~~~~~~~~~~~

Regression tests run automatically during ``pytest`` and catch performance
regressions beyond a 15% threshold:

.. code-block:: bash

   # Run regression tests (included in full pytest runs)
   pytest -m regression

   # Run quick benchmarks with pytest-benchmark
   pytest tests/performance/test_quick_benchmarks.py -v

Baselines are machine-specific and must be updated manually after confirmed
improvements. These tests are skipped in CI due to virtualization variance.

ASV (Airspeed Velocity)
~~~~~~~~~~~~~~~~~~~~~~~

For detailed cross-commit performance tracking:

.. code-block:: bash

   cd asv_benchmarks

   # Run benchmarks on current commit
   asv run

   # Check for regressions between commits
   asv continuous HEAD~1 HEAD

   # Target a specific suite
   asv continuous -b SimulationSuite HEAD~1 HEAD

   # Publish and view results
   asv publish && asv preview

Report significant changes (>5% improvement or regression) in pull request
descriptions.

Profiling
~~~~~~~~~

For detailed function-level profiling:

.. code-block:: bash

   python benchmarks/profile_simulation.py
   snakeviz benchmarks/simulation_profile.prof

See :doc:`performance` for comprehensive profiling guidance, optimization tips,
and current benchmark results.

Validation & Calibration
------------------------

BAM Engine includes two packages for model validation and parameter calibration.

Validation Package
~~~~~~~~~~~~~~~~~~

The ``validation/`` package validates simulation results against empirical targets
from Delli Gatti et al. (2011).

.. code-block:: python

   from validation import run_validation, run_stability_test

   # Single validation run
   result = run_validation(seed=42, n_periods=1000)
   print(f"Score: {result.total_score:.3f}, Passed: {result.passed}")

   # Multi-seed stability test
   stability = run_stability_test(seeds=[0, 42, 123, 456, 789])
   print(f"Mean: {stability.mean_score:.3f} ± {stability.std_score:.3f}")
   print(f"Pass rate: {stability.pass_rate:.0%}")

**Scenarios:**

* **baseline** (Section 3.9.1): Standard BAM model - validates unemployment, inflation,
  GDP, Phillips/Okun/Beveridge curves, firm size distribution
* **growth_plus** (Section 3.8): Endogenous productivity growth via R&D investment

.. code-block:: python

   # Growth+ scenario
   from validation import run_growth_plus_validation, run_growth_plus_stability_test

   result = run_growth_plus_validation(seed=42, n_periods=1000)
   stability = run_growth_plus_stability_test(seeds=[0, 42, 123, 456, 789])

**Key functions:**

* ``run_validation()`` / ``run_growth_plus_validation()`` - Single run validation
* ``run_stability_test()`` / ``run_growth_plus_stability_test()`` - Multi-seed testing
* ``print_validation_report()`` / ``print_growth_plus_report()`` - Formatted output

**Weight-based fail escalation:**

Status checks use a weight-derived escalation multiplier to adjust the WARN→FAIL
boundary. High-weight metrics (e.g., 3.0) fail more strictly, while low-weight
metrics (e.g., 0.5) require extreme deviations to FAIL. BOOLEAN checks are exempt.
See ``validation/README.md`` for the full mapping table.

See ``validation/README.md`` for the full API reference.

Calibration Package
~~~~~~~~~~~~~~~~~~~

The ``calibration/`` package finds optimal parameter values through sensitivity
analysis and focused grid search.

**Command line usage:**

.. code-block:: bash

   # Run sensitivity analysis only
   python -m calibration --phase sensitivity --workers 10

   # Full calibration (baseline scenario)
   python -m calibration --workers 10 --periods 1000

   # Calibrate Growth+ scenario
   python -m calibration --scenario growth_plus --workers 10

**Programmatic usage:**

.. code-block:: python

   from calibration import (
       run_sensitivity_analysis,
       build_focused_grid,
       run_focused_calibration,
       print_sensitivity_report,
   )

   # Phase 1: Sensitivity analysis
   sensitivity = run_sensitivity_analysis(scenario="baseline", n_workers=10)
   print_sensitivity_report(sensitivity)

   # Phase 2: Build focused grid
   grid, fixed = build_focused_grid(sensitivity)

   # Phases 3-4: Grid search + tiered stability (default tiers: 100:10, 50:20, 10:100)
   results = run_focused_calibration(grid, fixed)
   print(f"Best: {results[0].combined_score:.4f}")

**Calibration process:**

1. **Sensitivity Analysis**: One-at-a-time (OAT) testing to identify impactful parameters
2. **Build Focused Grid**: Categorize parameters by sensitivity (HIGH/MEDIUM/LOW)
3. **Grid Search Screening**: Test combinations using single seed
4. **Stability Testing**: Multi-seed validation of top candidates

See ``calibration/README.md`` for CLI options and full API reference.

Building Documentation
----------------------

Build the documentation locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

   # View in browser
   open _build/html/index.html  # macOS
   # Or: xdg-open _build/html/index.html  # Linux

Contributing
------------

.. note::

   This project is currently not accepting external contributions as it is part
   of ongoing thesis work. Once the thesis is submitted, contribution guidelines
   will be published.

For bug reports and feature requests, please open an issue on
`GitHub <https://github.com/kganitis/bam-engine/issues>`_.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

When contributions are open:

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/my-feature``)
3. Make changes and add tests
4. Run the full test suite (``pytest``)
5. Run code quality checks (``ruff format . && ruff check --fix . && mypy src/``)
6. Commit with a descriptive message
7. Push and open a pull request

Docstring Style
~~~~~~~~~~~~~~~

BAM Engine uses NumPy-style docstrings. See the
`NumPy docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
for the format specification.

Example:

.. code-block:: python

   def my_function(param1: float, param2: int) -> bool:
       """
       Short description of the function.

       Longer description if needed.

       Parameters
       ----------
       param1 : float
           Description of param1.
       param2 : int
           Description of param2.

       Returns
       -------
       bool
           Description of return value.

       Examples
       --------
       >>> my_function(1.0, 2)
       True
       """
       pass
