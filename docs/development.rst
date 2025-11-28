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

* `Black <https://github.com/psf/black>`_ for code formatting
* `Ruff <https://github.com/astral-sh/ruff>`_ for linting
* `Mypy <http://mypy-lang.org/>`_ for type checking

Run all code quality checks:

.. code-block:: bash

   # Format code (auto-fixes)
   black .

   # Lint code (auto-fixes when possible)
   ruff check --fix .

   # Type checking
   mypy src/

   # All checks in one command
   black . && ruff check --fix . && mypy src/

The configuration in ``pyproject.toml`` automatically includes ``src/``, ``tests/``,
``benchmarks/``, ``examples/``, and ``docs/conf.py`` while excluding generated files.

Testing
-------

Run the test suite with pytest:

.. code-block:: bash

   # Run all tests with coverage (requires 95% coverage)
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

Benchmarking
------------

BAM Engine uses `ASV (Airspeed Velocity) <https://asv.readthedocs.io/>`_ for
performance benchmarking with historical tracking.

.. code-block:: bash

   cd asv_benchmarks

   # Run benchmarks on current commit
   asv run

   # Check for regressions between commits
   asv continuous HEAD~1 HEAD

   # Publish and view results
   asv publish && asv preview

For quick local benchmarks and profiling:

.. code-block:: bash

   # Macro-benchmarks (full simulation)
   python benchmarks/bench_full_simulation.py

   # Profiling with cProfile
   python benchmarks/profile_simulation.py

See :doc:`performance` for current benchmark results.

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
5. Run code quality checks (``black . && ruff check --fix . && mypy src/``)
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
