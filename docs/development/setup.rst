Development Setup & Code Style
================================

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
   mypy

   # All checks in one command
   ruff format . && ruff check --fix . && mypy

The configuration in ``pyproject.toml`` automatically includes all Python packages
(``src/`` covers ``bamengine``, ``validation``, ``extensions``, ``calibration``;
plus ``tests/``, ``diagnostics/``, ``benchmarks/``, ``examples/``, and ``docs/conf.py``)
while excluding generated files.


Building Documentation
----------------------

Build the documentation locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

   # View in browser
   open _build/html/index.html  # macOS
   # Or: xdg-open _build/html/index.html  # Linux
