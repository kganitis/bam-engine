Contributing
============

Thank you for your interest in contributing to BAM Engine!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a branch for your changes
5. Make your changes and add tests
6. Run the test suite:

   .. code-block:: bash

      pytest

7. Submit a pull request

Code Style
----------

We use:

* **Black** for code formatting
* **Ruff** for linting
* **Mypy** for type checking

Run all checks with:

.. code-block:: bash

   black src/ tests/ && ruff check --fix src/ tests/ && mypy src/
