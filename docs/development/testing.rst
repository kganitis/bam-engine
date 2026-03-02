Testing
=======

BAM Engine uses pytest with comprehensive coverage requirements.


Running Tests
-------------

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


Test Categories
---------------

* **Unit tests** (``tests/unit/``): Test individual components in isolation
* **Integration tests** (``tests/integration/``): Test event chains and full simulations
* **Property-based tests** (``tests/property/``): Hypothesis-driven randomized testing
* **Performance tests** (``tests/performance/``): Benchmark regression testing

Test Markers
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Description
   * - ``slow``
     - Long-running tests (integration, validation)
   * - ``regression``
     - Performance regression tests (skipped in CI)
   * - ``invariants``
     - Model invariant checks


Coverage
--------

The project maintains ~99% code coverage. Coverage is measured automatically
during ``pytest`` runs. Performance tests automatically disable coverage
instrumentation (see ``tests/performance/conftest.py``) to avoid measurement
distortion from ``sys.settrace`` overhead.
