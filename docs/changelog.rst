Changelog
=========

All notable changes to BAM Engine are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. note::

   Pre-1.0 releases (0.x.x) may introduce breaking changes between minor versions.

[0.2.0] - 2026-01-20
--------------------

This is a major release introducing validation and calibration frameworks, significant performance
improvements, and API enhancements for model extensibility.

Added
~~~~~

**Validation Package** (``validation/``)

* Complete acceptance testing framework against empirical targets from Delli Gatti et al. (2011)
* Two validation scenarios: **Baseline** (Section 3.9.1) and **Growth+** (Section 3.8)
* Core API: ``run_validation()``, ``run_stability_test()``, ``run_growth_plus_validation()``
* Scoring system with ``ValidationScore``, ``StabilityResult``, ``MetricResult`` types
* YAML target definitions (``validation/targets/``) with inline documentation
* Multi-seed stability testing for calibration workflows
* Reporting functions: ``print_validation_report()``, ``print_stability_report()``

**Calibration Package** (``calibration/``)

* Parameter optimization framework with four-phase calibration process:

  1. One-at-a-time (OAT) sensitivity analysis to identify impactful parameters
  2. Focused grid building with HIGH/MEDIUM/LOW sensitivity categorization
  3. Grid search screening using single seed
  4. Multi-seed stability testing for top candidates

* CLI interface: ``python -m calibration --scenario baseline --workers 10``
* Programmatic API: ``run_sensitivity_analysis()``, ``build_focused_grid()``, ``run_focused_calibration()``
* Parallel execution with ``ProcessPoolExecutor`` for speed
* Supports both baseline and growth_plus scenarios

**Event Pipeline Hooks**

* New ``@event`` decorator parameters for automatic pipeline positioning:

  * ``after="event_name"`` - insert after specified event
  * ``before="event_name"`` - insert before specified event
  * ``replace="event_name"`` - replace specified event

* Enables clean model extensions without manual ``insert_after()`` calls
* Example usage::

    @event(after="firms_pay_dividends")
    class MyCustomEvent:
        def execute(self, sim):
            ...  # Automatically positioned in pipeline

**Data Collection Enhancements**

* Timed capture system: capture data after specific events via ``capture_after="event_name"``
* Pipeline callback registration: ``pipeline.register_after_event(event_name, callback)``
* New ``SimulationResults.get_array()`` method for convenient data access::

    results.get_array("Producer", "price", aggregate="mean")

* New ``SimulationResults.data`` property for unified role + economy access
* Support for Economy as pseudo-role in data access

**Configuration Parameters**

* New ratio-based initialization parameters:

  * ``min_wage_ratio`` - minimum wage relative to mean wage
  * ``net_worth_ratio`` - net worth relative to production level
  * ``labor_productivity`` - goods produced per worker

* New firm entry parameters: ``new_firm_size_factor``, ``new_firm_production_factor``,
  ``new_firm_wage_factor``, ``new_firm_price_markup``
* New implementation variant parameters:

  * ``loan_priority_method`` - bank ranking: "by_leverage" | "by_net_worth" | "by_appearance"
  * ``firing_method`` - worker selection: "random" | "expensive"
  * ``matching_method`` - labor market: "sequential" | "simultaneous"
  * ``job_search_method`` - sampling: "vacancies_only" | "all_firms"

* New credit constraint parameters: ``max_loan_to_net_worth``, ``max_leverage``

**Operations Module**

* ``ops.exp()`` - exponential function for growth models
* ``ops.array()`` - create array copies (vs ``asarray()`` which doesn't copy)

**Benchmarking**

* Seven comprehensive ASV benchmark suites: SimulationSuite, PipelineSuite, MemorySuite,
  CriticalEventSuite, InitSuite, LoanBookSuite, ScalingSuite
* Quick pytest-benchmark suite for fast local feedback
* Baseline comparison functionality for performance regression detection

**CI/CD**

* Dedicated validation workflow (``.github/workflows/validation.yml``)
* New pytest marker ``@pytest.mark.validation`` for acceptance tests

**Examples**

* Growth+ extension example (850+ lines) demonstrating R&D-driven productivity growth,
  pipeline hooks, and extension parameters
* Enhanced baseline example (600+ lines) with full validation integration and 8-panel visualization

Changed
~~~~~~~

**Breaking: Configuration System**

* Replaced absolute initialization values with ratio-based parameters:

  * ``min_wage`` → ``min_wage_ratio``
  * ``net_worth_init`` → ``net_worth_ratio``
  * ``production_init`` removed (calculated from ``labor_productivity``)
  * ``wage_offer_init`` removed (calculated from balance sheets)

* Default agent counts changed: ``n_firms``: 100 → 300, ``n_households``: 500 → 3000
* Default dividend rate changed: ``delta``: 0.40 → 0.10
* Default bank capital ratio changed: ``v``: 0.06 → 0.10

**Breaking: Results API**

* ``results.economy`` renamed to ``results.economy_data``
* Data collection now requires explicit ``collect=True`` parameter in ``run()``

**Event System**

* Renamed ``FirmsCalcCreditMetrics`` → ``FirmsCalcFinancialFragility`` for clarity
* Reordered price events: breakeven calculation → price adjustment → production → average price update

**Performance**

* Optimized logging with per-event enabled/disabled flags (~75-80% faster event execution)
* Vectorized firm selection in labor and goods markets
* Optimized hot loops in goods and labor market events
* Overall throughput improved 50-80% compared to v0.1.2

  * Small (100 firms): ~500 periods/s
  * Medium (200 firms): ~240 periods/s
  * Large (500 firms): ~77 periods/s

**Testing**

* Achieved ~100% test coverage with targeted tests and pragma markers
* Added comprehensive tests for edge cases, error paths, and event hooks

**Documentation**

* New ``validation/README.md`` and ``calibration/README.md`` package documentation
* Expanded ``docs/development.rst`` with validation and calibration workflows
* Updated ``docs/user_guide/data_collection.rst`` with timed capture examples
* Updated ``docs/user_guide/pipelines.rst`` with pipeline hooks documentation

**Dependencies**

* Added ``scipy>=1.11`` for validation metrics (skewness, percentiles)
* Added ``pandas-stubs>=2.0`` for mypy type checking

Fixed
~~~~~

**Labor Market**

* Fixed worker wages not updating when minimum wage increases
* Fixed excess labor firing event placement in pipeline
* Fixed labor market worker shuffling for proper sequential matching

**Goods Market**

* Fixed loyalty and preferential attachment behavior in consumer firm selection
* Fixed price event ordering to use projected output for breakeven price calculation
* Fixed average market price update to account for current period's actual production

**Core**

* Fixed bank equity test for total market collapse scenarios

[0.1.2] - 2025-12-05
--------------------

Added
~~~~~

* **Documentation**: New section for API reference, Examples, Performance & Profiling, available on `Read the Docs <https://bam-engine.readthedocs.io>`_
* **Examples**: Basic and advanced examples using Sphinx Gallery format
* **Benchmarking**: ASV benchmarking workflow with `GitHub Pages deployment <https://kganitis.github.io/bam-engine/#/>`_
* **CI/CD**: `sp-repo-review <https://github.com/scientific-python/cookie>`_ compliance for Scientific Python standards
* **CI/CD**: pre-commit.ci and Dependabot integration

Changed
~~~~~~~

* Replaced black formatter with ruff-format
* Refactored all examples to use ``bam.ops`` instead of raw NumPy operations
* Improved docstrings throughout codebase for Sphinx compatibility

[0.1.1] - 2024-11-14
--------------------

Added
~~~~~

* **CI/CD**: GitHub Actions workflows for automated testing and PyPI publishing

  * Cross-platform testing on Ubuntu, macOS, and Windows
  * Multi-version testing for Python 3.11, 3.12, 3.13
  * Automated PyPI publishing on GitHub releases

Changed
~~~~~~~

* **Breaking**: Dropped Python 3.10 support. Minimum required Python version is now 3.11+

  * Python 3.10 had complex metaclass compatibility issues in decorators that are not worth maintaining
  * Supported versions: Python 3.11, 3.12, 3.13

Fixed
~~~~~

* Windows compatibility in test suite (file handle management, path separators)

[0.1.0] - 2025-11-13
--------------------

This release is feature-complete for the core BAM model but APIs may change in future releases.
Designed for academic research and policy analysis experiments.

Added
~~~~~

**Core ECS Architecture**

* **Role (Component)**: Dataclasses with NumPy arrays for agent state, auto-registration via ``__init_subclass__``
* **Event (System)**: Pure functions wrapped in Event classes, YAML-configurable pipeline execution
* **Relationship**: Many-to-many connections with COO sparse format (e.g., LoanBook for loans)
* **Registry**: Global auto-registration for roles, events, and relationships
* **Pipeline**: YAML-based event ordering with special syntax (repeat, interleave, parameter substitution)
* **Simulation engine**: ``Simulation`` class managing agents, roles, relationships, and event execution

**User-Facing API**

* **Simplified definitions**: ``@role``, ``@event``, ``@relationship`` decorators for easy creation of components
* **Type aliases**: ``Float``, ``Int``, ``Bool``, ``AgentId`` for bam-engine type hints
* **Operations module**: 30+ operations (``ops.add``, ``ops.divide``, etc.) that wrap NumPy operations with safe handling
* **Read methods**: ``sim.get_role()``, ``sim.get_event()``, ``sim.get_relationship()`` with case-insensitive lookup

**BAM Model Implementation**

* **3 agent populations**: Firms, Households, Banks
* **6 agent roles**: Producer, Employer, Borrower, Worker, Consumer, Lender
* **8 event modules**: Planning, labor/credit/goods markets, production, revenue, bankruptcy (39 total events)

**Configuration & Validation**

* **Three-tier configuration**: Package defaults → User YAML → Keyword arguments
* **Centralized validation**: Type checking, range validation, relationship constraints
* **Custom pipeline support**: Load custom event sequences via ``pipeline_path`` parameter
* **Logging configuration**: Global and per-event log levels (DEBUG, INFO, WARNING, ERROR, TRACE)

**Performance & Testing**

* **Vectorized Operations**: NumPy-based computations for efficient large-scale simulations
* **Sparse Relationships**: COO sparse matrix format for memory-efficient relationship storage
* **Benchmarking**: 172 periods/s (small), 96 periods/s (medium), 40 periods/s (large) - 4-9x faster than targets
* **Testing**: Unit tests, integration tests, property-based tests, performance regression tests, covering >99% of codebase
* **Deterministic RNG**: Reproducible simulations with seed control

----

*See the* `GitHub Releases <https://github.com/kganitis/bam-engine/releases>`_ *for complete release history.*
