Changelog
=========

All notable changes to BAM Engine are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. note::

   Pre-1.0 releases (0.x.x) may introduce breaking changes between minor versions.

[Unreleased]
------------

Added
~~~~~

* Robustness analysis package (Section 3.10 of Delli Gatti et al., 2011):

  * ``validation/robustness/`` package with internal validity and sensitivity analysis
  * Internal validity: multi-seed simulation (20 seeds) verifying cross-simulation
    stability of co-movements, AR structure, firm size distributions, and empirical curves
  * Sensitivity analysis: univariate parameter sweeps across 5 experiment groups —
    credit markets (H), goods markets (Z), labor markets (M), contract length (theta),
    and economy size/composition
  * Statistical tools: Hodrick-Prescott filter, lead-lag cross-correlations,
    AR model fitting (OLS), impulse-response function computation
  * Visualization: co-movement plots (Figure 3.9 replica), IRF comparison,
    sensitivity co-movement comparison
  * Text reporting: formatted cross-simulation variance tables, co-movement
    summaries, AR fit results, empirical curve persistence
  * CLI entry point: ``python -m validation.robustness``
  * Example: ``examples/advanced/example_robustness.py``

Changed
~~~~~~~

* Default agent counts unified to match book setup: ``n_firms``: 300 → 100,
  ``n_households``: 3000 → 500
* Default new-firm entry parameters: ``new_firm_size_factor`` 0.8 → 0.5,
  ``new_firm_production_factor`` 0.9 → 0.5, ``new_firm_wage_factor`` 0.9 → 0.5,
  ``new_firm_price_markup`` 1.0 → 1.5
* Default ``max_loan_to_net_worth``: 2 → 5, ``job_search_method``:
  ``"vacancies_only"`` → ``"all_firms"``
* ``Scenario.default_config`` now optional (defaults to ``{}``)
* Growth+ and buffer-stock scenarios no longer override agent counts
* Baseline targets: updated log GDP and unemployment bounds for 100-firm/500-household economy
* Renamed 'destroyed' flag to 'collapsed' in ``Simulation``

Fixed
~~~~~

* Bankruptcy counter now captures correct counts with basic ``collect=True``.
  ``exiting_firms`` / ``exiting_banks`` arrays are no longer cleared in
  ``spawn_replacement_*``, so they persist through end-of-period data capture.
* Spawned replacement firms set ``production = production_prev`` to avoid
  immediate ghost-firm re-bankruptcy on the next period.

Removed
~~~~~~~

* ``SMALL_ECONOMY_CONFIG`` and ``validation/scenarios/_configs.py``
  (redundant with unified defaults)
* ``capture_timing`` workarounds for ``Economy.n_firm_bankruptcies`` /
  ``Economy.n_bank_bankruptcies`` in scenarios and examples (no longer needed)


[0.3.0] - 2026-02-11
--------------------

This release adds the buffer-stock consumption extension (Section 3.9.4 of
Delli Gatti et al., 2011), completing all three BAM scenarios from the book,
and introduces a composable extension API (``use_events``, ``use_config``,
``use_role`` with ``n_agents``) that replaces the global hook registry.

Added
~~~~~

**Consumption & Buffer-stock Extension**

* Consumption & buffer-stock extension (Chapter 3.9.4 of Delli Gatti et al., 2011):

  * ``extensions/buffer_stock/`` package with ``BufferStock`` role and replacement events
  * Individual adaptive MPC based on buffer-stock saving theory (Eq. 3.20)
  * Validation scenario with ~30 metrics: wealth CCDF fitting (Singh-Maddala, Dagum, GB2),
    Gini coefficient, MPC distribution, baseline macro dynamics
  * 8-panel visualization with CCDF plot (Figure 3.8 replica)
  * Example: ``examples/extensions/example_buffer_stock.py``

* ``Shareholder`` role — built-in household role tracking per-period dividends, used to
  adjust buffer-stock MPC metrics for the dividend artifact

**Composable Extension API**

* ``Simulation.use_config(config)`` — apply extension default configuration with "don't overwrite"
  semantics (user kwargs and earlier ``use_config()`` calls take precedence)
* ``Simulation.use_events(*event_classes)`` — explicitly apply pipeline hooks from event classes
* ``Simulation.use_role(cls, n_agents=N)`` — ``n_agents`` parameter for non-firm roles
* ``Pipeline.apply_hooks(*event_classes)`` — read hook metadata from classes and apply to pipeline

Changed
~~~~~~~

**Breaking**

* ``@event(after=..., before=..., replace=...)`` now stores hook metadata as class attributes
  instead of writing to a global dict. Hooks are applied explicitly via ``sim.use_events()``.
  Removed ``_EVENT_HOOKS`` global dict, ``register_event_hook()``, ``get_event_hooks()``,
  and the ``apply_hooks`` parameter from ``Pipeline.from_yaml()`` /
  ``Pipeline.from_event_list()``.
* Default ``aggregate`` for dict-form ``collect`` changed from ``"mean"`` to ``None``.
  Callers using ``collect={...}`` without an ``"aggregate"`` key now get full per-agent
  data. Add ``"aggregate": "mean"`` explicitly for the previous behavior.
  ``collect=True`` and list-form ``collect`` are unchanged (still aggregate with mean).

**Scenario Validation**

* Weight-based fail escalation for validation status checks — high-weight metrics
  fail more easily, low-weight metrics are more lenient. New ``escalation``
  parameter on status check functions. This also resolves the growth+ seed stability
  test failures in v0.2.2 by widening the WARN zone for lower-weight metrics.
* Validation scenarios restructured into self-contained directory packages
  (``validation/scenarios/{baseline,growth_plus,buffer_stock}/``), each
  co-locating code, visualization, targets, and output.
  ``Scenario.targets_file`` renamed to ``Scenario.targets_path: Path``;
  new ``SCENARIO_REGISTRY`` dict and ``get_scenario()`` lookup function.
* Widened some validation targets to reduce false negatives from seed sensitivity

[0.2.2] - 2026-02-09
--------------------

This release completes the Growth+ scenario (Section 3.9.2 of Delli Gatti et al., 2011), bringing its
validation metrics, financial dynamics analysis, and visualizations in line with the book, and re-calibrates
model defaults accordingly.

Added
~~~~~

* Growth+ financial dynamics metrics: Minsky classification, recession detection, GDP cyclicality
  correlations, Laplace tent-shape R², CV-based dispersion checks
* ``max_leverage`` and ``inflation_method`` configuration parameters
* Relationship data collection in ``SimulationResults``; new ``n_firm_bankruptcies``, ``n_bank_bankruptcies`` economy metrics
* ``extensions/rnd/`` standalone package (R&D extension extracted from validation)

Changed
~~~~~~~

**Breaking**

* Re-calibrated defaults: ``new_firm_size_factor`` 0.9→0.8, ``max_loan_to_net_worth`` 100→2,
  ``max_leverage`` uncapped→10, ``matching_method`` "sequential"→"simultaneous";
  removed ``cap_factor`` and ``fragility_cap_method``
* Restructured validation package: ``validation.core``/``runners``/``metrics`` → ``validation.types``/``scoring``/``engine``
* ``CalcAnnualInflationRate`` → ``CalcInflationRate`` (pipeline key: ``calc_inflation_rate``)
* R&D import path: ``from extensions.rnd import RnD`` (was in validation package)

**Events Pipeline**

* ``firms_calc_breakeven_price`` and ``firms_adjust_price`` events moved to production phase (before ``firms_run_production``)

Known Issues
~~~~~~~~~~~~

* Growth+ seed stability test failures are a calibration issue and will be addressed in a future release.
  The baseline scenario passes the stability test at 100%.

[0.2.1] - 2026-01-21
--------------------

This patch release simplifies examples and reorganizes the validation package for better maintainability.

Changed
~~~~~~~

**Validation Package**

* Restructured validation package into subpackages:

  * ``validation/metrics/`` - metric computation from simulation results (``baseline.py``, ``growth_plus.py``)
  * ``validation/scenarios/`` - detailed visualizations with target bounds
  * ``validation/targets/`` - YAML target definitions (unchanged)

* Moved RnD role and events to ``validation/scenarios/growth_plus_extension.py``

**Examples Simplification**

* Simplified ``example_baseline_scenario.py`` and ``example_growth_plus.py`` to be self-contained
* Examples focus on teaching core BAM Engine concepts; users are directed to validation package for detailed analysis with bounds

Fixed
~~~~~

* Fixed Sphinx Gallery compatibility by handling missing ``__file__`` in exec() environment

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
