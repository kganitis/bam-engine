Release History
===============

All notable changes to BAM Engine are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. note::

   Pre-1.0 releases (0.x.x) may introduce breaking changes between minor versions.

[0.9.1] - 2026-03-19
---------------------

This patch adds a convenience ``log_level`` parameter to ``Simulation.init()``
for simpler logging configuration.

Added
~~~~~

* ``log_level`` keyword argument on ``Simulation.init()``: shorthand for
  ``logging={"default_level": "..."}``
  (e.g., ``Simulation.init(seed=42, log_level="WARNING")``).

[0.9.0] - 2026-03-19
---------------------

This release redesigns the results collection API with intuitive access patterns,
automatic economy collection, and unaggregated data by default.

Added
~~~~~

**Results API**

* String-key access: ``results["Producer.production"]``,
  ``results["Economy.inflation"]``.
* Attribute access: ``results.Producer.production``,
  ``results.Economy.inflation``.
* ``results.get("Producer", "production", aggregate="mean")`` for on-demand
  aggregation, replacing the deprecated ``get_array()`` method.
* ``results.available()`` lists all collected data fields.
* ``sim.collectables()`` lists what data can be collected before calling ``run()``.

Changed
~~~~~~~

**Results Collection Defaults**

* ``sim.run()`` now returns results by default (``collect=True``).
* ``collect=True`` and list-form ``collect`` now produce **unaggregated 2D
  arrays** with shape ``(n_periods, n_agents)`` instead of mean-aggregated 1D
  arrays. Use ``results.get(..., aggregate="mean")`` for the previous behavior.
* Economy metrics (average price, inflation, etc.) are always collected automatically
  when collection is active.

Deprecated
~~~~~~~~~~

* ``SimulationResults.get_array()`` is deprecated in favor of
  ``SimulationResults.get()``.

Removed
~~~~~~~

* Dead ``unemp_rate_history`` field from ``Economy`` (was never updated after
  the ``CalcUnemploymentRate`` event was removed in v0.5.0).

[0.8.0] - 2026-03-17
---------------------

This release extends the calibration toolkit with four new composable tools,
recalibrates all validation targets from extracted book figure data,
and redesigns buffer-stock validation around aggregate improvement over Growth+.

Added
~~~~~

**Calibration Tools**

* Four new composable CLI tools for multi-scenario calibration workflows:
  :doc:`rescreen </calibration/rescreen>` (second-pass Morris screening with locked parameters),
  :doc:`cost </calibration/cost_analysis>` (budget-aware cost classification),
  :doc:`cross-eval </calibration/cross_eval>` (cross-scenario ranking with harmonic/geometric/min strategies),
  and :doc:`sweep </calibration/sweep>` (multi-stage parameter sweep with carry-forward winners).

Changed
~~~~~~~

**Validation Targets**

* All three scenarios recalibrated from pixel-level extraction of book figures
  (see :doc:`/validation/targets`).

**Buffer-Stock Validation**

* Replaced the duplicated 30-metric system with a focused two-layer approach:
  8 unique buffer-stock metrics (wealth fits, Gini, MPC, dissaving) for per-seed
  validation, plus aggregate improvement over Growth+ assessed at stability level.

**Calibrated Defaults**

* ``new_firm_price_markup``: 1.15 → 1.20.

Fixed
~~~~~

* Division-by-zero in ``score_outlier_penalty`` when ``max_outlier_pct=0``.
* Collapsed buffer-stock result incorrectly reporting ``passed=True``.

[0.7.0] - 2026-03-13
---------------------

This release replaces the v0.6.0 batch-sequential goods market with a pure
sequential implementation.

Changed
~~~~~~~

**Sequential Goods Market**

* ``goods_market_round`` now processes consumers **one at a time** using a
  sequential Python-list loop, replacing the batch-sequential approach (~10
  randomized consumer batches) from v0.6.0. Each consumer completes all shopping
  visits before the next consumer starts, eliminating many "phantom goods"
  overselling events per period caused by within-batch inventory collisions.
  See :ref:`decision-sequential-shopping` for design rationale.

**Validation Metrics**

* ``price_ratio_floor`` (Growth+): changed from global minimum to 1st
  percentile, which filters transient demand surges at full employment while still
  catching genuine deflationary spirals; weight also lowered from 3.0 to 2.0.

Performance
~~~~~~~~~~~

* Sequential goods market is **6.5% faster** than batch-sequential for full
  simulations. ``goods_market_round`` itself dropped **35%**.

Removed
~~~~~~~

* ``n_batches`` config parameter (only needed for batch-sequential goods market).

Development
~~~~~~~~~~~

**Seed-Stability Testing Overhaul**

* Seed stability tests upgraded from 20 to **100 seeds** per scenario, with parallel
  execution via new ``n_workers`` parameter.
* New ``benchmarks/bench_seed_stability.py``, a parallelized 1000-seed benchmark
  runner with JSON output, git-worktree support for historical commits, and CLI
  (``--scenario``, ``--seeds``, ``--tags``, ``--commits``). Results committed to
  ``benchmarks/results/``.
* New ``.github/workflows/validation-status.yml`` replaces ``validation.yml`` and
  ``growth-plus-stability.yml``; reads pre-computed JSON in ``benchmarks/results/``
  with **zero simulation** in CI.

[0.6.1] - 2026-03-10
---------------------

This patch eliminates the last Python loops in grouped random selection
(``resolve_conflicts``, ``firms_fire_excess_workers``, ``firms_fire_workers``)
by replacing per-group ``rng.choice``/``rng.permutation`` calls with a
vectorized random-priority + lexsort pattern.

Performance
~~~~~~~~~~~

* **Vectorized** ``resolve_conflicts``: replaced per-target Python loop +
  ``rng.choice`` with ``rng.random()`` + ``np.lexsort`` + grouped rank
  comparison. **3.4x faster** (0.100s → 0.029s, 7.5% → 2.3% of runtime).
* **Vectorized** ``firms_fire_excess_workers``: replaced per-firm
  ``rng.permutation`` loop with ``_flatten_and_shuffle_groups`` helper +
  rank threshold.
* **Vectorized** ``firms_fire_workers``: replaced per-firm cumsum loop with
  ``_flatten_and_shuffle_groups`` + ``grouped_cumsum`` +
  ``np.minimum.reduceat``.
* New shared helper ``_flatten_and_shuffle_groups`` extracts items from ragged
  group boundaries into a flat shuffled array using random-priority lexsort.

[0.6.0] - 2026-03-09
---------------------

This release consolidates two parallel market matching implementations (sequential
Python loops vs. vectorized NumPy batches) into a single vectorized implementation.
Simulation behavior is preserved.

Changed
~~~~~~~

**Vectorized Market Matching**

* **Labor market**: ``labor_market_round`` replaces ``workers_send_one_round`` +
  ``firms_hire_workers``. Batch conflict resolution via ``resolve_conflicts()`` utility.
  Pipeline calls it ``max_M`` times (one round per call).
* **Credit market**: ``credit_market_round`` replaces ``firms_send_one_loan_app`` +
  ``banks_provide_loans``. Firms sorted by ``projected_fragility`` (ascending leverage)
  within each target bank; ``grouped_cumsum()`` tracks per-bank supply exhaustion.
  Pipeline calls it ``max_H`` times.
* **Goods market**: ``goods_market_round`` replaces ``consumers_shop_sequential``.
  Uses **batch-sequential processing** (~10 randomized consumer batches, each completing
  all Z visits before the next starts). Pipeline calls it once (handles visits internally).
  See :ref:`decision-batch-sequential-shopping` for design rationale.

Performance
~~~~~~~~~~~

* Vectorized batch matching yields **~30% faster** full simulation runs across all
  economy sizes. See `ASV benchmarks <https://kganitis.github.io/bam-engine/>`_.

Removed
~~~~~~~

* **Sequential event classes**: ``WorkersSendOneRound``, ``FirmsHireWorkers``,
  ``FirmsSendOneLoanApp``, ``BanksProvideLoans``, ``ConsumersShopSequential``.
* **Interleave syntax (``<->``)** from pipeline YAML. Only ``event x N`` repetition
  remains.

[0.5.1] - 2026-03-04
---------------------

Performance
~~~~~~~~~~~

* Vectorized ``workers_decide_firms_to_apply``: replaced three Python for-loops
  (per-worker ``rng.choice()``, per-loyal-worker move-to-front, per-worker buffer
  write) with batch NumPy operations (random priorities + ``argpartition``, vectorized
  loyalty shift, slice assignment). ~3.5x faster event execution.

[0.5.0] - 2026-03-04
---------------------

Added
~~~~~

**Extension Bundles**

* :class:`~bamengine.Extension` dataclass: bundles roles, events, relationships,
  and config into a single object.
* :meth:`~bamengine.Simulation.use`: one-call extension activation
  (replaces the manual ``use_role``/``use_events``/``use_config`` pattern).
* Pre-built bundles: ``RND``, ``BUFFER_STOCK``, ``TAXATION`` in their
  respective ``extensions.*`` packages.
* Collect-config dicts: ``BASELINE_COLLECT``, ``RND_COLLECT``,
  ``BUFFER_STOCK_COLLECT``, suggested data-collection configs for
  ``sim.run(collect=...)``.

**Documentation**

* Extended User Guide to guide user through simulations, configuration,
  custom data collection, custom roles/events/relationships, pipelines,
  operations, extensions, validation, calibration, and best practices.
* New pages: Related Projects, Roadmap, Glossary, About.

Removed
~~~~~~~

* **Deprecated config parameters**: ``price_cut_allow_increase``, ``inflation_method``,
  ``labor_matching``, ``credit_matching``, ``min_wage_ratchet``, ``pricing_phase``,
  ``matching_method``. Behavior is now hardcoded to the previously active defaults.
* **Deprecated event classes**: ``WorkersApplyToFirms``, ``WorkersApplyToBestFirm``,
  ``FirmsApplyForLoans``, ``CalcUnemploymentRate``, ``ConsumersShopOneRound``,
  ``FirmsCalcBreakevenPrice``, ``FirmsAdjustPrice``.
* **Deprecated internal functions**: cascade matching (labor and credit), simultaneous
  matching, annualized inflation, ratchet minimum wage, production-phase pricing,
  round-robin shopping.

Fixed
~~~~~

* Calibration modules now fall back to serial execution when ``n_workers=1``,
  fixing a ``ProcessPoolExecutor`` spawn crash on macOS.

[0.4.0] - 2026-02-25
--------------------

This release adds the robustness analysis package (Section 3.10 of Delli Gatti
et al., 2011), overhauls the calibration package, re-calibrates defaults, and
fixes several issues in internal event logic.

Added
~~~~~

**Robustness Analysis**

* ``validation/robustness/`` package implementing Sections 3.10.1–3.10.2:

  * **Exploration of the parameter space** (Section 3.10.1)

    * **Internal validity**: 20-seed stability of co-movements, AR structure,
      firm size distributions, and empirical curves (HP filter, cross-correlations,
      AR fitting, impulse-response).
    * **Sensitivity analysis**: univariate sweeps across credit (H), goods (Z),
      labor (M), contract length (theta), and economy size.

  * **Preferential attachment in consumption and the entry mechanism** (Section 3.10.2)

    * **Preferential attachment**: New experiment (``run_pa_experiment``) that removes
      PA by setting new config parameter ``consumer_matching``: ``loyalty`` → ``random``.
    * **Entry mechanism**: New experiment (``run_entry_experiment``) that uses the
      new taxation extension (``extensions/taxation/``).

  * **CLI**: ``python -m validation.robustness --help``
  * **Example**: ``examples/extensions/example_robustness.py``

**Packaging**

* ``extensions/``, ``validation/``, ``calibration/`` are now pip-installable
  sibling packages (``pip install bamengine[validation]``, etc.).

Fixed
~~~~~

* New ``FirmsPlanBreakevenPrice`` / ``FirmsPlanPrice`` events (mutually exclusive
  with production-phase pair). Source relocation: ``events/planning.py`` →
  ``events/production.py`` (pipeline keys unchanged).
* Loan records retained through planning/labor phases (purged at credit market
  opening) for previous-period interest access.
* ``projected_fragility`` pre-filled with ``max_leverage`` for NW ≤ 0 firms.
* Loan rate now uses per-bank ``opex_shock`` (φ_k) instead of constant ``h_phi``
  upper bound, restoring bank heterogeneity (book eq. 3.7).
* Multi-lender: firms accumulate loans from multiple banks across credit rounds.
* ``LoanBook.principal_per_borrower()`` new method for efficient per-firm
  principal aggregation.
* Removed spurious ``wage_bill`` recalculation in ``workers_update_contracts``
  (overstated gross profit).
* Bad debt: ``clip(frac × NW, 0, principal)`` computes *recovery*, was
  subtracted as *loss*. Now ``loss = principal - recovery``.
* R&D deduction moved before dividends (book Section 3.8); operates on
  ``net_profit`` so dividends = δ(1−σ)π.
* ``net_worth_ratio`` formula: ``prod × ratio`` → ``prod × price × ratio``
  (dimensional fix).
* Bankruptcy counters persist through end-of-period data capture.
* Spawned firms set ``production = production_prev`` (prevents re-bankruptcy).

Changed
~~~~~~~

**Validation**

* Unemployment metrics restructured: ``unemployment_pct_in_bounds`` → separate
  floor/ceiling checks; ``unemployment_hard_ceiling`` → ``unemployment_absolute_ceiling``
  (30% → 20%, BOOLEAN in baseline). Removed ``unemployment_autocorrelation``.

**Calibration**

* Rebuilt into focused modules (``analysis``, ``grid``, ``screening``,
  ``stability``, ``io``, ``reporting``), replacing monolithic ``optimizer.py``.
* **Morris Method screening** (default): multi-trajectory OAT (Morris 1991)
  with dual-threshold classification (mu* and sigma). OAT via ``--method oat``.
* **Tiered stability**: incremental tournament (100×10 → 50×20 → 10×100 seeds).
* **Pairwise interaction analysis** for synergy/conflict detection.
* Phase-based CLI (``--phase``, ``--resume``, ``--rank-by``, ``--grid``,
  ``--fixed``), markdown reports, config export, before/after comparison.

**Breaking Changes**

* **Calibrated defaults**: n_firms 300→100, n_households 3000→500,
  New-firm entry: size 0.8→0.5, production 0.9→0.5, wage 0.9→0.5, markup 1.0→1.5,
  ``job_search_method``:  ``"vacancies_only"`` → ``"all_firms"``,
  ``matching_method``: ``"simultaneous"`` → ``"sequential"`` *(also removed from config)*
* **Config removed**: ``contract_poisson_mean``, ``loan_priority_method``,
  ``firing_method``, ``matching_method``
* **Config deprecated**: ``price_cut_allow_increase``, ``inflation_method``
* **Net worth**: ``net_worth_ratio`` semantics → "multiple of initial revenue";
  default 3.0 → 6.0
* **Renamed**: ``destroyed`` → ``collapsed`` on ``Simulation``
* **Calibration CLI**: ``--phase`` replaces ``--sensitivity-only``; new
  ``--resume``, ``--rank-by``, ``--k-factor``, ``--output-dir``

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

* ``Shareholder`` role: built-in household role tracking per-period dividends, used to
  adjust buffer-stock MPC metrics for the dividend artifact

**Composable Extension API**

* ``Simulation.use_config(config)``: apply extension default configuration with "don't overwrite"
  semantics (user kwargs and earlier ``use_config()`` calls take precedence)
* ``Simulation.use_events(*event_classes)``: explicitly apply pipeline hooks from event classes
* ``Simulation.use_role(cls, n_agents=N)``: ``n_agents`` parameter for non-firm roles
* ``Pipeline.apply_hooks(*event_classes)``: read hook metadata from classes and apply to pipeline

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

* Weight-based fail escalation for validation status checks: high-weight metrics
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
* **8 event modules**: Planning, labor/credit/goods markets, production, revenue, bankruptcy, economy stats (43 total events)

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
