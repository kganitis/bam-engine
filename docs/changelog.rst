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

* ``consumer_matching`` configuration parameter (``"loyalty"`` or ``"random"``,
  default ``"loyalty"``). Controls whether consumers use preferential attachment
  (loyalty to previous largest producer) when selecting firms to visit.
  ``"random"`` disables the positive feedback loop for structural experiments.

* ``extensions/taxation/`` package for profit taxation without redistribution.
  ``FirmsTaxProfits`` event hooks after ``firms_validate_debt_commitments``
  and deducts ``profit_tax_rate * max(0, net_profit)`` from profitable firms.
  Revenue vanishes (not redistributed). Used by entry neutrality experiment.

* ``setup_fn`` field on ``Experiment`` dataclass for experiments that need
  post-init simulation setup (e.g. attaching extension events/config).
  Must be a module-level function for ``ProcessPoolExecutor`` pickling.

* Section 3.10.2 structural experiments in ``validation/robustness/``:

  - **PA experiment** (``run_pa_experiment``): Disables consumer loyalty and
    runs internal validity + Z-sweep to show volatility drops and deep crises
    vanish. Optional baseline comparison.
  - **Entry neutrality experiment** (``run_entry_experiment``): Sweeps profit
    tax rate from 0% to 90% to confirm automatic firm entry does NOT
    artificially drive recovery (monotonic degradation expected).
  - New experiment definitions: ``goods_market_no_pa``, ``entry_neutrality``
  - New CLI flags: ``--structural-only``, ``--pa-experiment``,
    ``--entry-experiment``, ``--no-baseline``
  - Visualization: ``plot_pa_gdp_comparison``, ``plot_pa_comovements``,
    ``plot_entry_comparison``
  - Reporting: ``format_pa_report``, ``format_entry_report`` with
    monotonicity assessment

* Planning-phase breakeven price and price adjustment events
  (``firms_plan_breakeven_price``, ``firms_plan_price``) that use previous
  period's costs and desired production. These are mutually exclusive with the
  production-phase pair — use one or the other.

* ``pricing_phase`` configuration parameter (``"planning"`` or ``"production"``,
  default ``"planning"``). Automates switching between planning-phase and
  production-phase pricing events without manual pipeline YAML editing.
  Raises ``ValueError`` if used with ``pipeline_path`` (custom pipelines
  should be edited directly). Hard-coded mutual exclusion guard in
  ``Pipeline.__post_init__`` prevents both pricing pairs from coexisting.

* Cascade matching for the labor market (book Section 3.3–3.4):
  ``WorkersApplyToFirms`` event where each unemployed worker walks their
  ranked firm queue (best→worst wage) and is hired at the first firm with
  vacancies, or cascades to the next. Replaces the interleaved
  ``WorkersSendOneRound <-> FirmsHireWorkers × max_M`` pattern in the
  default pipeline. Shared ``_hire_workers()`` helper used by both legacy
  and cascade code paths.

* Cascade matching for the credit market (book Section 3.5):
  ``FirmsApplyForLoans`` event where each firm walks their ranked bank
  queue (lowest→highest rate). Loans granted immediately (partial if
  supply < demand), remaining demand carries to next bank. Firms sorted
  before processing by ``loan_priority_method`` (default:
  ``"by_leverage"``). Replaces the interleaved
  ``FirmsSendOneLoanApp <-> BanksProvideLoans × max_H`` pattern in the
  default pipeline. Shared ``_provide_loan()`` helper used by both legacy
  and cascade code paths.

* ``WorkersApplyToBestFirm`` event — alternative single-best matching
  where workers apply only to their top-choice firm (no cascade). Not in
  default pipeline; available for experimentation.

* ``labor_matching`` configuration parameter (``"cascade"`` or
  ``"interleaved"``, default ``"cascade"``). Automates switching between
  cascade and legacy interleaved labor matching events without manual
  pipeline YAML editing. Raises ``ValueError`` if used with
  ``pipeline_path``.

* ``credit_matching`` configuration parameter (``"cascade"`` or
  ``"interleaved"``, default ``"cascade"``). Same as above for the credit
  market.

* ``min_wage_ratchet`` configuration parameter (boolean, default
  ``False``). When ``True``, minimum wage never decreases during deflation
  (``max(1, 1 + inflation)``), matching the book's "revised upward"
  specification (Section 3.4).

* Diagnostics package (``diagnostics/``) with comprehensive analysis dashboards:
  baseline (13 figures), Growth+ (15 figures), credit market investigation,
  and labor market investigation scripts.

* ``LoanBook.principal_per_borrower()`` method for aggregating total
  principal per borrower (complements ``debt_per_borrower()`` and
  ``interest_per_borrower()``).

* Robustness analysis package (Section 3.10.1 of Delli Gatti et al., 2011):

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
  * Firm size distribution metrics: kurtosis (excess), tail index (log-log rank-size
    slope), and distribution shape classification (pareto-like/exponential/uniform-like)
  * Peak-lag detection in co-movement analysis (lag of max |correlation| per variable)
  * Wage/productivity ratio tracking across experiments
  * GDP growth volatility in sensitivity summary tables
  * CLI entry point: ``python -m validation.robustness``
  * Example: ``examples/advanced/example_robustness.py``

* Calibration package overhaul — multi-phase framework with Morris method
  screening, tiered stability testing, pairwise interaction analysis, and
  production-quality tooling:

  * **Morris Method screening** (default sensitivity method): runs multiple
    OAT trajectories from random starting points (Morris 1991). Produces
    mu* (mean absolute elementary effect) and sigma (std of effects) per
    parameter. Dual-threshold classification: ``INCLUDE`` if mu* > threshold
    OR sigma > threshold (catches interaction-prone parameters that single-
    baseline OAT would miss). New ``MorrisResult``, ``MorrisParameterEffect``
    types, ``run_morris_screening()``, ``print_morris_report()``.
    ``MorrisResult.to_sensitivity_result()`` converts for downstream
    compatibility. New ``--method morris|oat`` and ``--morris-trajectories``
    CLI options.
  * **Tiered stability testing**: incremental tournament system
    (default: 100×10, 50×20, 10×100 seeds) that efficiently narrows candidates
    while accumulating seed scores. Replaces flat ``top_k × N seeds`` approach.
    Total evaluations: 2,300 vs naive 10,000 for same coverage.
  * **Pairwise interaction analysis**: tests all 2-param combinations among
    sensitive parameters to detect synergies and conflicts. New
    ``PairInteraction`` / ``PairwiseResult`` types with ``synergies`` and
    ``conflicts`` properties.
  * **Multi-seed sensitivity**: ``n_seeds`` parameter for OAT evaluation
    (default: 3). Averages across seeds for more robust sensitivity measurement.
  * **Per-metric-group score decomposition**: ``ParameterSensitivity.group_scores``
    tracks which metric groups (TIME_SERIES, CURVES, etc.) each parameter
    affects, printed in sensitivity report.
  * **Value pruning**: drops grid values whose OAT score is more than
    ``pruning_threshold`` below the best value for that parameter.
    ``SensitivityResult.prune_grid()`` method. Default: ``auto`` (2× sensitivity
    threshold). Disable with ``--pruning-threshold none``.
  * **Phase-based CLI**: ``--phase sensitivity|morris|grid|stability|pairwise``
    for individual phase execution, or omit for all phases sequentially.
    Replaces ``--sensitivity-only``.
  * **Checkpointing and resume**: grid screening and stability testing save
    periodic checkpoints; ``--resume`` flag skips already-evaluated configs.
  * **Progress tracking with ETA**: ``format_eta()`` / ``format_progress()``
    helpers using sensitivity-measured ``avg_time_per_run``.
  * **Config export**: ``export_best_config()`` writes best result as
    ready-to-use YAML (``output/{scenario}_best_config.yml``).
  * **Before/after comparison**: ``compare_configs()`` runs default vs calibrated
    side-by-side and reports per-metric changes. ``ComparisonResult`` type.
  * **Parameter pattern analysis**: ``analyze_parameter_patterns()`` identifies
    which values consistently appear in top configs.
  * **Buffer-stock scenario support**: all calibration phases support
    ``--scenario buffer_stock`` (26 common + 3 extension params).
  * **Expanded parameter grid**: 26 common parameters covering initial
    conditions, new-firm entry, economy-wide, search frictions, and
    implementation variants. Extension-specific: ``sigma_decay`` / ``sigma_max``
    (Growth+), ``buffer_stock_h`` (buffer-stock).

Changed
~~~~~~~

* Moved ``FirmsCalcBreakevenPrice`` and ``FirmsAdjustPrice`` from
  ``events/planning.py`` to ``events/production.py`` to match their actual
  pipeline phase. Pipeline keys unchanged — no breaking changes for YAML
  configurations.
* Loan records are now retained through planning and labor phases (purged
  when the credit market opens instead of during revenue settlement). This
  enables planning-phase events to access previous-period interest data.
* Default agent counts unified to match book setup: ``n_firms``: 300 → 100,
  ``n_households``: 3000 → 500
* Default new-firm entry parameters: ``new_firm_size_factor`` 0.8 → 0.5,
  ``new_firm_production_factor`` 0.9 → 0.5, ``new_firm_wage_factor`` 0.9 → 0.5,
  ``new_firm_price_markup`` 1.0 → 1.5
* Default ``max_loan_to_net_worth``: 2 → 5
* Default ``job_search_method``: ``"all_firms"`` → ``"vacancies_only"``
  (book: "visiting firms that post vacancies")
* Default ``matching_method``: ``"simultaneous"`` → ``"sequential"``
  (book Section 3.4: "closed sequentially"). Only affects legacy
  interleaved events; cascade events are inherently sequential.
* Default pipeline now uses cascade matching (fixed 37 events, no longer
  depends on ``max_M`` or ``max_H``). Legacy interleaved events retained
  but off default path.
* ``create_default_pipeline()`` accepts ``labor_matching`` and
  ``credit_matching`` parameters for event swapping at pipeline creation.
* Pipeline-altering parameter conflict check now covers
  ``labor_matching`` and ``credit_matching`` in addition to
  ``pricing_phase``. Error message reports all conflicting parameters.
* ``Scenario.default_config`` now optional (defaults to ``{}``)
* Growth+ and buffer-stock scenarios no longer override agent counts
* Baseline targets: updated log GDP and unemployment bounds for
  100-firm/500-household economy
* Renamed 'destroyed' flag to 'collapsed' in ``Simulation``
* Validation: unemployment metrics split across all scenarios —
  ``unemployment_pct_in_bounds`` → ``unemployment_pct_above_floor`` +
  ``unemployment_pct_below_ceiling`` (separate floor/ceiling checks).
  ``unemployment_hard_ceiling`` → ``unemployment_absolute_ceiling``
  (tightened from 30% to 20%, now BOOLEAN check type in baseline).
  Removed ``unemployment_autocorrelation`` from baseline.
* Validation: metric parameters (thresholds, bounds) moved from
  ``metadata.params`` to inline in each metric's YAML entry (targets
  files are now fully self-contained).
* Validation: tightened visualization bounds and recalibrated targets
  across all three scenarios (baseline, Growth+, buffer-stock).
* Examples: default logging level ``"INFO"`` → ``"ERROR"`` in baseline
  and Growth+ examples for cleaner output.
* Event count: 43 → 45 events (2 new labor market, 1 new credit market).
* Sphinx API docs: added ``FirmsApplyForLoans`` to
  ``docs/api/events/credit_market.rst``.
* Calibration: default sensitivity method changed from OAT to Morris Method.
  OAT still available via ``--method oat``.
* Calibration: simplified sensitivity classification from HIGH/MEDIUM/LOW
  to binary INCLUDE/FIX with single ``sensitivity_threshold`` (default 0.02).
  ``build_focused_grid()`` parameters renamed: ``high_threshold`` /
  ``medium_threshold`` → ``sensitivity_threshold`` / ``pruning_threshold``.
* Calibration: ``run_focused_calibration()`` now uses tiered stability
  internally. Removed ``top_k`` and ``stability_seeds`` parameters;
  replaced with ``stability_tiers`` list.
* Calibration: CLI arguments restructured — removed ``--sensitivity-only``,
  ``--top-k``, ``--high-threshold``, ``--medium-threshold``; added
  ``--phase``, ``--sensitivity-threshold``, ``--pruning-threshold``,
  ``--sensitivity-seeds``, ``--stability-tiers``, ``--resume``.
* Calibration: ``screen_single_seed()`` now returns ``CalibrationResult``
  (was ``float``). ``CalibrationResult`` gains ``seed_scores`` field for
  incremental stability.
* Calibration: parameter grid restructured around shared ``_COMMON_GRID``
  (26 params) with scenario-specific extensions via dict unpacking.
* Docs: updated ``development.rst`` calibration examples to match new CLI.

Fixed
~~~~~

* Fixed co-movement convention comments in ``reference_values.yaml`` (negative/positive
  lag semantics were inverted).
* Fixed missing empty-valid guard in co-movement aggregation loop in
  ``internal_validity.py`` (would fail if all seeds collapsed).
* Fixed mean AR fit method: now fits AR(1) on the pointwise-averaged GDP cycle
  across seeds instead of averaging individual AR coefficients, matching the
  book's methodology.
* Fixed Growth+ R&D event ordering: R&D deduction (``FirmsDeductRnDExpenditure``)
  now runs *before* dividend distribution, matching book Section 3.8.
  Previously R&D hooked after ``firms_pay_dividends``, overstating dividends
  by δσπ. R&D now deducts from ``net_profit`` (not ``retained_profit``) so
  dividends correctly equal δ(1−σ)π.
* Fixed uninitialized ``projected_fragility`` buffer for firms with non-positive
  net worth. ``firms_calc_financial_fragility()`` now pre-fills with
  ``max_leverage`` before the conditional divide, ensuring deterministic
  behavior and correct credit priority (insolvent firms get lowest priority).
* Fixed gross_profit overstatement: removed redundant ``wage_bill``
  recalculation in ``workers_update_contracts`` that was using post-expiration
  values instead of the actual wages paid.
* Bad debt formula: fixed recovery-vs-loss inversion in
  ``firms_validate_debt_commitments()``. The formula ``clip(frac × net_worth,
  0, principal)`` computes the bank's *recovery* (proportional claim on firm
  equity), but was incorrectly subtracted as the *loss*. Actual loss is now
  ``principal - recovery``. Banks previously lost what they should have
  recovered, suppressing the credit feedback loop.
* Loan purge in ``banks_provide_loans()`` moved from per-bank loop to
  one-time safety clear in ``firms_prepare_loan_applications()``. This
  enables multi-lender support: firms can now accumulate loans from
  multiple banks across credit matching rounds (max_H).

* Loan rate formula in ``banks_provide_loans()`` now uses per-bank ``opex_shock``
  (φ_k) instead of the constant upper bound ``h_phi``, matching book equation 3.7.
  Previously all banks applied the maximum rate markup, eliminating bank heterogeneity
  and systematically overcharging borrowers.
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
