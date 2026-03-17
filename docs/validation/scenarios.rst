Validation Scenarios
====================

Three built-in scenarios validate different model configurations against the
reference book.


Baseline (Section 3.9.1)
-------------------------

Standard BAM model behavior — 25 metrics across 3 categories:

- **TIME_SERIES** (10): Unemployment, inflation, GDP trend/growth, vacancy rates
- **CURVES** (6): Phillips, Okun, Beveridge curve correlations
- **DISTRIBUTION** (4): Firm size metrics (skewness, tail ratios)

.. code-block:: python

   from validation import run_validation, run_stability_test

   result = run_validation(seed=42, n_periods=1000)
   stability = run_stability_test(seeds=list(range(100)))


Growth+ (Section 3.9.2)
-------------------------

Endogenous productivity growth via R&D investment — 65 metrics across 6
categories:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Category
     - Count
     - Key Metrics
   * - TIME_SERIES
     - 14
     - Unemployment, inflation, GDP trend/growth, vacancy rates
   * - CURVES
     - 6
     - Phillips, Okun, Beveridge correlations
   * - DISTRIBUTION
     - 4
     - Firm size metrics (skewness, tail ratios)
   * - GROWTH
     - 11
     - Productivity/wage growth, co-movement, recession detection
   * - FINANCIAL
     - 20
     - Interest rates, fragility, price ratio, Minsky classification
   * - GROWTH_RATE_DIST
     - 10
     - Tent-shape R², bounds checks, outlier percentages

.. code-block:: python

   from validation import run_growth_plus_validation

   result = run_growth_plus_validation(seed=42, n_periods=1000)


Buffer-Stock (Section 3.9.4)
------------------------------

Buffer-stock consumption with individual adaptive MPC. Uses a **two-layer
validation** approach:

1. **Unique metrics** (8, per-seed): Wealth distribution fitting (Figure 3.8)
   and MPC behavioral metrics. These determine per-seed PASS/FAIL.
2. **Improvement over Growth+** (aggregate): Mean score deltas across all seeds
   checked after stability testing. Catches systematic degradation without
   being affected by per-seed noise.

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Category
     - Count
     - Key Metrics
   * - DISTRIBUTION
     - 6
     - Wealth CCDF fitting (Singh-Maddala, Dagum, GB2 R²), Gini, skewness
   * - FINANCIAL
     - 2
     - Mean MPC, percent dissaving

.. code-block:: python

   from validation import run_buffer_stock_validation

   # Automatic: runs Growth+ internally as baseline
   result = run_buffer_stock_validation(seed=42, n_periods=1000)

   # With reuse: pass pre-computed Growth+ result
   from validation import run_growth_plus_validation

   gp = run_growth_plus_validation(seed=42, n_periods=1000)
   result = run_buffer_stock_validation(seed=42, growth_plus_result=gp)

   # Access improvement data
   print(result.improvement_deltas)  # dict of metric -> delta
   print(result.degraded_metrics)  # metrics that degraded beyond threshold
   print(result.baseline_score)  # Growth+ ValidationScore

.. autoclass:: validation.types.BufferStockValidationScore
   :members:
   :undoc-members:


Target File Structure
---------------------

Target values are defined in co-located ``targets.yaml`` files with standardized
keys per check type:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - CheckType
     - YAML Keys
     - Purpose
   * - ``MEAN_TOLERANCE``
     - ``target``, ``tolerance``
     - Value within target ± tolerance
   * - ``RANGE``
     - ``min``, ``max``
     - Value within [min, max]
   * - ``PCT_WITHIN``
     - ``target``, ``min``
     - Percentage meeting threshold
   * - ``OUTLIER``
     - ``max_outlier``, ``penalty_weight``
     - Penalize excess outliers
   * - ``BOOLEAN``
     - ``threshold`` (in MetricSpec)
     - Simple > or < check


Core Types
----------

.. autoclass:: validation.types.ValidationScore
   :members:
   :undoc-members:

.. autoclass:: validation.types.StabilityResult
   :members:
   :undoc-members:

.. autoclass:: validation.types.MetricResult
   :members:
   :undoc-members:

.. autoclass:: validation.types.MetricSpec
   :members:
   :undoc-members:

.. autoclass:: validation.types.Scenario
   :members:
   :undoc-members:
