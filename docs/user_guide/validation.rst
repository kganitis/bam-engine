Validation & Analysis
=====================

BAM Engine includes a validation framework for comparing simulation output
against target values from Delli Gatti et al. (2011). This ensures the model
reproduces the reference results and helps detect parameter configurations
that deviate from expected behavior.


Running Validation
------------------

The simplest way to validate is with ``run_validation()``:

.. code-block:: python

   from validation import run_validation

   result = run_validation(seed=42, n_periods=1000)

   print(f"Score: {result.total_score:.3f}")
   print(f"Passed: {result.passed}")
   print(f"Failures: {result.n_fail}")

The result object contains:

- ``total_score`` — Weighted score from 0.0 (worst) to 1.0 (perfect)
- ``passed`` — ``True`` if zero FAIL-status metrics
- ``n_pass``, ``n_warn``, ``n_fail`` — Count of metrics by status
- ``metric_results`` — Detailed per-metric breakdown


Validation Scenarios
--------------------

Three built-in scenarios correspond to sections of the reference book:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Scenario
     - Book Section
     - What It Validates
   * - ``baseline``
     - Section 3.9.1
     - Core model: unemployment, inflation, firm dynamics, business cycles
   * - ``growth_plus``
     - Section 3.9.2
     - R&D extension: productivity growth, firm size distribution
   * - ``buffer_stock``
     - Section 3.9.4
     - Buffer-stock extension: savings behavior, wealth distribution

Run a specific scenario:

.. code-block:: python

   from validation import run_validation, run_growth_plus_validation

   # Baseline (default)
   baseline_result = run_validation(seed=42, n_periods=1000)

   # Growth+ scenario
   growth_result = run_growth_plus_validation(seed=42, n_periods=1000)

Each scenario has its own targets (defined in ``targets.yaml`` files) and
metric weights tuned to the phenomena that matter most for that model variant.


Understanding Scores
--------------------

Validation uses a **two-layer system**:

**Status checks** (categorical):

- **PASS** — Metric is within acceptable range
- **WARN** — Metric is borderline (outside target but within tolerance)
- **FAIL** — Metric significantly deviates from target

**Scores** (continuous, 0 to 1):

Each metric produces a score between 0.0 and 1.0. The ``total_score`` is a
weighted average across all metrics. Metric weights range from 0.5 (low
importance) to 5.0 (critical).

**Weight-based fail escalation**: High-weight metrics have stricter WARN/FAIL
thresholds. The escalation formula
(:math:`\text{clamp}(5 - 2w, 0.5, 5.0)`) means a weight-3.0 metric fails at
deviations that would only warn for a weight-0.5 metric.

**Metric types**:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - How It Works
   * - ``RANGE``
     - Value must fall within [min, max] range
   * - ``TOLERANCE``
     - Value must be within percentage of target
   * - ``PCT_WITHIN``
     - Percentage of time series within a band
   * - ``OUTLIER_PENALTY``
     - Penalizes extreme values in distribution
   * - ``BOOLEAN``
     - Binary pass/fail check (e.g., "economy did not collapse")


Robustness Analysis
-------------------

The robustness package tests whether results hold across multiple random seeds
and parameter variations:

**Internal validity** — Multi-seed stability:

.. code-block:: python

   from validation.robustness import run_internal_validity

   result = run_internal_validity(n_seeds=20, n_periods=1000)
   # Tests: co-movement structure, AR fit, firm size distribution

**Sensitivity analysis** — Parameter sweeps:

.. code-block:: python

   from validation.robustness import run_sensitivity_analysis

   sa = run_sensitivity_analysis(experiments=["credit_market", "contract_length"])

**CLI interface:**

.. code-block:: bash

   # Full robustness analysis
   python -m validation.robustness

   # Internal validity only
   python -m validation.robustness --internal-only

   # Sensitivity analysis only
   python -m validation.robustness --sensitivity-only

   # Structural experiments (matching variants)
   python -m validation.robustness --structural-only


Parameter Calibration
---------------------

The calibration package finds parameters that maximize validation scores
through a multi-phase pipeline:

1. **Morris screening** — Identify which parameters matter most
2. **Grid search** — Search the reduced parameter space
3. **Stability testing** — Verify results hold across many seeds

.. code-block:: bash

   # Run the default calibration pipeline
   python -m calibration --scenario baseline --workers 10

See :doc:`calibration` for the full tutorial.


Visualization
-------------

**Scenario plots** — Run a scenario with visualization:

.. code-block:: python

   from validation.scenarios.baseline import run_scenario

   run_scenario(seed=0, show_plot=True)

**Diagnostic dashboards** — Comprehensive multi-figure analysis:

.. code-block:: bash

   python diagnostics/baseline_diagnostics.py
   python diagnostics/growth_plus_diagnostics.py

These generate 13-15 figures covering GDP, unemployment, inflation, firm
distributions, credit market dynamics, and more.


.. seealso::

   - :doc:`calibration` for the full calibration tutorial
   - :doc:`extensions` for setting up model extensions before validation
   - :doc:`configuration` for parameter definitions
