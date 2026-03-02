Scoring System
==============

Validation uses a **two-layer system**: status checks determine pass/fail,
while scores provide a continuous 0–1 measure for optimization.


Two-Layer Validation
--------------------

**Status checks** (categorical):

- **PASS** — Metric is within acceptable range
- **WARN** — Metric is borderline (outside target but within tolerance)
- **FAIL** — Metric significantly deviates from target

**Scores** (continuous, 0 to 1):

Each metric produces a score between 0.0 and 1.0. The ``total_score`` is a
weighted average across all metrics. A simulation *passes* if it has zero
FAIL-status metrics (``n_fail == 0``).


Weight-Based Fail Escalation
-----------------------------

High-weight metrics have stricter WARN→FAIL thresholds. The escalation
multiplier is computed as:

.. math::

   m = \text{clamp}(5 - 2w, \; 0.5, \; 5.0)

where :math:`w` is the metric weight. This multiplier scales the
WARN→FAIL boundary in each check function.

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Weight
     - Multiplier
     - Effect (MEAN_TOLERANCE, normal FAIL at 2× tol)
   * - 3.0
     - 0.5
     - FAIL at 1× tolerance (stricter)
   * - 2.0
     - 1.0
     - FAIL at 2× tolerance (normal)
   * - 1.5
     - 2.0
     - FAIL at 4× tolerance
   * - 1.0
     - 3.0
     - FAIL at 6× tolerance
   * - 0.5
     - 4.0
     - FAIL at 8× tolerance (lenient)

BOOLEAN checks are exempt from escalation — they always have natural PASS/FAIL
behavior.


Metric Types
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - How It Works
   * - ``RANGE``
     - Value must fall within [min, max] range
   * - ``MEAN_TOLERANCE``
     - Value must be within percentage of target
   * - ``PCT_WITHIN``
     - Percentage of time series within a band
   * - ``OUTLIER``
     - Penalizes extreme values in distribution
   * - ``BOOLEAN``
     - Binary pass/fail check (e.g., "economy did not collapse")


API Reference
-------------

.. automodule:: validation.scoring
   :members:
   :undoc-members:
