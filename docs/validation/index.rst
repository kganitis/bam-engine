Validation
==========

The ``validation`` package validates BAM simulation results against empirical
targets from Delli Gatti et al. (2011). It provides scenario-based validation,
multi-seed stability testing, and comprehensive robustness analysis.


Quick Start
-----------

**Single validation run:**

.. code-block:: python

   from validation import run_validation

   result = run_validation(seed=42, n_periods=1000)
   print(f"Score: {result.total_score:.3f}, Passed: {result.passed}")

**Multi-seed stability test:**

.. code-block:: python

   from validation import run_stability_test

   stability = run_stability_test(seeds=[0, 42, 123, 456, 789])
   print(f"Mean: {stability.mean_score:.3f} ± {stability.std_score:.3f}")
   print(f"Pass rate: {stability.pass_rate:.0%}")

**Scenario visualization:**

.. code-block:: bash

   # Baseline scenario (Section 3.9.1)
   python -m validation.scenarios.baseline

   # Growth+ scenario (Section 3.9.2)
   python -m validation.scenarios.growth_plus

   # Buffer-stock scenario (Section 3.9.4)
   python -m validation.scenarios.buffer_stock


Architecture
------------

The validation package uses a **MetricSpec abstraction** for declarative metric
validation:

.. code-block:: python

   MetricSpec(
       name="unemployment_rate_mean",
       field="unemployment_mean",
       check_type=CheckType.MEAN_TOLERANCE,
       target_path="metrics.unemployment_rate_mean",
       weight=1.5,
       group=MetricGroup.TIME_SERIES,
   )

The generic ``validate()`` engine:

1. Loads targets from YAML
2. Runs simulation with scenario config
3. Computes metrics using scenario's compute function
4. Evaluates each MetricSpec against targets
5. Returns weighted ``ValidationScore``


Module Structure
----------------

::

   validation/
   ├── __init__.py              # Registry-driven package exports
   ├── types.py                 # Core types: MetricSpec, Scenario, CheckType
   ├── scoring.py               # Scoring and status check functions
   ├── engine.py                # Generic validate() and stability_test()
   ├── reporting.py             # Report printing functions
   ├── scenarios/
   │   ├── __init__.py          # Scenario registry + get_scenario()
   │   ├── _utils.py            # Shared utilities
   │   ├── baseline/            # Section 3.9.1
   │   ├── growth_plus/         # Section 3.9.2
   │   └── buffer_stock/        # Section 3.9.4
   └── robustness/              # Section 3.10

.. toctree::
   :maxdepth: 2
   :hidden:

   scenarios
   targets
   scoring
   cli
   robustness/index
