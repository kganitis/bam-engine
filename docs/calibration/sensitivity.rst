Phase 1: Sensitivity Analysis
==============================

The first phase identifies which parameters significantly affect the
validation score.


Morris Method (Default)
-----------------------

Morris Method (Morris, 1991) runs multiple OAT trajectories from random
starting points across the parameter space, producing two measures per
parameter:

- **mu*** (mean absolute elementary effect): Average importance across
  different parameter contexts
- **sigma** (std of elementary effects): Interaction/nonlinearity indicator —
  how much the effect varies depending on other parameters

**Dual-threshold classification:**

- ``INCLUDE``: mu* > threshold **or** sigma > threshold — important or
  interaction-prone
- ``FIX``: mu* ≤ threshold **and** sigma ≤ threshold — truly unimportant

.. code-block:: python

   from calibration import run_morris_screening, print_morris_report

   morris = run_morris_screening(
       scenario="baseline",
       n_workers=10,
       n_seeds=3,
       n_trajectories=10,
   )
   print_morris_report(morris)

   # Convert for downstream use
   sensitivity = morris.to_sensitivity_result()


OAT (One-at-a-Time)
--------------------

Traditional OAT tests each parameter while holding others at defaults. Faster
but sensitive to baseline choice — parameter rankings can change depending on
which defaults are used.

.. code-block:: python

   from calibration import run_sensitivity_analysis

   sa = run_sensitivity_analysis(
       scenario="baseline",
       n_workers=10,
       n_seeds=3,
   )


Morris vs OAT
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - Morris
     - OAT
   * - Interaction detection
     - Yes (via sigma)
     - No
   * - Baseline dependency
     - Minimal (random starts)
     - High (single baseline)
   * - Cost (baseline, 3 seeds)
     - ~660 sim runs (~2.5 min)
     - ~216 sim runs (~50s)
   * - Recommended for
     - Production calibration
     - Quick exploration


Pairwise Interaction Analysis
------------------------------

After OAT identifies sensitive params, pairwise analysis tests all 2-param
combinations to find synergies and conflicts:

.. code-block:: python

   from calibration import run_pairwise_analysis

   pairs = run_pairwise_analysis(
       params=["h_rho", "delta", "theta"],
       scenario="baseline",
       n_workers=10,
   )
