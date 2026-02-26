Parameter Calibration
=====================

Calibrating an agent-based model means finding parameter values that reproduce
empirical targets — in BAM Engine's case, the results from Delli Gatti et al.
(2011). With 40+ parameters and nonlinear interactions, brute-force search is
impractical. BAM Engine's calibration package uses a multi-phase pipeline:
screen for important parameters, search a reduced grid, then stress-test the
best candidates.


Calibration Concepts
--------------------

**Objective function.** Calibration maximizes the validation score — a weighted
average of metric-level scores (0 to 1) comparing simulation output to
reference targets. See :doc:`validation` for details on the scoring system.

**The multi-phase approach:**

1. **Screening** — Identify which parameters significantly affect the score
   (typically 5-8 out of 40+)
2. **Grid search** — Exhaustively search combinations of important parameters
3. **Stability testing** — Verify that top candidates perform consistently
   across many random seeds

This pipeline reduces the search space by orders of magnitude while
maintaining confidence in results.


Quick Start
-----------

**CLI** (simplest):

.. code-block:: bash

   # Run the full calibration pipeline for the baseline scenario
   python -m calibration --scenario baseline --workers 10

   # Growth+ scenario
   python -m calibration --scenario growth_plus --workers 10

**Python API:**

.. code-block:: python

   from calibration import (
       run_morris_screening,
       build_focused_grid,
       run_focused_calibration,
   )

   # Phase 1: Screen parameters
   morris = run_morris_screening(scenario="baseline", n_workers=10, n_seeds=3)
   sensitivity = morris.to_sensitivity_result()

   # Phase 2: Build and search grid
   grid, fixed = build_focused_grid(sensitivity)
   results = run_focused_calibration(grid, fixed, scenario="baseline", n_workers=10)

   # Phase 3: Stability test
   from calibration import run_tiered_stability

   stable = run_tiered_stability(results[:10], scenario="baseline", n_workers=10)


Phase 1: Parameter Screening (Morris Method)
---------------------------------------------

The Morris Method (Morris, 1991) efficiently identifies which parameters matter
by running multiple one-at-a-time (OAT) trajectories from random starting
points. For each parameter, it computes:

- **mu*** (importance) — How much the parameter affects the score on average
- **sigma** (interaction) — How much the effect varies depending on other
  parameter values

.. code-block:: python

   from calibration import run_morris_screening

   morris = run_morris_screening(
       scenario="baseline",
       n_workers=10,  # Parallel workers
       n_seeds=3,  # Seeds per evaluation (more = more reliable)
       n_trajectories=10,  # Morris trajectories
   )

   # View results
   for p in morris.effects:
       print(
           f"{p.name}: mu*={p.mu_star:.4f}, sigma={p.sigma:.4f}, class={p.classification}"
       )

**Dual-threshold classification.** A parameter is classified as ``INCLUDE``
(important) if mu* > threshold **or** sigma > threshold. This catches
parameters that are individually important *and* those that interact strongly
with other parameters.

**Converting to sensitivity result:**

.. code-block:: python

   sensitivity = morris.to_sensitivity_result()
   # sensitivity.important_params — list of parameter names to include in grid
   # sensitivity.fixed_params — parameters to fix at their default values


Phase 2: Grid Search
--------------------

Build a focused grid from screening results and search exhaustively:

.. code-block:: python

   from calibration import build_focused_grid, generate_combinations, count_combinations

   # Automatic grid from screening results
   grid, fixed = build_focused_grid(sensitivity)
   print(f"Grid has {count_combinations(grid)} combinations")

   # Or define a custom grid
   custom_grid = {
       "h_rho": [0.05, 0.10, 0.15, 0.20],
       "delta": [0.05, 0.10, 0.15],
       "max_M": [2, 4, 6],
   }
   custom_fixed = {"theta": 8, "beta": 2.5}

**Running the search:**

.. code-block:: python

   from calibration import run_focused_calibration

   results = run_focused_calibration(
       grid,
       fixed,
       scenario="baseline",
       n_workers=10,
       n_seeds=3,
   )

   # Results sorted by score (best first)
   best = results[0]
   print(f"Best score: {best.score:.4f}")
   print(f"Parameters: {best.params}")

Each ``CalibrationResult`` contains the parameter combination, validation
score, and per-metric breakdowns.

**Grid from YAML:**

.. code-block:: yaml

   # calibration_grid.yml
   grid:
     h_rho: [0.05, 0.10, 0.15]
     delta: [0.05, 0.10]
   fixed:
     theta: 8

.. code-block:: bash

   python -m calibration --scenario baseline --grid calibration_grid.yml


Phase 3: Stability Testing
---------------------------

A parameter set that scores well with one seed may fail with others. Stability
testing verifies robustness using a **tiered tournament**:

.. code-block:: python

   from calibration import run_tiered_stability

   stable = run_tiered_stability(
       candidates=results[:20],  # Top 20 from grid search
       scenario="baseline",
       n_workers=10,
   )

   # Results ranked by stability
   for r in stable[:5]:
       print(f"Score: {r.mean_score:.4f} ± {r.std_score:.4f}")

**Default tiers:**

1. **100 seeds x 10 periods** — Quick filter, eliminates fragile configurations
2. **50 seeds x 20 periods** — Medium test, catches short-run instabilities
3. **10 seeds x 100 periods** — Full test, verifies long-run behavior

Each tier eliminates the bottom half of candidates.

**Ranking strategies:**

- ``"combined"`` (default) — Score = mean - k * std (penalizes variance)
- ``"stability"`` — Minimize standard deviation (most consistent)
- ``"mean"`` — Maximize mean score (highest average performance)

.. code-block:: bash

   python -m calibration --scenario baseline --rank-by combined --k-factor 1.5


Calibrating Custom Extensions
------------------------------

To calibrate extension parameters, define a custom grid that includes both
base model and extension parameters:

.. code-block:: python

   from calibration import run_focused_calibration

   # Grid with both base and extension parameters
   grid = {
       "delta": [0.05, 0.10, 0.15],
       "sigma_min": [0.0, 0.01, 0.02],
       "sigma_max": [0.05, 0.10, 0.15],
       "sigma_decay": [-0.5, -1.0, -2.0],
   }
   fixed = {"h_rho": 0.10, "theta": 8}

   results = run_focused_calibration(
       grid,
       fixed,
       scenario="growth_plus",
       n_workers=10,
   )

Extension parameters are passed as ``extra_params`` to the simulation and
must match the names expected by the extension events.


Sensitivity Analysis
--------------------

Beyond Morris screening, the calibration package provides traditional
one-at-a-time (OAT) sensitivity analysis:

.. code-block:: python

   from calibration import run_sensitivity_analysis

   sa = run_sensitivity_analysis(
       scenario="baseline",
       n_workers=10,
       n_seeds=3,
   )

   # View parameter importance
   for name, effect in sa.effects.items():
       print(f"{name}: range={effect.score_range:.4f}")

**Pairwise analysis** detects parameter interactions:

.. code-block:: python

   from calibration import run_pairwise_analysis

   pairs = run_pairwise_analysis(
       params=["h_rho", "delta", "theta"],
       scenario="baseline",
       n_workers=10,
   )


CLI Reference
-------------

.. code-block:: bash

   python -m calibration [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--scenario NAME``
     - Scenario to calibrate (``baseline``, ``growth_plus``, ``buffer_stock``)
   * - ``--phase PHASE``
     - Run specific phase (``screening``, ``grid``, ``stability``)
   * - ``--workers N``
     - Number of parallel workers
   * - ``--seeds N``
     - Seeds per evaluation (default: 3)
   * - ``--periods N``
     - Simulation periods (default: 1000)
   * - ``--grid PATH``
     - Custom grid YAML file
   * - ``--fixed YAML``
     - Fixed parameters as inline YAML
   * - ``--rank-by STRATEGY``
     - Ranking strategy (``combined``, ``stability``, ``mean``)
   * - ``--k-factor FLOAT``
     - Penalty factor for variance in combined ranking (default: 1.0)
   * - ``--output-dir PATH``
     - Output directory for results and reports


Tips
----

- **Start with screening**: Always run Morris screening first — don't skip to
  grid search. Most parameters have negligible effect and can be fixed.
- **Use multiple seeds**: Screening with ``n_seeds=1`` is fast but unreliable.
  Use at least 3 seeds for stable importance rankings.
- **Watch the grid size**: ``count_combinations(grid)`` before running.
  A grid with 5 parameters x 5 values each = 3,125 combinations x 3 seeds =
  9,375 simulations.
- **Stability testing catches fragile optima**: A configuration that scores
  0.95 on one seed but 0.60 on another is not useful. Always stability-test.
- **Extension parameters interact with base parameters**: When calibrating
  extensions, include key base parameters (like ``delta``) in the grid too.


.. seealso::

   - :doc:`validation` for understanding validation scores
   - :doc:`extensions` for extension parameter definitions
   - :doc:`configuration` for all parameter descriptions
   - :doc:`/calibration` for the development-oriented calibration reference
