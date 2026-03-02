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


The Three Phases
-----------------

**Phase 1: Parameter Screening (Morris Method).** The Morris Method
identifies which parameters significantly affect the score by computing
mu* (importance) and sigma (interaction effects). Parameters are classified
as INCLUDE (important) or FIX (unimportant).

**Phase 2: Grid Search.** Builds a focused grid from screening results and
evaluates all combinations using single-seed screening with checkpointing
for resumability.

**Phase 3: Stability Testing.** A tiered tournament progressively narrows
candidates through increasing numbers of seeds (e.g., 100×10 → 50×20 →
10×100), ranking by combined score, stability, or mean.


Tips
----

- **Start with screening**: Always run Morris screening first. Most parameters
  have negligible effect and can be fixed.
- **Use multiple seeds**: Screening with ``n_seeds=1`` is fast but unreliable.
  Use at least 3 seeds.
- **Stability testing catches fragile optima**: A configuration that scores
  0.95 on one seed but 0.60 on another is not useful. Always stability-test.
- **Extension parameters interact with base parameters**: When calibrating
  extensions, include key base parameters (like ``delta``) in the grid too.


.. seealso::

   Full calibration reference with detailed phase documentation, CLI options,
   and complete API:

   - :doc:`/calibration/index` — Calibration overview and guide
   - :doc:`/calibration/sensitivity` — Morris Method and OAT details
   - :doc:`/calibration/grid_search` — Grid building and screening
   - :doc:`/calibration/stability` — Tiered stability and ranking
   - :doc:`/calibration/cli` — Full CLI reference
   - :doc:`validation` for understanding validation scores
   - :doc:`extensions` for extension parameter definitions
   - :doc:`configuration` for all parameter descriptions
