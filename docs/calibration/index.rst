Calibration
===========

The ``calibration`` package optimizes BAM model parameters via a multi-phase
pipeline: sensitivity analysis → focused grid screening → tiered stability
testing.


Quick Start
-----------

**CLI:**

.. code-block:: bash

   # Run full pipeline (Morris → grid → stability)
   python -m calibration --scenario baseline --workers 10

   # Individual phases
   python -m calibration --phase sensitivity --scenario baseline
   python -m calibration --phase grid --scenario baseline
   python -m calibration --phase stability --scenario baseline

**Python API:**

.. code-block:: python

   from calibration import (
       run_morris_screening,
       build_focused_grid,
       run_focused_calibration,
       export_best_config,
   )

   # Phase 1: Morris screening
   morris = run_morris_screening(scenario="baseline", n_workers=10, n_seeds=3)
   sensitivity = morris.to_sensitivity_result()

   # Phase 2: Build focused grid
   grid, fixed = build_focused_grid(sensitivity)

   # Phases 3-4: Screening + tiered stability
   results = run_focused_calibration(grid, fixed, rank_by="combined")

   # Export best config
   export_best_config(results[0], "baseline")


Calibration Process
-------------------

::

   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ Phase 1          │     │ Phase 2          │     │ Phase 3          │
   │ Sensitivity      │────>│ Grid Screening   │────>│ Tiered Stability │
   │ (Morris or OAT)  │     │ (single seed)    │     │ (multi-seed)     │
   └──────────────────┘     └──────────────────┘     └──────────────────┘
    Identifies important      Evaluates all           Tournament-style
    parameters, fixes         combinations,           multi-seed testing
    unimportant ones          prunes poor values       with ranking


Scenarios
---------

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Scenario
     - Parameters
     - Description
   * - ``baseline``
     - 14 common
     - Standard BAM model (Section 3.9.1)
   * - ``growth_plus``
     - 14 + 2 R&D
     - Endogenous R&D growth (Section 3.9.2)
   * - ``buffer_stock``
     - 14 + 3
     - Buffer-stock consumption (Section 3.9.4)


Calibration Guide
-----------------

**Morris vs OAT:** Morris Method catches interaction effects that single-baseline
OAT misses. Mean sigma/mu* ratio of ~1.58 indicates pervasive interactions in
the BAM model — Morris is strongly recommended for production calibration.

**Parameter importance:** Economy-wide params (``beta``, ``max_M``) are
consistently the most important. Initial conditions are largely irrelevant —
the Kalecki attractor erases them in ~50 periods.

**Multi-seed stability:** Single-seed screening overfits to the specific random
draw. A config ranking 1st with seed=0 may rank 50th across 100 seeds. Use
at least 10 seeds for tier 1 and 100+ for final selection.

**Multi-pass workflow:** Run sensitivity → grid → stability for primary
parameters, fix them at optimal values, then re-run the pipeline for
previously fixed parameters. See :doc:`recipe` for the full recommended
workflow.


.. toctree::
   :maxdepth: 2
   :hidden:

   sensitivity
   grid_search
   stability
   rescreen
   cost_analysis
   cross_eval
   sweep
   recipe
   cli
   api
