Calibration
===========

The ``calibration/`` package optimizes BAM model parameters via a multi-phase
pipeline: sensitivity analysis (Morris Method or OAT) → focused grid screening
→ tiered stability testing.

Overview
--------

The calibration process identifies which parameters matter, searches over their
values, and selects the most stable configuration across many random seeds:

1. **Sensitivity analysis** identifies important parameters using the Morris
   Method (default) or one-at-a-time (OAT) sweeps. Unimportant parameters are
   fixed at their best values.

2. **Grid screening** evaluates all combinations of the important parameters
   using a single seed, with checkpointing for resumability.

3. **Tiered stability testing** progressively narrows candidates through
   increasing numbers of seeds (e.g., 100 configs × 10 seeds → 50 × 20 →
   10 × 100), ranking by combined score, stability, or mean.

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

   # Custom grid input
   python -m calibration --phase grid --grid custom_grid.yaml

   # Ranking strategy
   python -m calibration --phase stability --rank-by stability --k-factor 1.5

**Programmatic:**

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

Module Structure
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Purpose
   * - ``cli.py``
     - Argument parsing and phase dispatch
   * - ``analysis.py``
     - ``CalibrationResult`` type, patterns, export, comparison
   * - ``grid.py``
     - Grid building, YAML loading, validation, combination generation
   * - ``screening.py``
     - Single-seed grid screening with checkpointing
   * - ``stability.py``
     - Tiered stability testing with configurable ranking
   * - ``io.py``
     - Save/load for all result types, timestamped output directories
   * - ``reporting.py``
     - Auto-generated markdown reports
   * - ``morris.py``
     - Morris Method screening (elementary effects)
   * - ``sensitivity.py``
     - OAT sensitivity analysis and pairwise interaction testing
   * - ``parameter_space.py``
     - Parameter grids for all three scenarios

For full API documentation, see ``calibration/README.md``.

See Also
--------

* ``calibration/README.md`` — comprehensive API reference and calibration guide
* ``examples/advanced/example_calibration.py`` — programmatic usage example
