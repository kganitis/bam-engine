Phase 2: Grid Search & Screening
==================================

The second phase builds a focused parameter grid and searches all
combinations using single-seed evaluation.


Grid Building
-------------

Parameters are categorized by sensitivity analysis results:

- **INCLUDE** (Δ > threshold): Include in grid search with all values
- **FIX** (Δ ≤ threshold): Fix at best value from sensitivity analysis

.. code-block:: python

   from calibration import build_focused_grid, count_combinations

   grid, fixed = build_focused_grid(sensitivity)
   print(f"Grid: {count_combinations(grid)} combinations")
   print(f"Fixed: {fixed}")

**Value pruning** (optional): Drops grid values whose OAT score is more than
``pruning_threshold`` below the best value for that parameter. Default:
``auto`` (2× sensitivity threshold). Use ``--pruning-threshold none`` to
disable.


Grid from YAML
--------------

.. code-block:: yaml

   # calibration_grid.yml
   grid:
     h_rho: [0.05, 0.10, 0.15]
     delta: [0.05, 0.10]
   fixed:
     theta: 8

.. code-block:: bash

   python -m calibration --phase grid --grid calibration_grid.yml


Parameter Grid
--------------

The 14 common parameters cover:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Parameters
   * - Initial conditions
     - ``price_init``, ``min_wage_ratio``, ``net_worth_ratio``,
       ``equity_base_init``, ``savings_init``
   * - New firm entry
     - ``new_firm_size_factor``, ``new_firm_production_factor``,
       ``new_firm_wage_factor``, ``new_firm_price_markup``
   * - Economy-wide
     - ``beta``
   * - Search frictions
     - ``max_M``
   * - Credit
     - ``max_loan_to_net_worth``, ``max_leverage``
   * - Labor
     - ``job_search_method``

Extension-specific: Growth+ adds ``sigma_decay``, ``sigma_max``;
buffer-stock adds ``buffer_stock_h``.


Checkpointing
-------------

Grid screening supports automatic checkpointing for resumability:

.. code-block:: bash

   # Start a grid search (auto-checkpoints)
   python -m calibration --phase grid --scenario baseline

   # Resume after interruption
   python -m calibration --phase grid --scenario baseline --resume

Checkpoints save completed results incrementally, so interrupted runs lose
only the current evaluation.
