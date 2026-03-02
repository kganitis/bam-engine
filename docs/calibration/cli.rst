CLI Reference
=============

.. code-block:: bash

   python -m calibration [OPTIONS]


Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--scenario``
     - baseline
     - Scenario: ``baseline``, ``growth_plus``, ``buffer_stock``
   * - ``--phase``
     - all
     - Phase: ``sensitivity``, ``morris``, ``grid``, ``stability``, ``pairwise``
   * - ``--method``
     - morris
     - Sensitivity method: ``morris`` or ``oat``
   * - ``--morris-trajectories``
     - 10
     - Number of Morris trajectories
   * - ``--workers``
     - 10
     - Parallel workers
   * - ``--periods``
     - 1000
     - Simulation periods
   * - ``--sensitivity-threshold``
     - 0.02
     - Minimum Δ for INCLUDE in grid search
   * - ``--pruning-threshold``
     - auto
     - Max score gap for keeping values: ``auto``, ``none``, or float
   * - ``--sensitivity-seeds``
     - 3
     - Seeds per sensitivity evaluation
   * - ``--stability-tiers``
     - 100:10,50:20,10:100
     - Tiers as ``configs:seeds,...``
   * - ``--resume``
     - false
     - Resume from checkpoint
   * - ``--grid``
     - (none)
     - Load custom grid from YAML/JSON file
   * - ``--fixed``
     - (none)
     - Fix a parameter: ``KEY=VALUE`` (repeatable)
   * - ``--rank-by``
     - combined
     - Ranking strategy: ``combined``, ``stability``, ``mean``
   * - ``--k-factor``
     - 1.0
     - k in mean−k×std formula (combined mode)
   * - ``--output-dir``
     - (timestamped)
     - Custom output directory


Output Structure
----------------

Results are saved to timestamped directories:

::

   output/2026-02-24_143052_baseline/
   ├── sensitivity.json          # Sensitivity analysis results
   ├── sensitivity_report.md     # Sensitivity markdown report
   ├── morris.json               # Morris method results
   ├── screening.json            # Grid screening results
   ├── screening_report.md       # Screening markdown report
   ├── stability.json            # Stability-tested results
   ├── stability_report.md       # Stability markdown report
   ├── full_report.md            # Combined report
   ├── best_config.yml           # Best config as YAML
   └── pairwise.json             # Pairwise interactions


Common Workflows
----------------

.. code-block:: bash

   # Full pipeline with defaults
   python -m calibration --scenario baseline --workers 10

   # Morris screening only
   python -m calibration --phase sensitivity --workers 10

   # OAT instead of Morris
   python -m calibration --phase sensitivity --method oat --workers 10

   # Custom grid input
   python -m calibration --phase grid --grid custom_grid.yaml

   # Custom stability tiers
   python -m calibration --phase stability --stability-tiers "50:10,20:30,5:100"

   # Growth+ scenario
   python -m calibration --scenario growth_plus --workers 10
