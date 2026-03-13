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
     - Phase: ``sensitivity``, ``morris``, ``grid``, ``stability``, ``pairwise``,
       ``rescreen``, ``cost``, ``cross-eval``, ``sweep``
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
   * - ``--sensitivity-seeds``, ``--seeds``
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
     - Ranking strategy: ``combined``, ``stability``, ``mean``,
       ``stability-first``, ``score-first``, ``balanced``
   * - ``--k-factor``
     - 1.0
     - k in mean−k×std formula (combined mode)
   * - ``--output-dir``
     - (timestamped)
     - Custom output directory
   * - ``--fix-from``
     - (none)
     - Load fixed params from stability JSON (rescreen)
   * - ``--params``
     - (none)
     - Parameter group or comma-separated names (rescreen)
   * - ``--base``
     - (none)
     - Base config from stability JSON or YAML (cost, sweep)
   * - ``--swaps``
     - (none)
     - Swap values: ``param=v1,v2`` (repeatable, cost)
   * - ``--combo-grid``
     - false
     - Run combo grid of cheap swaps (cost)
   * - ``--scenarios``
     - (none)
     - Comma-separated scenario list (cross-eval)
   * - ``--configs``
     - (none)
     - Configs from result JSON (cross-eval)
   * - ``--stages``
     - (none)
     - Stage definitions: ``LABEL:param=v1,v2`` (sweep)
   * - ``--cross-scenario``
     - (none)
     - Cross-evaluate against this scenario (sweep)


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

   # Second-pass Morris screening
   python -m calibration --phase rescreen --scenario baseline \
     --fix-from output/baseline_stability.json --params behavioral

   # Targeted cost analysis
   python -m calibration --phase cost --scenario baseline \
     --base output/baseline_stability.json --swaps "beta=2.5" --seeds 20

   # Cross-scenario evaluation
   python -m calibration --phase cross-eval \
     --scenarios baseline,growth_plus --configs output/stability.json

   # Structured parameter sweep
   python -m calibration --phase sweep --scenario baseline \
     --base output/stability.json \
     --stages "A:beta=0.5,1.0" "B:max_M=2,4" --cross-scenario growth_plus
