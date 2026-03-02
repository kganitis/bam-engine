CLI Reference
=============

Validation scenarios can be run from the command line for quick visualization
and assessment.


Scenario Visualization
----------------------

.. code-block:: bash

   # Baseline scenario (Section 3.9.1) — 8-panel figure
   python -m validation.scenarios.baseline

   # Growth+ scenario (Section 3.9.2) — 16-panel figure
   python -m validation.scenarios.growth_plus

   # Buffer-stock scenario (Section 3.9.4) — 8-panel figure + CCDF
   python -m validation.scenarios.buffer_stock

Each scenario generates multi-panel visualizations with target bounds,
statistical annotations, and validation status indicators.


Programmatic Scenario Running
-----------------------------

.. code-block:: python

   from validation import (
       run_baseline_scenario,
       run_growth_plus_scenario,
       run_buffer_stock_scenario,
   )

   # Run with visualization
   run_baseline_scenario(seed=0, show_plot=True)

   # Run without visualization (returns metrics)
   metrics = run_baseline_scenario(seed=0, show_plot=False)


.. seealso::

   - :doc:`/validation/robustness/cli` for robustness analysis CLI
   - :doc:`/calibration/cli` for calibration CLI
