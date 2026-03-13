Second-Pass Screening (rescreen)
=================================

After locking optimized parameters from the main calibration pipeline,
second-pass Morris screening tests whether previously fixed parameters
become important when the primary ones are held at their optimal values.

This implements Lesson L3 (*calibration order matters*): behavioral parameters
first, structural second, with second-pass Morris to confirm sensitivity
collapse.


CLI Usage
---------

.. code-block:: bash

   # Screen structural params after locking the behavioral winner
   python -m calibration --phase rescreen --scenario growth_plus \
     --fix-from output/growth_plus_stability.json --params initial_conditions

   # Screen a comma-separated list of specific params
   python -m calibration --phase rescreen --scenario baseline \
     --fix-from output/baseline_stability.json --params beta,max_M

Required flags:

- ``--fix-from``: Path to stability result JSON (loads #1-ranked config)
- ``--params``: Parameter group name (from ``PARAM_GROUPS``) or comma-separated names


Python API
----------

.. code-block:: python

   from calibration.rescreen import run_rescreen, compute_sensitivity_collapse

   result, collapse = run_rescreen(
       scenario="baseline",
       fix_from=Path("output/baseline_stability.json"),
       params=["price_init", "min_wage_ratio"],
       n_seeds=5,
   )

   for name, data in collapse.items():
       print(f"{name}: {data['collapse_pct']:.1f}% collapse")


API Reference
-------------

.. automodule:: calibration.rescreen
   :members:
   :undoc-members:
   :no-index:
