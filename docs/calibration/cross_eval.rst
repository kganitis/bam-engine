Cross-Scenario Evaluation
==========================

Evaluates parameter configurations across multiple scenarios simultaneously
and ranks using cross-scenario criteria.

Three ranking strategies are available:

- **stability-first**: Sort by minimum pass rate across scenarios, then total
  fails, then minimum combined score. Best when you need all scenarios to pass.
- **score-first**: Sort by minimum combined score across scenarios. Best when
  you want the highest floor on quality.
- **balanced**: Sort by geometric mean of combined scores. Best when you want
  a balanced tradeoff across scenarios.

This implements Lesson L4 (*cross-scenario needs different ranking*).


CLI Usage
---------

.. code-block:: bash

   # Evaluate top configs across baseline and growth_plus
   python -m calibration --phase cross-eval \
     --scenarios baseline,growth_plus \
     --configs output/baseline_stability.json \
     --seeds 100 --rank-by stability-first

Required flags:

- ``--scenarios``: Comma-separated list of scenario names
- ``--configs``: Path to stability/screening result JSON


Python API
----------

.. code-block:: python

   from calibration.cross_eval import evaluate_cross_scenario, rank_cross_scenario

   results = evaluate_cross_scenario(
       configs=[{"beta": 5.0, "max_M": 4}],
       scenarios=["baseline", "growth_plus"],
       n_seeds=100,
   )

   ranked = rank_cross_scenario(results, strategy="stability-first")


Scenario Tension
^^^^^^^^^^^^^^^^

Use ``compute_scenario_tension`` to identify parameters where different
scenarios prefer different values:

.. code-block:: python

   from calibration.cross_eval import compute_scenario_tension

   tension = compute_scenario_tension(results, ["baseline", "growth_plus"])
   for param, details in tension.items():
       print(f"{param}: scenarios disagree on best value")


API Reference
-------------

.. automodule:: calibration.cross_eval
   :members:
   :undoc-members:
   :no-index:
