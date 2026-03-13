Targeted Cost Analysis
=======================

Quantifies the score cost of substituting preferred parameter values into
an optimized base configuration. Each swap is classified by impact:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Classification
     - Absolute Delta
     - Meaning
   * - ``FREE``
     - < 0.002
     - No measurable impact
   * - ``CHEAP``
     - < 0.005
     - Negligible impact
   * - ``MODERATE``
     - < 0.010
     - Small but detectable
   * - ``EXPENSIVE``
     - >= 0.010
     - Significant impact

This implements Lesson L5 (*targeted cost analysis for endgame*): most
parameter preferences turn out to be FREE or CHEAP.


CLI Usage
---------

.. code-block:: bash

   # Test swapping preferred values
   python -m calibration --phase cost --scenario baseline \
     --base output/baseline_stability.json \
     --swaps "price_init=2.0,1.5" "min_wage_ratio=0.3" --seeds 20

   # Run combo grid of all cheap swaps
   python -m calibration --phase cost --scenario baseline \
     --base output/baseline_stability.json \
     --swaps "price_init=2.0" "beta=2.5" --combo-grid

Required flags:

- ``--base``: Path to stability result JSON or YAML with base config
- ``--swaps``: One or more ``param=v1,v2`` specifications


Python API
----------

.. code-block:: python

   from calibration.cost import run_cost_analysis, classify_cost

   results = run_cost_analysis(
       base_params={"beta": 5.0, "max_M": 4},
       swaps={"beta": [2.5], "price_init": [2.0, 1.5]},
       scenario="baseline",
       n_seeds=20,
   )

   for r in results:
       print(f"{r.param}={r.value}: {r.classification} (delta={r.delta:+.4f})")


API Reference
-------------

.. automodule:: calibration.cost
   :members:
   :undoc-members:
   :no-index:
