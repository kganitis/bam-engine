Structured Parameter Sweep
===========================

Runs a multi-stage parameter sweep where each stage tests a category of
parameters and the winner's values carry forward to subsequent stages.
This is more efficient than a full re-grid when only a few parameters
need adjustment.

This implements Lesson L7 (*structured sweep > full re-grid*): test by
category (entry, behavioral, initial, credit) rather than all at once.


Stage Syntax
------------

Each stage is defined as ``LABEL:param1=v1,v2 param2=v3,v4``:

.. code-block:: bash

   --stages "entry:new_firm_size_factor=0.1,0.2 new_firm_price_markup=0.05,0.1" \
            "behavioral:beta=0.5,1.0,2.5 max_M=2,4"

The label is used for reporting. Parameters within a stage form a grid;
stages execute sequentially with winners carried forward.


CLI Usage
---------

.. code-block:: bash

   # Two-stage sweep with cross-scenario check
   python -m calibration --phase sweep --scenario growth_plus \
     --base output/growth_plus_stability.json \
     --stages "A:beta=0.5,1.0,2.5" "B:max_leverage=5,10,20" \
     --cross-scenario baseline

Required flags:

- ``--base``: Path to stability result JSON or YAML with base config
- ``--stages``: One or more stage definitions

Optional:

- ``--cross-scenario``: Cross-evaluate against this scenario at each stage


Python API
----------

.. code-block:: python

   from calibration.sweep import run_sweep, parse_stages

   stages = parse_stages(
       [
           "entry:new_firm_size_factor=0.1,0.2",
           "behavioral:beta=0.5,1.0,2.5 max_M=2,4",
       ]
   )

   results = run_sweep(
       base_params={"beta": 5.0, "max_M": 4},
       stages=stages,
       scenario="baseline",
       n_workers=10,
   )

   for stage in results:
       print(f"Stage {stage.label}: {'improved' if stage.improved else 'no change'}")
       print(f"  Winner: {stage.winner_params}")


API Reference
-------------

.. automodule:: calibration.sweep
   :members:
   :undoc-members:
   :no-index:
