Calibration Recipe
==================

Recommended workflow for calibrating BAM Engine parameters,
distilled from four calibration campaigns (~80K simulation runs).


Phase 1: Morris Screening
---------------------------

Always use Morris Method, never OAT alone (Lesson L1). Morris catches
interaction effects that single-baseline OAT misses. The BAM model has
pervasive parameter interactions (mean sigma/mu* ratio ~1.58).

.. code-block:: bash

   python -m calibration --phase morris --scenario baseline --seeds 5

Review the Morris report. If more than 15 parameters are classified
INCLUDE, fix the lower-sensitivity ones at their current defaults
(Lesson L9: fix at defaults, not Morris-best).


Phase 2: Grid Search
----------------------

Build a focused grid from Morris INCLUDE parameters. Fix all others
at their **current defaults** (Lesson L9), not at Morris-best values.

.. code-block:: bash

   python -m calibration --phase grid --scenario baseline


Phase 3: Tiered Stability
---------------------------

Single-seed screening overfits (Lesson L2). Always run multi-seed
stability testing. Check whether the screening winner survives --
if it doesn't, that's expected.

.. code-block:: bash

   python -m calibration --phase stability --scenario baseline


Phase 4: Second-Pass Screening
--------------------------------

If parameters were fixed in Phase 2, run second-pass Morris to
confirm their sensitivity has collapsed (Lesson L3: behavioral
first, structural second).

.. code-block:: bash

   python -m calibration --phase rescreen --scenario baseline \
     --fix-from output/baseline_stability.json --params initial_conditions

If any parameter still shows significant sensitivity, run a targeted
grid + stability for those parameters.


Phase 5: Targeted Cost Analysis
---------------------------------

Ask what parameter values are preferred, then quantify the cost of
using them (Lesson L5). Most preferences are FREE or CHEAP.

.. code-block:: bash

   python -m calibration --phase cost --scenario baseline \
     --base output/baseline_stability.json \
     --swaps "price_init=2.0,1.5" --seeds 20

If cheap combos exist, test them together with ``--combo-grid``.


Phase 6: Cross-Scenario Validation
------------------------------------

For multiple scenarios, use ``cross-eval`` with stability-first
ranking (Lesson L4).

.. code-block:: bash

   python -m calibration --phase cross-eval \
     --scenarios baseline,growth_plus \
     --configs output/baseline_stability.json \
     --rank-by stability-first

If pass rate < 90% on any scenario, use a structured sweep:

.. code-block:: bash

   python -m calibration --phase sweep --scenario growth_plus \
     --base output/stability.json \
     --stages "A:beta=0.5,1.0" "B:max_M=2,4" \
     --cross-scenario baseline


Key Principles
--------------

- **L6** -- Initial conditions are mostly irrelevant: the Kalecki
  attractor erases them in ~50 periods. Save for last.
- **L7** -- Structured sweep > full re-grid: test by category
  (entry, behavioral, initial, credit) rather than all at once.
- **L8** -- Narrow sweet spot: broad exploration is required.
  Hill-climbing does not work in this parameter space.
- **L10** -- Respect parameter coupling: ``job_search_method``
  and ``max_M`` always co-vary.
