Phase 3: Stability Testing & Ranking
======================================

The final phase progressively narrows candidates through increasing numbers
of seeds using a tiered tournament.


Tiered Tournament
-----------------

::

   Default tiers: [(100, 10), (50, 20), (10, 100)]

     Tier 1: top 100 configs × 10 seeds → rank → keep top 50
     Tier 2: top 50 × +10 new seeds (20 total) → rank → keep top 10
     Tier 3: top 10 × +80 new seeds (100 total) → final ranking

Each tier only runs **new** seeds (incremental), accumulating all prior
results. Total: 100×10 + 50×10 + 10×80 = 2,300 vs naive 100×100 = 10,000.

.. code-block:: python

   from calibration import run_tiered_stability

   stable = run_tiered_stability(
       candidates=results[:20],
       scenario="baseline",
       n_workers=10,
   )

   for r in stable[:5]:
       print(f"Score: {r.mean_score:.4f} ± {r.std_score:.4f}")


Ranking Strategies
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Strategy
     - Formula
     - Best For
   * - ``combined``
     - ``mean_score × (1 - k × std_score)``
     - Balance of quality and stability
   * - ``stability``
     - ``pass_rate DESC, n_fail ASC, combined DESC``
     - Maximizing reproducibility
   * - ``mean``
     - ``mean_score DESC``
     - Ignoring variance, best average

The ``--k-factor`` parameter (default 1.0) controls variance penalty in
``combined`` mode. Higher k penalizes high-variance configs more heavily.

.. code-block:: bash

   python -m calibration --phase stability --rank-by stability --k-factor 1.5


Multi-Seed Guidance
-------------------

Single-seed screening is necessary for computational efficiency but
**overfits** to the specific random draw. A config ranking 1st with seed=0
may rank 50th across 100 seeds. Recommendations:

- **Tier 1**: At least 10 seeds for initial screening
- **Tier 2**: 20–30 seeds for medium confidence
- **Final selection**: 100+ seeds for publication-quality results


Multi-Pass Workflow
-------------------

For thorough calibration:

1. Run sensitivity → grid → stability for the primary parameters
2. Fix the now-calibrated parameters at their optimal values
3. Re-run the pipeline for previously fixed parameters
4. Compare results and iterate as needed
