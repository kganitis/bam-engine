Structural Experiments
======================

*Section 3.10.2 of Delli Gatti et al. (2011)*

Structural experiments test the model's **mechanisms** rather than sweeping
parameter values. Two experiments are implemented.


PA (Preferential Attachment) Experiment
---------------------------------------

Tests the effect of consumer loyalty on economic dynamics by disabling the
"rich get richer" mechanism:

1. **Phase 1**: Run internal validity with ``consumer_matching="random"``
   (PA off)
2. **Phase 2**: Run Z-sweep with PA off
3. **Optional baseline**: Run internal validity with PA on for comparison

**Expected findings from the book:**

- GDP volatility drops sharply
- Deep crises disappear
- Wages/prices become lagging or acyclical
- AR persistence drops from ~0.8 to ~0.4
- Firm size distribution becomes more uniform

.. code-block:: python

   from validation.robustness import (
       run_pa_experiment,
       print_pa_report,
       plot_pa_gdp_comparison,
       plot_pa_comovements,
   )

   pa = run_pa_experiment(n_seeds=20, n_periods=1000, include_baseline=True)
   print_pa_report(pa)
   plot_pa_gdp_comparison(pa, show=True)
   plot_pa_comovements(pa, show=True)


Entry Neutrality Experiment
----------------------------

Tests whether automatic firm entry artificially drives recovery by imposing
heavy profit taxation without redistribution:

- Sweeps ``profit_tax_rate`` from 0% to 90%
- Uses the :doc:`/extensions/taxation` extension
- **Expected finding**: Monotonic degradation of economic performance,
  confirming that the business cycle is genuinely endogenous

.. code-block:: python

   from validation.robustness import (
       run_entry_experiment,
       print_entry_report,
       plot_entry_comparison,
   )

   entry = run_entry_experiment(n_seeds=20, n_periods=1000)
   print_entry_report(entry)
   plot_entry_comparison(entry, show=True)


Result Structures
-----------------

``PAExperimentResult``:

- ``pa_off_validity``: Internal validity with PA off
- ``pa_off_z_sweep``: Z-sweep sensitivity with PA off
- ``baseline_validity``: Optional PA-on baseline for comparison

``EntryExperimentResult``:

- ``tax_sweep``: Tax rate sweep results


API Reference
-------------

.. automodule:: validation.robustness.structural
   :members:
   :undoc-members:
