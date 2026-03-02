Visualization
=============

The ``viz`` module provides publication-quality plots for robustness analysis
results.


Co-Movement Plot (Figure 3.9)
------------------------------

Creates a 3x2 grid showing cross-correlations at leads/lags (-4 to +4) for
five variables: unemployment, productivity, price index, real interest rate,
and real wage.

.. code-block:: python

   from validation.robustness import plot_comovements

   plot_comovements(iv_result, output_dir="output/", show=True)


Impulse-Response Function Plot
-------------------------------

Compares baseline AR(2) IRF (dashed) with cross-simulation mean AR(1) IRF
(solid).

.. code-block:: python

   from validation.robustness import plot_irf

   plot_irf(iv_result, show=True)


Sensitivity Co-Movement Comparison
------------------------------------

Shows how co-movement structure changes across parameter values for each
experiment.

.. code-block:: python

   from validation.robustness import plot_sensitivity_comovements

   for exp_result in sa.experiments.values():
       plot_sensitivity_comovements(exp_result, show=True)


PA Experiment Plots
-------------------

**GDP comparison** (Figure 3.10): Side-by-side time series of GDP with and
without preferential attachment.

.. code-block:: python

   from validation.robustness import plot_pa_gdp_comparison, plot_pa_comovements

   plot_pa_gdp_comparison(pa, show=True)
   plot_pa_comovements(pa, show=True)


Entry Experiment Plots
----------------------

GDP growth and bankruptcy rates across tax rate levels.

.. code-block:: python

   from validation.robustness import plot_entry_comparison

   plot_entry_comparison(entry, show=True)


API Reference
-------------

.. automodule:: validation.robustness.viz
   :members:
   :undoc-members:
