Sensitivity Analysis
====================

*Section 3.10.1, Part 2*

Univariate sensitivity analysis varies one parameter at a time while holding
all others at baseline values. For each parameter value, it runs multiple
simulations and computes the same statistics as the internal validity analysis.


The Five Experiments
--------------------

.. list-table::
   :header-rows: 1
   :widths: 5 20 12 25 12

   * - #
     - Experiment
     - Parameter
     - Values
     - Baseline
   * - i
     - Credit market
     - ``max_H``
     - 1, 2, 3, 4, 6
     - 2
   * - ii
     - Goods market
     - ``max_Z``
     - 2, 3, 4, 5, 6
     - 2
   * - iii
     - Labor applications
     - ``max_M``
     - 2, 3, 4, 5, 6
     - 4
   * - iv
     - Contract length
     - ``theta``
     - 1, 4, 6, 8, 10, 12, 14
     - 8
   * - v
     - Economy size
     - multi-param
     - 7 configurations
     - 100/500/10


Key Findings from the Book
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Experiment
     - Finding
   * - **Credit market (H)**
     - General properties stable. As H increases, price index becomes coincident
       with output; net worth distribution becomes more Pareto-like.
   * - **Goods market (Z)**
     - As Z increases, competition rises, production smooths, firm size
       kurtosis decreases. Real wages become lagging.
   * - **Labor (M)**
     - As M decreases, prices become pro-cyclical/lagging, instability rises.
       As M increases, wages are pushed above productivity.
   * - **Contract length** (:math:`\theta`)
     - Extreme values cause degenerate dynamics. :math:`\theta=1`: collapse. :math:`\theta \geq 12`:
       supply-side breakdown. :math:`\theta=6`--10: stable.
   * - **Economy size**
     - Proportional scaling preserves co-movements but smooths fluctuations.


Usage
-----

.. code-block:: python

   from validation.robustness import (
       run_sensitivity_analysis,
       print_sensitivity_report,
       plot_sensitivity_comovements,
   )

   sa = run_sensitivity_analysis(
       experiments=["credit_market", "contract_length"],
       n_seeds=20,
       n_periods=1000,
   )

   print_sensitivity_report(sa)
   for exp in sa.experiments.values():
       plot_sensitivity_comovements(exp, show=True)


Custom Experiments
------------------

Define custom experiments using the ``Experiment`` dataclass:

.. code-block:: python

   from validation.robustness.experiments import Experiment, EXPERIMENTS

   my_experiment = Experiment(
       name="custom_delta",
       description="Sensitivity to depreciation rate",
       param="delta",
       values=[0.05, 0.10, 0.15, 0.20],
       baseline_value=0.10,
   )

   EXPERIMENTS["custom_delta"] = my_experiment
   result = run_sensitivity_analysis(experiments=["custom_delta"])


API Reference
-------------

.. automodule:: validation.robustness.sensitivity
   :members:
   :undoc-members:

.. automodule:: validation.robustness.experiments
   :members:
   :undoc-members:
