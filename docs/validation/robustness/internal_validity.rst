Internal Validity
=================

*Section 3.10.1, Part 1*

Internal validity tests whether the model produces qualitatively similar results
regardless of the random seed. It runs multiple simulations (default 20) with
default parameters and performs five checks.


Checks
------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Check
     - Description
   * - Cross-simulation variance
     - Key macro statistics should have small variance across seeds
   * - Co-movement structure
     - Cross-correlations between HP-filtered GDP and other variables at
       leads/lags should be consistent (Figure 3.9)
   * - AR model fit
     - Individual seeds fit AR(2); cross-seed average fits AR(1) with
       parameter ~0.8
   * - Firm size distributions
     - Should remain non-normal and positively skewed across all seeds
   * - Empirical curves
     - Phillips, Okun, and Beveridge curves should emerge from each simulation


Usage
-----

.. code-block:: python

   from validation.robustness import (
       run_internal_validity,
       print_internal_validity_report,
       plot_comovements,
       plot_irf,
   )

   result = run_internal_validity(
       n_seeds=20,
       n_periods=1000,
       burn_in=500,
       n_workers=10,
   )

   print_internal_validity_report(result)
   plot_comovements(result, show=True)
   plot_irf(result, show=True)


Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``n_seeds``
     - 20
     - Number of random seeds to test
   * - ``n_periods``
     - 1000
     - Simulation periods per seed
   * - ``burn_in``
     - 500
     - Burn-in periods to discard
   * - ``n_workers``
     - 10
     - Parallel workers
   * - ``max_lag``
     - 4
     - Maximum lead/lag for cross-correlations
   * - ``ar_order_single``
     - 2
     - AR order for individual seeds
   * - ``ar_order_mean``
     - 1
     - AR order for cross-seed average
   * - ``setup_hook``
     - None
     - Extension setup callable (e.g., ``setup_growth_plus``)


Result Structure
----------------

``InternalValidityResult`` contains:

- ``seed_analyses``: Per-seed results (co-movements, AR coefficients, statistics)
- ``mean_comovements`` / ``std_comovements``: Cross-seed aggregates
- ``mean_ar_coeffs``, ``mean_ar_r_squared``: Mean AR parameters
- ``mean_irf``: Mean impulse-response function
- ``cross_sim_stats``: Summary statistics with CV across seeds
- ``n_collapsed``, ``n_degenerate``: Problematic simulation counts

Each ``SeedAnalysis`` contains co-movement correlations (5 variables × 9 lags),
AR coefficients, IRF, summary statistics, empirical curve correlations, and
firm size distribution metrics.


API Reference
-------------

.. automodule:: validation.robustness.internal_validity
   :members:
   :undoc-members:
