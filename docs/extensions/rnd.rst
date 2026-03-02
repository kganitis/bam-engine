Growth+ (R&D) Extension
========================

*Based on Section 3.8 of Delli Gatti et al. (2011)*

The Growth+ extension adds **endogenous productivity growth** via R&D investment.
Profitable firms invest a fraction of their profits in R&D, and their labor
productivity grows stochastically based on R&D intensity. This creates persistent
firm-level heterogeneity and aggregate productivity growth.


Quick Start
-----------

.. code-block:: python

   import bamengine as bam
   from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

   sim = bam.Simulation.init(seed=42)
   sim.use_role(RnD)
   sim.use_events(*RND_EVENTS)
   sim.use_config(RND_CONFIG)

   results = sim.run(n_periods=1000, collect=True)


Role Fields
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``sigma``
     - R&D share of profits (0 to ``sigma_max``), decreasing with financial fragility
   * - ``rnd_intensity``
     - Expected productivity gain (:math:`\mu`), derived from sigma and profit
   * - ``productivity_increment``
     - Actual productivity gain drawn from exponential distribution each period
   * - ``fragility``
     - Financial fragility metric (wage bill / net worth)


Events
------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Event
     - Hook
     - Description
   * - ``FirmsComputeRnDIntensity``
     - after ``firms_validate_debt_commitments``
     - Compute R&D share :math:`\sigma` and intensity :math:`\mu` from profits
       and fragility
   * - ``FirmsApplyProductivityGrowth``
     - after ``firms_compute_rnd_intensity``
     - Draw productivity increment from :math:`\text{Exp}(\mu)` and add to
       labor productivity
   * - ``FirmsDeductRnDExpenditure``
     - after ``firms_apply_productivity_growth``
     - Reduce net profit by :math:`(1 - \sigma)` factor before dividends


Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Description
   * - ``sigma_min``
     - 0.0
     - Minimum R&D share (for highly fragile firms)
   * - ``sigma_max``
     - 0.1
     - Maximum R&D share (for financially healthy firms)
   * - ``sigma_decay``
     - -1.0
     - Decay rate for sigma as a function of fragility

**Expected behavior:** Rising average labor productivity over time, increasing
firm size heterogeneity, and a positive long-run growth trend in output.


API Reference
-------------

.. automodule:: extensions.rnd
   :members:
   :undoc-members:

.. automodule:: extensions.rnd.role
   :members:
   :undoc-members:

.. automodule:: extensions.rnd.events
   :members:
   :undoc-members:


.. seealso::

   - :doc:`/user_guide/extensions` for writing custom extensions
   - Growth+ validation: ``python -m validation.scenarios.growth_plus``
   - :doc:`/auto_examples/extensions/example_growth_plus`
