Taxation Extension
==================

*Used in Section 3.10.2 structural experiments of Delli Gatti et al. (2011)*

The taxation extension adds **profit taxation** to firms. Tax revenue is
removed from the economy (no redistribution) — this is designed for structural
experiments testing the role of automatic stabilizers and entry dynamics.


Quick Start
-----------

.. code-block:: python

   import bamengine as bam
   from extensions.taxation import FirmsTaxProfits, TAXATION_CONFIG

   sim = bam.Simulation.init(seed=42, profit_tax_rate=0.20)
   sim.use_events(FirmsTaxProfits)

   results = sim.run(n_periods=1000, collect=True)

.. note::

   The taxation extension has no role — it operates directly on the existing
   ``Borrower`` role fields (``net_profit`` and ``total_funds``).


Events
------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Event
     - Hook
     - Description
   * - ``FirmsTaxProfits``
     - after ``firms_validate_debt_commitments``
     - Deduct ``profit_tax_rate * max(0, net_profit)`` from firm funds


Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Description
   * - ``profit_tax_rate``
     - 0.0
     - Tax rate on positive net profit (0.0 = no tax)


API Reference
-------------

.. automodule:: extensions.taxation
   :members:
   :undoc-members:

.. automodule:: extensions.taxation.events
   :members:
   :undoc-members:


.. seealso::

   - :doc:`/validation/robustness/structural` for entry neutrality experiments using taxation
   - :doc:`/user_guide/extensions` for extension development patterns
