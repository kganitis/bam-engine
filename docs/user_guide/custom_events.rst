Custom Events
=============

Events are the behavioral rules of the simulation — the "systems" in BAM Engine's
ECS architecture. Each event reads role data, performs calculations, and writes
results back. Custom events let you add new economic behaviors or modify existing
ones.


Quick Example
-------------

.. code-block:: python

   from bamengine import event, ops, Simulation


   @event(after="firms_collect_revenue")
   class FirmsPayBonus:
       """Pay a 5% bonus to firms with positive net profit."""

       def execute(self, sim: Simulation) -> None:
           bor = sim.get_role("Borrower")
           bonus = ops.where(bor.net_profit > 0, ops.multiply(bor.net_profit, 0.05), 0)
           ops.assign(bor.total_funds, ops.add(bor.total_funds, bonus))


The ``@event`` Decorator
------------------------

The ``@event`` decorator registers an event class and optionally specifies where
it should be inserted in the pipeline.

**Standalone event** (no automatic placement):

.. code-block:: python

   @event
   class MyEvent:
       def execute(self, sim): ...

**Hook after** an existing event:

.. code-block:: python

   @event(after="firms_pay_dividends")
   class AfterDividends:
       def execute(self, sim): ...

**Hook before** an existing event:

.. code-block:: python

   @event(before="firms_run_production")
   class BeforeProduction:
       def execute(self, sim): ...

**Replace** an existing event entirely:

.. code-block:: python

   @event(replace="consumers_calc_propensity")
   class MyPropensityRule:
       def execute(self, sim): ...

**Custom name** (defaults to class name converted to snake_case):

.. code-block:: python

   @event(name="custom_tax_event", after="firms_validate_debt_commitments")
   class FirmsTaxProfits:
       def execute(self, sim): ...


The ``execute()`` Method
------------------------

Every event must implement an ``execute`` method:

.. code-block:: python

   def execute(self, sim: Simulation) -> None: ...

The ``sim`` parameter provides access to all simulation state. The method should
not return a value — all effects happen through mutations via ``ops.assign()``
or relationship methods.


Accessing Simulation State
--------------------------

Inside ``execute()``, the ``sim`` object exposes:

**Roles** — agent data arrays:

.. code-block:: python

   prod = sim.get_role("Producer")  # or sim.prod
   wrk = sim.get_role("Worker")  # or sim.wrk
   emp = sim.get_role("Employer")  # or sim.emp
   bor = sim.get_role("Borrower")  # or sim.bor
   con = sim.get_role("Consumer")  # or sim.con
   sh = sim.get_role("Shareholder")  # or sim.sh
   lend = sim.get_role("Lender")  # or sim.lend

**Economy** — aggregate state:

.. code-block:: python

   sim.ec.avg_mkt_price  # Current average market price
   sim.ec.min_wage  # Current minimum wage
   sim.ec.collapsed  # Whether the economy has collapsed

**Configuration** — model parameters:

.. code-block:: python

   sim.theta  # Contract length
   sim.delta  # Dividend payout ratio
   sim.h_rho  # Production shock width
   sim.n_firms  # Number of firms

**RNG** — random number generator:

.. code-block:: python

   shock = sim.rng.uniform(0, 0.1, size=sim.n_firms)
   # or: shock = ops.uniform(sim.rng, 0, 0.1, size=sim.n_firms)

**Relationships** — loan data:

.. code-block:: python

   loans = sim.get_relationship("LoanBook")  # or sim.lb

**Extension parameters** — custom parameters passed at init:

.. code-block:: python

   sigma_min = sim.sigma_min  # Accesses extra_params["sigma_min"]


Pipeline Integration
--------------------

**Hook activation is explicit.** Declaring ``@event(after="...")`` stores the
hook as class metadata but does **not** modify the pipeline. You must call
:meth:`~bamengine.Simulation.use_events` to apply hooks:

.. code-block:: python

   import bamengine as bam

   sim = bam.Simulation.init(seed=42)
   sim.use_events(FirmsPayBonus)  # NOW the hook is applied

   # Multiple events at once
   sim.use_events(EventA, EventB, EventC)

.. warning::

   Forgetting ``sim.use_events()`` is the most common mistake when working with
   custom events. The event will be registered but never executed.

**Manual pipeline methods** (for events without hooks):

.. code-block:: python

   sim.pipeline.insert_after("target_event", "my_event")
   sim.pipeline.insert_before("target_event", "my_event")
   sim.pipeline.remove("event_to_remove")
   sim.pipeline.replace("old_event", "new_event")


Worked Example: Inventory Carrying Cost
----------------------------------------

This complete example adds an inventory carrying cost to firms — a charge for
holding unsold goods. It demonstrates the full workflow: define a role, define
an event, hook it into the pipeline, and run.

.. code-block:: python

   import bamengine as bam
   from bamengine import event, ops, role
   from bamengine.typing import Float


   # 1. Define a role to track carrying costs
   @role
   class InventoryCost:
       carrying_cost: Float


   # 2. Define an event that computes and deducts carrying costs
   @event(after="firms_run_production")
   class FirmsPayCarryingCost:
       """Charge firms for holding unsold inventory."""

       def execute(self, sim):
           prod = sim.prod
           bor = sim.bor
           ic = sim.get_role("InventoryCost")

           # 2% of inventory value per period
           cost = ops.multiply(prod.inventory, ops.multiply(prod.price, 0.02))
           ops.assign(ic.carrying_cost, cost)

           # Deduct from available funds
           ops.assign(bor.total_funds, ops.subtract(bor.total_funds, cost))


   # 3. Set up and run
   sim = bam.Simulation.init(seed=42)
   sim.use_role(InventoryCost)
   sim.use_events(FirmsPayCarryingCost)

   results = sim.run(
       n_periods=100,
       collect={"InventoryCost": True, "Economy": True, "aggregate": "mean"},
   )


Built-in Events
---------------

BAM Engine includes 39 built-in events organized in 8 phases. Below is a
summary — see :doc:`bam_model` for the economic logic of each phase.

**Phase 1: Planning**

.. list-table::
   :widths: 40 60

   * - ``firms_decide_desired_production``
     - Set production targets from demand/inventory signals
   * - ``firms_plan_breakeven_price``
     - Calculate cost-covering price floor (planning phase)
   * - ``firms_plan_price``
     - Adjust price based on inventory and market position
   * - ``firms_decide_desired_labor``
     - Calculate workforce needed for production target
   * - ``firms_decide_vacancies``
     - Post vacancies to fill labor gap
   * - ``firms_fire_excess_workers``
     - Fire workers when desired labor < current labor

**Phase 2: Labor Market**

.. list-table::
   :widths: 40 60

   * - ``calc_inflation_rate``
     - Compute economy-wide inflation rate
   * - ``adjust_minimum_wage``
     - Revise minimum wage for inflation
   * - ``firms_decide_wage_offer``
     - Firms set wage offers with random markup
   * - ``workers_decide_firms_to_apply``
     - Workers select firms to apply to
   * - ``workers_send_one_round``
     - Workers send one application (interleaved matching)
   * - ``firms_hire_workers``
     - Firms process applications and hire
   * - ``firms_calc_wage_bill``
     - Calculate total wage obligations

**Phase 3: Credit Market**

.. list-table::
   :widths: 40 60

   * - ``banks_decide_credit_supply``
     - Banks set lending capacity from equity
   * - ``banks_decide_interest_rate``
     - Banks set interest rates with cost shock
   * - ``firms_decide_credit_demand``
     - Firms calculate borrowing needs
   * - ``firms_calc_financial_fragility``
     - Calculate leverage ratio for credit evaluation
   * - ``firms_prepare_loan_applications``
     - Firms rank banks by interest rate
   * - ``firms_send_one_loan_app``
     - Firms send one loan application (interleaved matching)
   * - ``banks_provide_loans``
     - Banks process applications by fragility order
   * - ``firms_fire_workers``
     - Fire workers if credit insufficient

**Phase 4: Production**

.. list-table::
   :widths: 40 60

   * - ``firms_pay_wages``
     - Deduct wage bill from firm funds
   * - ``workers_receive_wage``
     - Workers receive wages as income
   * - ``firms_run_production``
     - Produce goods: output = productivity x labor
   * - ``update_avg_mkt_price``
     - Update economy-wide average price
   * - ``workers_update_contracts``
     - Decrement contract duration, handle expiration

**Phase 5: Goods Market**

.. list-table::
   :widths: 40 60

   * - ``consumers_calc_propensity``
     - Calculate consumption propensity from savings ratio
   * - ``consumers_decide_income_to_spend``
     - Allocate spending budget
   * - ``consumers_decide_firms_to_visit``
     - Select firms to visit (loyalty + random)
   * - ``consumers_shop_sequential``
     - Sequential shopping across selected firms
   * - ``consumers_finalize_purchases``
     - Save unspent budget

**Phase 6: Revenue**

.. list-table::
   :widths: 40 60

   * - ``firms_collect_revenue``
     - Collect sales revenue, calculate gross profit
   * - ``firms_validate_debt_commitments``
     - Repay loans, calculate net profit
   * - ``firms_pay_dividends``
     - Distribute dividends from positive profits

**Phase 7: Bankruptcy**

.. list-table::
   :widths: 40 60

   * - ``firms_update_net_worth``
     - Add retained profits to net worth
   * - ``mark_bankrupt_firms``
     - Detect and remove insolvent firms
   * - ``mark_bankrupt_banks``
     - Detect and remove insolvent banks

**Phase 8: Entry**

.. list-table::
   :widths: 40 60

   * - ``spawn_replacement_firms``
     - Create new firms to replace bankrupt ones
   * - ``spawn_replacement_banks``
     - Create new banks to replace bankrupt ones


Tips
----

- **Always use ``ops.assign()``** for role mutations — direct assignment
  (``role.field = value``) silently fails to update the shared array
- **Always use ``sim.rng``** for randomness — never ``numpy.random`` directly
- **Events are stateless**: Don't store state on ``self``. If you need
  state that persists across periods, use a role field.
- **Hook activation is explicit**: Call ``sim.use_events(MyEvent)`` after
  ``Simulation.init()`` — hooks declared in ``@event(after=...)`` are just
  metadata until activated


.. seealso::

   - :doc:`operations` for the ops module reference
   - :doc:`pipelines` for pipeline customization details
   - :doc:`custom_roles` for defining new agent state
   - :doc:`extensions` for complete extension examples
