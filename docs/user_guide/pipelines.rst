Pipeline Customization
======================

The pipeline manages event execution order during each simulation period.
BAM Engine uses **explicit ordering** — events run in the exact order listed,
with no automatic dependency resolution. This matches the original BAM model
specification and makes the execution flow predictable and debuggable.


The Default Pipeline
--------------------

The default pipeline executes 8 phases per period. Within the labor, credit,
and goods market phases, matching events are **interleaved** — seekers and
providers alternate turns for multiple rounds.

**Phase 1: Planning** (6 events)

::

   firms_decide_desired_production
   firms_plan_breakeven_price
   firms_plan_price
   firms_decide_desired_labor
   firms_decide_vacancies
   firms_fire_excess_workers

**Phase 2: Labor Market** (6 events, with interleaved matching)

::

   calc_inflation_rate
   adjust_minimum_wage
   firms_decide_wage_offer
   workers_decide_firms_to_apply
   workers_send_one_round <-> firms_hire_workers  x max_M
   firms_calc_wage_bill

The ``workers_send_one_round <-> firms_hire_workers x max_M`` expands to
``max_M`` alternating rounds: send, hire, send, hire, ... (default: 4 pairs =
8 events).

**Phase 3: Credit Market** (7 events, with interleaved matching)

::

   banks_decide_credit_supply
   banks_decide_interest_rate
   firms_decide_credit_demand
   firms_calc_financial_fragility
   firms_prepare_loan_applications
   firms_send_one_loan_app <-> banks_provide_loans  x max_H
   firms_fire_workers

**Phase 4: Production** (5 events)

::

   firms_pay_wages
   workers_receive_wage
   firms_run_production
   update_avg_mkt_price
   workers_update_contracts

**Phase 5: Goods Market** (5 events)

::

   consumers_calc_propensity
   consumers_decide_income_to_spend
   consumers_decide_firms_to_visit
   consumers_shop_sequential
   consumers_finalize_purchases

**Phase 6: Revenue** (3 events)

::

   firms_collect_revenue
   firms_validate_debt_commitments
   firms_pay_dividends

**Phase 7: Bankruptcy** (3 events)

::

   firms_update_net_worth
   mark_bankrupt_firms
   mark_bankrupt_banks

**Phase 8: Entry** (2 events)

::

   spawn_replacement_firms
   spawn_replacement_banks


Pipeline Hooks (Recommended)
----------------------------

The simplest way to customize the pipeline is using **pipeline hooks**.
Declare where your custom event should be inserted directly in the decorator:

.. code-block:: python

   from bamengine import event, Simulation
   import bamengine as bam


   @event(after="firms_pay_dividends")
   class MyCustomEvent:
       """Positioned after firms_pay_dividends via hook metadata."""

       def execute(self, sim: Simulation) -> None:
           pass


   # Hooks are NOT auto-applied; use sim.use_events() to activate them
   sim = bam.Simulation.init(seed=42)
   sim.use_events(MyCustomEvent)

**Hook types:**

- ``after="event_name"`` — Insert immediately after the target event
- ``before="event_name"`` — Insert immediately before the target event
- ``replace="event_name"`` — Replace the target event entirely

**Key points:**

- Hooks are stored as class attributes, NOT applied automatically
- Call ``sim.use_events(*event_classes)`` to apply hooks to the pipeline
- Multiple events targeting the same point are inserted in the order passed
  to ``use_events()``
- No import ordering constraints — import extensions anywhere before calling
  ``use_events()``


Manual Pipeline Modification
----------------------------

For more control, modify the pipeline directly after initialization:

.. code-block:: python

   import bamengine as bam

   sim = bam.Simulation.init(seed=42)

   # Insert events at specific positions
   sim.pipeline.insert_after("firms_adjust_price", "my_custom_event")
   sim.pipeline.insert_before("firms_adjust_price", "pre_pricing_check")

   # Remove an event
   sim.pipeline.remove("workers_send_one_round_0")

   # Replace an event
   sim.pipeline.replace("firms_decide_desired_production", "my_production_rule")

All pipeline modifications should happen **before** calling ``sim.run()`` or
``sim.step()``.


Custom Pipeline YAML
--------------------

For full control over the event sequence, create a custom pipeline YAML file:

.. code-block:: python

   sim = bam.Simulation.init(pipeline_path="custom_pipeline.yml", seed=42)

**YAML syntax:**

.. code-block:: yaml

   events:
     # Simple event — execute once per period
     - firms_decide_desired_production

     # Repeated event — execute N times
     - consumers_shop_one_round x 4

     # Interleaved events — alternate between two events, N rounds
     - workers_send_one_round <-> firms_hire_workers x 6

     # Parameter substitution — use config values
     - workers_send_one_round <-> firms_hire_workers x {max_M}
     - firms_send_one_loan_app <-> banks_provide_loans x {max_H}
     - consumers_shop_one_round x {max_Z}

**Parameter substitution** replaces ``{param_name}`` with the corresponding
configuration value at pipeline load time. Available substitutions:

- ``{max_M}`` — Number of labor market matching rounds
- ``{max_H}`` — Number of credit market matching rounds
- ``{max_Z}`` — Number of goods market shopping rounds


Planning-Phase Pricing (Alternative)
-------------------------------------

BAM Engine provides an alternative pair of pricing events that run during the
planning phase instead of the production phase:

- ``firms_plan_breakeven_price`` — Breakeven from previous period's costs / desired production
- ``firms_plan_price`` — Price adjustment with breakeven floor

These are **mutually exclusive** with the production-phase pair
(``firms_calc_breakeven_price``, ``firms_adjust_price``). The default pipeline
uses planning-phase pricing. To switch to production-phase pricing, create a
custom pipeline that moves the pricing events:

.. code-block:: yaml

   events:
     # Planning — no pricing events
     - firms_decide_desired_production
     - firms_decide_desired_labor
     - firms_decide_vacancies
     - firms_fire_excess_workers

     # ... labor and credit market phases ...

     # Production — pricing moved here
     - firms_pay_wages
     - workers_receive_wage
     - firms_calc_breakeven_price
     - firms_adjust_price
     - firms_run_production
     - update_avg_mkt_price
     - workers_update_contracts


Inspecting the Pipeline
-----------------------

View the current pipeline to verify event ordering:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)

   # List all events in execution order
   for entry in sim.pipeline:
       print(entry.name)

   # Check total event count
   print(f"Pipeline has {len(sim.pipeline)} events")

This is especially useful after applying hooks to verify insertion points.


Tips
----

- **Explicit ordering is intentional**: BAM Engine deliberately avoids
  topological sorting. The execution order is a core part of the model
  specification.
- **Hook activation is explicit**: Always call ``sim.use_events()`` after
  init — declaring ``@event(after=...)`` alone does nothing.
- **Modify before running**: Pipeline changes should happen between
  ``Simulation.init()`` and ``sim.run()``, not during execution.
- **Interleaved matching is hardcoded**: The default pipeline always uses
  interleaved matching for labor and credit markets, regardless of deprecated
  config parameters.


.. seealso::

   - :doc:`custom_events` for creating events to hook into the pipeline
   - :doc:`bam_model` for the economic logic of each phase
   - :doc:`extensions` for how built-in extensions modify the pipeline
