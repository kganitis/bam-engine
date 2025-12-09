Pipeline Customization
======================

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for pipeline customization patterns.

The Pipeline manages event execution order during simulation.

Pipeline Hooks (Recommended)
----------------------------

The simplest way to customize the pipeline is using **pipeline hooks**.
Declare where your custom event should be inserted directly in the decorator:

.. code-block:: python

   from bamengine import event, Simulation


   @event(after="firms_pay_dividends")
   class MyCustomEvent:
       """Automatically inserted after firms_pay_dividends."""

       def execute(self, sim: Simulation) -> None:
           # Your custom logic here
           pass


   # Events with hooks are automatically inserted when the pipeline is created
   sim = bam.Simulation.init(seed=42)
   # MyCustomEvent is already in the pipeline!

**Hook Types:**

- ``after="event_name"``: Insert immediately after the target event
- ``before="event_name"``: Insert immediately before the target event
- ``replace="event_name"``: Replace the target event entirely

**Key Points:**

- Hooks are applied automatically during ``Simulation.init()``
- Events must be imported before ``Simulation.init()`` for hooks to work
- Multiple events targeting the same point are inserted in registration order

Manual Pipeline Modification
----------------------------

You can also modify the pipeline manually after initialization:

.. code-block:: python

   import bamengine as bam

   sim = bam.Simulation.init(seed=42)

   # Insert events after a target
   sim.pipeline.insert_after("firms_adjust_price", "my_custom_event")

   # Insert events before a target
   sim.pipeline.insert_before("firms_adjust_price", "pre_pricing_check")

   # Remove an event
   sim.pipeline.remove("workers_send_one_round_0")

   # Replace an event
   sim.pipeline.replace("firms_decide_desired_production", "my_production_rule")

Custom Pipeline YAML
--------------------

You can also specify a completely custom pipeline via YAML:

.. code-block:: python

   sim = bam.Simulation.init(pipeline_path="custom_pipeline.yml", seed=42)

Topics to be covered:

* Default pipeline structure
* YAML pipeline syntax
* Repeat and interleave patterns
* Parameter substitution
