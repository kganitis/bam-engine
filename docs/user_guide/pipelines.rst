Pipeline Customization
======================

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for pipeline customization patterns.

The Pipeline manages event execution order during simulation.

Quick Example
-------------

.. code-block:: python

   import bamengine as bam

   # Use custom pipeline
   sim = bam.Simulation.init(
       pipeline_path="custom_pipeline.yml",
       seed=42
   )

   # Or modify the default pipeline
   sim.pipeline.insert_after("firms_adjust_price", "my_custom_event")
   sim.pipeline.remove("workers_send_one_round_0")

Topics to be covered:

* Default pipeline structure
* YAML pipeline syntax
* Repeat and interleave patterns
* Pipeline modification methods
* Parameter substitution
