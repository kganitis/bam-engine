Best Practices
==============

This guide covers recommended practices for working with BAM Engine effectively.


Reproducibility
---------------

Always set a seed for reproducible results:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)

   # Results will be identical across runs with the same seed
   results1 = sim.run(n_periods=100)

   sim2 = bam.Simulation.init(seed=42)
   results2 = sim2.run(n_periods=100)
   # results1 and results2 are identical

All randomness flows through ``sim.rng``. Custom events must use this generator
instead of ``numpy.random`` or Python's ``random`` module.


Performance Tips
----------------

Logging Level
~~~~~~~~~~~~~

For production runs, disable verbose logging to improve performance:

.. code-block:: python

   sim = bam.Simulation.init(logging={"default_level": "ERROR"}, seed=42)

This can provide 10-20% speedup for long simulations by avoiding log message
formatting overhead.

Agent Count Selection
~~~~~~~~~~~~~~~~~~~~~

Choose agent counts based on your research phase:

* **Exploratory phase** (100 firms): Fast feedback loop for testing ideas
* **Parameter tuning** (200 firms): Balance of speed and statistical accuracy
* **Final runs** (500+ firms): Publication-quality results

The model's emergent properties are stable across agent counts, so smaller
configurations are valid for development. See :ref:`choosing-agent-counts`
for detailed guidance.

Vectorized Operations
~~~~~~~~~~~~~~~~~~~~~

When writing custom events, use the ``ops`` module for array operations:

.. code-block:: python

   from bamengine import ops

   # Good: vectorized operation
   ops.assign(role.price, ops.multiply(role.price, 1.1))

   # Bad: Python loop (much slower)
   for i in range(len(role.price)):
       role.price[i] *= 1.1

Batch Simulations
~~~~~~~~~~~~~~~~~

For parameter sweeps, run simulations in separate processes to utilize
multiple CPU cores:

.. code-block:: python

   from multiprocessing import Pool
   import bamengine as bam


   def run_simulation(seed):
       sim = bam.Simulation.init(seed=seed)
       return sim.run(n_periods=100)


   with Pool(4) as pool:
       results = pool.map(run_simulation, range(10))

Memory Management
~~~~~~~~~~~~~~~~~

For very long simulations or parameter sweeps, consider periodic data extraction
to manage memory:

.. code-block:: python

   for batch in range(10):
       sim = bam.Simulation.init(seed=batch)
       results = sim.run(n_periods=100)
       results.to_dataframe().to_parquet(f"results_{batch}.parquet")
       # Results object is garbage collected after each iteration


Common Pitfalls
---------------

Reproducibility Issues
~~~~~~~~~~~~~~~~~~~~~~

* **Forgetting to set a seed**: Always use ``seed=`` for reproducible results
* **Different seeds across runs**: Use the same seed when comparing configurations

Array Modification Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Direct array modification**: Always use ``ops.assign()`` for in-place updates
* **Creating new arrays in loops**: Pre-allocate scratch buffers in custom roles

Resource Issues
~~~~~~~~~~~~~~~

* **Too many agents**: Start small and scale up as needed
* **Long simulations without checkpoints**: Save intermediate results for recovery


Extension Best Practices
------------------------

Activation Order
~~~~~~~~~~~~~~~~

Always activate extensions in the correct order — role, events, config:

.. code-block:: python

   # Correct order
   sim.use_role(RnD)
   sim.use_events(*RND_EVENTS)
   sim.use_config(RND_CONFIG)

   # Missing use_events() is the #1 extension mistake!

Explicit Hook Activation
~~~~~~~~~~~~~~~~~~~~~~~~

The ``@event(after="...")`` decorator stores hook metadata but does **not**
modify the pipeline. You must call ``sim.use_events()`` to apply hooks:

.. code-block:: python

   # This does nothing by itself:
   @event(after="firms_pay_dividends")
   class MyEvent:
       def execute(self, sim): ...


   # You MUST activate it:
   sim.use_events(MyEvent)

Testing Extensions in Isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test custom events independently before integrating into a full simulation:

.. code-block:: python

   # Quick smoke test for an extension
   sim = bam.Simulation.init(seed=42)
   sim.use_role(MyRole)
   sim.use_events(MyEvent)

   sim.run(n_periods=10)  # Short run to verify no errors

   role = sim.get_role("MyRole")
   assert role.my_field.sum() > 0, "Extension had no effect"


Debugging Tips
--------------

TRACE Logging
~~~~~~~~~~~~~

Use ``TRACE`` level logging to see exactly what happens inside events:

.. code-block:: python

   sim = bam.Simulation.init(
       logging={
           "default_level": "WARNING",
           "events": {
               "firms_hire_workers": "TRACE",  # Verbose for this event only
           },
       },
       seed=42,
   )

Single-Step Inspection
~~~~~~~~~~~~~~~~~~~~~~

Step through the simulation one period at a time to inspect intermediate state:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)

   for t in range(10):
       sim.step()
       print(f"Period {t}:")
       print(f"  Avg price:    {sim.prod.price.mean():.4f}")
       print(f"  Unemployment: {(~sim.wrk.employed).mean():.2%}")
       print(f"  Total loans:  {sim.lb.size}")
       print(f"  Bankruptcies: {len(sim.ec.exiting_firms)}")

Checking Pipeline Order
~~~~~~~~~~~~~~~~~~~~~~~

Verify that custom events are correctly placed in the pipeline:

.. code-block:: python

   sim = bam.Simulation.init(seed=42)
   sim.use_events(MyCustomEvent)

   # Print full pipeline
   for i, entry in enumerate(sim.pipeline):
       print(f"  {i:3d}: {entry.name}")

Inspecting Role State
~~~~~~~~~~~~~~~~~~~~~

Check agent state at any point using NumPy array operations:

.. code-block:: python

   # Distribution summary
   import numpy as np

   print(
       f"Price: mean={sim.prod.price.mean():.3f}, "
       f"std={sim.prod.price.std():.3f}, "
       f"min={sim.prod.price.min():.3f}, "
       f"max={sim.prod.price.max():.3f}"
   )

   # Find extreme agents
   richest = np.argmax(sim.bor.net_worth)
   print(f"Richest firm #{richest}: NW={sim.bor.net_worth[richest]:.2f}")


Project Organization
--------------------

For research projects using BAM Engine, a suggested directory structure:

::

   my_research/
   ├── configs/           # YAML configuration files
   │   ├── baseline.yml
   │   └── experiment_1.yml
   ├── extensions/        # Custom roles and events
   │   ├── __init__.py
   │   └── my_extension.py
   ├── analysis/          # Analysis and plotting scripts
   │   ├── run_experiment.py
   │   └── plot_results.py
   ├── results/           # Output data (gitignored)
   │   ├── baseline/
   │   └── experiment_1/
   └── README.md

Keep configuration files separate from analysis scripts, and save results
to a dedicated output directory that can be regenerated from the configs.


.. seealso::

   - :doc:`running_simulations` for initialization and execution patterns
   - :doc:`operations` for the ops module reference
   - :doc:`extensions` for built-in extension usage
   - :doc:`configuration` for all configuration options
