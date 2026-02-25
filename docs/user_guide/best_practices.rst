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
