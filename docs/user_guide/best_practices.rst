Best Practices
==============

.. note::

   This section is under construction.

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

- Use vectorized operations via the ``ops`` module
- Avoid Python loops over agents when possible
- Pre-allocate arrays for scratch buffers in custom roles
- Use appropriate agent counts for your analysis needs

Common Pitfalls
---------------

- Forgetting to set a seed (non-reproducible results)
- Modifying role arrays without using ``ops.assign()``
- Creating too many agents for available memory

Topics to be covered:

* Reproducibility best practices
* Performance optimization
* Memory management
* Debugging simulations
* Validation and testing custom components
* Common mistakes and how to avoid them
