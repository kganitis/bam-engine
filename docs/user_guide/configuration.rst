Configuration
=============

.. note::

   This section is under construction. See the :doc:`examples </auto_examples/index>`
   for configuration usage patterns.

BAM Engine uses a three-tier configuration system:

1. **Package defaults** (``src/bamengine/config/defaults.yml``)
2. **User YAML file** (custom configuration)
3. **Keyword arguments** (highest priority)

Quick Example
-------------

.. code-block:: python

   import bamengine as bam

   # Use defaults
   sim = bam.Simulation.init()

   # Override with kwargs
   sim = bam.Simulation.init(n_firms=200, seed=42)

   # Override with YAML
   sim = bam.Simulation.init(config="my_config.yml")

Topics to be covered:

* Configuration parameters reference
* YAML configuration files
* Configuration validation
* Logging configuration

Extension Parameters
--------------------

BAM Engine supports passing custom parameters to simulations for use by
extension events. Any keyword argument passed to ``Simulation.init()`` that
is not a core configuration parameter will be stored in ``extra_params``
and accessible as an attribute on the simulation object.

.. code-block:: python

    import bamengine as bam

    # Pass custom parameters for a model extension
    sim = bam.Simulation.init(
        n_firms=100,
        n_households=500,
        seed=42,
        # Custom extension parameters
        sigma_min=0.0,
        sigma_max=0.1,
        sigma_decay=-1.0,
    )

    # Access directly in custom events
    sigma_min = sim.sigma_min
    sigma_max = sim.sigma_max
    sigma_decay = sim.sigma_decay

    # Also available via extra_params dict
    print(sim.extra_params)  # {'sigma_min': 0.0, 'sigma_max': 0.1, 'sigma_decay': -1.0}

This feature enables:

* **Model extensions**: Pass parameters for custom events without modifying core code
* **Parameter sweeps**: Easily vary extension parameters across simulation runs
* **Clean syntax**: Access parameters directly as ``sim.param_name``

See the :doc:`Growth+ example </auto_examples/extensions/example_growth_plus>` for a
complete demonstration of extension parameters in action.
