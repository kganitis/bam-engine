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
