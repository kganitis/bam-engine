"""
Configuration system for BAM Engine.

This package provides a three-tier configuration system with centralized
validation:

1. **Package defaults** (`src/bamengine/config/defaults.yml`)
2. **User config file** (YAML path or dict)
3. **Keyword arguments** (highest priority)

The Config dataclass groups all simulation hyperparameters in one immutable
object. The ConfigValidator performs all validation once at Simulation.init()
to ensure fail-fast behavior with clear error messages.

Components
----------
Config : dataclass
    Immutable configuration for simulation parameters.
ConfigValidator : class
    Centralized validation for all configuration parameters.

Examples
--------
Use defaults:

>>> import bamengine as be
>>> sim = be.Simulation.init()

Override with YAML:

>>> sim = be.Simulation.init(config="my_config.yml")

Override with kwargs (highest priority):

>>> sim = be.Simulation.init(n_firms=200, seed=42)

Mix YAML and kwargs:

>>> sim = be.Simulation.init(config="base.yml", n_firms=200)

Custom pipeline:

>>> sim = be.Simulation.init(n_firms=100, pipeline_path="custom_pipeline.yml", seed=42)

Custom logging:

>>> log_config = {
...     "default_level": "DEBUG",
...     "events": {
...         "workers_send_one_round": "WARNING",
...         "firms_hire_workers": "INFO",
...     },
... }
>>> sim = be.Simulation.init(logging=log_config)

Validation
----------
All configuration parameters are validated at Simulation.init():

- **Type checking**: Ensures correct types (int, float, str, etc.)
- **Range validation**: Ensures parameters within valid ranges
- **Relationship constraints**: Validates cross-parameter dependencies
- **Pipeline validation**: Validates custom pipeline YAML files
- **Logging validation**: Validates log levels and event names

Invalid configurations are rejected immediately with clear error messages:

>>> sim = be.Simulation.init(n_firms="100")  # doctest: +SKIP
ValueError: Config parameter 'n_firms' must be int, got str

>>> sim = be.Simulation.init(h_rho=1.5)  # doctest: +SKIP
ValueError: Config parameter 'h_rho' must be <= 1.0, got 1.5

See Also
--------
:class:`~bamengine.config.Config` : Immutable configuration dataclass
:class:`~bamengine.config.ConfigValidator` : Centralized validation class
:meth:`bamengine.Simulation.init` : Initialize simulation with configuration
"""

from bamengine.config.schema import Config
from bamengine.config.validator import ConfigValidator

__all__ = ["Config", "ConfigValidator"]
