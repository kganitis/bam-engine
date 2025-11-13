"""
BAM Engine - Bottom-Up Adaptive Macroeconomics Simulation Framework
====================================================================

BAM Engine is a Python implementation of the BAM (Bottom-Up Adaptive
Macroeconomics) model from Delli Gatti et al. (2008). It provides a
high-performance, vectorized agent-based macroeconomic simulation framework
for studying complex economic dynamics.

Quick Start
-----------
Basic simulation with default configuration:

>>> import bamengine as be
>>> sim = be.Simulation.init(seed=42)
>>> sim.run(n_periods=100)
>>> unemployment = sim.ec.unemp_rate_history[-1]
>>> print(f"Final unemployment: {unemployment:.2%}")

Custom configuration via kwargs:

>>> sim = be.Simulation.init(
...     n_firms=200,
...     n_households=1000,
...     n_banks=20,
...     seed=42
... )
>>> sim.run(n_periods=100)

Custom configuration via YAML file:

>>> sim = be.Simulation.init(config="my_config.yml", seed=42)
>>> sim.run(n_periods=100)

Key Concepts
------------
**Agents and Roles**
  Agents have multiple roles (components). Firms are Producer + Employer + Borrower.
  Households are Worker + Consumer. Banks are Lender.

**Event Pipeline**
  Each period executes 40+ events in fixed order: Planning → Labor Market →
  Credit Market → Production → Goods Market → Revenue → Bankruptcy → Entry.

**Vectorized Operations**
  All agent state stored in NumPy arrays for performance. Population-level
  operations execute in parallel.

**Deterministic RNG**
  Fixed seed ensures reproducible simulations for scientific research.

Public API
----------
**Core Classes**

Simulation
    Main simulation facade for running BAM simulations.
Role
    Base class for defining custom agent components.
Event
    Base class for defining custom economic events.
Relationship
    Base class for defining agent-to-agent relationships.
Economy
    Container for economy-wide state (prices, wages, unemployment).

**Decorators**

role
    Decorator for defining custom role classes (simplified syntax).
event
    Decorator for defining custom event classes (simplified syntax).
relationship
    Decorator for defining custom relationship classes.

**Type Aliases**

Float, Int, Bool, AgentId
    User-friendly type aliases for defining custom roles without NumPy knowledge.
Rng
    Type alias for numpy.random.Generator (random number generator).

**Utility Modules**

ops
    NumPy-free operations for writing custom events (add, multiply, divide, etc.).
logging
    Custom logging with TRACE level and per-event log configuration.

**Registry Functions**

get_role, get_event, get_relationship
    Retrieve registered roles/events/relationships by name.
list_roles, list_events, list_relationships
    List all registered roles/events/relationships.

Examples
--------
Access time-series data after simulation:

>>> sim = be.Simulation.init(seed=42)
>>> sim.run(n_periods=100)
>>> import matplotlib.pyplot as plt
>>> plt.plot(sim.ec.inflation_history)
>>> plt.title("Inflation Over Time")

Define a custom role:

>>> @be.role
... class Inventory:
...     goods_on_hand: be.Float
...     reorder_point: be.Float
...     supplier_id: be.AgentId

Define a custom event:

>>> @be.event
... class CustomPricing:
...     def execute(self, sim):
...         prod = sim.get_role("Producer")
...         be.ops.multiply(prod.price, 1.1, out=prod.price)

Step through simulation manually:

>>> sim = be.Simulation.init(seed=42)
>>> for period in range(10):
...     sim.step()
...     if period % 5 == 0:
...         unemp = sim.ec.unemp_rate_history[-1]
...         print(f"Period {period}: Unemployment = {unemp:.2%}")

Module Organization
-------------------
**Public API** (stable, documented, recommended for users):
  - `bamengine.Simulation` : Main simulation class
  - `bamengine.Role`, `bamengine.Event`, `bamengine.Relationship` : Base classes
  - `bamengine.role`, `bamengine.event`, `bamengine.relationship` : Decorators
  - `bamengine.ops` : NumPy-free operations
  - `bamengine.logging` : Custom logging
  - `bamengine.Float`, `bamengine.Int`, `bamengine.Bool`,
    `bamengine.AgentId` : Type aliases

**Internal Modules** (implementation details, subject to change):
  - `bamengine.roles` : Built-in role implementations (Producer, Worker, etc.)
  - `bamengine.events` : Built-in event implementations (37 events)
  - `bamengine.relationships` : Built-in relationships (LoanBook)
  - `bamengine.config` : Configuration and validation
  - `bamengine.core` : ECS infrastructure (registry, pipeline)
  - `bamengine.utils` : Internal utilities

References
----------
Delli Gatti, D., Desiderio, S., Gaffeo, E., Cirillo, P., & Gallegati, M. (2011).
    Macroeconomics from the Bottom-up (Vol. 1). Springer Science & Business Media.

See Also
--------
bamengine.simulation : Main simulation facade
bamengine.ops : User-friendly operations
defaults.yml : Default configuration parameters
default_pipeline.yml : Default event execution order

Notes
-----
- Time scale: 1 period = 1 quarter (4 periods = 1 year)
- All simulations are deterministic when seed is specified
- Configuration precedence: defaults.yml → user config → kwargs
- Pipeline events execute in explicit order (no automatic dependency resolution)

Version
-------
Current version: 0.0.0 (development)
"""

from __future__ import annotations

__version__: str = "0.1.0"

# ============================================================================
# Standard library imports
# ============================================================================
from typing import TypeAlias

import numpy as np

# ============================================================================
# Type system for user extensions (must be before Simulation import)
# ============================================================================
from .typing import Agent as AgentId
from .typing import Bool, Float, Int

# Type alias for RNG (must be before Simulation import)
Rng: TypeAlias = np.random.Generator

# ============================================================================
# User-facing utilities (must be before Simulation import)
# ============================================================================
from . import logging, ops  # noqa: E402 (circular‑safe)


def make_rng(seed: int | None = None) -> Rng:
    """Create a new random number generator.

    This is the recommended way to create RNGs for use with BAM Engine.
    Under the hood, this uses NumPy's `default_rng`, which provides the
    modern Generator API with better statistical properties than the
    legacy RandomState.

    Parameters
    ----------
    seed : int | None
        Seed for reproducibility. If `None`, uses a random seed.

    Returns
    -------
    Rng
        A NumPy random number generator (np.random.Generator).

    Examples
    --------
    >>> import bamengine as be
    >>> rng = be.make_rng(42)  # Reproducible
    >>> rng.normal(0, 1, size=10)  # Use standard NumPy methods
    >>> rng2 = be.make_rng()  # Random seed

    See Also
    --------
    numpy.random.default_rng : The underlying NumPy function
    """
    return np.random.default_rng(seed)


# ============================================================================
# Core simulation facade (imports after dependencies)
# ============================================================================
# ============================================================================
# ECS extensibility components
# ============================================================================
from .core import (  # noqa: E402 (circular‑safe)
    Agent,
    AgentType,
    Event,
    Relationship,
    Role,
    event,
    get_event,
    get_relationship,
    get_role,
    list_events,
    list_relationships,
    list_roles,
    relationship,
    role,
)
from .economy import Economy  # noqa: E402 (circular‑safe)
from .simulation import Simulation  # noqa: E402  (circular‑safe)

# ============================================================================
# Public API exports
# ============================================================================
__all__ = [
    "Simulation",
    "__version__",
    # Core ECS components
    "Agent",
    "AgentType",
    "Role",
    "Economy",
    "Event",
    "Relationship",
    "event",
    "role",
    "get_event",
    "get_role",
    "get_relationship",
    "list_events",
    "list_roles",
    "list_relationships",
    "relationship",
    # Type system
    "Float",
    "Int",
    "Bool",
    "AgentId",
    "Rng",
    # Utilities
    "make_rng",
    "ops",
    "logging",
]
