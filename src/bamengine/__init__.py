# src/bamengine/__init__.py
"""
BAM Engine – an agent‑based macro framework
==========================================

Public surface
--------------

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=10, n_households=50)
>>> sim.step()

Everything else (`bamengine.roles`, `bamengine.systems`, …) is **internal**
and may change without notice.
"""

from __future__ import annotations

__version__: str = "0.0.0"

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
from . import logging, ops


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
from .simulation import Simulation  # noqa: E402  (circular‑safe)

# ============================================================================
# ECS extensibility components
# ============================================================================
from .core import (
    Agent,
    AgentType,
    Event,
    Role,
    event,
    get_event,
    get_role,
    list_events,
    list_roles,
    role,
)

# ============================================================================
# Public API exports
# ============================================================================
__all__: list[str] = [
    # Core
    "Simulation",
    "__version__",
    # ECS components
    "Agent",
    "AgentType",
    "Role",
    "Event",
    "event",
    "role",
    "get_event",
    "get_role",
    "list_events",
    "list_roles",
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
