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

# --------------------------------------------------------------------- #
# semantic‑versioning string kept in one place
# --------------------------------------------------------------------- #
__version__: str = "0.0.0"

# --------------------------------------------------------------------- #
# re‑export the one *true* facade
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# core ECS components for extensibility
# --------------------------------------------------------------------- #
from .core import Agent, AgentType, Event, Role, event
from .simulation import Simulation  # noqa: E402  (circular‑safe)

# --------------------------------------------------------------------- #
# user-facing API: operations and types for custom roles/events
# --------------------------------------------------------------------- #
from . import ops
from .typing import Agent as AgentId
from .typing import Bool, Float, Int

__all__: list[str] = [
    "Simulation",
    "__version__",
    # Core ECS components
    "Agent",
    "AgentType",
    "Role",
    "Event",
    "event",
    # User-facing API
    "ops",
    "Float",
    "Int",
    "Bool",
    "AgentId",
]
