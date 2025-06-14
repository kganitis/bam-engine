# src/bamengine/__init__.py
"""
BAM Engine – an agent‑based macro framework
==========================================

Public surface
--------------

>>> import bamengine as be
>>> sched = be.Scheduler.init(n_firms=10, n_households=50)
>>> sched.step()

Everything else (`bamengine.components`, `bamengine.systems`, …) is **internal**
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
from .scheduler import Scheduler  # noqa: E402  (circular‑safe)

__all__: list[str] = [
    "Scheduler",
    "__version__",
]
