# src/_testing/__init__.py
"""
Private helpers used *only* by the test-suite.
They should disappear once all real events are implemented.
"""

from typing import TYPE_CHECKING

# import numpy as np

if TYPE_CHECKING:
    from bamengine.scheduler import Scheduler


def advance_stub_state(sched: "Scheduler") -> None:
    """
    One-shot placeholder that nudges the state forward so multi-period
    tests have fresh, non-degenerate arrays.
    """
    pass
