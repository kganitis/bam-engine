"""R&D extension for endogenous productivity growth.

This extension implements Section 3.8 of Delli Gatti et al. (2011),
adding endogenous productivity growth through R&D investment.

Usage::

    from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG

    sim = bam.Simulation.init(**config)
    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    sim.use_config(RND_CONFIG)
    results = sim.run()

Components:
    - RnD: Role tracking R&D investment and productivity increments
    - FirmsComputeRnDIntensity: Event computing R&D share and intensity
    - FirmsApplyProductivityGrowth: Event applying productivity gains
    - FirmsDeductRnDExpenditure: Event adjusting retained profits
    - RND_EVENTS: List of all R&D event classes for use with ``sim.use_events()``
    - RND_CONFIG: Default R&D parameters for use with ``sim.use_config()``
"""

from __future__ import annotations

from extensions.rnd.events import (
    FirmsApplyProductivityGrowth,
    FirmsComputeRnDIntensity,
    FirmsDeductRnDExpenditure,
)
from extensions.rnd.role import RnD

RND_EVENTS = [
    FirmsComputeRnDIntensity,
    FirmsApplyProductivityGrowth,
    FirmsDeductRnDExpenditure,
]

RND_CONFIG = {
    "sigma_min": 0.0,
    "sigma_max": 0.1,
    "sigma_decay": -1.0,
}

__all__ = [
    "RnD",
    "RND_EVENTS",
    "RND_CONFIG",
    "FirmsComputeRnDIntensity",
    "FirmsApplyProductivityGrowth",
    "FirmsDeductRnDExpenditure",
]
