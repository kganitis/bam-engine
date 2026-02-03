"""R&D extension for endogenous productivity growth.

This extension implements Section 3.8 of Delli Gatti et al. (2011),
adding endogenous productivity growth through R&D investment.

Usage:
    from extensions.rnd import RnD

    # Import BEFORE creating simulation
    sim = bam.Simulation.init(**config)
    sim.use_role(RnD)
    results = sim.run()

Components:
    - RnD: Role tracking R&D investment and productivity increments
    - FirmsComputeRnDIntensity: Event computing R&D share and intensity
    - FirmsApplyProductivityGrowth: Event applying productivity gains
    - FirmsDeductRnDExpenditure: Event adjusting retained profits
"""

from __future__ import annotations

from extensions.rnd.events import (
    FirmsApplyProductivityGrowth,
    FirmsComputeRnDIntensity,
    FirmsDeductRnDExpenditure,
)
from extensions.rnd.role import RnD

__all__ = [
    "RnD",
    "FirmsComputeRnDIntensity",
    "FirmsApplyProductivityGrowth",
    "FirmsDeductRnDExpenditure",
]
