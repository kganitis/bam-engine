"""R&D role definition for Growth+ extension.

This module defines the RnD role that tracks R&D investment decisions
and productivity increments for firms.
"""

from __future__ import annotations

from bamengine import Float, role


@role
class RnD:
    """R&D state for Growth+ extension.

    Tracks R&D investment decisions and productivity increments for firms.

    Parameters
    ----------
    sigma : Float
        R&D share of profits (0.0 to 0.1). Higher values mean more
        investment in R&D. Decreases with financial fragility.
    rnd_intensity : Float
        Expected productivity gain (mu). Scale parameter for the
        exponential distribution from which actual gains are drawn.
    productivity_increment : Float
        Actual productivity increment (z) drawn each period.
        Added to labor_productivity.
    fragility : Float
        Financial fragility metric (W/A = wage_bill / net_worth).
        High fragility leads to lower R&D investment.
    """

    sigma: Float
    rnd_intensity: Float
    productivity_increment: Float
    fragility: Float
