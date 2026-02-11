"""Buffer-stock consumption role definition.

This module defines the BufferStock role that tracks previous-period income
and buffer-stock propensity for each household.

See Section 3.9.4 of Delli Gatti et al. (2011).
"""

from __future__ import annotations

from bamengine import Float, role


@role
class BufferStock:
    """Buffer-stock consumption state for households.

    Tracks previous-period income and buffer-stock propensity for each
    household. Used by the buffer-stock consumption extension to compute
    individual MPC based on adaptive savings rules.

    Parameters
    ----------
    prev_income : Float
        Previous period income (W_{t-1}). Used to compute income growth
        rate for the buffer-stock MPC formula.
    propensity : Float
        Buffer-stock MPC (c_t). May exceed 1.0 when households dissave.
        Separate from ``Consumer.propensity`` to avoid conflicts when
        the extension is not active.
    """

    prev_income: Float
    propensity: Float
