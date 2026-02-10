"""
Shareholder role for households.

Tracks per-period dividend income received by each household from firm
profit distribution. This role serves as a foundation for a future
Capitalist extension with explicit firm ownership relationships.
"""

from bamengine.core.decorators import role
from bamengine.typing import Float1D


@role
class Shareholder:
    """
    Shareholder role for households.

    Tracks per-period dividend income for each household. The ``dividends``
    array is overwritten each period by ``firms_pay_dividends`` with the
    current period's dividend amount (total dividends / n_households).

    Parameters
    ----------
    dividends : Float1D
        Per-period dividend received by each household (overwritten each
        period, not cumulative).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_households=500, seed=42)
    >>> sh = sim.sh
    >>> sh.dividends.shape
    (500,)

    Notes
    -----
    The Shareholder role is one of three roles assigned to households:

    - Worker: employment and labor supply (see Worker)
    - Consumer: consumption and savings
    - Shareholder: per-period dividend income tracking

    Currently dividends are distributed equally to all households. A future
    Capitalist extension will introduce firm ownership relationships to
    allow dividends to flow to specific households (shareholders).

    See Also
    --------
    :class:`~bamengine.roles.Consumer` : Consumption role for households
    :class:`~bamengine.roles.Worker` : Employment role for households
    :class:`~bamengine.events.revenue.FirmsPayDividends` : Dividend distribution event
    """

    dividends: Float1D
