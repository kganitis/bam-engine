"""
Consumer role for households.

Represents the consumption and savings aspect of household behavior
in the BAM model. Each household receives income and purchases consumer goods.
"""

from bamengine.core.decorators import role
from bamengine.typing import Float1D, Idx1D, Idx2D


@role
class Consumer:
    """
    Consumer role for households.

    Represents the consumption and savings state for households. Each array index
    corresponds to a household ID (0 to n_households-1).

    Parameters
    ----------
    income : Float1D
        Total income received this period from wages.
    savings : Float1D
        Accumulated savings from previous periods.
    income_to_spend : Float1D
        Portion of income allocated for consumption this period.
    propensity : Float1D
        Propensity to consume (calculated from income and savings).
    largest_prod_prev : Idx1D
        ID of firm with highest production in previous period (search hint).
    shop_visits_head : Idx1D
        Queue head pointer for shop visits.
    shop_visits_targets : Idx2D
        Queue of firm IDs to visit, shape (n_households, max_Z).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_households=500, seed=42)
    >>> con = sim.con
    >>> con.income.shape
    (500,)
    >>> con.savings.shape
    (500,)

    Calculate aggregate consumption spending:

    >>> import numpy as np
    >>> total_spending = con.income_to_spend.sum()
    >>> total_spending  # doctest: +SKIP
    525.5

    Find households with high savings:

    >>> high_savers = con.savings > 100
    >>> high_savers.sum()  # doctest: +SKIP
    45

    Check propensity to consume distribution:

    >>> mean_propensity = con.propensity.mean()
    >>> mean_propensity  # doctest: +SKIP
    0.75

    Find households targeting specific firm:

    >>> target_firm = 5
    >>> targeting = con.largest_prod_prev == target_firm
    >>> targeting.sum()  # doctest: +SKIP
    12

    Notes
    -----
    The Consumer role is one of two roles assigned to households:

    - Worker: employment and labor supply (see Worker)
    - Consumer: consumption and savings

    The propensity to consume is calculated using the formula:
    propensity = income^beta / (income^beta + savings^beta)
    where beta is a configuration parameter (typically beta=2.5).

    See Also
    --------
    :class:`~bamengine.roles.Worker` : Employment role for households
    :class:`~bamengine.roles.Producer` : Production role for firms
    :mod:`bamengine.events._internal.goods_market` : Goods market logic
    """

    income: Float1D
    savings: Float1D
    income_to_spend: Float1D
    propensity: Float1D
    largest_prod_prev: Idx1D

    # Scratch queues
    shop_visits_head: Idx1D
    shop_visits_targets: Idx2D  # shape (n_households, Z)
