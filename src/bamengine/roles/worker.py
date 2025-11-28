"""
Worker role for households.

Represents the employment and labor supply aspect of household behavior
in the BAM model. Each household supplies labor and earns wages.
"""

from bamengine.core.decorators import role
from bamengine.typing import Bool1D, Float1D, Idx1D, Idx2D, Int1D


@role
class Worker:
    """
    Worker role for households.

    Represents the employment and wage state for households. Each array index
    corresponds to a household ID (0 to n_households-1).

    Parameters
    ----------
    employer : Idx1D
        Current employer firm ID (-1 if unemployed).
    employer_prev : Idx1D
        Previous employer firm ID (for tracking job switches).
    wage : Float1D
        Current wage earned from employment.
    periods_left : Int1D
        Periods remaining in current employment contract.
    contract_expired : Bool1D
        True if current contract has expired.
    fired : Bool1D
        True if worker was fired this period.
    job_apps_head : Idx1D
        Queue head pointer for job applications.
    job_apps_targets : Idx2D
        Queue of firm IDs to apply to, shape (n_households, max_M).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_households=500, seed=42)
    >>> wrk = sim.wrk
    >>> wrk.wage.shape
    (500,)
    >>> wrk.employer.shape
    (500,)

    Check employment status:

    >>> import numpy as np
    >>> employed = wrk.employed
    >>> employed.sum()  # doctest: +SKIP
    480

    Find unemployed workers:

    >>> unemployed_mask = ~wrk.employed
    >>> unemployed_ids = np.where(unemployed_mask)[0]
    >>> unemployed_ids.shape  # doctest: +SKIP
    (20,)

    Calculate average wage of employed workers:

    >>> employed_wages = wrk.wage[wrk.employed]
    >>> employed_wages.mean()  # doctest: +SKIP
    1.15

    Check contract expiration:

    >>> expiring_soon = wrk.periods_left <= 1
    >>> expiring_soon.sum()  # doctest: +SKIP
    35

    Notes
    -----
    The Worker role is one of two roles assigned to households:

    - Worker: employment and labor supply
    - Consumer: consumption and savings (see Consumer)

    Employment status is computed via the `employed` property, which checks
    if employer >= 0. Unemployed workers have employer = -1.

    See Also
    --------
    :class:`~bamengine.roles.Consumer` : Consumption role for households
    :class:`~bamengine.roles.Employer` : Labor hiring role for firms
    :mod:`bamengine.events._internal.labor_market` : Labor market logic
    """

    employer: Idx1D
    employer_prev: Idx1D
    wage: Float1D
    periods_left: Int1D
    contract_expired: Bool1D
    fired: Bool1D

    # Scratch queues
    job_apps_head: Idx1D
    job_apps_targets: Idx2D

    @property
    def employed(self) -> Bool1D:
        """
        Compute employment status from employer ID.

        Returns True for workers with employer >= 0, False otherwise.

        Returns
        -------
        Bool1D
            Boolean array where True indicates employed, False indicates unemployed.

        Examples
        --------
        >>> import bamengine as bam
        >>> sim = bam.Simulation.init(n_households=500, seed=42)
        >>> wrk = sim.wrk
        >>> employed_count = wrk.employed.sum()
        >>> employed_count  # doctest: +SKIP
        480
        >>> unemployment_rate = 1.0 - (employed_count / 500)
        >>> unemployment_rate  # doctest: +SKIP
        0.04
        """
        return self.employer >= 0
