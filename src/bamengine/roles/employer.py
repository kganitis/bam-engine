"""
Employer role for firms.

Represents the labor hiring and wage management aspect of firm behavior
in the BAM model. Each firm hires workers and pays wages.
"""

from dataclasses import field

from bamengine.core.decorators import role
from bamengine.typing import Float1D, Idx1D, Idx2D, Int1D


@role
class Employer:
    """
    Employer role for firms.

    Represents the labor hiring and wage state for firms. Each array index
    corresponds to a firm ID (0 to n_firms-1).

    Parameters
    ----------
    desired_labor : Int1D
        Target number of workers needed (based on production plans).
    current_labor : Int1D
        Current number of employed workers.
    wage_offer : Float1D
        Wage offered to potential workers.
    wage_bill : Float1D
        Total wages paid this period (shared with Borrower role).
    n_vacancies : Int1D
        Number of open positions (desired_labor - current_labor).
    total_funds : Float1D
        Available funds for hiring (shared view with Borrower.total_funds).
    recv_job_apps_head : Idx1D
        Queue head pointer for received job applications.
    recv_job_apps : Idx2D
        Queue of job application IDs, shape (n_firms, n_households).
    wage_shock : Float1D, optional
        Scratch buffer for wage shock calculations (not persisted).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> emp = sim.emp
    >>> emp.wage_offer.shape
    (100,)
    >>> emp.current_labor.sum()  # doctest: +SKIP
    480

    Find firms with vacancies:

    >>> import numpy as np
    >>> has_vacancies = emp.n_vacancies > 0
    >>> has_vacancies.sum()  # doctest: +SKIP
    25

    Calculate labor shortage:

    >>> shortage = emp.desired_labor - emp.current_labor
    >>> total_shortage = shortage.sum()
    >>> total_shortage  # doctest: +SKIP
    20

    Notes
    -----
    The Employer role is one of three roles assigned to firms:

    - Producer: production and pricing (see Producer)
    - Employer: labor hiring and wages
    - Borrower: finance and credit (see Borrower)

    The total_funds and wage_bill arrays are shared with Borrower role
    (same underlying NumPy array) for memory efficiency and consistency.

    See Also
    --------
    :class:`~bamengine.roles.Producer` : Production role for firms
    :class:`~bamengine.roles.Borrower` : Financial role for firms
    :class:`~bamengine.roles.Worker` : Employment role for households
    :mod:`bamengine.events._internal.labor_market` : Labor market logic
    """

    desired_labor: Int1D
    current_labor: Int1D
    wage_offer: Float1D
    wage_bill: Float1D
    n_vacancies: Int1D

    # Shared view with Borrower role (same array in memory)
    total_funds: Float1D

    # Scratch queues
    recv_job_apps_head: Idx1D
    recv_job_apps: Idx2D

    # Scratch buffer
    wage_shock: Float1D | None = field(default=None, repr=False)
