"""
Borrower role for firms.

Represents the financial and credit management aspect of firm behavior
in the BAM model. Each firm manages net worth, seeks credit, and collects revenues.
"""

from bamengine.core.decorators import role
from bamengine.typing import Float1D, Idx1D, Idx2D


@role
class Borrower:
    """
    Borrower role for firms.

    Represents the financial and credit state for firms. Each array index
    corresponds to a firm ID (0 to n_firms-1).

    Parameters
    ----------
    net_worth : Float1D
        Firm equity/net worth (A = assets - liabilities).
    total_funds : Float1D
        Available liquidity (shared view with Employer.total_funds).
    wage_bill : Float1D
        Total wages to be paid (W, shared with Employer.wage_bill).
    credit_demand : Float1D
        Amount of credit requested from banks (B).
    projected_fragility : Float1D
        Financial fragility metric (B / A, leverage).
    gross_profit : Float1D
        Revenue before debt service.
    net_profit : Float1D
        Profit after debt service (π).
    retained_profit : Float1D
        Profit retained after dividends.
    loan_apps_head : Idx1D
        Queue head pointer for loan applications.
    loan_apps_targets : Idx2D
        Queue of bank IDs to apply to, shape (n_firms, max_H).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> bor = sim.bor
    >>> bor.net_worth.shape
    (100,)
    >>> bor.net_worth.mean()  # doctest: +SKIP
    105.3

    Find firms with negative net worth (insolvent):

    >>> import numpy as np
    >>> insolvent = bor.net_worth < 0
    >>> insolvent.sum()  # doctest: +SKIP
    3

    Calculate aggregate credit demand:

    >>> total_credit_demand = bor.credit_demand.sum()
    >>> total_credit_demand  # doctest: +SKIP
    1250.0

    Check financial fragility distribution:

    >>> high_fragility = bor.projected_fragility > 0.5
    >>> high_fragility.sum()  # doctest: +SKIP
    15

    Notes
    -----
    The Borrower role is one of three roles assigned to firms:

    - Producer: production and pricing (see Producer)
    - Employer: labor hiring and wages (see Employer)
    - Borrower: finance and credit

    The total_funds and wage_bill arrays are shared with Employer role
    (same underlying NumPy array) for memory efficiency and consistency.

    See Also
    --------
    :class:`~bamengine.roles.Producer` : Production role for firms
    :class:`~bamengine.roles.Employer` : Labor hiring role for firms
    :class:`~bamengine.roles.Lender` : Credit supply role for banks
    :class:`~bamengine.relationships.LoanBook` : Loan relationship between borrowers and lenders
    :mod:`bamengine.events._internal.credit_market` : Credit market logic
    :mod:`bamengine.events._internal.revenue` : Revenue collection logic
    """

    # Finance
    net_worth: Float1D
    total_funds: Float1D
    wage_bill: Float1D

    # Credit
    credit_demand: Float1D
    projected_fragility: Float1D

    # Revenues
    gross_profit: Float1D
    net_profit: Float1D
    retained_profit: Float1D

    # Scratch queues
    loan_apps_head: Idx1D
    loan_apps_targets: Idx2D
