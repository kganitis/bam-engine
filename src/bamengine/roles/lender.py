"""
Lender role for banks.

Represents the credit supply and interest rate management for banks
in the BAM model. Each bank provides credit to firms and charges interest.
"""

from dataclasses import field

from bamengine.core.decorators import role
from bamengine.typing import Float1D, Idx1D, Idx2D


@role
class Lender:
    """
    Lender role for banks.

    Represents the credit supply and interest rate state for banks. Each array
    index corresponds to a bank ID (0 to n_banks-1).

    Parameters
    ----------
    equity_base : Float1D
        Bank equity/capital base.
    credit_supply : Float1D
        Maximum credit available to lend (based on equity and capital requirements).
    interest_rate : Float1D
        Interest rate charged on loans.
    recv_loan_apps_head : Idx1D
        Queue head pointer for received loan applications.
    recv_loan_apps : Idx2D
        Queue of borrower firm IDs, shape (n_banks, n_firms).
    opex_shock : Float1D, optional
        Scratch buffer for operational expense shock calculations (not persisted).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_banks=10, seed=42)
    >>> lend = sim.lend
    >>> lend.credit_supply.shape
    (10,)
    >>> lend.interest_rate.shape
    (10,)

    Calculate total credit supply:

    >>> import numpy as np
    >>> total_supply = lend.credit_supply.sum()
    >>> total_supply  # doctest: +SKIP
    1500.0

    Find banks with high interest rates:

    >>> high_rate = lend.interest_rate > 0.03
    >>> high_rate.sum()  # doctest: +SKIP
    3

    Check equity distribution:

    >>> mean_equity = lend.equity_base.mean()
    >>> mean_equity  # doctest: +SKIP
    150.0

    Find bank with highest credit supply:

    >>> max_supply_bank = np.argmax(lend.credit_supply)
    >>> max_supply_bank  # doctest: +SKIP
    5

    Notes
    -----
    Banks are the only agents with a single role (Lender). Firms have three
    roles (Producer, Employer, Borrower) and households have two (Worker, Consumer).

    Credit supply is constrained by bank equity and the capital requirement ratio (v):
    credit_supply = equity_base / v

    See Also
    --------
    :class:`~bamengine.roles.Borrower` : Credit demand role for firms
    :class:`~bamengine.relationships.LoanBook` : Loan relationship between borrowers and lenders
    :mod:`bamengine.events._internal.credit_market` : Credit market logic
    """

    equity_base: Float1D
    credit_supply: Float1D
    interest_rate: Float1D

    # Scratch queues
    recv_loan_apps_head: Idx1D
    recv_loan_apps: Idx2D

    # Scratch buffer
    opex_shock: Float1D | None = field(default=None, repr=False)
