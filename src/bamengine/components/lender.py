# src/bamengine/components/lender.py
from dataclasses import dataclass, field
from typing import Optional

from bamengine.core import Role
from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Lender(Role):
    """
    Lender role for agents.

    Represents an entity that is able to provide credit and receive interest.
    """

    equity_base: Float1D
    credit_supply: Float1D
    interest_rate: Float1D

    # Scratch queues
    recv_loan_apps_head: Idx1D
    recv_loan_apps: Idx2D  # shape (n_banks, n_firms)

    # Scratch buffer (optional for performance)
    opex_shock: Optional[Float1D] = field(default=None, repr=False)
