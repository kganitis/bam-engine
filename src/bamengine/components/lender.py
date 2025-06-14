# src/bamengine/components/lender.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Lender:

    equity_base: Float1D
    credit_supply: Float1D
    interest_rate: Float1D

    # Scratch queues
    recv_loan_apps_head: Idx1D
    recv_loan_apps: Idx2D  # shape (n_banks, n_firms)

    # Scratch buffer
    opex_shock: Float1D | None = field(default=None, repr=False)
