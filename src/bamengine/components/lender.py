# src/bamengine/components/lender.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Lender:
    """Dense state for lender agents."""

    equity_base: Float1D  # E_k
    credit_supply: Float1D  # C_k
    interest_rate: Float1D  # r_k

    # bounded FIFO queue of borrower indices that apply each round
    recv_apps_head: Idx1D  # ptr (−1 ⇒ empty)
    recv_apps: Idx2D  # shape (N_banks, H)

    opex_shock: Float1D | None = field(default=None, repr=False)  # scratch
