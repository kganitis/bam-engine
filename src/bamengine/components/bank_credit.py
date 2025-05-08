# src/bamengine/components/bank_credit.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Idx1D, Int1D


@dataclass(slots=True)
class BankCreditSupply:
    """
    Dense credit‑supply state (one row per bank).
    """

    equity_base: Float1D  # E_k   (constant for now)
    credit_supply: Float1D  # C_k   ← out


@dataclass(slots=True)
class BankInterestRate:
    """
    Dense nominal interest rate state (one row per bank).
    """

    interest_rate: Float1D  # r_k   ← out
    credit_shock: Float1D | None = field(default=None, repr=False)  # scratch


@dataclass(slots=True)
class BankProvideLoan:
    """ """

    credit_supply: Float1D  # (shared view)

    # bounded FIFO queue of firm indices that applied this round
    recv_apps_head: Int1D  # ptr (−1 ⇒ empty)
    recv_apps: Idx1D  # shape (N_banks, H)
