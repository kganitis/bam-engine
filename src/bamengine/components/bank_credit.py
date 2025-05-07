# src/bamengine/components/bank_credit.py
from dataclasses import dataclass, field

from bamengine.typing import FloatA, IntA, IdxA


@dataclass(slots=True)
class BankCreditSupply:
    """
    Dense credit‑supply state (one row per bank).
    """
    equity_base: FloatA        # E_k   (constant for now)
    credit_supply: FloatA      # C_k   ← out


@dataclass(slots=True)
class BankInterestRate:
    """
    Dense nominal interest rate state (one row per bank).
    """
    interest_rate: FloatA  # r_k   ← out
    credit_shock: FloatA | None = field(default=None, repr=False)  # scratch


@dataclass(slots=True)
class BankReceiveLoanApplication:
    """

    """
    # bounded FIFO queue of firm indices that applied this round
    recv_apps_head: IntA  # ptr (−1 ⇒ empty)
    recv_apps: IdxA  # shape (N_banks, H)

    # scratch
    contract_rate: FloatA | None = field(default=None, repr=False)  # temp per bank


@dataclass(slots=True)
class BankProvideLoan:
    """

    """
    credit_supply: FloatA
    recv_apps_head: IntA
    recv_apps: IdxA

    # scratch
    contract_rate: FloatA | None = field(default=None, repr=False)  # temp per bank
