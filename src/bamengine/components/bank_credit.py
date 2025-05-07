from dataclasses import dataclass, field

from bamengine.typing import FloatA, IntA, IdxA


@dataclass(slots=True)
class BankCreditSupply:
    """
    Dense credit‑supply state (one row per bank).
    """
    equity_base: FloatA        # E_k   (constant for now)
    credit_supply: FloatA      # C_k   ← out

    # bounded FIFO queue of firm indices that applied this round
    recv_apps_head: IntA       # ptr (−1 ⇒ empty)
    recv_apps: IdxA            # shape (N_banks, H)

    # scratch
    contract_rate: FloatA | None = field(default=None, repr=False)  # temp per bank


@dataclass(slots=True)
class BankInterestRate:
    """
    Dense nominal interest rate state (one row per bank).
    """
    interest_rate: FloatA  # r_k   ← out
    credit_shock: FloatA | None = field(default=None, repr=False)  # scratch
