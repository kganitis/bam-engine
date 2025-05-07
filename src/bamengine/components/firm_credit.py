# src/bamengine/components/firm_credit.py
from dataclasses import dataclass, field

from bamengine.typing import FloatA, IntA, BoolA


@dataclass(slots=True)
class FirmCreditDemand:
    """
    Dense finance state for *all* firms (one row per firm).

    Arrays are shared by reference where possible; write‑only outputs are
    allocated once and reused forever.
    """

    # persistent inputs
    net_worth: FloatA          # A_i
    wage_bill: FloatA          # W_i  (should be L_i * wage_i)
    credit_demand: FloatA      # B_i           ← out

    # ── permanent scratch buffers ──────────────────────────────────────
    apply_mask: BoolA | None = field(default=None, repr=False)


@dataclass(slots=True)
class FirmLoanApplication:
    """
    """
    credit_demand: FloatA  # B_i           ← input

    # output / scratch (reset every period)
    loan_apps_head: IntA  # ptr into loan_apps_targets
    loan_apps_targets: IntA  # shape (N_firms, H)


@dataclass(slots=True)
class FirmLoan:
    """
    """
    credit_demand: FloatA
