# src/bamengine/components/firm_credit.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Int1D


@dataclass(slots=True)
class FirmCreditDemand:
    """
    Dense finance state for *all* firms (one row per firm).

    Arrays are shared by reference where possible; write‑only outputs are
    allocated once and reused forever.
    """

    # persistent inputs
    net_worth: Float1D  # A_i
    wage_bill: Float1D  # W_i  (should be L_i * wage_i)
    credit_demand: Float1D  # B_i           ← out


@dataclass(slots=True)
class FirmCreditMetrics:
    """
    Per-firm finance metrics used only inside the credit-market event.
    All arrays are 1-D, length = n_firms.
    """

    credit_demand: Float1D  # B_i   (shared view)
    net_worth: Float1D  # A_i   (shared view)
    rnd_intensity: Float1D  # μ_i   (constant or slowly changing)
    projected_fragility: Float1D  # φ̂_i           ← out


@dataclass(slots=True)
class FirmLoanApplication:
    """
    One row per firm – keeps the outbound queue *and* the metrics
    used by the bank when deciding the contractual rate.
    """

    credit_demand: Float1D  # B_i   (live state, reused)
    projected_fragility: Float1D  # φ̂_i  (filled by calc-metrics)

    loan_apps_head: Int1D  # queue pointer  (-1 ⇒ empty)
    loan_apps_targets: Int1D  # shape (N_firms, H)  index buffer
