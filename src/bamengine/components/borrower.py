# src/bamengine/components/borrower.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Int1D


@dataclass(slots=True)
class Borrower:
    """Dense state for all borrowing firms."""

    # to ChatGPT: consider moving `net_worth`, `total_funds`,
    # `wage_bill`, rnd_intensity` to a separate component
    net_worth: Float1D  # A_i
    total_funds: Float1D
    wage_bill: Float1D  # W_i  (should be L_i * wage_i)
    credit_demand: Float1D  # B_i
    rnd_intensity: Float1D  # μ_i
    projected_fragility: Float1D  # φ̂_i

    # scratch arrays (reset every period)
    loan_apps_head: Int1D  # queue pointer  (-1 ⇒ empty)
    loan_apps_targets: Int1D  # shape (N_firms, H)  index buffer
