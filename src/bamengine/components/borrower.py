# src/bamengine/components/borrower.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Borrower:
    """Dense state for all borrowing firms."""

    rnd_intensity: Float1D  # μ_i

    # Finance
    net_worth: Float1D  # A_i
    total_funds: Float1D  # cash account
    wage_bill: Float1D  # W_i  (should be L_i * wage_i)

    # Credit
    credit_demand: Float1D  # B_i
    projected_fragility: Float1D  # φ̂_i

    # Revenues
    gross_profit: Float1D
    net_profit: Float1D
    retained_profit: Float1D

    # --------------- period-scratch queues ------------- #
    loan_apps_head: Idx1D  # queue pointer  (-1 ⇒ empty)
    loan_apps_targets: Idx2D  # shape (N_firms, H)  index buffer
