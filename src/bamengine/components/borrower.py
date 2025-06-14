# src/bamengine/components/borrower.py
from dataclasses import dataclass

from bamengine.typing import Float1D, Idx1D, Idx2D


@dataclass(slots=True)
class Borrower:

    rnd_intensity: Float1D  # μ

    # Finance
    net_worth: Float1D  # A
    total_funds: Float1D  # shared view with Employer
    wage_bill: Float1D  # W

    # Credit
    credit_demand: Float1D  # B
    projected_fragility: Float1D  # μ · B / A

    # Revenues
    gross_profit: Float1D
    net_profit: Float1D  # π
    retained_profit: Float1D

    # Scratch queues
    loan_apps_head: Idx1D
    loan_apps_targets: Idx2D  # shape (n_firms, H)
