# src/bamengine/components/employer.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Idx1D, Idx2D, Int1D


@dataclass(slots=True)
class Employer:

    desired_labor: Int1D
    current_labor: Int1D
    wage_offer: Float1D
    wage_bill: Float1D
    n_vacancies: Int1D

    total_funds: Float1D  # shared view with Borrower

    # Scratch queues
    recv_job_apps_head: Idx1D
    recv_job_apps: Idx2D  # shape (n_firms, n_households)

    # Scratch buffer
    wage_shock: Float1D | None = field(default=None, repr=False)
