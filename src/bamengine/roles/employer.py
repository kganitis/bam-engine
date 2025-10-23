# src/bamengine/roles/employer.py
from dataclasses import dataclass, field
from typing import Optional

from bamengine.core import Role
from bamengine.typing import Float1D, Idx1D, Idx2D, Int1D


@dataclass(slots=True)
class Employer(Role):
    """
    Employer role for agents.

    Represents an entity that is able to hire labor and pay wages.
    """

    desired_labor: Int1D
    current_labor: Int1D
    wage_offer: Float1D
    wage_bill: Float1D
    n_vacancies: Int1D

    # Shared view with Borrower role (same array in memory)
    total_funds: Float1D

    # Scratch queues
    recv_job_apps_head: Idx1D
    recv_job_apps: Idx2D  # shape (n_firms, n_households)

    # Scratch buffer (optional for performance)
    wage_shock: Optional[Float1D] = field(default=None, repr=False)
