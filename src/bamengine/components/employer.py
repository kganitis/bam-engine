# src/bamengine/components/employer.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Int1D, Idx1D, Idx2D


@dataclass(slots=True)
class Employer:
    """Dense state for *all* employer firms."""

    desired_labor: Int1D
    current_labor: Int1D
    wage_offer: Float1D
    wage_bill: Float1D
    n_vacancies: Int1D

    total_funds: Float1D  # shared view from other component?

    # scratch queues (reset every period)
    recv_job_apps_head: Idx1D  # head ptr into recv_job_apps (−1 ⇒ empty)
    recv_job_apps: Idx2D  # buffer of worker indices

    # permanent scratch buffer
    wage_shock: Float1D | None = field(default=None, repr=False)
