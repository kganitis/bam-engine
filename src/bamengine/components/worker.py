# src/bamengine/components/worker.py
from dataclasses import dataclass

from bamengine.typing import Bool1D, Float1D, Idx1D, Idx2D, Int1D


@dataclass(slots=True)
class Worker:
    """Dense state for *all* worker households."""

    # persistent
    employed: Bool1D  # 1 = employed, 0 = unemployed
    employer: Idx1D  # index of employer firm, -1 if None
    employer_prev: Idx1D  # index of previous employer firm, -1 if None
    wage: Float1D  # amount of wage received
    periods_left: Int1D  # number of periods left until contract expiry
    contract_expired: Bool1D  # 1 if contract **just** ended
    fired: Bool1D  # 1 if **just** fired

    # scratch arrays (reset every period)
    job_apps_head: Idx1D  # index into `job_apps_targets`, -1 when no apps
    job_apps_targets: Idx2D  # flat array of firm indices (size = max_M * N_workers)
