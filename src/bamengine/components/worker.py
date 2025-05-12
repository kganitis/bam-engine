# src/bamengine/components/worker.py
from dataclasses import dataclass

from bamengine.typing import Bool1D, Float1D, Int1D, Idx1D, Idx2D


@dataclass(slots=True)
class Worker:
    """Dense state for *all* worker households."""

    # persistent
    employed: Bool1D  # 1 = employed, 0 = unemployed          (bool-int)
    employer: Idx1D  # index of employer firm, -1 if None    (int64)
    employer_prev: Idx1D  # index of previous employer firm, -1 if None    (int64)
    wage: Float1D
    periods_left: Int1D
    contract_expired: Bool1D  # 1 if contract just ended              (bool-int)
    fired: Bool1D  # 1 if just fired                       (bool-int)

    # scratch arrays (reset every period)
    job_apps_head: Idx1D  # index into `job_apps_targets`, -1 when no apps
    job_apps_targets: Idx2D  # flat array of firm indices (size = max_M * N_workers)
