from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntA = NDArray[np.int64]
FloatA = NDArray[np.float64]


@dataclass(slots=True)
class WorkerJobSearch:
    """Per-period job–search state for *all* workers (dense arrays)."""

    # input / persistent
    employed: IntA  # 1 = employed, 0 = unemployed          (bool-int)
    employer_prev: IntA  # index of previous firm, -1 if None    (int64)
    contract_expired: IntA  # 1 if contract just ended              (bool-int)
    fired: IntA  # 1 if just fired                       (bool-int)

    # output / scratch (reset every period)
    apps_head: IntA  # index into `apps_targets`, -1 when no apps
    apps_targets: IntA  # flat array of firm indices (size = max_M * N_workers)
