from dataclasses import dataclass

from bamengine.typing import Float1D, Int1D


@dataclass(slots=True)
class WorkerJobSearch:
    """Per-period job–search state for *all* workers (dense arrays)."""

    # input / persistent
    employed: Int1D  # 1 = employed, 0 = unemployed          (bool-int)
    employer_prev: Int1D  # index of previous firm, -1 if None    (int64)
    contract_expired: Int1D  # 1 if contract just ended              (bool-int)
    fired: Int1D  # 1 if just fired                       (bool-int)

    # output / scratch (reset every period)
    apps_head: Int1D  # index into `apps_targets`, -1 when no apps
    apps_targets: Int1D  # flat array of firm indices (size = max_M * N_workers)


@dataclass(slots=True)
class WorkerContract:
    """Future‑proof container: wage grid + periods left per worker."""

    wage: Float1D  # shape (N_workers,)
    periods_left: Int1D  # shape (N_workers,)
