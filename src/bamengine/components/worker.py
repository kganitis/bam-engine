# src/bamengine/components/worker.py
from dataclasses import dataclass

from bamengine.typing import Bool1D, Float1D, Idx1D, Idx2D, Int1D


@dataclass(slots=True)
class Worker:

    employed: Bool1D
    employer: Idx1D
    employer_prev: Idx1D
    wage: Float1D
    periods_left: Int1D
    contract_expired: Bool1D
    fired: Bool1D

    # Scratch queues
    job_apps_head: Idx1D
    job_apps_targets: Idx2D  # shape (n_households, M)
