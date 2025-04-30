from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatA = NDArray[np.float64]
IntA = NDArray[np.int64]


@dataclass(slots=True)
class FirmWageOffer:
    """State needed to post a wage offer in the labour market.

    - `wage_prev` is the contractual wage paid to the last cohort hired.
    - `n_vacancies` comes from Event 1.
    - `wage_offer` (output) is what the firm will advertise this period.
    """

    wage_prev: FloatA  # w_{i,t-1}
    n_vacancies: IntA  # V_i  (read-only)
    wage_offer: FloatA  # w_i^b (output)


@dataclass(slots=True)
class FirmHiring:
    """All hiring-related state in dense form."""

    wage_offer: FloatA  # w_i^b (current period offer)
    n_vacancies: IntA  # V_i   (posted this period, already from planning)
    # scratch queues
    recv_apps_head: IntA  # head ptr into recv_apps array (–1 => empty)
    recv_apps: IntA  # flat buffer of worker indices
