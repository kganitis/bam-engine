from dataclasses import dataclass, field

from bamengine.typing import FloatA, IntA


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
    # ---- permanent scratch buffers ----
    wage_shock: FloatA | None = field(default=None, repr=False)


@dataclass(slots=True)
class FirmHiring:
    """All hiring‑related state in dense form."""

    wage_offer: FloatA  # w_i^b
    n_vacancies: IntA  # V_i   (remaining for this period)
    current_labor: IntA  # L_i   (shared view with FirmVacancies)

    # scratch queues
    recv_apps_head: IntA  # head ptr into recv_apps (−1 ⇒ empty)
    recv_apps: IntA  # flat buffer of worker indices
