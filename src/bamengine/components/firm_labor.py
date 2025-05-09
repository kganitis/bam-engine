from dataclasses import dataclass, field

from bamengine.typing import Float1D, Int1D


@dataclass(slots=True)
class FirmWageOffer:
    """State needed to post a wage offer in the labour market.

    - `wage_prev` is the contractual wage paid to the last cohort hired.
    - `n_vacancies` comes from Event 1.
    - `wage_offer` (output) is what the firm will advertise this period.
    """

    wage_prev: Float1D  # w_{i,t-1}
    n_vacancies: Int1D  # V_i  (read-only)
    wage_offer: Float1D  # w_i^b (output)
    # ---- permanent scratch buffers ----
    wage_shock: Float1D | None = field(default=None, repr=False)


@dataclass(slots=True)
class FirmHiring:
    """All hiring‑related state in dense form."""

    wage_offer: Float1D  # w_i^b
    n_vacancies: Int1D  # V_i   (remaining for this period)
    current_labor: Int1D  # L_i   (shared view with FirmVacancies)

    # scratch queues
    recv_apps_head: Int1D  # head ptr into recv_apps (−1 ⇒ empty)
    recv_apps: Int1D  # flat buffer of worker indices


@dataclass(slots=True)
class FirmWageBill:
    """ """

    current_labor: Int1D  # L_i
    wage: Float1D  # w_i
    wage_bill: Float1D  # W_i           ← out
