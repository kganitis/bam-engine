# BAM-Engine roadmap

### Immediate tasks

* Implement production components
* Implement production systems
* Scheduler wiring
* Unit tests
* Integration tests
* Raise test coverage
* Scheduler tests

---

### Incremental event implementation

Implement the next events one by one

* Specify the required components and systems
* Wire the event into the scheduler
* Write unit tests that cover edge cases and buffer reuse
* Write an event-level integration test
* Raise the test coverage
* Update the scheduler integration test

Repeat until the full event chain is implemented

---

### Visualization

* Logging and plotting
  * Add lightweight debug logging
  * Add minimal plotting helper
  * Guard with `log.isEnabledFor`

---

### Low-priority housekeeping

* Guard all heavy logging with level checks
* Keep the Ruff rule set minimal (`E, F, W, I` plus docstrings later)
* Introduce pytest-benchmark and fail the suite on measurable performance regressions
* Remove dead lint-ignore comments and finalise design diagrams once the API is frozen

---

### Performance milestone

* Profile representative sizes to establish baselines
* Remove remaining temporary allocations and other obvious hotspots
  * e.g. reuse permanent scratch buffers in `workers_decide_firms_to_apply` and `firms_prepare_loan_applications`
* Enable threaded NumPy or process-level sharding of firms when array sizes become very large
* Apply targeted `@njit` only after profiling shows sustained benefit

---

### Distribution and research API

* Package the library as a PyPI wheel once the public API stabilises
* Add `Scheduler.to_dict` and `Scheduler.from_dict` for checkpoint and restart workflows
* Provide read-only pandas views for downstream analysis without touching core arrays
* Replace the current hook set with a small plugin event bus (`Plugin.on_event(sched, tag)`)
* Finish NumPy-style docstrings and enable Ruff’s docstring lint rule

---

### Future research extensions

* Add reinforcement-learning agents for policy search experiments
* Investigate GPU or distributed back-ends once single-node scaling limits are reached


It's finally time to implement the next event: production.
I will provide you with the respective code from my older OOP implementation.
I want you to write an implementation that fits our current structure and logic.
Your output should be the necessary updated components and new systems.
We should use and update the existing Producer, Employer and Worker components. We won't need new ones.
For the systems write a new `production.py` file.

### Older OOP implementation
```
def pay_wages(self) -> None:
    for w in self.workers:
        w.receive_wage(self.wage)
    self.total_funds -= self.wage_bill

def run_production(self) -> None:
    self.production_Y = self.labor_productivity_a * self.labor_L
    self.inventory_S = self.production_Y

# Event 4 - Production
'''
Production takes one time period, regardless of the scale of production/firm’s size.
'''
for f in self.firms:
    f.pay_wages()
    f.run_production()
```

### Existing files needed for the ECS implementation
```
# src/bamengine/components/producer.py
from dataclasses import dataclass, field

from bamengine.typing import Bool1D, Float1D


@dataclass(slots=True)
class Producer:
    """Dense state needed for production firms."""

    production: Float1D  # Y_i  (carried from t-1)
    inventory: Float1D  # S_i  (carried from t-1)
    expected_demand: Float1D  # D̂_i
    desired_production: Float1D  # Yd_i
    labor_productivity: Float1D  # a_i   (can change with R&D later)

    price: Float1D  # p_i  (carried from t-1)   shared view from Seller component?

    # ---- permanent scratch buffers ----
    prod_shock: Float1D | None = field(default=None, repr=False)
    prod_mask_up: Bool1D | None = field(default=None, repr=False)
    prod_mask_dn: Bool1D | None = field(default=None, repr=False)


# src/bamengine/components/employer.py
from dataclasses import dataclass, field

from bamengine.typing import Float1D, Idx1D, Idx2D, Int1D


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


# src/bamengine/components/economy.py
from dataclasses import dataclass, field

import numpy as np

from bamengine.typing import Float1D, Int1D


@dataclass(slots=True)
class Economy:
    """Global, mutable scalars & time-series."""

    avg_mkt_price: float
    avg_mkt_price_history: Float1D  # P_0 … P_t   (append-only)
    min_wage: float  # ŵ_t
    min_wage_rev_period: int  # constant (e.g. 4)
    r_bar: float  # base interest rate
    v: float  # capital requirement coefficient
```