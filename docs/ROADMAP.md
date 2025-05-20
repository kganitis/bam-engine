# BAM-Engine roadmap

### Immediate tasks

* Scheduler wiring
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

---

A new error has been revealed by an older integration test:
### Specifics of the error:
====================================================== FAILURES ======================================================= 
______________________________________ test_scheduler_state_stable_over_time[10] ______________________________________ 

steps = 10

    @pytest.mark.parametrize("steps", [10])
    def test_scheduler_state_stable_over_time(steps: int) -> None:
        """
        Multi‑period smoke test: run consecutive Scheduler steps on a medium‑sized
        economy and assert that key invariants hold after **each** period.

        This catches state‑advancement bugs that single‑step tests can miss.
        """
        # Medium‑sized economy
        sch = Scheduler.init(
            n_firms=50,
            n_households=250,
            n_banks=5,
            seed=9,
        )

        for _ in range(steps):
            sch.step()
>           assert_basic_invariants(sch)

tests\integration\scheduler\test_scheduler.py:43:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

sch = Scheduler(rng=Generator(PCG64) at 0x1EDAF390BA0, ec=Economy(avg_mkt_price=1.4920143254959057, avg_mkt_price_histor
y=ar...t=array([], dtype=float64), capacity=128, size=0), h_rho=0.1, h_xi=0.05, h_phi=0.1, max_M=4, max_H=2, max_Z=2, theta=8)

    def assert_basic_invariants(sch: Scheduler) -> None:
        """
        Assertions that must hold after *any* ``Scheduler.step()`` call,
        regardless of economy size or number of periods.
        """
        # ------------------------------------------------------------------ #
        #  Planning & labor-market                                          #
        # ------------------------------------------------------------------ #
        # Production plans non-negative & finite
        assert not np.isnan(sch.prod.desired_production).any()
        assert (sch.prod.desired_production >= 0).all()

        # Vacancies never negative
        assert (sch.emp.n_vacancies >= 0).all()

        # Wage offers respect current minimum wage
        assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

        # Employment flags are boolean
        assert set(np.unique(sch.wrk.employed)).issubset({0, 1})

        # Every employed worker counted in current_labor, and vice-versa
>       assert sch.wrk.employed.sum() == sch.emp.current_labor.sum()
E       AssertionError

tests\helpers\invariants.py:28: AssertionError

### What I found:
I narrowed down the problem and it's caused the very moment I wire up the `firms_pay_wages` function, in combination with one of the following:
1. Deleting the production mocking in the function `advance_stub_state` of `src/_testing/__init__.py` while wiring up the rest of the production event:
``` 
firms_pay_wages(self.emp)
# the rest of the event is wired
workers_receive_wage(self.con, self.wrk)
firms_run_production(self.prod, self.emp)
workers_update_contracts(self.wrk, self.emp)
```
2. Modifying the mock to also mock production increase **without** wiring up the rest of the event:
``` 
# 1. --- mock production ---------------------------------------
sched.prod.production[:] *= sched.rng.uniform(
    0.7, 2.0, size=sched.prod.production.shape
)
```
``` 
firms_pay_wages(self.emp)
# rest of the event is unwired
# workers_receive_wage(self.con, self.wrk)
# firms_run_production(self.prod, self.emp)
# workers_update_contracts(self.wrk, self.emp)
```

I also modified firms_fire_workers by updating ALL of the worker fields:
```     
wrk.employed[victims] = 0
wrk.employer[victims] = -1
wrk.employer_prev[victims] = i
wrk.wage[victims] = 0.0
wrk.periods_left[victims] = 0
wrk.contract_expired[victims] = 0  # explicit: this was a firing
wrk.fired[victims] = 1
```
but the error persists.
I will provide you with all the necessary code involved:

### Current code
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
    """Dense state for *all* workers."""

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


# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.employer import Employer
from bamengine.components.producer import Producer

log = logging.getLogger(__name__)

CAP_LAB_PROD = 1.0e-6  # labor productivity cap if below from or equal to zero


def firms_decide_desired_production(  # noqa: C901  (still quite short)
    prod: Producer,
    *,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """
    Update `prod.expected_demand` and `prod.desired_production` **in‑place**.

    Rule
    ----
      if S_i == 0 and P_i ≥ p̄   → raise   by (1 + shock)
      if S_i  > 0 and P_i < p̄   → cut     by (1 − shock)
      otherwise                 → keep previous level
    """
    shape = prod.price.shape

    # ── 1. permanent scratches ---------------
    shock = prod.prod_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        prod.prod_shock = shock

    up_mask = prod.prod_mask_up
    if up_mask is None or up_mask.shape != shape:
        up_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_up = up_mask

    dn_mask = prod.prod_mask_dn
    if dn_mask is None or dn_mask.shape != shape:
        dn_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_dn = dn_mask

    # ── 2. fill buffers in‑place ---------------
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    # ── 3. core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    prod.desired_production[:] = prod.expected_demand


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    invalid = (~np.isfinite(prod.labor_productivity)) | (
        prod.labor_productivity <= CAP_LAB_PROD
    )
    if invalid.any():
        log.warning("labor productivity too low / non-finite – clamped")
        prod.labor_productivity[invalid] = CAP_LAB_PROD

    # core rule -----------------------------------------------------------
    ratio = prod.desired_production / prod.labor_productivity
    np.ceil(ratio, out=ratio)  # in-place ceiling

    # clip to int64 range to avoid overflow warnings
    int64_max = np.iinfo(np.int64).max
    np.clip(ratio, 0, int64_max, out=ratio)

    emp.desired_labor[:] = ratio.astype(np.int64)  # safe int cast


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Vector rule: V_i = max( Ld_i – L_i , 0 )
    """
    np.subtract(
        emp.desired_labor,
        emp.current_labor,
        out=emp.n_vacancies,
        dtype=np.int64,
        casting="unsafe",  # makes MyPy/NumPy on Windows happy
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)


# src/bamengine/systems/labor_market.py
import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.economy import Economy
from bamengine.components.employer import Employer
from bamengine.components.worker import Worker
from bamengine.typing import Float1D, Idx1D

log = logging.getLogger(__name__)


def adjust_minimum_wage(ec: Economy) -> None:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation:

        π = (P_{t-1} - P_{t-m}) / P_{t-m}
        ŵ_t = ŵ_{t-1} * (1 + π)
    """
    m = ec.min_wage_rev_period
    if ec.avg_mkt_price_history.size <= m:
        return  # not enough data yet
    if (ec.avg_mkt_price_history.size - 1) % m != 0:
        return  # not a revision step

    p_now = ec.avg_mkt_price_history[-1]  # price of period t-1
    p_prev = ec.avg_mkt_price_history[-m - 1]  # price of period t-m
    inflation = (p_now - p_prev) / p_prev

    ec.min_wage *= 1.0 + inflation


def firms_decide_wage_offer(
    emp: Employer,
    *,
    w_min: float,
    h_xi: float,
    rng: Generator,
) -> None:
    """
    Vector rule:

        shock_i ~ U(0, h_xi)  if V_i>0 else 0
        w_i^b   = max( w_min , w_{i,t-1} * (1 + shock_i) )

    Works fully in-place, no temporary allocations.
    """
    shape = emp.wage_offer.shape

    # permanent scratch
    shock = emp.wage_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        emp.wage_shock = shock

    # Draw one shock per firm, then mask where V_i==0.
    shock[:] = rng.uniform(0.0, h_xi, size=shape)
    shock[emp.n_vacancies == 0] = 0.0

    # core rule
    np.multiply(emp.wage_offer, 1.0 + shock, out=emp.wage_offer)
    np.maximum(emp.wage_offer, w_min, out=emp.wage_offer)


# --------------------------------------------------------------------------- #
def _topk_indices_desc(values: Float1D, k: int) -> Idx1D:
    """
    Return indices of the *k* largest elements along the last axis
    (unsorted, descending).

    Complexity
    ----------
    argpartition : O(n)   – finds the split position,
                          - ascending -> we call it on **‑values**
    slicing      : O(k)   – keeps only the first k
    Total        : O(n + k)  vs. full argsort O(n logn)
    """
    if k >= values.shape[-1]:  # degenerate: keep all
        return np.argpartition(-values, kth=0, axis=-1)
    part = np.argpartition(-values, kth=k - 1, axis=-1)  # top‑k to the left
    return part[..., :k]  # [:, :k] for 2‑D case, same ndim as input


# ---------------------------------------------------------------------
def workers_decide_firms_to_apply(
    wrk: Worker,
    emp: Employer,
    *,
    max_M: int,
    rng: Generator,
) -> None:
    n_firms = emp.wage_offer.size
    unem = np.where(wrk.employed == 0)[0]  # unemployed ids

    if unem.size == 0:  # early-exit → nothing to do
        wrk.job_apps_head.fill(-1)
        return

    # -------- sample M random emp per worker -----------------------
    sample = rng.integers(0, n_firms, size=(unem.size, max_M), dtype=np.int64)

    loyal = (
        (wrk.contract_expired[unem] == 1)
        & (wrk.fired[unem] == 0)
        & (wrk.employer_prev[unem] >= 0)
    )
    if loyal.any():
        sample[loyal, 0] = wrk.employer_prev[unem[loyal]]

    # -------- wage‑descending *partial* sort ----------------------------
    topk = _topk_indices_desc(emp.wage_offer[sample], k=max_M)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    #
    # -------- loyalty: ensure previous employer is always in column 0 ---
    if loyal.any():
        # indices of loyal workers in the `unem` array
        loyal_rows = np.where(loyal)[0]

        # swap previous‑employer into col 0 when it got shuffled away
        for r in loyal_rows:
            prev = wrk.employer_prev[unem[r]]
            row = sorted_sample[r]

            if row[0] != prev:  # not covered by tests
                # find where prev employer ended up (guaranteed to exist)
                j = np.where(row == prev)[0][0]
                row[0], row[j] = row[j], row[0]

    stride = max_M
    for k, w in enumerate(unem):
        wrk.job_apps_targets[w, :stride] = sorted_sample[k]
        wrk.job_apps_head[w] = w * stride  # first slot of that row

    # reset flags
    wrk.contract_expired[unem] = 0
    wrk.fired[unem] = 0


# ---------------------------------------------------------------------
def workers_send_one_round(wrk: Worker, emp: Employer) -> None:
    stride = wrk.job_apps_targets.shape[1]

    for w in np.where(wrk.employed == 0)[0]:
        h = wrk.job_apps_head[w]
        if h < 0:
            continue
        row, col = divmod(h, stride)
        firm_idx = wrk.job_apps_targets[row, col]
        if firm_idx < 0:  # exhausted list
            wrk.job_apps_head[w] = -1
            continue

        # bounded queue
        ptr = emp.recv_job_apps_head[firm_idx] + 1
        if ptr >= emp.recv_job_apps.shape[1]:
            continue  # queue full – drop
        emp.recv_job_apps_head[firm_idx] = ptr
        emp.recv_job_apps[firm_idx, ptr] = w

        # advance pointer & clear slot
        wrk.job_apps_head[w] = h + 1
        wrk.job_apps_targets[row, col] = -1


# ---------------------------------------------------------------------
def firms_hire_workers(
    wrk: Worker,
    emp: Employer,
    *,
    theta: int,
) -> None:
    """Match firms with queued applicants and update all related state."""
    for i in np.where(emp.n_vacancies > 0)[0]:
        n_recv = emp.recv_job_apps_head[i] + 1  # queue length (−1 ⇒ 0)
        if n_recv <= 0:
            continue

        n_hire = int(min(n_recv, emp.n_vacancies[i]))
        hires = emp.recv_job_apps[i, :n_hire]
        hires = hires[hires >= 0]  # drop sentinel slots
        if hires.size == 0:
            emp.recv_job_apps_head[i] = -1  # clear queue
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        # ---- worker‑side updates ----------------------------------------
        wrk.employed[hires] = 1
        wrk.employer[hires] = i  # update contract

        # if wages become worker-specific replace with np.put(…) / gather logic
        wrk.wage[hires] = emp.wage_offer[i]

        wrk.periods_left[hires] = theta
        wrk.contract_expired[hires] = 0
        wrk.fired[hires] = 0

        wrk.job_apps_head[hires] = -1  # clear queue
        wrk.job_apps_targets[hires, :] = -1

        # ---- firm‑side updates ------------------------------------------
        emp.current_labor[i] += hires.size
        emp.n_vacancies[i] -= hires.size

        emp.recv_job_apps_head[i] = -1  # clear queue
        emp.recv_job_apps[i, :n_recv] = -1


def firms_calc_wage_bill(emp: Employer) -> None:
    """
    W_i = L_i · w_i
    """
    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
    

# src/bamengine/systems/credit_market.py
...
def firms_fire_workers(
    emp: Employer,
    wrk: Worker,
    *,
    rng: Generator,
) -> None:
    """
    If `wage_bill[i]` exceeds `total_funds[i]` the firm lays off just enough
    workers to close the gap:

        n_fire = ceil( gap / wage[i] )   but never more than current labor.

    Workers are picked **uniformly at random** from the current workforce.
    """
    n_firms = emp.current_labor.size

    for i in range(n_firms):
        gap = emp.wage_bill[i] - emp.total_funds[i]
        if gap <= 0.0 or emp.current_labor[i] == 0:
            continue

        # how many heads must roll?
        needed = int(
            np.ceil(gap / float(emp.wage_offer[i]))
        )  # heads needed to close gap
        capacity = int(emp.current_labor[i])  # heads currently employed
        n_fire = min(capacity, needed)
        if n_fire == 0:  # float quirks
            continue

        # choose victims uniformly
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        if workforce.size == 0:
            continue
        victims = rng.choice(workforce, size=n_fire, replace=False)

        # --- worker-side -------------------------------------------
        wrk.employed[victims] = 0
        wrk.employer[victims] = -1
        wrk.employer_prev[victims] = i
        wrk.wage[victims] = 0.0
        wrk.periods_left[victims] = 0
        wrk.contract_expired[victims] = 0  # explicit: this was a firing
        wrk.fired[victims] = 1

        # --- firm-side ---------------------------------------------
        emp.current_labor[i] -= n_fire
        emp.wage_bill[i] -= n_fire * emp.wage_offer[i]


# src/bamengine/systems/production.py
"""
Event-4 – Production systems (vectorised, zero allocations)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.components import Consumer, Employer, Producer, Worker
from bamengine.typing import Idx1D


# --------------------------------------------------------------------- #
# 1.  Wage payment – firm-side only                                     #
# --------------------------------------------------------------------- #
def firms_pay_wages(emp: Employer) -> None:
    """
    Debit each firm’s cash account by its wage-bill (vectorised).
    """
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)


# --------------------------------------------------------------------- #
# 2.  Wage receipt – household side                                     #
# --------------------------------------------------------------------- #
def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income by the wage that *employed* workers earned:

        income_h += wage_h · employed_h
    """
    inc = wrk.wage * wrk.employed
    np.add(con.income, inc, out=con.income)


# --------------------------------------------------------------------- #
# 3.  Physical production                                               #
# --------------------------------------------------------------------- #
def firms_run_production(prod: Producer, emp: Employer) -> None:
    """
    Compute current-period output and replace inventories:

        Y_i  =  a_i · L_i
        S_i  ←  Y_i
    """
    np.multiply(prod.labor_productivity, emp.current_labor, out=prod.production)
    prod.inventory[:] = prod.production  # overwrite, do **not** add


# --------------------------------------------------------------------- #
# 4.  Contract-expiration mechanic                                      #
# --------------------------------------------------------------------- #
def workers_update_contracts(wrk: Worker, emp: Employer) -> None:
    """
    Decrease `periods_left` for every *employed* worker.
    When the counter reaches 0 the contract expires:

        • worker becomes unemployed (`employed = 0`)
        • `contract_expired = 1`
        • `employer_prev` stores the firm index
        • firm labour count `current_labor` is decremented
        • queue-related scratch is left untouched (will be filled later)

    All updates are fully vectorised except the firm-side labour
    adjustment, which uses a single `np.bincount`.
    """

    # --- step 1: tick down only for currently employed -----------------
    mask_emp: NDArray[np.bool_] = wrk.employed == 1
    if not mask_emp.any():
        return  # nothing to do

    wrk.periods_left[mask_emp] -= 1

    # --- step 2: detect expirations -----------------------------------
    expired: NDArray[np.bool_] = mask_emp & (wrk.periods_left == 0)
    if not expired.any():
        return

    # snapshot firm indices before we overwrite them
    firms: Idx1D = wrk.employer[expired]

    # worker-side state
    wrk.employed[expired] = 0
    wrk.employer[expired] = -1
    wrk.employer_prev[expired] = firms
    wrk.wage[expired] = 0.0
    wrk.contract_expired[expired] = 1
    wrk.fired[expired] = 0  # explicit: this was an expiration

    # firm-side labour book-keeping
    delta = np.bincount(firms, minlength=emp.current_labor.size)
    emp.current_labor[: delta.size] -= delta
    # guard against numerical slip
    assert (emp.current_labor >= 0).all(), "negative labour after expirations"


# src/_testing/__init__.py
"""
Private helpers used *only* by the test-suite.
They should disappear once all real events are implemented.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bamengine.scheduler import Scheduler


def advance_stub_state(sched: "Scheduler") -> None:
    """
    One-shot placeholder that nudges the state forward so multi-period
    tests have fresh, non-degenerate arrays.
    """

    # 1. --- mock production ---------------------------------------
    sched.prod.production[:] *= sched.rng.uniform(
        0.7, 1.0, size=sched.prod.production.shape
    )

    # 2. --- mock prices -------------------------------------------
    sched.prod.price[:] *= sched.rng.uniform(0.95, 1.05, size=sched.prod.price.shape)
    sched.ec.avg_mkt_price = float(sched.prod.price.mean())
    sched.ec.avg_mkt_price_history = np.append(
        sched.ec.avg_mkt_price_history, sched.ec.avg_mkt_price
    )

    # ------ mock goods market -------------------------------------
    sched.prod.inventory[:] = sched.rng.integers(0, 6, size=sched.prod.inventory.shape)


# src/bamengine/scheduler.py
"""BAM Engine – tiny driver that wires components ↔ systems for 1 period."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, TypeAlias

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import (
    Borrower,
    Consumer,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.systems.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    firms_calc_wage_bill,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.systems.production import (
    firms_pay_wages,
    firms_run_production,
    workers_receive_wage,
)
from bamengine.typing import Bool1D, Float1D, Int1D

__all__ = ["Scheduler", "HOOK_NAMES"]

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Hook infrastructure                                               #
# ------------------------------------------------------------------ #

# A hook is any callable that receives the ``Scheduler`` instance.
SchedulerHook: TypeAlias = Callable[["Scheduler"], None]

# Central list that defines **all** available hook points and their order.
# Add / remove / reorder entries here and both the `step` method and the
# tests pick the change up automatically.
HOOK_NAMES: tuple[str, ...] = (
    "before_planning",
    "after_planning",
    "before_labor_market",
    "after_labor_market",
    "before_credit_market",
    "after_credit_market",
    "before_production",
    "after_production",
    "after_stub",
)


# --------------------------------------------------------------------- #
#                               Scheduler                               #
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class Scheduler:
    """
    Facade that drives a BAM economy for one or more periods.
    """

    rng: Generator
    ec: Economy
    prod: Producer
    wrk: Worker
    emp: Employer
    bor: Borrower
    lend: Lender
    con: Consumer
    lb: LoanBook

    # global parameters
    h_rho: float  # max production-growth shock
    h_xi: float  # max wage-growth shock
    h_phi: float  # max bank operational costs shock
    max_M: int  # max job applications per unemployed worker
    max_H: int  # max loan applications per firm
    max_Z: int  # max firm visits per consumer
    theta: int  # job contract length

    # --------------------------------------------------------------------- #
    #                            constructor                                #
    # --------------------------------------------------------------------- #
    @classmethod
    def init(
        cls,
        *,
        n_firms: int,
        n_households: int,
        n_banks: int,
        h_rho: float = 0.1,
        h_xi: float = 0.05,
        h_phi: float = 0.1,
        max_M: int = 4,
        max_H: int = 2,
        max_Z: int = 2,
        theta: int = 8,
        seed: int | np.random.Generator = 0,
    ) -> "Scheduler":
        rng = seed if isinstance(seed, Generator) else default_rng(seed)

        # finance vectors
        net_worth = np.full(n_firms, 10.0)
        total_funds = np.copy(net_worth)
        rnd_intensity = np.ones(n_firms)

        # producer vectors
        production = np.ones(n_firms)
        inventory = np.zeros_like(production)
        expected_demand = np.ones_like(production)
        desired_production = np.zeros_like(production)
        labor_productivity = np.ones_like(production)

        price = np.full(n_firms, 1.5)

        # employer vectors
        labor = np.zeros(n_firms, dtype=np.int64)
        desired_labor = np.zeros_like(labor)
        wage_offer = np.ones(n_firms)
        wage_bill = np.zeros_like(wage_offer)
        n_vacancies = np.zeros_like(desired_labor)
        recv_job_apps_head = np.full(n_firms, -1, dtype=np.int64)
        recv_job_apps = np.full((n_firms, max_M), -1, dtype=np.int64)

        # worker vectors
        employed = np.zeros(n_households, dtype=np.bool_)
        employer = np.full(n_households, -1, dtype=np.int64)
        employer_prev = np.full_like(employer, -1)
        periods_left = np.zeros(n_households, dtype=np.int64)
        contract_expired = np.zeros_like(employed)
        fired = np.zeros_like(employed)
        wage = np.zeros(n_households)
        job_apps_head = np.full(n_households, -1, dtype=np.int64)
        job_apps_targets = np.full((n_households, max_M), -1, dtype=np.int64)

        # borrower vectors
        credit_demand = np.zeros_like(net_worth)
        projected_fragility = np.zeros(n_firms)
        loan_apps_head = np.full(n_firms, -1, dtype=np.int64)
        loan_apps_targets = np.full((n_firms, max_H), -1, dtype=np.int64)

        # lender vectors
        equity_base = np.full(n_banks, 10_000.00)
        credit_supply = np.zeros_like(equity_base)
        interest_rate = np.zeros(n_banks)
        recv_loan_apps_head = np.full(n_banks, -1, dtype=np.int64)
        recv_loan_apps = np.full((n_banks, max_H), -1, dtype=np.int64)

        # consumer vectors
        income = np.zeros(n_households)

        # ---------- wrap into components ----------------------------------
        ec = Economy(
            min_wage=1.0,
            min_wage_rev_period=4,
            avg_mkt_price=1.5,
            avg_mkt_price_history=np.array([1.5]),
            r_bar=0.07,
            v=0.23,
        )
        prod = Producer(
            price=price,
            production=production,
            inventory=inventory,
            expected_demand=expected_demand,
            desired_production=desired_production,
            labor_productivity=labor_productivity,
        )
        wrk = Worker(
            employed=employed,
            employer=employer,
            employer_prev=employer_prev,
            wage=wage,
            periods_left=periods_left,
            contract_expired=contract_expired,
            fired=fired,
            job_apps_head=job_apps_head,
            job_apps_targets=job_apps_targets,
        )
        emp = Employer(
            desired_labor=desired_labor,
            current_labor=labor,
            wage_offer=wage_offer,
            wage_bill=wage_bill,
            n_vacancies=n_vacancies,
            total_funds=total_funds,
            recv_job_apps_head=recv_job_apps_head,
            recv_job_apps=recv_job_apps,
        )
        bor = Borrower(
            net_worth=net_worth,
            total_funds=total_funds,
            wage_bill=wage_bill,
            credit_demand=credit_demand,
            rnd_intensity=rnd_intensity,
            projected_fragility=projected_fragility,
            loan_apps_head=loan_apps_head,
            loan_apps_targets=loan_apps_targets,
        )
        lend = Lender(
            equity_base=equity_base,
            credit_supply=credit_supply,
            interest_rate=interest_rate,
            recv_apps_head=recv_loan_apps_head,
            recv_apps=recv_loan_apps,
        )
        lb = LoanBook()
        con = Consumer(
            income=income,
        )

        return cls(
            rng=rng,
            ec=ec,
            prod=prod,
            wrk=wrk,
            emp=emp,
            bor=bor,
            lend=lend,
            lb=lb,
            con=con,
            h_rho=h_rho,
            h_xi=h_xi,
            h_phi=h_phi,
            max_M=max_M,
            max_H=max_H,
            max_Z=max_Z,
            theta=theta,
        )

    # ------------------------------------------------------------------ #
    #                               one step                             #
    # ------------------------------------------------------------------ #
    def step(self, **hooks: SchedulerHook) -> None:
        """Advance the economy by one period.

        Any keyword whose name appears in ``HOOK_NAMES`` may be supplied
        with a callable that is executed at the documented point.
        """

        def _call(name: str) -> None:
            hook = hooks.get(name)
            if hook is not None:
                hook(self)

        # ===== Event 1 – planning ======================================
        _call("before_planning")

        firms_decide_desired_production(
            self.prod, p_avg=self.ec.avg_mkt_price, h_rho=self.h_rho, rng=self.rng
        )
        firms_decide_desired_labor(self.prod, self.emp)
        firms_decide_vacancies(self.emp)

        _call("after_planning")
        # ===============================================================

        # ===== Event 2 – labor-market ==================================
        _call("before_labor_market")

        adjust_minimum_wage(self.ec)
        firms_decide_wage_offer(
            self.emp, w_min=self.ec.min_wage, h_xi=self.h_xi, rng=self.rng
        )
        workers_decide_firms_to_apply(
            self.wrk, self.emp, max_M=self.max_M, rng=self.rng
        )
        for _ in range(self.max_M):
            workers_send_one_round(self.wrk, self.emp)
            firms_hire_workers(self.wrk, self.emp, theta=self.theta)
        firms_calc_wage_bill(self.emp)

        _call("after_labor_market")
        # ===============================================================

        # ===== Event 3 – credit-market =================================
        _call("before_credit_market")

        banks_decide_credit_supply(self.lend, v=self.ec.v)
        banks_decide_interest_rate(
            self.lend, r_bar=self.ec.r_bar, h_phi=self.h_phi, rng=self.rng
        )
        firms_decide_credit_demand(self.bor)
        firms_calc_credit_metrics(self.bor)
        firms_prepare_loan_applications(
            self.bor, self.lend, max_H=self.max_H, rng=self.rng
        )
        for _ in range(self.max_H):
            firms_send_one_loan_app(self.bor, self.lend)
            banks_provide_loans(self.bor, self.lb, self.lend, r_bar=self.ec.r_bar)
        firms_fire_workers(self.emp, self.wrk, rng=self.rng)

        _call("after_credit_market")
        # ===============================================================

        # ===== Event 4 – production ====================================
        _call("before_production")

        firms_pay_wages(self.emp)
        workers_receive_wage(self.con, self.wrk)
        firms_run_production(self.prod, self.emp)

        _call("after_production")
        # ===============================================================

        # Stub state advance – internal deterministic bookkeeping
        import _testing

        _testing.advance_stub_state(self)

        _call("after_stub")

    # --------------------------------------------------------------------- #
    #                               snapshot                                #
    # --------------------------------------------------------------------- #
    def snapshot(
        self, *, copy: bool = True
    ) -> Dict[str, Float1D | Int1D | Bool1D | float]:
        """ "
        Return a read‑only view (or copy) of key state arrays.

        Parameters
        ----------
        copy : bool, default True
            * True  – return **copies** so the caller can mutate freely.
            * False – return **views**; cheap but mutation‑unsafe.
        """
        cp = np.copy if copy else lambda x: x  # cheap inline helper

        return {
            "net_worth": cp(self.bor.net_worth),
            "price": cp(self.prod.price),
            "inventory": cp(self.prod.inventory),
            "labor": cp(self.emp.current_labor),
            "min_wage": float(self.ec.min_wage),
            "wage": cp(self.wrk.wage),
            "wage_bill": cp(self.emp.wage_bill),
            "employed": cp(self.wrk.employed),
            "debt": cp(self.lb.debt),
            "production": cp(self.prod.production),
            "income": cp(self.con.income),
            "avg_mkt_price": float(self.ec.avg_mkt_price),
        }


# tests/helpers/invariants.py
import numpy as np

from bamengine.scheduler import Scheduler


def assert_basic_invariants(sch: Scheduler) -> None:
    """
    Assertions that must hold after *any* ``Scheduler.step()`` call,
    regardless of economy size or number of periods.
    """
    # ------------------------------------------------------------------ #
    #  Planning & labor-market                                          #
    # ------------------------------------------------------------------ #
    # Production plans non-negative & finite
    assert not np.isnan(sch.prod.desired_production).any()
    assert (sch.prod.desired_production >= 0).all()

    # Vacancies never negative
    assert (sch.emp.n_vacancies >= 0).all()

    # Wage offers respect current minimum wage
    assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

    # Employment flags are boolean
    assert set(np.unique(sch.wrk.employed)).issubset({0, 1})

    # Every employed worker counted in current_labor, and vice-versa
    assert sch.wrk.employed.sum() == sch.emp.current_labor.sum()

    # ------------------------------------------------------------------ #
    #  Credit-market                                                     #
    # ------------------------------------------------------------------ #
    # Borrower demand and lender supply never negative
    assert (sch.bor.credit_demand >= -1e-9).all()
    assert (sch.lend.credit_supply >= -1e-9).all()

    # Ledger capacity never exceeded
    assert sch.lb.size <= sch.lb.capacity

    # Ledger indices within component bounds
    n_borrowers = sch.bor.net_worth.size
    n_lenders = sch.lend.credit_supply.size
    assert np.all((sch.lb.borrower[: sch.lb.size] < n_borrowers))
    assert np.all((sch.lb.lender[: sch.lb.size] < n_lenders))

    # Basic ledger algebra:  interest = principal * rate,
    #                        debt     = principal * (1 + rate)
    if sch.lb.size:  # skip empty ledger
        principal = sch.lb.principal[: sch.lb.size]
        rate = sch.lb.rate[: sch.lb.size]
        np.testing.assert_allclose(
            sch.lb.interest[: sch.lb.size], principal * rate, rtol=1e-8
        )
        np.testing.assert_allclose(
            sch.lb.debt[: sch.lb.size], principal * (1.0 + rate), rtol=1e-8
        )

    #  Bank-side sanity: remaining pot never goes negative *and*
    #  never exceeds the initial regulatory cap  (E_k · v).
    cap_per_bank = sch.lend.equity_base * sch.ec.v
    assert np.all(
        (0.0 <= sch.lend.credit_supply)
        & (sch.lend.credit_supply <= cap_per_bank + 1e-9)
    )

```
