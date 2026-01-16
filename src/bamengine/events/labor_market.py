"""
Labor market events for wage setting, applications, and hiring.

This module defines the labor market phase events that execute after planning.
Firms post wage offers, unemployed workers apply to firms, and firms hire
workers through a sequential matching process.

Event Sequence
--------------
The labor market events execute in this order:

1. CalcAnnualInflationRate - Calculate year-over-year inflation rate
2. AdjustMinimumWage - Update minimum wage based on inflation (periodic)
3. FirmsDecideWageOffer - Firms post wage offers with random markup
4. WorkersDecideFirmsToApply - Unemployed workers select firms to apply to
5. WorkersSendOneRound ↔ FirmsHireWorkers - Interleaved application/hiring rounds (max_M times)
6. FirmsCalcWageBill - Calculate total wage bill from employed workers

The send/hire rounds are interleaved to simulate sequential matching: workers
send one application, firms process it and hire if possible, then the next
round begins. This repeats max_M times to process all applications.

Design Notes
------------
- Events operate on employer and worker roles (Employer, Worker)
- Economy-level state (min_wage, inflation) updated by CalcAnnualInflationRate/AdjustMinimumWage
- Loyalty rule: workers whose contracts expired (not fired) apply to previous employer first
- Wage offers constrained by minimum wage floor
- Contract duration: θ + Poisson(λ=10) periods

Examples
--------
Execute labor market events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
>>> # Labor market events run as part of default pipeline
>>> sim.step()

Execute individual labor market event:

>>> event = sim.get_event("firms_decide_wage_offer")
>>> event.execute(sim)
>>> sim.emp.wage_offer.mean()  # doctest: +SKIP
1.15

Check unemployment after hiring:

>>> employed_count = sim.wrk.employed.sum()
>>> unemployment_rate = 1.0 - (employed_count / 500)
>>> unemployment_rate  # doctest: +SKIP
0.04

See Also
--------
bamengine.events._internal.labor_market : System function implementations
Employer : Labor hiring state
Worker : Employment state
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class CalcAnnualInflationRate:
    """
    Calculate and store the annual inflation rate for the current period.

    The inflation rate measures the year-over-year change in the average market
    price level. This is used by AdjustMinimumWage to index the minimum wage to
    inflation. Requires at least 5 periods of price history.

    Algorithm
    ---------
    1. Check if price history has at least 5 periods (:math:`t \\geq 4`)
    2. If insufficient history, set :math:`\\pi_t = 0` and skip
    3. Otherwise, calculate: :math:`\\pi_t = (\\bar{P}_t - \\bar{P}_{t-4}) / \\bar{P}_{t-4}`
    4. Append :math:`\\pi_t` to inflation history

    Mathematical Notation
    ---------------------
    .. math::
        \\pi_t = \\frac{\\bar{P}_t - \\bar{P}_{t-4}}{\\bar{P}_{t-4}}

    where:

    - :math:`\\pi_t`: annual inflation rate at period t
    - :math:`\\bar{P}_t`: average market price at period t
    - :math:`\\bar{P}_{t-4}`: average market price 4 periods ago (year-over-year)

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("calc_annual_inflation_rate")
    >>> event.execute(sim)

    Check inflation history:

    >>> # Need at least 5 periods for non-zero inflation
    >>> for _ in range(5):
    ...     sim.step()
    >>> sim.ec.inflation_history[-1]  # doctest: +SKIP
    0.023

    Inflation requires 5 periods of history:

    >>> sim = be.Simulation.init(n_firms=10, seed=42)
    >>> event = sim.get_event("calc_annual_inflation_rate")
    >>> event.execute(sim)
    >>> sim.ec.inflation_history[-1]
    0.0

    Notes
    -----
    This event must execute before AdjustMinimumWage in each period.

    During the first 4 periods (t < 4), inflation is set to 0.0 since there is
    insufficient history for year-over-year calculation.

    The inflation rate is stored in Economy.inflation_history for later use by
    minimum wage adjustment.

    See Also
    --------
    AdjustMinimumWage : Uses inflation to update minimum wage
    Economy : Global economy state with price/inflation history
    bamengine.events._internal.labor_market.calc_annual_inflation_rate : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import calc_annual_inflation_rate

        calc_annual_inflation_rate(sim.ec)


@event
class AdjustMinimumWage:
    """
    Periodically update the minimum wage based on realized inflation.

    The minimum wage is indexed to inflation to maintain real purchasing power.
    Updates occur every `min_wage_rev_period` periods (e.g., every 4 periods
    for annual revision). Only updates after sufficient price history exists.

    Algorithm
    ---------
    1. Check if current period is a revision period: :math:`(t+1) \\mod \\text{min\\_wage\\_rev\\_period} = 0`
    2. Check if sufficient price history exists: :math:`t > \\text{min\\_wage\\_rev\\_period}`
    3. If both conditions met:
       - Retrieve most recent inflation rate: :math:`\\pi_t = \\text{inflation\\_history}[-1]`
       - Update minimum wage: :math:`\\hat{w}_t = \\hat{w}_{t-1} \\times (1 + \\pi_t)`
    4. Otherwise, skip revision (:math:`\\hat{w}_t = \\hat{w}_{t-1}`)

    Mathematical Notation
    ---------------------
    .. math::
        \\hat{w}_t = \\hat{w}_{t-1} \\times (1 + \\pi_t)

    where:

    - :math:`\\hat{w}_t`: minimum wage at period t
    - :math:`\\pi_t`: annual inflation rate
    - Revision occurs only when: :math:`(t+1) \\mod M = 0`, where :math:`M` = min_wage_rev_period

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("adjust_minimum_wage")
    >>> event.execute(sim)

    Check minimum wage after revision period:

    >>> sim = be.Simulation.init(n_firms=10, seed=42, min_wage_rev_period=4)
    >>> initial_wage = sim.ec.min_wage
    >>> # Run 5 periods to trigger first revision
    >>> for _ in range(5):
    ...     sim.step()
    >>> sim.ec.min_wage >= initial_wage  # Should increase with positive inflation
    True

    Minimum wage unchanged before revision period:

    >>> sim = be.Simulation.init(n_firms=10, seed=42, min_wage_rev_period=4)
    >>> initial_wage = sim.ec.min_wage
    >>> sim.step()
    >>> sim.ec.min_wage == initial_wage
    True

    Notes
    -----
    This event must execute after CalcAnnualInflationRate in each period.

    The revision only occurs on specific periods determined by min_wage_rev_period
    (e.g., if min_wage_rev_period=4, revisions occur at t=4, 8, 12, ...).

    If inflation is negative (deflation), the minimum wage will decrease accordingly.

    See Also
    --------
    CalcAnnualInflationRate : Calculates inflation rate used for adjustment
    FirmsDecideWageOffer : Wage offers must satisfy minimum wage constraint
    Economy : Global economy state with min_wage field
    bamengine.events._internal.labor_market.adjust_minimum_wage : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import adjust_minimum_wage

        adjust_minimum_wage(sim.ec, sim.wrk)


@event
class FirmsDecideWageOffer:
    """
    Firms with vacancies post wage offers with random markup over previous offer.

    Firms that have open vacancies increase their wage offers to attract workers.
    The wage increase is a random shock, and the final offer must satisfy the
    minimum wage constraint. Firms without vacancies leave their wage unchanged.

    Algorithm
    ---------
    For each firm i:

    1. Generate wage shock: :math:`\\varepsilon_i \\sim U(0, h_\\xi)`
    2. If firm has vacancies (:math:`V_i > 0`):
       - Apply markup: :math:`w'_i = w_{i,t-1} \\times (1 + \\varepsilon_i)`
       - Enforce floor: :math:`w_i = \\max(w'_i, \\hat{w}_t)`
    3. Otherwise: :math:`w_i = w_{i,t-1}` (unchanged)

    Mathematical Notation
    ---------------------
    .. math::
        w_{i,t} = \\begin{cases}
            \\max(\\hat{w}_t, w_{i,t-1} \\times (1 + \\varepsilon_i)) & \\text{if } V_i > 0 \\\\
            w_{i,t-1} & \\text{otherwise}
        \\end{cases}

    where:

    - :math:`w_i`: wage offer by firm i
    - :math:`\\hat{w}_t`: minimum wage at period t
    - :math:`\\varepsilon_i`: wage shock :math:`\\sim U(0, h_\\xi)`
    - :math:`h_\\xi`: maximum wage growth rate parameter (config)
    - :math:`V_i`: number of vacancies at firm i

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_decide_wage_offer")
    >>> event.execute(sim)

    Check wage offers satisfy minimum wage:

    >>> (sim.emp.wage_offer >= sim.ec.min_wage).all()
    True

    Find firms with vacancies:

    >>> import numpy as np
    >>> has_vacancies = sim.emp.n_vacancies > 0
    >>> has_vacancies.sum()  # doctest: +SKIP
    25

    Average wage offer from hiring firms:

    >>> hiring_mask = sim.emp.n_vacancies > 0
    >>> sim.emp.wage_offer[hiring_mask].mean()  # doctest: +SKIP
    1.15

    Notes
    -----
    This event must execute after FirmsDecideVacancies and AdjustMinimumWage.

    Only firms with vacancies adjust their wage offers. Firms without vacancies
    retain their previous wage offer (though it may not be used until they post
    vacancies again).

    The wage shock prevents firms from settling into static wage levels and
    introduces competition for workers.

    See Also
    --------
    AdjustMinimumWage : Updates minimum wage floor
    FirmsDecideVacancies : Determines which firms have open positions
    WorkersDecideFirmsToApply : Workers select firms based on wage offers
    bamengine.events._internal.labor_market.firms_decide_wage_offer : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import firms_decide_wage_offer

        firms_decide_wage_offer(
            sim.emp,
            w_min=sim.ec.min_wage,
            h_xi=sim.config.h_xi,
            rng=sim.rng,
        )


@event
class WorkersDecideFirmsToApply:
    """
    Unemployed workers choose up to max_M firms to apply to, sorted by wage.

    Unemployed workers build an application queue by sampling firms and sorting
    them by wage offer (descending). Workers apply the loyalty rule: if their
    contract expired (not fired), they prioritize their previous employer.

    The ``job_search_method`` config parameter controls which firms are sampled:

    - ``"vacancies_only"`` (default): Sample only from firms with open vacancies.
    - ``"all_firms"``: Sample from ALL firms. Applications to firms without
      vacancies are "wasted" (the firm simply doesn't hire).

    Algorithm
    ---------
    For each unemployed worker j:

    1. Sample min(max_M, n_firms) firms randomly from eligible pool
    2. Sort sampled firms by wage offer (descending)
    3. Apply loyalty rule:
       - If worker's contract expired (not fired) AND previous employer is hiring:
         Move previous employer to position 0 (top priority)
    4. Store sorted application queue in worker's buffer
    5. Reset contract_expired and fired flags

    Mathematical Notation
    ---------------------
    Let :math:`H` be the set of eligible firms:

    - If ``job_search_method="vacancies_only"``: :math:`H = \\{i : V_i > 0\\}` (firms with vacancies)
    - If ``job_search_method="all_firms"``: :math:`H = \\{1, ..., N\\}` (all firms)

    For unemployed worker j:

    .. math::
        \\text{Sample}_j \\sim \\text{Random}(H, k=\\min(M, |H|), \\text{replace}=False)

    Then sort by wage:

    .. math::
        \\text{Queue}_j = \\text{argsort}_{\\text{desc}}(w_i \\text{ for } i \\in \\text{Sample}_j)

    If loyalty applies (contract expired, not fired, previous employer hiring):

    .. math::
        \\text{Queue}_j[0] = \\text{employer\\_prev}_j

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> event = sim.get_event("workers_decide_firms_to_apply")
    >>> event.execute(sim)

    Check unemployed workers prepared applications:

    >>> import numpy as np
    >>> unemployed_mask = ~sim.wrk.employed
    >>> unemployed_mask.sum()  # doctest: +SKIP
    20

    Inspect application queue for unemployed worker:

    >>> unemployed_ids = np.where(~sim.wrk.employed)[0]
    >>> if len(unemployed_ids) > 0:
    ...     worker_id = unemployed_ids[0]
    ...     targets = sim.wrk.job_apps_targets[worker_id]
    ...     # First 3 targets (may include -1 if fewer than max_M firms hiring)
    ...     targets[:3]  # doctest: +SKIP
    array([45, 23, 67])

    Check loyalty rule application:

    >>> # Workers with expired contracts should have previous employer at position 0
    >>> # (if previous employer is hiring)
    >>> expired_mask = (sim.wrk.contract_expired == 1) & (~sim.wrk.employed)
    >>> expired_mask.sum()  # doctest: +SKIP
    5

    Notes
    -----
    This event must execute after FirmsDecideWageOffer (need wage offers for sorting).

    Only unemployed workers prepare applications. Employed workers are skipped.

    The loyalty rule implements realistic worker behavior: workers whose contracts
    expired naturally (not fired) prefer to stay with their previous employer if
    possible.

    Workers sample firms randomly then sort by wage. This means workers may miss
    the highest-wage firms if they are not in the random sample. The max_M parameter
    controls how many firms each worker can apply to.

    See Also
    --------
    FirmsDecideWageOffer : Determines wage offers used for sorting
    WorkersSendOneRound : Processes applications from queue
    Worker : Employment state with application queue
    bamengine.events._internal.labor_market.workers_decide_firms_to_apply : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import (
            workers_decide_firms_to_apply,
        )

        workers_decide_firms_to_apply(
            wrk=sim.wrk,
            emp=sim.emp,
            max_M=sim.config.max_M,
            job_search_method=sim.config.job_search_method,
            rng=sim.rng,
        )


@event
class WorkersSendOneRound:
    """
    Workers send one round of job applications to firms.

    Each unemployed worker sends one application from their queue to the
    corresponding firm. Applications may be dropped if the firm's application
    queue is full. This event is repeated max_M times in the pipeline,
    interleaved with FirmsHireWorkers, to process all applications sequentially.

    Algorithm
    ---------
    For each unemployed worker j:

    1. Check if worker has applications remaining in queue (head pointer >= 0)
    2. Pop next target firm from worker's application queue
    3. Check if target firm still has vacancies (:math:`V_i > 0`)
    4. If yes, append worker ID to firm's application queue
    5. Advance worker's queue pointer

    Mathematical Notation
    ---------------------
    Sequential matching process over R rounds (:math:`R` = max_M):

    .. math::
        \\text{Round } r: \\text{ For each unemployed } j, \\text{ send to firm } \\text{Queue}_j[r]

    Applications processed: :math:`A_i(r) = \\{j : \\text{Queue}_j[r] = i \\text{ and } V_i > 0\\}`

    Examples
    --------
    Execute this event (typically in loop with FirmsHireWorkers):

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # First prepare applications
    >>> event_prepare = sim.get_event("workers_decide_firms_to_apply")
    >>> event_prepare.execute(sim)
    >>> # Now process one round
    >>> event_send = sim.get_event("workers_send_one_round")
    >>> event_send.execute(sim)

    Process all application rounds:

    >>> max_M = sim.config.max_M
    >>> for _ in range(max_M):
    ...     sim.get_event("workers_send_one_round")().execute(sim)
    ...     sim.get_event("firms_hire_workers")().execute(sim)

    Notes
    -----
    This event must execute after WorkersDecideFirmsToApply (need application queues).

    This event is typically repeated max_M times, interleaved with FirmsHireWorkers,
    to simulate sequential matching where workers send one application at a time
    and firms can hire immediately.

    Applications may be dropped silently if:
    - Target firm's application queue is full (unlikely with large buffers)
    - Target firm has no vacancies remaining (filled by earlier hires)

    The interleaved send/hire pattern prevents workers from "holding" multiple
    offers simultaneously.

    See Also
    --------
    WorkersDecideFirmsToApply : Prepares application queues
    FirmsHireWorkers : Processes applications and hires workers
    Worker : Employment state with application queue
    bamengine.events._internal.labor_market.workers_send_one_round : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import workers_send_one_round

        workers_send_one_round(
            sim.wrk, sim.emp, rng=sim.rng, matching_method=sim.config.matching_method
        )


@event
class FirmsHireWorkers:
    """
    Firms process applications and hire workers.

    Each firm with vacancies processes its application queue, hires up to the
    number of available vacancies, and updates worker state (employment, wage,
    contract duration). This event is repeated max_M times in the pipeline,
    interleaved with WorkersSendOneRound, to process all application rounds.

    Algorithm
    ---------
    For each firm i with vacancies (V_i > 0):

    1. Pop one application from firm's application queue
    2. Check if worker is still unemployed (may have been hired by another firm)
    3. If worker unemployed:
       - Set worker's employer = firm ID
       - Set worker's wage = firm's wage offer
       - Generate contract duration: θ + Poisson(λ=10)
       - Set worker's periods_left = contract duration
       - Increment firm's current_labor count
       - Decrement firm's vacancies count
    4. If no vacancies remain, clear firm's application queue

    Mathematical Notation
    ---------------------
    For firm i hiring worker j:

    .. math::
        \\text{employer}_j \\leftarrow i

        \\text{wage}_j \\leftarrow w_i

        \\text{periods\\_left}_j \\leftarrow \\theta + \\text{Poisson}(\\lambda=10)

        L_i \\leftarrow L_i + 1

        V_i \\leftarrow V_i - 1

    where:

    - :math:`L_i`: current labor force at firm i
    - :math:`V_i`: remaining vacancies at firm i
    - :math:`w_i`: wage offer by firm i
    - :math:`\\theta`: minimum contract duration (config parameter)

    Examples
    --------
    Execute this event (typically in loop with WorkersSendOneRound):

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # Prepare applications
    >>> sim.get_event("workers_decide_firms_to_apply")().execute(sim)
    >>> # Process one round
    >>> sim.get_event("workers_send_one_round")().execute(sim)
    >>> sim.get_event("firms_hire_workers")().execute(sim)

    Check hired workers:

    >>> import numpy as np
    >>> employed_count = sim.wrk.employed.sum()
    >>> employed_count  # doctest: +SKIP
    480

    Verify contract durations:

    >>> employed_mask = sim.wrk.employed
    >>> contract_lengths = sim.wrk.periods_left[employed_mask]
    >>> (contract_lengths >= sim.config.theta).all()
    True

    Check labor counts match:

    >>> # Worker count should match firm labor count
    >>> worker_counts = np.bincount(sim.wrk.employer[sim.wrk.employed], minlength=100)
    >>> np.array_equal(worker_counts, sim.emp.current_labor)
    True

    Notes
    -----
    This event must execute after WorkersSendOneRound (need applications to process).

    This event is typically repeated max_M times, interleaved with WorkersSendOneRound,
    to simulate sequential matching where firms can hire immediately after receiving
    applications.

    Contract duration is stochastic: θ + Poisson(λ=10), where θ is the minimum
    contract duration. This introduces heterogeneity in contract lengths.

    Firms hire workers on a first-come, first-served basis from their application
    queue. Workers who arrive later may find vacancies already filled.

    See Also
    --------
    WorkersSendOneRound : Sends applications to firms
    WorkersDecideFirmsToApply : Prepares application queues
    Worker : Employment state updated by hiring
    Employer : Labor force state updated by hiring
    bamengine.events._internal.labor_market.firms_hire_workers : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import firms_hire_workers

        firms_hire_workers(
            wrk=sim.wrk,
            emp=sim.emp,
            theta=sim.config.theta,
            contract_poisson_mean=sim.config.contract_poisson_mean,
            matching_method=sim.config.matching_method,
            rng=sim.rng,
        )


@event
class FirmsCalcWageBill:
    """
    Firms calculate total wage bill based on currently employed workers.

    The wage bill is the sum of wages for all workers employed by the firm.
    This is used by FirmsCalcBreakevenPrice to determine production costs.

    Algorithm
    ---------
    For each firm i:

    1. Find all workers employed by firm i: :math:`E_i = \\{j : \\text{employer}_j = i\\}`
    2. Sum wages: :math:`W_i = \\sum_{j \\in E_i} w_j`
    3. Store in firm's wage_bill field

    Uses vectorized aggregation via np.bincount for efficiency.

    Mathematical Notation
    ---------------------
    .. math::
        W_i = \\sum_{j \\in E_i} w_j

    where:

    - :math:`W_i`: total wage bill for firm i
    - :math:`E_i`: set of workers employed by firm i
    - :math:`w_j`: wage of worker j

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> event = sim.get_event("firms_calc_wage_bill")
    >>> event.execute(sim)

    Check total wage bill:

    >>> sim.emp.wage_bill.sum()  # doctest: +SKIP
    552.5

    Verify wage bill matches sum of worker wages:

    >>> import numpy as np
    >>> employed_mask = sim.wrk.employed
    >>> total_wages = sim.wrk.wage[employed_mask].sum()
    >>> total_wage_bill = sim.emp.wage_bill.sum()
    >>> abs(total_wages - total_wage_bill) < 1e-10  # Allow floating point tolerance
    True

    Check firms with no employees have zero wage bill:

    >>> no_labor_mask = sim.emp.current_labor == 0
    >>> (sim.emp.wage_bill[no_labor_mask] == 0).all()
    True

    Notes
    -----
    This event must execute after all FirmsHireWorkers rounds complete.

    The wage bill represents the total labor cost that will be paid in the
    production phase (FirmsPayWages event).

    The wage bill is shared with Borrower role (same underlying array) for
    memory efficiency and consistency with credit demand calculations.

    See Also
    --------
    FirmsHireWorkers : Hires workers and sets their wages
    FirmsCalcBreakevenPrice : Uses wage bill to calculate production costs
    FirmsPayWages : Pays wages based on wage_bill
    Employer : Labor hiring state with wage_bill field
    bamengine.events._internal.labor_market.firms_calc_wage_bill : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import firms_calc_wage_bill

        firms_calc_wage_bill(sim.emp, sim.wrk)
