"""Labor market events for wage setting, applications, and hiring.

This module contains Event classes that wrap labor market system functions.
Each event encapsulates wage setting, job applications, hiring, and related
labor market operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class CalcAnnualInflationRate:
    """
    Calculate annual inflation rate based on price history.

    Inflation is calculated as:
        π_t = (P̄_t - P̄_{t-4}) / P̄_{t-4}

    where P̄ is the average market price. Requires at least 5 periods of history.

    This event wraps `bamengine.systems.labor_market.calc_annual_inflation_rate`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute inflation calculation."""
        from bamengine.systems.labor_market import calc_annual_inflation_rate

        calc_annual_inflation_rate(sim.ec)


@event
class AdjustMinimumWage:
    """
    Adjust minimum wage based on realized inflation.

    Every `min_wage_rev_period` periods, the minimum wage is updated:
        ŵ_t = ŵ_{t-1} · (1 + π_t)

    where π_t is the most recent annual inflation rate.

    This event wraps `bamengine.systems.labor_market.adjust_minimum_wage`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute minimum wage adjustment."""
        from bamengine.systems.labor_market import adjust_minimum_wage

        adjust_minimum_wage(sim.ec)


@event
class FirmsDecideWageOffer:
    """
    Firms with vacancies set wage offers as markup over previous wage.

    Wage offer rule:
        shock ~ U(0, h_ξ)
        w_t = max(ŵ_t, w_{t-1} · (1 + shock))

    where ŵ is the minimum wage and h_ξ is the maximum wage shock parameter.
    Firms with zero vacancies keep their previous wage offer unchanged.

    This event wraps `bamengine.systems.labor_market.firms_decide_wage_offer`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute wage offer decision."""
        from bamengine.systems.labor_market import firms_decide_wage_offer

        firms_decide_wage_offer(
            sim.emp,
            w_min=sim.ec.min_wage,
            h_xi=sim.config.h_xi,
            rng=sim.rng,
        )


@event
class WorkersDecideFirmsToApply:
    """
    Unemployed workers choose firms to apply to, sorted by wage offer.

    Workers select up to max_M firms with open vacancies, ranked by wage offer
    (descending). Workers remain loyal to their previous employer if their
    contract expired naturally (not fired) and that employer is hiring.

    This event wraps `bamengine.systems.labor_market.workers_decide_firms_to_apply`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute worker application decision."""
        from bamengine.systems.labor_market import workers_decide_firms_to_apply

        workers_decide_firms_to_apply(
            wrk=sim.wrk,
            emp=sim.emp,
            max_M=sim.config.max_M,
            rng=sim.rng,
        )


@event
class WorkersSendOneRound:
    """
    Workers send one round of job applications to firms.

    Each unemployed worker sends one application from their queue to the
    corresponding firm. Applications are dropped if the firm's queue is full
    or if vacancies are exhausted.

    Note: This event is typically repeated max_M times in the pipeline to
    process all applications.

    This event wraps `bamengine.systems.labor_market.workers_send_one_round`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one application sending round."""
        from bamengine.systems.labor_market import workers_send_one_round

        workers_send_one_round(sim.wrk, sim.emp, rng=sim.rng)


@event
class FirmsHireWorkers:
    """
    Firms process applications and hire workers.

    Each firm with vacancies processes its application queue, hires up to the
    number of available vacancies, and updates worker state (employment,
    wage, contract duration).

    Note: This event is typically repeated max_M times in the pipeline,
    alternating with WorkersSendOneRound.

    This event wraps `bamengine.systems.labor_market.firms_hire_workers`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute hiring decision."""
        from bamengine.systems.labor_market import firms_hire_workers

        firms_hire_workers(
            wrk=sim.wrk,
            emp=sim.emp,
            theta=sim.config.theta,
            rng=sim.rng,
        )


@event
class FirmsCalcWageBill:
    """
    Firms calculate total wage bill based on currently employed workers.

    Wage bill is calculated as:
        W_i = Σ w_j for all j employed by firm i

    where w_j is the individual wage of worker j.

    This event wraps `bamengine.systems.labor_market.firms_calc_wage_bill`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute wage bill calculation."""
        from bamengine.systems.labor_market import firms_calc_wage_bill

        firms_calc_wage_bill(sim.emp, sim.wrk)
