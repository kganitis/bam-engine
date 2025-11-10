"""
Labor market events for wage setting, applications, and hiring.

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
    Calculate and store the annual inflation rate for the current period.

    Rule
    ----
        π_t = (P̄_{t} - P̄_{t-4}) / P̄_{t-4}

    t: Current Period, π: Annual Inflation Rate, P̄: Average Market Price
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import calc_annual_inflation_rate

        calc_annual_inflation_rate(sim.ec)


@event
class AdjustMinimumWage:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation π.

    Rule
    ----
        ŵ_t = ŵ_{t-1} · π_t

    t: Current Period, ŵ: Minimum Wage, π: Annual Inflation Rate
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import adjust_minimum_wage

        adjust_minimum_wage(sim.ec)


@event
class FirmsDecideWageOffer:
    """
    Firms with vacancies post a wage offer as a markup over their previous offer.

    Rule
    ----
        shock ~ U(0, h_ξ)
        w_t = max( ŵ_t, w_{t-1} · (1 + shock) )

    t: Current Period, w: Offered Wage, ŵ: Minimum Wage, h_ξ: Max Wage Growth
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
    Unemployed workers choose up to `max_M` firms to apply to, sorted by wage.
    Workers remain loyal to their last employer if their contract has just expired.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import (
            workers_decide_firms_to_apply,
        )

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
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import workers_send_one_round

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
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import firms_hire_workers

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

    Rule
    ----
        W_i = Σ w_j for all j employed by firm i

    W: Wage Bill, w: (Individual) Wage
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.labor_market import firms_calc_wage_bill

        firms_calc_wage_bill(sim.emp, sim.wrk)
