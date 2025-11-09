"""Credit market events for credit supply, demand, and loan provision.

This module contains Event classes that wrap credit market system functions.
Each event encapsulates credit decisions, loan applications, and related
credit market operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class BanksDecideCreditSupply:
    """
    Banks decide total credit supply based on equity and leverage ratio.

    Credit supply is calculated as:
        S_i = v · E_i

    where v is the bank's leverage ratio and E_i is equity.

    This event wraps `bamengine.systems.credit_market.banks_decide_credit_supply`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit supply decision."""
        from bamengine.systems.credit_market import banks_decide_credit_supply

        banks_decide_credit_supply(sim.lend, v=sim.ec.v)


@event
class BanksDecideInterestRate:
    """
    Banks set interest rates as markup over base rate with random shock.

    Interest rate rule:
        shock ~ U(-h_φ, h_φ)
        r_t = r̄ · (1 + shock)

    where r̄ is the base interest rate.

    This event wraps `bamengine.systems.credit_market.banks_decide_interest_rate`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute interest rate decision."""
        from bamengine.systems.credit_market import banks_decide_interest_rate

        banks_decide_interest_rate(
            sim.lend,
            r_bar=sim.ec.r_bar,
            h_phi=sim.config.h_phi,
            rng=sim.rng,
        )


@event
class FirmsDecideCreditDemand:
    """
    Firms decide desired credit based on funding needs.

    Credit demand covers production costs:
        D_i = max(wage_bill_i - NW_i, 0)

    This event wraps `bamengine.systems.credit_market.firms_decide_credit_demand`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit demand decision."""
        from bamengine.systems.credit_market import firms_decide_credit_demand

        firms_decide_credit_demand(sim.bor)


@event
class FirmsCalcCreditMetrics:
    """
    Firms calculate credit metrics for loan applications.

    Calculates metrics such as leverage ratio, creditworthiness indicators,
    and other measures used by banks to evaluate loan applications.

    This event wraps `bamengine.systems.credit_market.firms_calc_credit_metrics`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit metrics calculation."""
        from bamengine.systems.credit_market import firms_calc_credit_metrics

        firms_calc_credit_metrics(sim.bor)


@event
class FirmsPrepareLoanApplications:
    """
    Firms select banks to apply to for loans.

    Firms with positive credit demand choose up to max_H banks to apply to,
    ranked by interest rate (ascending - prefer lower rates).

    This event wraps `bamengine.systems.credit_market.firms_prepare_loan_applications`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute loan application preparation."""
        from bamengine.systems.credit_market import firms_prepare_loan_applications

        firms_prepare_loan_applications(
            sim.bor,
            sim.lend,
            max_H=sim.config.max_H,
            rng=sim.rng,
        )


@event
class FirmsSendOneLoanApp:
    """
    Firms send one round of loan applications to banks.

    Each firm with credit demand sends one application from their queue to
    the corresponding bank. Applications are dropped if the bank's queue is
    full or credit supply is exhausted.

    Note: This event is typically repeated max_H times in the pipeline.

    This event wraps `bamengine.systems.credit_market.firms_send_one_loan_app`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one loan application sending round."""
        from bamengine.systems.credit_market import firms_send_one_loan_app

        firms_send_one_loan_app(sim.bor, sim.lend, rng=sim.rng)


@event
class BanksProvideLoans:
    """
    Banks process loan applications and provide credit.

    Each bank evaluates applications based on net worth, provides loans up to
    available credit supply, and updates the loan book with new loans.

    Note: This event is typically repeated max_H times in the pipeline,
    alternating with FirmsSendOneLoanApp.

    This event wraps `bamengine.systems.credit_market.banks_provide_loans`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute loan provision."""
        from bamengine.systems.credit_market import banks_provide_loans

        banks_provide_loans(
            sim.bor,
            sim.lb,
            sim.lend,
            r_bar=sim.ec.r_bar,
            h_phi=sim.config.h_phi,
        )


@event
class FirmsFireWorkers:
    """
    Firms that failed to obtain sufficient credit fire workers.

    Firms with unfunded wage bill lay off the most expensive workers first
    until wage commitments are within available funds.

    This event wraps `bamengine.systems.credit_market.firms_fire_workers`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute worker firing decision."""
        from bamengine.systems.credit_market import firms_fire_workers

        firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)
