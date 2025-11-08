"""Credit market events for credit supply, demand, and loan provision.

This module contains Event classes that wrap credit market system functions.
Each event encapsulates credit decisions, loan applications, and related
credit market operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bamengine.core import Event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class BanksDecideCreditSupply(Event):
    """
    Banks decide total credit supply based on equity and leverage ratio.

    Credit supply is calculated as:
        S_i = v · E_i

    where v is the bank's leverage ratio and E_i is equity.

    This event wraps `bamengine.systems.credit_market.banks_decide_credit_supply`.

    Dependencies
    ------------
    None (first event in credit market phase)

    See Also
    --------
    bamengine.systems.credit_market.banks_decide_credit_supply : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit supply decision."""
        from bamengine.systems.credit_market import banks_decide_credit_supply

        banks_decide_credit_supply(sim.lend, v=sim.ec.v)


@dataclass(slots=True)
class BanksDecideInterestRate(Event):
    """
    Banks set interest rates as markup over base rate with random shock.

    Interest rate rule:
        shock ~ U(-h_φ, h_φ)
        r_t = r̄ · (1 + shock)

    where r̄ is the base interest rate.

    This event wraps `bamengine.systems.credit_market.banks_decide_interest_rate`.

    Dependencies
    ------------
    - banks_decide_credit_supply : Credit supply decision

    See Also
    --------
    bamengine.systems.credit_market.banks_decide_interest_rate : Underlying logic
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


@dataclass(slots=True)
class FirmsDecideCreditDemand(Event):
    """
    Firms decide desired credit based on funding needs.

    Credit demand covers production costs:
        D_i = max(wage_bill_i - NW_i, 0)

    This event wraps `bamengine.systems.credit_market.firms_decide_credit_demand`.

    Dependencies
    ------------
    - firms_calc_wage_bill : Uses wage bill for credit calculation

    See Also
    --------
    bamengine.systems.credit_market.firms_decide_credit_demand : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit demand decision."""
        from bamengine.systems.credit_market import firms_decide_credit_demand

        firms_decide_credit_demand(sim.bor)


@dataclass(slots=True)
class FirmsCalcCreditMetrics(Event):
    """
    Firms calculate credit metrics for loan applications.

    Calculates metrics such as leverage ratio, creditworthiness indicators,
    and other measures used by banks to evaluate loan applications.

    This event wraps `bamengine.systems.credit_market.firms_calc_credit_metrics`.

    Dependencies
    ------------
    - firms_decide_credit_demand : Uses credit demand

    See Also
    --------
    bamengine.systems.credit_market.firms_calc_credit_metrics : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute credit metrics calculation."""
        from bamengine.systems.credit_market import firms_calc_credit_metrics

        firms_calc_credit_metrics(sim.bor)


@dataclass(slots=True)
class FirmsPrepareLoanApplications(Event):
    """
    Firms select banks to apply to for loans.

    Firms with positive credit demand choose up to max_H banks to apply to,
    ranked by interest rate (ascending - prefer lower rates).

    This event wraps `bamengine.systems.credit_market.firms_prepare_loan_applications`.

    Dependencies
    ------------
    - firms_calc_credit_metrics : Credit metrics calculated
    - banks_decide_interest_rate : Interest rates available

    See Also
    --------
    bamengine.systems.credit_market.firms_prepare_loan_applications : Underlying logic
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


@dataclass(slots=True)
class FirmsSendOneLoanApp(Event):
    """
    Firms send one round of loan applications to banks.

    Each firm with credit demand sends one application from their queue to
    the corresponding bank. Applications are dropped if the bank's queue is
    full or credit supply is exhausted.

    Note: This event is typically repeated max_H times in the pipeline.

    This event wraps `bamengine.systems.credit_market.firms_send_one_loan_app`.

    Dependencies
    ------------
    - firms_prepare_loan_applications : Uses loan application queues

    See Also
    --------
    bamengine.systems.credit_market.firms_send_one_loan_app : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute one loan application sending round."""
        from bamengine.systems.credit_market import firms_send_one_loan_app

        firms_send_one_loan_app(sim.bor, sim.lend, rng=sim.rng)


@dataclass(slots=True)
class BanksProvideLoans(Event):
    """
    Banks process loan applications and provide credit.

    Each bank evaluates applications based on net worth, provides loans up to
    available credit supply, and updates the loan book with new loans.

    Note: This event is typically repeated max_H times in the pipeline,
    alternating with FirmsSendOneLoanApp.

    This event wraps `bamengine.systems.credit_market.banks_provide_loans`.

    Dependencies
    ------------
    - firms_send_one_loan_app : Processes applications sent in current round

    See Also
    --------
    bamengine.systems.credit_market.banks_provide_loans : Underlying logic
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


@dataclass(slots=True)
class FirmsFireWorkers(Event):
    """
    Firms that failed to obtain sufficient credit fire workers.

    Firms with unfunded wage bill lay off the most expensive workers first
    until wage commitments are within available funds.

    This event wraps `bamengine.systems.credit_market.firms_fire_workers`.

    Dependencies
    ------------
    - banks_provide_loans : Uses final credit provision state

    See Also
    --------
    bamengine.systems.credit_market.firms_fire_workers : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute worker firing decision."""
        from bamengine.systems.credit_market import firms_fire_workers

        firms_fire_workers(sim.emp, sim.wrk, rng=sim.rng)
