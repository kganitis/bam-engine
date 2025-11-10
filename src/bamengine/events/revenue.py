"""Revenue events for collection, debt repayment, and dividends.

Each event encapsulates revenue collection, debt validation and repayment,
and dividend distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class FirmsCollectRevenue:
    """
    Collect revenue from sales and calculate gross profit for firms.

    Rule
    ----
        R            = P · (Y − S)
        gross_profit = R − W
        funds        += R

    R: Revenue, P: Individual Price, Y: Actual Production, S: Inventory, W: Wage Bill
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_collect_revenue

        firms_collect_revenue(sim.prod, sim.bor)


@event
class FirmsValidateDebtCommitments:
    """
    Validate debt commitments and process repayments or write-offs.

    Rule
    ----
        If total_funds ≥ total_debt  →  full repayment (principal+interest).
        Else: proportional write-off up to net-worth
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_validate_debt_commitments

        firms_validate_debt_commitments(sim.bor, sim.lend, sim.lb)


@event
class FirmsPayDividends:
    """
    Calculate dividends and retained earnings, then update firm funds accounts.

    Rule
    ----
        retained = π           ( ≤ 0 case)
                 = π · (1-δ)   ( > 0 case)
        Div      = π · δ       ( > 0 case)

    π: Net Profit, Div: Dividends, δ: Dividend Payout Ratio

    Notes
    -----
    • Cash decreases by dividends paid
    • Net-worth is **not** updated here
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_pay_dividends

        firms_pay_dividends(sim.bor, delta=sim.config.delta)
