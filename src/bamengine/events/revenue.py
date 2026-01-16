"""
Revenue events for collection, debt repayment, and dividend distribution.

This module defines the revenue phase events that execute after goods market.
Firms collect sales revenue, repay or write off debts, and distribute dividends
to shareholders.

Event Sequence
--------------
The revenue events execute in this order:

1. FirmsCollectRevenue - Calculate revenue from sales and add to funds
2. FirmsValidateDebtCommitments - Repay debts or write off if insufficient funds
3. FirmsPayDividends - Distribute profits as dividends (if positive)

Design Notes
------------
- Events operate on borrower, lender, producer, and loanbook
- Revenue: R = P × (Y - S) where Y - S = units sold
- Gross profit: R - W (revenue minus wage bill)
- Net profit: gross_profit - interest_paid
- Dividend payout: δ × net_profit (if positive), else retain all losses
- Debt write-off: if funds < debt, proportional reduction up to net worth

Examples
--------
Execute revenue events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, seed=42)
>>> # Revenue events run as part of default pipeline
>>> sim.step()

Execute individual revenue event:

>>> event = sim.get_event("firms_collect_revenue")
>>> event.execute(sim)
>>> sim.bor.gross_profit.sum()  # doctest: +SKIP
2450.0

See Also
--------
bamengine.events._internal : System function implementations (in revenue module)
Borrower : Financial state with profits
Producer : Production state with revenue calculation
LoanBook : Debt relationships
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class FirmsCollectRevenue:
    """
    Firms collect revenue from sales and calculate gross profit.

    Revenue is calculated from goods sold (production minus remaining inventory).
    Gross profit is revenue minus wage costs. Funds increase by revenue amount.

    Algorithm
    ---------
    For each firm i:

    1. Calculate units sold: :math:`Q_i = Y_i - S_i`
    2. Calculate revenue: :math:`R_i = P_i \\times Q_i`
    3. Calculate gross profit: :math:`GP_i = R_i - W_i`
    4. Add revenue to funds: :math:`A_i \\leftarrow A_i + R_i`

    Mathematical Notation
    ---------------------
    .. math::
        Q_i = Y_i - S_i

        R_i = P_i \\times Q_i

        GP_i = R_i - W_i

        A_i \\leftarrow A_i + R_i

    where :math:`Y_i` = production, :math:`S_i` = inventory, :math:`P_i` = price, :math:`W_i` = wage_bill, :math:`A_i` = total_funds.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_collect_revenue")
    >>> event.execute(sim)

    Check total revenue:

    >>> total_revenue = sim.bor.gross_profit.sum() + sim.bor.wage_bill.sum()
    >>> total_revenue  # doctest: +SKIP
    5200.0

    Firms with positive sales have positive revenue:

    >>> import numpy as np
    >>> units_sold = sim.prod.production - sim.prod.inventory
    >>> firms_with_sales = units_sold > 0
    >>> revenue = sim.prod.price * units_sold
    >>> (revenue[firms_with_sales] > 0).all()
    True

    Notes
    -----
    This event must execute after all goods market events (need final inventory).

    Gross profit can be negative if wage bill exceeds revenue (operating loss).

    Net profit (calculated later) further subtracts interest payments.

    See Also
    --------
    FirmsPayDividends : Uses net_profit for dividend calculation
    bamengine.events._internal.firms_collect_revenue : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_collect_revenue

        firms_collect_revenue(sim.prod, sim.bor)


@event
class FirmsValidateDebtCommitments:
    """
    Firms repay debts or write off if insufficient funds.

    Firms attempt to repay all outstanding debt (principal + interest). If
    funds are insufficient, debts are proportionally written off up to net worth.
    Banks absorb losses from write-offs.

    Algorithm
    ---------
    For each firm i:

    1. Calculate total debt: :math:`D_i = \\sum \\text{debt}` for all loans to firm i
    2. If :math:`A_i \\geq D_i` (can repay):
       - Pay full debt: :math:`A_i \\leftarrow A_i - D_i`
       - Bank receives payment
       - Remove loans from LoanBook
       - Net profit: :math:`NP_i = GP_i - \\text{interest\\_paid}`
    3. Else (cannot repay):
       - Calculate write-off: :math:`W = \\min(D_i - A_i, \\max(0, A_i))`
       - Proportionally reduce debt by :math:`W`
       - Pay remainder: :math:`A_i \\leftarrow 0`
       - Bank absorbs loss
       - Net profit: :math:`NP_i = GP_i - \\text{interest\\_paid}` (partial)

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> event = sim.get_event("firms_validate_debt_commitments")
    >>> event.execute(sim)

    Check firms that repaid fully:

    >>> import numpy as np
    >>> # Firms with zero debt repaid successfully
    >>> total_debt = sim.lb.debt_per_borrower(n_borrowers=100)
    >>> fully_repaid = total_debt == 0
    >>> fully_repaid.sum()  # doctest: +SKIP
    65

    Check net profit after interest:

    >>> sim.bor.net_profit.sum()  # doctest: +SKIP
    1850.0

    Notes
    -----
    This event must execute after FirmsCollectRevenue (need funds from sales).

    Write-offs reduce bank equity (banks absorb losses from defaults).

    Net profit = gross_profit - interest_paid (after debt service).

    See Also
    --------
    FirmsCollectRevenue : Provides funds for debt repayment
    LoanBook : Stores debt relationships
    bamengine.events._internal.firms_validate_debt_commitments : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_validate_debt_commitments

        firms_validate_debt_commitments(
            sim.bor,
            sim.lend,
            sim.lb,
        )


@event
class FirmsPayDividends:
    """
    Firms distribute dividends from positive profits to households.

    Profitable firms (net_profit > 0) pay dividends. Unprofitable firms retain
    all losses. Dividends are distributed equally to all households, maintaining
    stock-flow consistency in the model.

    Algorithm
    ---------
    For each firm i:

    1. If :math:`NP_i > 0` (profitable):
       - Dividends: :math:`Div_i = \\delta \\times NP_i`
       - Retained: :math:`RP_i = (1 - \\delta) \\times NP_i`
       - Pay dividends: :math:`A_i \\leftarrow A_i - Div_i`
    2. Else (unprofitable):
       - Retained: :math:`RP_i = NP_i` (retain all losses)
       - No dividends paid

    For households:

    3. Total dividends distributed equally: :math:`div_j = \\sum Div_i / N_H`
    4. Household savings increased: :math:`SA_j \\leftarrow SA_j + div_j`

    Mathematical Notation
    ---------------------
    .. math::
        \\text{If } NP_i > 0:

        \\quad Div_i = \\delta \\times NP_i

        \\quad RP_i = (1 - \\delta) \\times NP_i

        \\quad A_i \\leftarrow A_i - Div_i

        \\text{Else:}

        \\quad RP_i = NP_i

        \\text{Dividend distribution to households:}

        \\quad div_j = \\frac{\\sum_i Div_i}{N_H} \\quad \\forall j

        \\quad SA_j \\leftarrow SA_j + div_j

    where :math:`\\delta` = dividend payout ratio (config), :math:`N_H` = number
    of households.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> initial_funds = sim.bor.total_funds.copy()
    >>> initial_savings = sim.cons.savings.copy()
    >>> event = sim.get_event("firms_pay_dividends")
    >>> event.execute(sim)

    Check total dividends paid:

    >>> import numpy as np
    >>> profitable = sim.bor.net_profit > 0
    >>> dividends = sim.config.delta * sim.bor.net_profit[profitable]
    >>> total_dividends = dividends.sum()
    >>> total_dividends  # doctest: +SKIP
    420.0

    Verify funds decreased by dividends:

    >>> funds_decrease = initial_funds - sim.bor.total_funds
    >>> np.allclose(funds_decrease[profitable], dividends)
    True

    Verify household savings increased:

    >>> savings_increase = sim.cons.savings - initial_savings
    >>> np.allclose(savings_increase.sum(), total_dividends)  # doctest: +SKIP
    True

    Notes
    -----
    This event must execute after FirmsValidateDebtCommitments (need net_profit).

    Net worth is NOT updated here (happens in bankruptcy phase).

    Dividend payout ratio δ typically 0.1-0.3 (10-30% of profits).

    **Modeling Note**: Equal distribution of dividends to all households is a
    simplification that avoids introducing a separate "capitalist" role.
    Since all households share the same consumption function based on savings
    ratios, the specific distribution pattern does not meaningfully affect
    aggregate consumption dynamics. What matters for model validity is stock-flow
    consistency: dividends debited from firms are credited to households.

    See Also
    --------
    FirmsValidateDebtCommitments : Calculates net_profit
    bamengine.events._internal.firms_pay_dividends : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal import firms_pay_dividends

        firms_pay_dividends(sim.bor, sim.con, delta=sim.config.delta)
