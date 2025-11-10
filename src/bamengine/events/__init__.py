"""Event classes for BAM Engine ECS architecture.

This module contains Event classes that wrap existing system functions.
Events are auto-registered via __init_subclass__ hook and can be
composed into a Pipeline for execution.

Each event module corresponds to a system module:
- planning.py → wraps _internal/planning.py
- labor_market.py → wraps _internal/labor_market.py
- credit_market.py → wraps _internal/credit_market.py
- production.py → wraps _internal/production.py
- goods_market.py → wraps _internal/goods_market.py
- revenue.py → wraps _internal/revenue.py
- bankruptcy.py → wraps _internal/bankruptcy.py
- economy_stats.py → wraps economy-wide statistics
"""

# Import all events to trigger auto-registration
from bamengine.events.bankruptcy import (
    FirmsUpdateNetWorth,
    MarkBankruptBanks,
    MarkBankruptFirms,
    SpawnReplacementBanks,
    SpawnReplacementFirms,
)
from bamengine.events.credit_market import (
    BanksDecideCreditSupply,
    BanksDecideInterestRate,
    BanksProvideLoans,
    FirmsCalcCreditMetrics,
    FirmsDecideCreditDemand,
    FirmsFireWorkers,
    FirmsPrepareLoanApplications,
    FirmsSendOneLoanApp,
)
from bamengine.events.economy_stats import CalcUnemploymentRate, UpdateAvgMktPrice
from bamengine.events.goods_market import (
    ConsumersCalcPropensity,
    ConsumersDecideFirmsToVisit,
    ConsumersDecideIncomeToSpend,
    ConsumersFinalizePurchases,
    ConsumersShopOneRound,
)
from bamengine.events.labor_market import (
    AdjustMinimumWage,
    CalcAnnualInflationRate,
    FirmsCalcWageBill,
    FirmsDecideWageOffer,
    FirmsHireWorkers,
    WorkersDecideFirmsToApply,
    WorkersSendOneRound,
)
from bamengine.events.planning import (
    FirmsAdjustPrice,
    FirmsCalcBreakevenPrice,
    FirmsDecideDesiredLabor,
    FirmsDecideDesiredProduction,
    FirmsDecideVacancies,
)
from bamengine.events.production import (
    FirmsPayWages,
    FirmsRunProduction,
    WorkersReceiveWage,
    WorkersUpdateContracts,
)
from bamengine.events.revenue import (
    FirmsCollectRevenue,
    FirmsPayDividends,
    FirmsValidateDebtCommitments,
)

__all__ = [
    # Planning events (5)
    "FirmsDecideDesiredProduction",
    "FirmsCalcBreakevenPrice",
    "FirmsAdjustPrice",
    "FirmsDecideDesiredLabor",
    "FirmsDecideVacancies",
    # Labor market events (7)
    "CalcAnnualInflationRate",
    "AdjustMinimumWage",
    "FirmsDecideWageOffer",
    "WorkersDecideFirmsToApply",
    "WorkersSendOneRound",
    "FirmsHireWorkers",
    "FirmsCalcWageBill",
    # Credit market events (8)
    "BanksDecideCreditSupply",
    "BanksDecideInterestRate",
    "FirmsDecideCreditDemand",
    "FirmsCalcCreditMetrics",
    "FirmsPrepareLoanApplications",
    "FirmsSendOneLoanApp",
    "BanksProvideLoans",
    "FirmsFireWorkers",
    # Production events (4)
    "FirmsPayWages",
    "WorkersReceiveWage",
    "FirmsRunProduction",
    "WorkersUpdateContracts",
    # Goods market events (5)
    "ConsumersCalcPropensity",
    "ConsumersDecideIncomeToSpend",
    "ConsumersDecideFirmsToVisit",
    "ConsumersShopOneRound",
    "ConsumersFinalizePurchases",
    # Revenue events (3)
    "FirmsCollectRevenue",
    "FirmsValidateDebtCommitments",
    "FirmsPayDividends",
    # Bankruptcy events (5)
    "FirmsUpdateNetWorth",
    "MarkBankruptFirms",
    "MarkBankruptBanks",
    "SpawnReplacementFirms",
    "SpawnReplacementBanks",
    # Economy stats events (2)
    "UpdateAvgMktPrice",
    "CalcUnemploymentRate",
]
