"""
Event classes for BAM Engine simulation.

This package contains 39 event classes organized into 8 modules representing
different phases of the BAM economic model. Events are auto-registered via
__init_subclass__ hook and composed into a Pipeline for execution.

Event Organization
------------------
Events are organized by economic phase:

1. **Planning** (5 events): Firms plan production targets, calculate costs, set prices
2. **Labor Market** (7 events): Wage setting, job applications, hiring
3. **Credit Market** (8 events): Credit supply/demand, loan applications, provision
4. **Production** (4 events): Wage payments, production execution, contract updates
5. **Goods Market** (5 events): Consumption decisions, shopping
6. **Revenue** (3 events): Revenue collection, debt repayment, dividends
7. **Bankruptcy** (5 events): Insolvency detection, agent replacement
8. **Economy Stats** (2 events): Aggregate metrics (prices, unemployment)

Total: 39 events across 8 modules

Event Execution
---------------
Events execute in order specified by Pipeline (see config/default_pipeline.yml).
Each event wraps a system function from events._internal/ modules.

Key Design Patterns
-------------------
- **Auto-registration**: All events inherit from Event base class with __init_subclass__
- **Event-System separation**: Event classes (public API) wrap system functions (internal)
- **Pipeline composition**: Events composed via YAML configuration
- **Stateless execution**: Events receive Simulation object, operate on roles/relationships
- **Interleaved rounds**: Some events repeat (max_M, max_H, max_Z) for sequential matching

Event Naming Convention
-----------------------
- Agent action: FirmsDecideWageOffer, WorkersReceiveWage
- State update: UpdateAvgMktPrice, CalcUnemploymentRate
- Process round: WorkersSendOneRound, ConsumersShopOneRound

Examples
--------
Access event by name:

>>> import bamengine as bam
>>> sim = be.Simulation.init(seed=42)
>>> event = sim.get_event("firms_decide_desired_production")
>>> event.execute(sim)

Execute full pipeline:

>>> sim.step()  # Executes all 39 events in default order

See Also
--------
:class:`bamengine.core.Event` : Event base class
:class:`bamengine.core.Pipeline` : Pipeline composition
:mod:`bamengine.events._internal` : System function implementations
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
    FirmsCalcFinancialFragility,
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
    "FirmsCalcFinancialFragility",
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
