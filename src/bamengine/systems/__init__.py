"""
BAM Events internal implementations.
"""

from .credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)
from .goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_shop_one_round,
)
from .labor_market import (
    adjust_minimum_wage,
    calc_annual_inflation_rate,
    firms_calc_wage_bill,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from .planning import (
    firms_adjust_price,
    firms_calc_breakeven_price,
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from .production import (
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)
from .revenue import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)

__all__: list[str] = [
    # planning
    "firms_decide_desired_production",
    "firms_calc_breakeven_price",
    "firms_adjust_price",
    "firms_decide_desired_labor",
    "firms_decide_vacancies",
    # labor market
    "calc_annual_inflation_rate",
    "adjust_minimum_wage",
    "firms_calc_wage_bill",
    "firms_decide_wage_offer",
    "firms_hire_workers",
    "workers_decide_firms_to_apply",
    "workers_send_one_round",
    # credit market
    "banks_decide_credit_supply",
    "banks_decide_interest_rate",
    "banks_provide_loans",
    "firms_calc_credit_metrics",
    "firms_decide_credit_demand",
    "firms_fire_workers",
    "firms_prepare_loan_applications",
    "firms_send_one_loan_app",
    # production
    "firms_pay_wages",
    "firms_run_production",
    "update_avg_mkt_price",
    "workers_receive_wage",
    "workers_update_contracts",
    # goods market
    "consumers_calc_propensity",
    "consumers_decide_firms_to_visit",
    "consumers_decide_income_to_spend",
    "consumers_finalize_purchases",
    "consumers_shop_one_round",
    # revenue
    "firms_collect_revenue",
    "firms_pay_dividends",
    "firms_validate_debt_commitments",
]
