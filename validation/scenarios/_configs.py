"""Shared scenario configuration constants.

Both the Growth+ (Section 3.9.2) and Buffer-stock (Section 3.9.3) scenarios
use a smaller economy with adjusted new-firm-entry parameters. This module
extracts those shared calibration overrides into a single constant to
prevent silent drift between scenarios.
"""

from __future__ import annotations

from typing import Any

SMALL_ECONOMY_CONFIG: dict[str, Any] = {
    "n_firms": 100,
    "n_households": 500,
    "n_banks": 10,
    "new_firm_size_factor": 0.5,
    "new_firm_production_factor": 0.5,
    "new_firm_wage_factor": 0.5,
    "new_firm_price_markup": 1.5,
    "max_loan_to_net_worth": 5,
    "job_search_method": "all_firms",
}
