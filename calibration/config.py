"""
Calibration Configuration
=========================

Parameter grids and fixed parameters for BAM model calibration.
"""

from __future__ import annotations

from typing import Any

# ==============================================================================
# FIXED PARAMETERS
# ==============================================================================
# Parameters from book Section 3.9.1 and past sensitivity analysis.
# These remain constant during calibration.

FIXED_PARAMS: dict[str, Any] = {
    # Population sizes
    "n_firms": 100,
    "n_households": 500,
    "n_banks": 10,
    # Stochastic shock widths
    "h_rho": 0.10,
    "h_xi": 0.05,
    "h_phi": 0.10,
    "h_eta": 0.10,
    # Search frictions
    "max_M": 4,
    "max_H": 2,
    "max_Z": 2,
    # Structural parameters
    "min_wage_rev_period": 4,
    "theta": 8,
    "beta": 2.50,
    "delta": 0.10,
    "r_bar": 0.02,
    "labor_productivity": 0.50,
    "wage_offer_init": 1.0,
    # Implementation variants (empirically determined from past analysis)
    "loan_priority_method": "by_leverage",
    "v": 0.10,
    "savings_init": 3.0,
    "firing_method": "random",
    "price_cut_allow_increase": True,
    "cap_factor": 2.00,
    "fragility_cap_method": "credit_demand",
    # Low-impact params fixed to best values from sensitivity analysis
    "new_firm_size_factor": 0.8,
    "employed_capture_event": "workers_update_contracts",
    "vacancies_capture_event": "firms_fire_workers",
}


# ==============================================================================
# PARAMETER GRIDS
# ==============================================================================

# One-at-a-time sensitivity analysis grid (includes extreme values)
OAT_PARAM_GRID: dict[str, list] = {
    "contract_poisson_mean": [0, 10],
    "net_worth_init": [1.0, 10.0, 100.0],
    "equity_base_init": [5.0, 10.0, 10000.0],
    "price_init_offset": [0.1, 0.5, 1.0],
    "min_wage_ratio": [0.5, 0.9, 1.0],
    "new_firm_size_factor": [0.5, 0.8],
    "new_firm_production_factor": [0.5, 0.8, 1.0],
    "new_firm_wage_factor": [0.5, 0.8, 1.0],
    "new_firm_price_markup": [1.0, 1.25],
    # Capture timing parameters
    "employed_capture_event": [None, "workers_update_contracts"],
    "vacancies_capture_event": [None, "firms_fire_workers"],
}

# Calibration grid for full factorial sweep
# Total: 3 x 5 x 2 x 4 x 3 x 3 x 6 x 2 = 12,960 configs
CALIBRATION_PARAM_GRID: dict[str, list] = {
    # HIGH IMPACT
    "price_init_offset": [0.1, 0.25, 0.5],
    "min_wage_ratio": [0.5, 0.55, 0.6, 0.65, 0.7],
    # MEDIUM-HIGH IMPACT
    "contract_poisson_mean": [0, 10],
    "net_worth_init": [1.0, 2.0, 5.0, 10.0],
    "new_firm_price_markup": [1.0, 1.10, 1.25],
    "new_firm_production_factor": [0.8, 0.9, 1.0],
    "new_firm_wage_factor": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # LOWER IMPACT
    "equity_base_init": [5.0, 10.0],
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def apply_config_offsets(config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert offset-based parameters to absolute values.

    This centralizes the logic for converting relative params to absolute values:
    - price_init_offset -> price_init = wage_offer_init + offset
    - min_wage_ratio -> min_wage = wage_offer_init * ratio

    Parameters
    ----------
    config : dict
        Configuration dictionary with potential offset-based params.

    Returns
    -------
    dict
        Configuration with offsets converted to absolute values.
    """
    config = config.copy()
    wage_base = config.get("wage_offer_init", 1.0)

    if "price_init_offset" in config:
        config["price_init"] = wage_base + config.pop("price_init_offset")

    if "min_wage_ratio" in config:
        config["min_wage"] = wage_base * config.pop("min_wage_ratio")

    return config


def get_default_value(param_name: str) -> Any:
    """
    Get the default value for a parameter from defaults.yml.

    For parameters in FIXED_PARAMS, returns the fixed value.
    For offset-based params, returns values matching defaults.yml.

    Parameters
    ----------
    param_name : str
        Name of the parameter.

    Returns
    -------
    Any
        Default value for the parameter.
    """
    # Check if it's in FIXED_PARAMS
    if param_name in FIXED_PARAMS:
        return FIXED_PARAMS[param_name]

    # Default values for calibration parameters (from defaults.yml)
    defaults = {
        "contract_poisson_mean": 10,
        "net_worth_init": 10.0,
        "equity_base_init": 10.0,
        "price_init_offset": 0.5,  # price_init=1.5, wage_offer_init=1.0
        "min_wage_ratio": 0.5,  # min_wage=0.5, wage_offer_init=1.0
        "new_firm_production_factor": 0.8,
        "new_firm_wage_factor": 0.8,
        "new_firm_price_markup": 1.25,
    }

    return defaults.get(param_name)
