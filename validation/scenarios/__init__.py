"""Validation scenarios for BAM Engine.

This subpackage provides detailed visualization scenarios that reproduce
results from Delli Gatti et al. (2011).

Scenarios:
    baseline: Section 3.9.1 - Standard BAM model behavior
    growth_plus: Section 3.9.2 - Endogenous productivity growth via R&D

Usage:
    # Run scenario with visualization
    from validation.scenarios import run_baseline_scenario, run_growth_plus_scenario

    run_baseline_scenario(seed=0, show_plot=True)
    run_growth_plus_scenario(seed=2, show_plot=True)

    # Or run as module:
    # python -m validation.scenarios.baseline
    # python -m validation.scenarios.growth_plus
"""

from __future__ import annotations

# Baseline scenario
from validation.scenarios.baseline import run_scenario as run_baseline_scenario
from validation.scenarios.baseline import visualize_baseline_results

# Growth+ scenario
from validation.scenarios.growth_plus import run_scenario as run_growth_plus_scenario
from validation.scenarios.growth_plus import visualize_growth_plus_results

# Growth+ extension (RnD role and events)
from validation.scenarios.growth_plus_extension import (
    FirmsApplyProductivityGrowth,
    FirmsComputeRnDIntensity,
    FirmsDeductRnDExpenditure,
    RnD,
)

__all__ = [
    # Baseline
    "run_baseline_scenario",
    "visualize_baseline_results",
    # Growth+ extension
    "RnD",
    "FirmsComputeRnDIntensity",
    "FirmsApplyProductivityGrowth",
    "FirmsDeductRnDExpenditure",
    # Growth+
    "run_growth_plus_scenario",
    "visualize_growth_plus_results",
]
