"""Parameter calibration package for BAM Engine.

This package provides tools for finding optimal parameter values using
sensitivity analysis followed by focused grid search.

Usage:
    # Run full calibration from command line
    python -m calibration --workers 10 --periods 1000

    # Run sensitivity analysis only
    python -m calibration --sensitivity-only

    # Programmatic usage
    from calibration import run_sensitivity_analysis, run_focused_calibration

    sensitivity = run_sensitivity_analysis(n_workers=10)
    grid, fixed = build_focused_grid(sensitivity)
    results = run_focused_calibration(grid, fixed)
"""

from calibration.optimizer import (
    CalibrationResult,
    build_focused_grid,
    evaluate_stability,
    run_focused_calibration,
    screen_single_seed,
)
from calibration.parameter_space import (
    DEFAULT_VALUES,
    PARAMETER_GRID,
    count_combinations,
    generate_combinations,
)
from calibration.sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
    print_sensitivity_report,
    run_sensitivity_analysis,
)

# Re-export from validation for backwards compatibility
from validation import compute_combined_score

__all__ = [
    # Parameter space
    "PARAMETER_GRID",
    "DEFAULT_VALUES",
    "generate_combinations",
    "count_combinations",
    # Sensitivity analysis
    "ParameterSensitivity",
    "SensitivityResult",
    "run_sensitivity_analysis",
    "print_sensitivity_report",
    # Optimization
    "CalibrationResult",
    "build_focused_grid",
    "compute_combined_score",
    "screen_single_seed",
    "evaluate_stability",
    "run_focused_calibration",
]
