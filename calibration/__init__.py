"""Parameter calibration package for BAM Engine.

This package provides tools for finding optimal parameter values using
Morris Method screening (or OAT sensitivity) followed by focused grid
search with tiered stability testing.

Usage:
    # Run full calibration from command line (Morris is default)
    python -m calibration --workers 10 --periods 1000

    # Run individual phases
    python -m calibration --phase sensitivity --scenario baseline
    python -m calibration --phase grid --scenario baseline
    python -m calibration --phase stability --scenario baseline

    # Programmatic usage
    from calibration import run_morris_screening, build_focused_grid, run_focused_calibration

    morris = run_morris_screening(n_workers=10, n_seeds=3)
    sensitivity = morris.to_sensitivity_result()
    grid, fixed = build_focused_grid(sensitivity)
    results = run_focused_calibration(grid, fixed)  # default tiers: 100:10, 50:20, 10:100
"""

from calibration.morris import (
    MorrisParameterEffect,
    MorrisResult,
    print_morris_report,
    run_morris_screening,
)
from calibration.optimizer import (
    CalibrationResult,
    ComparisonResult,
    analyze_parameter_patterns,
    build_focused_grid,
    compare_configs,
    evaluate_stability,
    export_best_config,
    format_eta,
    format_progress,
    parse_stability_tiers,
    run_focused_calibration,
    run_screening,
    run_tiered_stability,
    screen_single_seed,
)
from calibration.parameter_space import (
    DEFAULT_VALUES,
    PARAMETER_GRID,
    count_combinations,
    generate_combinations,
    get_default_values,
    get_parameter_grid,
)
from calibration.sensitivity import (
    PairInteraction,
    PairwiseResult,
    ParameterSensitivity,
    SensitivityResult,
    print_pairwise_report,
    print_sensitivity_report,
    run_pairwise_analysis,
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
    "get_parameter_grid",
    "get_default_values",
    # Morris method screening
    "MorrisParameterEffect",
    "MorrisResult",
    "run_morris_screening",
    "print_morris_report",
    # OAT sensitivity analysis
    "ParameterSensitivity",
    "SensitivityResult",
    "run_sensitivity_analysis",
    "print_sensitivity_report",
    # Pairwise interaction
    "PairInteraction",
    "PairwiseResult",
    "run_pairwise_analysis",
    "print_pairwise_report",
    # Optimization
    "CalibrationResult",
    "ComparisonResult",
    "build_focused_grid",
    "compute_combined_score",
    "screen_single_seed",
    "evaluate_stability",
    "run_focused_calibration",
    "run_screening",
    "run_tiered_stability",
    "parse_stability_tiers",
    "analyze_parameter_patterns",
    "export_best_config",
    "compare_configs",
    # Progress
    "format_eta",
    "format_progress",
]
