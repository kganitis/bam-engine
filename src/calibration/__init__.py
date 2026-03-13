"""Parameter calibration package for BAM Engine.

This package provides tools for finding optimal parameter values using
Morris Method screening (or OAT sensitivity) followed by focused grid
search with tiered stability testing, plus composable tools for
second-pass screening, cost analysis, cross-scenario evaluation,
and structured parameter sweeps.

Usage:
    # Run full calibration from command line (Morris is default)
    python -m calibration --workers 10 --periods 1000

    # Run individual phases
    python -m calibration --phase sensitivity --scenario baseline
    python -m calibration --phase grid --scenario baseline
    python -m calibration --phase stability --scenario baseline

    # Extended phases
    python -m calibration --phase rescreen --fix-from stability.json --params behavioral
    python -m calibration --phase cost --base stability.json --swaps "beta=5,2.5"
    python -m calibration --phase cross-eval --scenarios baseline,growth_plus --configs stability.json
    python -m calibration --phase sweep --stages "A:beta=0.5,1.0" "B:max_M=2,4"

    # Programmatic usage
    from calibration import run_morris_screening, build_focused_grid, run_focused_calibration

    morris = run_morris_screening(n_workers=10, n_seeds=3)
    sensitivity = morris.to_sensitivity_result()
    grid, fixed = build_focused_grid(sensitivity)
    results = run_focused_calibration(grid, fixed)  # default tiers: 100:10, 50:20, 10:100
"""

# Analysis (types and utilities)
from calibration.analysis import (
    CalibrationResult,
    ComparisonResult,
    ScenarioResult,
    analyze_parameter_patterns,
    compare_configs,
    export_best_config,
    format_eta,
    format_progress,
    print_comparison,
    print_parameter_patterns,
)

# Cost
from calibration.cost import (
    SwapResult,
    classify_cost,
    parse_swaps,
    run_cost_analysis,
    save_cost_results,
)

# Cross-eval
from calibration.cross_eval import (
    compute_scenario_tension,
    evaluate_cross_scenario,
    rank_cross_scenario,
)

# Grid
from calibration.grid import (
    build_focused_grid,
    count_combinations,
    generate_combinations,
    load_grid,
    validate_grid,
)

# IO
from calibration.io import (
    create_run_dir,
    load_morris,
    load_pairwise,
    load_screening,
    load_sensitivity,
    load_stability,
    save_morris,
    save_pairwise,
    save_screening,
    save_sensitivity,
    save_stability,
)

# Morris
from calibration.morris import (
    MorrisParameterEffect,
    MorrisResult,
    print_morris_report,
    run_morris_screening,
)

# Parameter space
from calibration.parameter_space import (
    DEFAULT_VALUES,
    PARAM_GROUPS,
    PARAMETER_GRID,
    get_default_values,
    get_parameter_grid,
)

# Reporting
from calibration.reporting import (
    generate_full_report,
    generate_screening_report,
    generate_sensitivity_report,
    generate_stability_report,
)

# Rescreen
from calibration.rescreen import (
    compute_sensitivity_collapse,
    load_fixed_from_result,
    resolve_params,
    run_rescreen,
)

# Screening
from calibration.screening import run_screening, screen_single_seed

# Sensitivity
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

# Stability
from calibration.stability import (
    evaluate_stability,
    parse_stability_tiers,
    run_focused_calibration,
    run_tiered_stability,
)

# Sweep
from calibration.sweep import StageResult, parse_stage, parse_stages, run_sweep

__all__ = [
    # Analysis (types)
    "CalibrationResult",
    "ComparisonResult",
    "ScenarioResult",
    # Grid
    "build_focused_grid",
    "count_combinations",
    "generate_combinations",
    "load_grid",
    "validate_grid",
    # Screening
    "screen_single_seed",
    "run_screening",
    # Stability
    "evaluate_stability",
    "run_tiered_stability",
    "run_focused_calibration",
    "parse_stability_tiers",
    # Morris
    "MorrisParameterEffect",
    "MorrisResult",
    "run_morris_screening",
    "print_morris_report",
    # Sensitivity
    "ParameterSensitivity",
    "SensitivityResult",
    "run_sensitivity_analysis",
    "print_sensitivity_report",
    # Pairwise
    "PairInteraction",
    "PairwiseResult",
    "run_pairwise_analysis",
    "print_pairwise_report",
    # IO
    "create_run_dir",
    "save_sensitivity",
    "load_sensitivity",
    "save_morris",
    "load_morris",
    "save_screening",
    "load_screening",
    "save_stability",
    "load_stability",
    "save_pairwise",
    "load_pairwise",
    # Reporting
    "generate_sensitivity_report",
    "generate_screening_report",
    "generate_stability_report",
    "generate_full_report",
    # Analysis
    "analyze_parameter_patterns",
    "print_parameter_patterns",
    "export_best_config",
    "compare_configs",
    "print_comparison",
    # Progress
    "format_eta",
    "format_progress",
    # Parameter space
    "PARAMETER_GRID",
    "DEFAULT_VALUES",
    "PARAM_GROUPS",
    "get_parameter_grid",
    "get_default_values",
    # Cost
    "SwapResult",
    "classify_cost",
    "parse_swaps",
    "run_cost_analysis",
    "save_cost_results",
    # Cross-eval
    "compute_scenario_tension",
    "evaluate_cross_scenario",
    "rank_cross_scenario",
    # Rescreen
    "compute_sensitivity_collapse",
    "load_fixed_from_result",
    "resolve_params",
    "run_rescreen",
    # Sweep
    "StageResult",
    "parse_stage",
    "parse_stages",
    "run_sweep",
]
