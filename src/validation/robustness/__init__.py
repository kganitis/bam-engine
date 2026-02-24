"""Robustness analysis for BAM Engine (Section 3.10).

This package implements the robustness checking procedures from
Section 3.10 of Delli Gatti et al. (2011), consisting of:

1. **Internal validity**: Multi-seed simulation with default parameters
   to verify cross-simulation stability of co-movements, distributions,
   and empirical curves (Section 3.10.1, Part 1).

2. **Sensitivity analysis**: Univariate parameter sweeps across five
   parameter groups (H, Z, M, theta, economy size) to assess output
   robustness to input variation (Section 3.10.1, Part 2).

3. **Structural experiments**: Mechanism tests (Section 3.10.2):
   PA toggle (consumer loyalty) and entry neutrality (profit taxation).

Quick start::

    from validation.robustness import run_internal_validity

    result = run_internal_validity(n_seeds=20, n_periods=1000)
    print_internal_validity_report(result)
    plot_comovements(result)

    from validation.robustness import run_sensitivity_analysis

    sa = run_sensitivity_analysis(experiments=["credit_market"])
    print_sensitivity_report(sa)

    from validation.robustness import run_pa_experiment, run_entry_experiment

    pa = run_pa_experiment(n_seeds=10, n_periods=500)
    print_pa_report(pa)

    entry = run_entry_experiment(n_seeds=10, n_periods=500)
    print_entry_report(entry)

CLI usage::

    python -m validation.robustness                    # Full analysis
    python -m validation.robustness --internal-only    # Internal validity only
    python -m validation.robustness --sensitivity-only # Sensitivity only
    python -m validation.robustness --structural-only  # Structural experiments only
    python -m validation.robustness --pa-experiment    # PA experiment only
    python -m validation.robustness --entry-experiment # Entry experiment only
    python -m validation.robustness --experiments credit_market,contract_length
"""

from validation.robustness.experiments import (
    ALL_EXPERIMENT_NAMES,
    EXPERIMENTS,
    PARAMETER_EXPERIMENT_NAMES,
    PARAMETER_EXPERIMENTS,
    STRUCTURAL_EXPERIMENT_NAMES,
    STRUCTURAL_EXPERIMENTS,
    Experiment,
)
from validation.robustness.internal_validity import (
    COMOVEMENT_VARIABLES,
    GROWTH_PLUS_COLLECT_CONFIG,
    InternalValidityResult,
    SeedAnalysis,
    run_internal_validity,
    setup_growth_plus,
)
from validation.robustness.reporting import (
    format_entry_report,
    format_internal_validity_report,
    format_pa_report,
    format_sensitivity_report,
    print_entry_report,
    print_internal_validity_report,
    print_pa_report,
    print_sensitivity_report,
)
from validation.robustness.sensitivity import (
    ExperimentResult,
    SensitivityResult,
    ValueResult,
    run_sensitivity_analysis,
)
from validation.robustness.stats import (
    cross_correlation,
    fit_ar,
    hp_filter,
    impulse_response,
)
from validation.robustness.structural import (
    EntryExperimentResult,
    PAExperimentResult,
    run_entry_experiment,
    run_pa_experiment,
)
from validation.robustness.viz import (
    plot_comovements,
    plot_entry_comparison,
    plot_irf,
    plot_pa_comovements,
    plot_pa_gdp_comparison,
    plot_sensitivity_comovements,
)

__all__ = [
    # Internal validity
    "run_internal_validity",
    "InternalValidityResult",
    "SeedAnalysis",
    "COMOVEMENT_VARIABLES",
    # Growth+ support
    "setup_growth_plus",
    "GROWTH_PLUS_COLLECT_CONFIG",
    # Sensitivity analysis
    "run_sensitivity_analysis",
    "SensitivityResult",
    "ExperimentResult",
    "ValueResult",
    # Experiments
    "EXPERIMENTS",
    "ALL_EXPERIMENT_NAMES",
    "PARAMETER_EXPERIMENTS",
    "PARAMETER_EXPERIMENT_NAMES",
    "STRUCTURAL_EXPERIMENTS",
    "STRUCTURAL_EXPERIMENT_NAMES",
    "Experiment",
    # Structural experiments (Section 3.10.2)
    "run_pa_experiment",
    "run_entry_experiment",
    "PAExperimentResult",
    "EntryExperimentResult",
    # Statistical tools
    "hp_filter",
    "cross_correlation",
    "fit_ar",
    "impulse_response",
    # Reporting
    "print_internal_validity_report",
    "print_sensitivity_report",
    "format_internal_validity_report",
    "format_sensitivity_report",
    "print_pa_report",
    "print_entry_report",
    "format_pa_report",
    "format_entry_report",
    # Visualization
    "plot_comovements",
    "plot_irf",
    "plot_sensitivity_comovements",
    "plot_pa_gdp_comparison",
    "plot_pa_comovements",
    "plot_entry_comparison",
]
