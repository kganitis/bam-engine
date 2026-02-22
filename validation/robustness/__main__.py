"""Entry point for: python -m validation.robustness

Runs internal validity analysis, sensitivity analysis (Section 3.10.1),
and/or structural experiments (Section 3.10.2) from Delli Gatti et al. (2011).
"""

from __future__ import annotations

import argparse

from validation.robustness.experiments import ALL_EXPERIMENT_NAMES
from validation.robustness.internal_validity import run_internal_validity
from validation.robustness.reporting import (
    print_entry_report,
    print_internal_validity_report,
    print_pa_report,
    print_sensitivity_report,
)
from validation.robustness.sensitivity import run_sensitivity_analysis
from validation.robustness.structural import run_entry_experiment, run_pa_experiment
from validation.robustness.viz import (
    plot_comovements,
    plot_entry_comparison,
    plot_irf,
    plot_pa_comovements,
    plot_pa_gdp_comparison,
    plot_sensitivity_comovements,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m validation.robustness",
        description="Robustness analysis (Section 3.10)",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Run internal validity only (skip sensitivity and structural).",
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Run sensitivity analysis only (skip internal validity and structural).",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Run structural experiments only (Section 3.10.2).",
    )
    parser.add_argument(
        "--pa-experiment",
        action="store_true",
        help="Run PA (preferential attachment) experiment only.",
    )
    parser.add_argument(
        "--entry-experiment",
        action="store_true",
        help="Run entry neutrality experiment only.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help=(
            "Comma-separated experiment names for sensitivity analysis. "
            f"Available: {', '.join(ALL_EXPERIMENT_NAMES)}"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of random seeds per configuration (default: 20).",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Simulation periods (default: 1000).",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=500,
        help="Burn-in periods to discard (default: 500).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison in PA experiment.",
    )
    parser.add_argument(
        "--growth-plus",
        action="store_true",
        help="Use Growth+ model (R&D extension) instead of baseline.",
    )

    args = parser.parse_args(argv)

    # Determine which analyses to run
    specific_structural = args.pa_experiment or args.entry_experiment
    if args.structural_only or specific_structural:
        run_internal = False
        run_sensitivity = False
        run_structural = True
    elif args.internal_only:
        run_internal = True
        run_sensitivity = False
        run_structural = False
    elif args.sensitivity_only:
        run_internal = False
        run_sensitivity = True
        run_structural = False
    else:
        # Default: run everything
        run_internal = True
        run_sensitivity = True
        run_structural = True

    experiments = None
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]

    # Growth+ setup hook and collect config
    setup_hook = None
    collect_config = None
    model_label = "baseline"

    if args.growth_plus:
        from validation.robustness.internal_validity import (
            GROWTH_PLUS_COLLECT_CONFIG,
            setup_growth_plus,
        )

        setup_hook = setup_growth_plus
        collect_config = GROWTH_PLUS_COLLECT_CONFIG
        model_label = "Growth+ (R&D)"

    common_kwargs = {
        "n_seeds": args.seeds,
        "n_periods": args.periods,
        "burn_in": args.burn_in,
        "n_workers": args.workers,
        "setup_hook": setup_hook,
        "collect_config": collect_config,
    }

    # ── Internal validity ────────────────────────────────────────────
    if run_internal:
        print("\n" + "=" * 60)
        print(f"  INTERNAL VALIDITY ANALYSIS ({model_label})")
        print("=" * 60)

        iv_result = run_internal_validity(**common_kwargs)

        print_internal_validity_report(iv_result)

        if not args.no_plots:
            plot_comovements(iv_result, show=True)
            plot_irf(iv_result, show=True)

    # ── Sensitivity analysis ─────────────────────────────────────────
    if run_sensitivity:
        print("\n" + "=" * 60)
        print(f"  SENSITIVITY ANALYSIS ({model_label})")
        print("=" * 60)

        sa_result = run_sensitivity_analysis(experiments=experiments, **common_kwargs)

        print_sensitivity_report(sa_result)

        if not args.no_plots:
            for exp_result in sa_result.experiments.values():
                plot_sensitivity_comovements(exp_result, show=True)

    # ── Structural experiments (Section 3.10.2) ──────────────────────
    if run_structural:
        run_pa = not specific_structural or args.pa_experiment
        run_entry = not specific_structural or args.entry_experiment

        if run_pa:
            pa_result = run_pa_experiment(
                include_baseline=not args.no_baseline,
                verbose=True,
                **common_kwargs,
            )

            print_pa_report(pa_result)

            if not args.no_plots:
                plot_pa_gdp_comparison(pa_result, show=True)
                plot_pa_comovements(pa_result, show=True)

        if run_entry:
            entry_result = run_entry_experiment(
                verbose=True,
                **common_kwargs,
            )

            print_entry_report(entry_result)

            if not args.no_plots:
                plot_entry_comparison(entry_result, show=True)


if __name__ == "__main__":
    main()
