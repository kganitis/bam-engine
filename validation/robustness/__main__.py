"""Entry point for: python -m validation.robustness

Runs internal validity analysis and/or sensitivity analysis from
Section 3.10.1 of Delli Gatti et al. (2011).
"""

from __future__ import annotations

import argparse

from validation.robustness.experiments import ALL_EXPERIMENT_NAMES
from validation.robustness.internal_validity import run_internal_validity
from validation.robustness.reporting import (
    print_internal_validity_report,
    print_sensitivity_report,
)
from validation.robustness.sensitivity import run_sensitivity_analysis
from validation.robustness.viz import (
    plot_comovements,
    plot_irf,
    plot_sensitivity_comovements,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m validation.robustness",
        description="Robustness analysis (Section 3.10.1)",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Run internal validity only (skip sensitivity).",
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Run sensitivity analysis only (skip internal validity).",
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
        "--growth-plus",
        action="store_true",
        help="Use Growth+ model (R&D extension) instead of baseline.",
    )

    args = parser.parse_args(argv)

    run_internal = not args.sensitivity_only
    run_sensitivity = not args.internal_only

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

    # ── Internal validity ────────────────────────────────────────────
    if run_internal:
        print("\n" + "=" * 60)
        print(f"  INTERNAL VALIDITY ANALYSIS ({model_label})")
        print("=" * 60)

        iv_result = run_internal_validity(
            n_seeds=args.seeds,
            n_periods=args.periods,
            burn_in=args.burn_in,
            n_workers=args.workers,
            setup_hook=setup_hook,
            collect_config=collect_config,
        )

        print_internal_validity_report(iv_result)

        if not args.no_plots:
            plot_comovements(iv_result, show=True)
            plot_irf(iv_result, show=True)

    # ── Sensitivity analysis ─────────────────────────────────────────
    if run_sensitivity:
        print("\n" + "=" * 60)
        print(f"  SENSITIVITY ANALYSIS ({model_label})")
        print("=" * 60)

        sa_result = run_sensitivity_analysis(
            experiments=experiments,
            n_seeds=args.seeds,
            n_periods=args.periods,
            burn_in=args.burn_in,
            n_workers=args.workers,
            setup_hook=setup_hook,
            collect_config=collect_config,
        )

        print_sensitivity_report(sa_result)

        if not args.no_plots:
            for exp_result in sa_result.experiments.values():
                plot_sensitivity_comovements(exp_result, show=True)


if __name__ == "__main__":
    main()
