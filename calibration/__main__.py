"""
BAM Engine Calibration CLI
==========================

Command-line interface for BAM model parameter calibration.

Usage::

    # Run sensitivity analysis first
    python -m calibration --sensitivity

    # Review results, edit CALIBRATION_PARAM_GRID in config.py, then run grid search
    python -m calibration --calibrate

    # Or run the main calibration pipeline (grid + local sweep + BO)
    python -m calibration --full

    # Or run individual stages
    python -m calibration --local-sweep  # requires checkpoint
    python -m calibration --bayesian     # requires checkpoint

    # Resume from checkpoint if interrupted
    python -m calibration --full --resume

    # Finally, run consistency analysis on top configs
    python -m calibration --consistency --max-score 50 --consistency-seeds 10

    # Visualize results
    python -m calibration --visualize 1
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import cpu_count
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="BAM Engine Parameter Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sensitivity analysis (~10 min)
  python -m calibration --sensitivity

  # Run full calibration pipeline (grid + local sweep + BO, ~8 hours)
  python -m calibration --full

  # Run grid search only
  python -m calibration --calibrate

  # Run local sweep on existing checkpoint
  python -m calibration --local-sweep --resume

  # Run Bayesian optimization on existing checkpoint
  python -m calibration --bayesian --resume

  # Resume interrupted full calibration
  python -m calibration --full --resume

  # Run consistency analysis on top configurations
  python -m calibration --consistency --max-score 50 --consistency-seeds 10

  # Run with custom number of workers and seeds
  python -m calibration --full -j 8 --seeds 5

  # Specify custom checkpoint and output directory
  python -m calibration --calibrate --checkpoint ./my_checkpoint.pkl --output-dir ./results

  # Show grid configuration info
  python -m calibration --info

  # Single run with defaults
  python -m calibration --baseline

  # Visualize best configuration
  python -m calibration --visualize 1
""",
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run OAT sensitivity analysis only (~10 minutes)",
    )
    mode_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Run grid search only",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full calibration (grid + local sweep + BO, ~8 hours)",
    )
    mode_group.add_argument(
        "--local-sweep",
        action="store_true",
        dest="local_sweep",
        help="Run local sensitivity sweep (requires existing checkpoint)",
    )
    mode_group.add_argument(
        "--bayesian",
        action="store_true",
        help="Run Bayesian optimization (requires existing checkpoint)",
    )
    mode_group.add_argument(
        "--baseline",
        action="store_true",
        help="Single run with default parameters (for comparison)",
    )
    mode_group.add_argument(
        "--visualize",
        nargs="?",
        type=int,
        const=1,
        metavar="RANK",
        help="Visualize a configuration by rank (default: 1 = best)",
    )
    mode_group.add_argument(
        "--info",
        action="store_true",
        help="Show grid configuration info and exit",
    )
    mode_group.add_argument(
        "--consistency",
        action="store_true",
        help="Run consistency analysis on top configurations (10 seeds each)",
    )

    # Common options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Parallel workers (default: {max(1, cpu_count() - 1)})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Seeds per configuration (default: 3)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=1000,
        help="Simulation periods (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Number of top configs to sweep in local sweep (default: 300)",
    )
    parser.add_argument(
        "--bo-iterations",
        type=int,
        default=300,
        help="Number of Bayesian optimization iterations (default: 300)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=50.0,
        help="Max score threshold for consistency analysis (default: 50.0)",
    )
    parser.add_argument(
        "--consistency-seeds",
        type=int,
        default=10,
        help="Seeds per config for consistency analysis (default: 10)",
    )

    return parser


def run_sensitivity_mode(args):
    """Run OAT sensitivity analysis."""
    from .sensitivity import run_oat_sensitivity_analysis

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_oat_sensitivity_analysis(
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        output_dir=output_dir,
    )


def run_calibrate_mode(args):
    """Run grid search only."""
    from .checkpoint import CheckpointManager, get_default_checkpoint_path
    from .grid_search import run_grid_search
    from .visualization import visualize_configuration

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    checkpoint = CheckpointManager(checkpoint_path, resume=args.resume)
    checkpoint.set_metadata("n_seeds", args.seeds)
    checkpoint.set_metadata("n_periods", args.periods)
    checkpoint.set_metadata("burn_in", args.periods // 2)

    top_configs = run_grid_search(
        checkpoint=checkpoint,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_workers=args.workers,
    )

    # Save final checkpoint
    checkpoint.data["top_configurations"] = top_configs[:10]
    checkpoint.save()

    print(f"\nResults saved to: {checkpoint_path}")

    # Visualize best configuration
    if top_configs:
        print("\nGenerating visualization for best configuration...")
        output_dir = (
            Path(args.output_dir) if args.output_dir else checkpoint_path.parent
        )
        visualize_configuration(
            params=top_configs[0]["params"],
            seed=0,
            n_periods=args.periods,
            burn_in=args.periods // 2,
            title="BAM Calibration - Best Configuration",
            save_path=output_dir / "best_config_visualization.png",
        )


def run_full_mode(args):
    """Run full calibration pipeline (grid + local sweep + BO)."""
    from .bayesian_opt import run_bayesian_optimization
    from .checkpoint import CheckpointManager, get_default_checkpoint_path
    from .grid_search import run_grid_search
    from .local_sweep import run_local_sensitivity_sweep
    from .visualization import visualize_configuration

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    checkpoint = CheckpointManager(checkpoint_path, resume=args.resume)
    checkpoint.set_metadata("n_seeds", args.seeds)
    checkpoint.set_metadata("n_periods", args.periods)
    checkpoint.set_metadata("burn_in", args.periods // 2)

    print("\n" + "=" * 70)
    print("FULL CALIBRATION PIPELINE")
    print("=" * 70)
    print("\nStages:")
    print("  1. Grid Search (~12,960 configs)")
    print("  2. Local Sensitivity Sweep (top 300)")
    print("  3. Bayesian Optimization (300 iterations)")
    print("=" * 70)

    # Stage 1: Grid Search
    top_configs = run_grid_search(
        checkpoint=checkpoint,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_workers=args.workers,
    )

    # Stage 2: Local Sensitivity Sweep
    top_configs = run_local_sensitivity_sweep(
        checkpoint=checkpoint,
        top_configs=top_configs,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_workers=args.workers,
        top_k=args.top_k,
    )

    # Stage 3: Bayesian Optimization
    run_bayesian_optimization(
        checkpoint=checkpoint,
        top_configs=top_configs,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_calls=args.bo_iterations,
        n_workers=args.workers,
    )

    # Save final checkpoint
    checkpoint.data["top_configurations"] = top_configs[:10]
    checkpoint.save()

    print("\n" + "=" * 70)
    print("FULL CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {checkpoint_path}")
    print(
        f"Total configurations evaluated: {checkpoint.data['metadata']['total_configs_evaluated']}"
    )

    # Get final best configuration
    final_top = checkpoint.get_top_configs(1)
    if final_top:
        best = final_top[0]
        print(f"\nBest configuration score: {best['scores']['total']:.2f}")
        print("Best parameters:")
        for k, v in best["params"].items():
            print(f"  {k}: {v}")

    # Visualize best configuration
    if final_top:
        print("\nGenerating visualization for best configuration...")
        output_dir = (
            Path(args.output_dir) if args.output_dir else checkpoint_path.parent
        )
        visualize_configuration(
            params=final_top[0]["params"],
            seed=0,
            n_periods=args.periods,
            burn_in=args.periods // 2,
            title="BAM Calibration - Best Configuration",
            save_path=output_dir / "best_config_visualization.png",
        )


def run_local_sweep_mode(args):
    """Run local sensitivity sweep on existing checkpoint."""
    from .checkpoint import CheckpointManager, get_default_checkpoint_path
    from .local_sweep import run_local_sensitivity_sweep

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run grid search first: python -m calibration --calibrate")
        sys.exit(1)

    checkpoint = CheckpointManager(checkpoint_path, resume=True)
    top_configs = checkpoint.get_top_configs(args.top_k)

    if not top_configs:
        print("Error: No configurations found in checkpoint")
        sys.exit(1)

    print(f"Loaded {len(top_configs)} top configs from checkpoint")

    run_local_sensitivity_sweep(
        checkpoint=checkpoint,
        top_configs=top_configs,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_workers=args.workers,
        top_k=args.top_k,
    )

    print(f"\nResults saved to: {checkpoint_path}")


def run_bayesian_mode(args):
    """Run Bayesian optimization on existing checkpoint."""
    from .bayesian_opt import run_bayesian_optimization
    from .checkpoint import CheckpointManager, get_default_checkpoint_path

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run grid search first: python -m calibration --calibrate")
        sys.exit(1)

    checkpoint = CheckpointManager(checkpoint_path, resume=True)
    top_configs = checkpoint.get_top_configs(100)

    if not top_configs:
        print("Error: No configurations found in checkpoint")
        sys.exit(1)

    print(f"Loaded {len(top_configs)} top configs from checkpoint")

    run_bayesian_optimization(
        checkpoint=checkpoint,
        top_configs=top_configs,
        n_seeds=args.seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        n_calls=args.bo_iterations,
        n_workers=args.workers,
    )

    print(f"\nResults saved to: {checkpoint_path}")


def run_baseline_mode(args):
    """Run single simulation with defaults."""
    from .visualization import visualize_configuration

    print("=" * 60)
    print("BASELINE MODE: Running with default parameters")
    print("=" * 60)
    print("\nThis runs a single simulation with default parameters")
    print("to compare against calibration results.\n")

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_configuration(
        params={},  # Empty params = use defaults
        seed=0,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        title="BAM Calibration - Baseline (defaults)",
        save_path=output_dir / "baseline_visualization.png",
    )


def run_visualize_mode(args):
    """Visualize a configuration from checkpoint."""
    from .checkpoint import CheckpointManager, get_default_checkpoint_path
    from .visualization import visualize_configuration

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = CheckpointManager(checkpoint_path, resume=True)
    checkpoint.print_summary()

    rank = args.visualize
    config = checkpoint.get_config_by_rank(rank)

    if config is None:
        print(f"Error: Rank {rank} not found in checkpoint")
        sys.exit(1)

    print(f"\nVisualizing rank {rank} configuration")
    visualize_configuration(
        params=config["params"],
        seed=0,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        title=f"BAM Calibration - Rank {rank}",
    )


def run_info_mode(args):
    """Show grid configuration info."""
    from .bayesian_opt import print_bo_info
    from .grid_search import print_grid_info
    from .local_sweep import print_local_sweep_info

    print_grid_info(n_seeds=args.seeds)
    print_local_sweep_info(top_k=args.top_k)
    print_bo_info(n_calls=args.bo_iterations)


def run_consistency_mode(args):
    """Run consistency analysis on top configurations."""
    from .checkpoint import CheckpointManager, get_default_checkpoint_path
    from .consistency import run_consistency_analysis

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else get_default_checkpoint_path()
    )

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run calibration first: python -m calibration --calibrate")
        sys.exit(1)

    checkpoint = CheckpointManager(checkpoint_path, resume=True)

    # Use top_k if specified, otherwise use max_score threshold
    top_n = args.top_k if args.top_k else None

    results = run_consistency_analysis(
        checkpoint=checkpoint,
        n_seeds=args.consistency_seeds,
        n_periods=args.periods,
        burn_in=args.periods // 2,
        max_score=args.max_score,
        top_n=top_n,
        n_workers=args.workers,
    )

    print(f"\nResults saved to: {checkpoint_path}")
    print(
        f"Analyzed {len(results)} configurations with {args.consistency_seeds} seeds each"
    )


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Determine mode
    if args.sensitivity:
        run_sensitivity_mode(args)
    elif args.calibrate:
        run_calibrate_mode(args)
    elif args.full:
        run_full_mode(args)
    elif args.local_sweep:
        run_local_sweep_mode(args)
    elif args.bayesian:
        run_bayesian_mode(args)
    elif args.baseline:
        run_baseline_mode(args)
    elif args.visualize:
        run_visualize_mode(args)
    elif args.info:
        run_info_mode(args)
    elif args.consistency:
        run_consistency_mode(args)
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print("""
1. First, run sensitivity analysis:
   python -m calibration --sensitivity

2. Review results, edit CALIBRATION_PARAM_GRID in:
   calibration/config.py

3. Run full calibration pipeline:
   python -m calibration --full

   Or run grid search only:
   python -m calibration --calibrate

4. Visualize results:
   python -m calibration --visualize 1
""")


if __name__ == "__main__":
    main()
