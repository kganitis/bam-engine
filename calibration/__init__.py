"""
BAM Engine Calibration Package
==============================

This package provides tools for calibrating BAM model parameters to match
target macroeconomic patterns from Delli Gatti et al. (2011).

Usage
-----
Run calibration from command line::

    python -m calibration --sensitivity   # OAT sensitivity analysis
    python -m calibration --calibrate     # Grid search only
    python -m calibration --full          # Full pipeline (grid + sweep + BO)
    python -m calibration --local-sweep   # Local sweep only
    python -m calibration --bayesian      # Bayesian optimization only
    python -m calibration --consistency   # Consistency analysis (10 seeds)
    python -m calibration --baseline      # Single run with defaults
    python -m calibration --visualize 1   # Visualize best config

Or use programmatically::

    from calibration import (
        run_oat_sensitivity_analysis,
        run_grid_search,
        run_local_sensitivity_sweep,
        run_bayesian_optimization,
        visualize_configuration,
    )
"""

from __future__ import annotations

# Bayesian optimization
from .bayesian_opt import BO_SPACE_CONFIG, run_bayesian_optimization

# Checkpoint
from .checkpoint import CheckpointManager

# Configuration
from .config import (
    CALIBRATION_PARAM_GRID,
    FIXED_PARAMS,
    OAT_PARAM_GRID,
    apply_config_offsets,
)

# Consistency analysis
from .consistency import ConsistencyResult, run_consistency_analysis

# Grid search
from .grid_search import generate_combinations, run_grid_search

# Local sweep
from .local_sweep import LOCAL_SWEEP_PARAMS, run_local_sensitivity_sweep

# Progress tracking
from .progress import ProgressTracker

# Runner
from .runner import apply_unemployment_smoothing, run_ensemble, run_single_simulation

# Scoring
from .scoring import SCORE_TARGETS, SCORE_WEIGHTS, compute_all_scores

# Sensitivity analysis
from .sensitivity import run_oat_sensitivity_analysis

# Visualization
from .visualization import visualize_configuration

__all__ = [
    # Config
    "FIXED_PARAMS",
    "OAT_PARAM_GRID",
    "CALIBRATION_PARAM_GRID",
    "apply_config_offsets",
    # Scoring
    "SCORE_TARGETS",
    "SCORE_WEIGHTS",
    "compute_all_scores",
    # Runner
    "apply_unemployment_smoothing",
    "run_single_simulation",
    "run_ensemble",
    # Sensitivity
    "run_oat_sensitivity_analysis",
    # Grid search
    "generate_combinations",
    "run_grid_search",
    # Local sweep
    "LOCAL_SWEEP_PARAMS",
    "run_local_sensitivity_sweep",
    # Bayesian optimization
    "BO_SPACE_CONFIG",
    "run_bayesian_optimization",
    # Consistency analysis
    "ConsistencyResult",
    "run_consistency_analysis",
    # Progress
    "ProgressTracker",
    # Checkpoint
    "CheckpointManager",
    # Visualization
    "visualize_configuration",
]
