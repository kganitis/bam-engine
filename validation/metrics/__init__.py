"""Validation metrics computation for BAM Engine.

This subpackage provides functions to compute validation metrics from simulation
results. These metrics are used both by scenario scripts for visualization
and by validation tests for comparing against book targets.

Modules:
    baseline: Metrics for baseline scenario (Section 3.9.1)
    growth_plus: Metrics for Growth+ scenario (Section 3.9.2)

The metrics correspond to the 8 figures in Delli Gatti et al. (2011):
    1. Log GDP (real output)
    2. Unemployment rate
    3. Inflation rate
    4. Productivity / Real wage
    5. Phillips curve (wage inflation vs unemployment)
    6. Okun curve (output growth vs unemployment growth)
    7. Beveridge curve (vacancy rate vs unemployment)
    8. Firm size distribution
"""

from __future__ import annotations

# Baseline metrics
from validation.metrics.baseline import (
    BASELINE_COLLECT_CONFIG,
    BaselineMetrics,
    compute_baseline_metrics,
    load_baseline_targets,
)

# Growth+ metrics
from validation.metrics.growth_plus import (
    GROWTH_PLUS_COLLECT_CONFIG,
    GrowthPlusMetrics,
    compute_growth_plus_metrics,
    load_growth_plus_targets,
)

__all__ = [
    # Baseline
    "BaselineMetrics",
    "compute_baseline_metrics",
    "BASELINE_COLLECT_CONFIG",
    "load_baseline_targets",
    # Growth+
    "GrowthPlusMetrics",
    "compute_growth_plus_metrics",
    "GROWTH_PLUS_COLLECT_CONFIG",
    "load_growth_plus_targets",
]
