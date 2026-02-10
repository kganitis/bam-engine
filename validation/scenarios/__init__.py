"""Validation scenarios for BAM Engine.

This subpackage provides scenario definitions and visualizations that reproduce
results from Delli Gatti et al. (2011).

Scenarios:
    baseline: Section 3.9.1 - Standard BAM model behavior
    growth_plus: Section 3.9.2 - Endogenous productivity growth via R&D
    buffer_stock: Section 3.9.3 - Buffer-stock consumption extension

Usage:
    # Run scenario with visualization
    from validation.scenarios.baseline import run_scenario
    run_scenario(seed=0, show_plot=True)

    from validation.scenarios.growth_plus import run_scenario
    run_scenario(seed=2, show_plot=True)

    from validation.scenarios.buffer_stock import run_scenario
    run_scenario(seed=0, show_plot=True)

    # Or run as module:
    # python -m validation.scenarios.baseline
    # python -m validation.scenarios.growth_plus
    # python -m validation.scenarios.buffer_stock

    # For extensions, import from extensions package:
    from extensions.rnd import RnD
    from extensions.buffer_stock import attach_buffer_stock
"""

from __future__ import annotations

# Baseline scenario
from validation.scenarios.baseline import COLLECT_CONFIG as BASELINE_COLLECT_CONFIG
from validation.scenarios.baseline import SCENARIO as BASELINE_SCENARIO
from validation.scenarios.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    load_baseline_targets,
)
from validation.scenarios.baseline import run_scenario as run_baseline_scenario

# Visualization functions (lazy import recommended to avoid matplotlib overhead)
# from validation.scenarios.baseline_viz import visualize_baseline_results
# from validation.scenarios.growth_plus_viz import visualize_growth_plus_results
# Buffer-stock scenario
from validation.scenarios.buffer_stock import (
    COLLECT_CONFIG as BUFFER_STOCK_COLLECT_CONFIG,
)
from validation.scenarios.buffer_stock import SCENARIO as BUFFER_STOCK_SCENARIO
from validation.scenarios.buffer_stock import (
    BufferStockMetrics,
    compute_buffer_stock_metrics,
    load_buffer_stock_targets,
)
from validation.scenarios.buffer_stock import run_scenario as run_buffer_stock_scenario

# Growth+ scenario
from validation.scenarios.growth_plus import (
    COLLECT_CONFIG as GROWTH_PLUS_COLLECT_CONFIG,
)
from validation.scenarios.growth_plus import SCENARIO as GROWTH_PLUS_SCENARIO
from validation.scenarios.growth_plus import (
    GrowthPlusMetrics,
    compute_growth_plus_metrics,
    load_growth_plus_targets,
)
from validation.scenarios.growth_plus import run_scenario as run_growth_plus_scenario

__all__ = [
    # Baseline
    "BASELINE_SCENARIO",
    "BASELINE_COLLECT_CONFIG",
    "BaselineMetrics",
    "compute_baseline_metrics",
    "load_baseline_targets",
    "run_baseline_scenario",
    # Growth+
    "GROWTH_PLUS_SCENARIO",
    "GROWTH_PLUS_COLLECT_CONFIG",
    "GrowthPlusMetrics",
    "compute_growth_plus_metrics",
    "load_growth_plus_targets",
    "run_growth_plus_scenario",
    # Buffer-stock
    "BUFFER_STOCK_SCENARIO",
    "BUFFER_STOCK_COLLECT_CONFIG",
    "BufferStockMetrics",
    "compute_buffer_stock_metrics",
    "load_buffer_stock_targets",
    "run_buffer_stock_scenario",
]
