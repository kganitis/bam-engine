"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Modules:
    metrics: Compute validation metrics from simulation results
    targets/: YAML files defining target values for different scenarios
"""

from validation.metrics import BaselineMetrics, compute_baseline_metrics

__all__ = ["BaselineMetrics", "compute_baseline_metrics"]
