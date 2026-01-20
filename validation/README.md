# Validation Package

Tools for validating BAM simulation results against empirical targets from Delli Gatti et al. (2011).

## Quick Start

```python
from validation import run_validation, run_stability_test

# Single validation run
result = run_validation(seed=42, n_periods=1000)
print(f"Score: {result.total_score:.3f}, Passed: {result.passed}")

# Multi-seed stability test
stability = run_stability_test(seeds=[0, 42, 123, 456, 789])
print(f"Mean: {stability.mean_score:.3f} ± {stability.std_score:.3f}")
print(f"Pass rate: {stability.pass_rate:.0%}")

# Growth+ scenario
from validation import run_growth_plus_validation

result = run_growth_plus_validation(seed=42)
```

## Scenarios

### Baseline (Section 3.9.1)

Standard BAM model behavior - validates unemployment, inflation, GDP, Phillips/Okun/Beveridge curves, firm size distribution.

### Growth+ (Section 3.8)

Endogenous productivity growth via R&D investment - adds productivity growth, wage growth, and trend validation.

## API Reference

### Validation Functions

- `run_validation(**kwargs) -> ValidationScore` - Single run baseline validation
- `run_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed baseline testing
- `run_growth_plus_validation(**kwargs) -> ValidationScore` - Single run Growth+ validation
- `run_growth_plus_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed Growth+ testing

### Report Functions

- `print_validation_report(result)` - Formatted baseline report
- `print_stability_report(result)` - Formatted stability report
- `print_growth_plus_report(result)` - Formatted Growth+ report
- `print_growth_plus_stability_report(result)` - Formatted Growth+ stability report

### Core Types

- `ValidationScore` - Single validation result with metrics and total score
- `StabilityResult` - Multi-seed result with mean/std/pass_rate
- `MetricResult` - Individual metric validation result
- `MetricStats` - Per-metric statistics across seeds

### Scoring Functions

- `score_mean_tolerance(actual, target, tolerance)` - Score based on distance from target
- `score_range(actual, min_val, max_val)` - Score based on range position
- `compute_combined_score(stability)` - Combined score for calibration ranking

### Constants

- `DEFAULT_STABILITY_SEEDS = [0, 42, 123, 456, 789]`
- `BASELINE_WEIGHTS` - Metric weights for baseline scenario
- `GROWTH_PLUS_WEIGHTS` - Metric weights for Growth+ scenario

## Target Files

Target values are defined in YAML files with inline documentation:

- `validation/targets/baseline.yaml` - Baseline scenario targets
- `validation/targets/growth_plus.yaml` - Growth+ scenario targets

## Module Structure

```
validation/
├── __init__.py      # Package exports
├── core.py          # Types, dataclasses, scoring functions
├── runners.py       # Validation runner functions
├── metrics.py       # Metrics computation from simulation results
└── targets/
    ├── baseline.yaml     # Baseline target values
    └── growth_plus.yaml  # Growth+ target values
```
