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

## Scenario Visualization

Run scenarios with detailed visualization including target bounds, statistical annotations, and validation status indicators:

```bash
# Baseline scenario (Section 3.9.1)
python -m validation.scenarios.baseline

# Growth+ scenario (Section 3.8)
python -m validation.scenarios.growth_plus
```

Or use programmatically:

```python
from validation import run_baseline_scenario, run_growth_plus_scenario

# Run with visualization
run_baseline_scenario(seed=0, show_plot=True)
run_growth_plus_scenario(seed=2, show_plot=True)

# Run without visualization (returns metrics)
metrics = run_baseline_scenario(seed=0, show_plot=False)
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

### Scenario Functions

- `run_baseline_scenario(seed, n_periods, burn_in, show_plot) -> BaselineMetrics` - Run baseline with visualization
- `run_growth_plus_scenario(seed, n_periods, burn_in, show_plot) -> GrowthPlusMetrics` - Run Growth+ with visualization
- `visualize_baseline_results(metrics, bounds, burn_in)` - Detailed baseline visualization
- `visualize_growth_plus_results(metrics, bounds, burn_in)` - Detailed Growth+ visualization

### Report Functions

- `print_validation_report(result)` - Formatted baseline report
- `print_stability_report(result)` - Formatted stability report
- `print_growth_plus_report(result)` - Formatted Growth+ report
- `print_growth_plus_stability_report(result)` - Formatted Growth+ stability report

### Growth+ Extension

The package includes the RnD role and custom events for the Growth+ scenario:

```python
from validation.scenarios.growth_plus_extension import RnD, FirmsComputeRnDIntensity

# Use in custom simulations
sim = bam.Simulation.init(sigma_min=0.0, sigma_max=0.1, sigma_decay=-1.0)
sim.use_role(RnD)
```

### Core Types

- `ValidationScore` - Single validation result with metrics and total score
- `StabilityResult` - Multi-seed result with mean/std/pass_rate
- `MetricResult` - Individual metric validation result
- `MetricStats` - Per-metric statistics across seeds
- `BaselineMetrics` - Computed metrics for baseline scenario
- `GrowthPlusMetrics` - Computed metrics for Growth+ scenario

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
├── __init__.py              # Package exports
├── core.py                  # Types, dataclasses, scoring functions
├── runners.py               # Validation runner functions
├── metrics/                 # Metrics computation subpackage
│   ├── __init__.py          # Re-exports all metrics
│   ├── _utils.py            # Shared utilities
│   ├── baseline.py          # Baseline scenario metrics
│   └── growth_plus.py       # Growth+ scenario metrics
├── scenarios/               # Scenario visualization subpackage
│   ├── __init__.py          # Re-exports all scenarios
│   ├── baseline.py          # Baseline scenario visualization
│   ├── growth_plus.py       # Growth+ scenario visualization
│   └── growth_plus_extension.py  # RnD role and custom events
└── targets/
    ├── baseline.yaml        # Baseline target values
    └── growth_plus.yaml     # Growth+ target values
```
