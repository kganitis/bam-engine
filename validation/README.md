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

# Growth+ scenario (Section 3.9.2)
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

### Growth+ (Section 3.9.2)

Endogenous productivity growth via R&D investment - adds productivity growth, wage growth, financial dynamics, and trend validation.

## API Reference

### Validation Functions

- `run_validation(**kwargs) -> ValidationScore` - Single run baseline validation
- `run_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed baseline testing
- `run_growth_plus_validation(**kwargs) -> ValidationScore` - Single run Growth+ validation
- `run_growth_plus_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed Growth+ testing

### Scenario Functions

- `run_baseline_scenario(seed, n_periods, burn_in, show_plot) -> BaselineMetrics` - Run baseline with visualization
- `run_growth_plus_scenario(seed, n_periods, burn_in, show_plot) -> GrowthPlusMetrics` - Run Growth+ with visualization

### Report Functions

- `print_validation_report(result)` - Formatted baseline report
- `print_stability_report(result)` - Formatted stability report
- `print_growth_plus_report(result)` - Formatted Growth+ report
- `print_growth_plus_stability_report(result)` - Formatted Growth+ stability report

### Growth+ Extension (RnD Role)

The R&D extension for Growth+ is in the `extensions/` package:

```python
from extensions.rnd import RnD, FirmsComputeRnDIntensity

# Use in custom simulations
sim = bam.Simulation.init(sigma_min=0.0, sigma_max=0.1, sigma_decay=-1.0)
sim.use_role(RnD)
```

### Core Types

- `ValidationScore` - Single validation result with metrics and total score
- `StabilityResult` - Multi-seed result with mean/std/pass_rate
- `MetricResult` - Individual metric validation result
- `MetricStats` - Per-metric statistics across seeds
- `MetricSpec` - Declarative specification for a validation metric
- `Scenario` - Bundles metric specs, config, and compute function
- `BaselineMetrics` - Computed metrics for baseline scenario
- `GrowthPlusMetrics` - Computed metrics for Growth+ scenario

### Scoring Functions

- `score_mean_tolerance(actual, target, tolerance)` - Score based on distance from target
- `score_range(actual, min_val, max_val)` - Score based on range position
- `score_pct_within_target(actual, target, min_pct)` - Score for percentage checks
- `score_outlier_penalty(outlier_pct, max_pct)` - Penalize excess outliers
- `compute_combined_score(stability)` - Combined score for calibration ranking

### Engine Functions

- `validate(scenario, seed, n_periods, **config)` - Generic validation engine
- `stability_test(scenario, seeds, n_periods, **config)` - Generic stability testing
- `evaluate_metric(spec, metrics, targets)` - Evaluate single metric
- `load_targets(scenario)` - Load YAML targets for scenario

### Constants

- `DEFAULT_STABILITY_SEEDS = list(range(20))` - 20 seeds for stability testing
- `BASELINE_WEIGHTS` - Metric weights for baseline scenario
- `GROWTH_PLUS_WEIGHTS` - Metric weights for Growth+ scenario

## Target Files

Target values are defined in YAML files with standardized keys:

- `validation/targets/baseline.yaml` - Baseline scenario targets (22 metrics)
- `validation/targets/growth_plus.yaml` - Growth+ scenario targets (39 metrics)

YAML structure uses standardized keys:

- `target`, `tolerance` for MEAN_TOLERANCE checks
- `min`, `max` for RANGE checks
- `target`, `min` for PCT_WITHIN checks
- `max_outlier`, `penalty_weight` for OUTLIER checks

## Module Structure

```
validation/
├── __init__.py              # Package exports and thin wrappers
├── types.py                 # Core types: MetricSpec, Scenario, CheckType, etc.
├── scoring.py               # Scoring and status check functions
├── engine.py                # Generic validate() and stability_test()
├── reporting.py             # Report printing functions
├── scenarios/
│   ├── __init__.py          # Re-exports scenarios
│   ├── baseline.py          # Baseline: metrics + computation + run_scenario()
│   ├── baseline_viz.py      # Baseline visualization functions
│   ├── growth_plus.py       # Growth+: metrics + computation + run_scenario()
│   └── growth_plus_viz.py   # Growth+ visualization functions
└── targets/
    ├── baseline.yaml        # Baseline target values
    └── growth_plus.yaml     # Growth+ target values

extensions/                  # Separate package for model extensions
└── rnd/
    ├── __init__.py          # Exports RnD role and events
    ├── role.py              # RnD role definition
    └── events.py            # R&D pipeline events
```

## Architecture

The validation package uses a **MetricSpec abstraction** for declarative metric validation:

```python
MetricSpec(
    name="unemployment_rate_mean",
    field="unemployment_mean",  # Attribute on Metrics dataclass
    check_type=CheckType.MEAN_TOLERANCE,
    target_path="metrics.unemployment_rate_mean",  # Path in YAML
    weight=1.5,
    group=MetricGroup.TIME_SERIES,
)
```

The generic `validate()` engine:

1. Loads targets from YAML
1. Runs simulation with scenario config
1. Computes metrics using scenario's compute function
1. Evaluates each MetricSpec against targets
1. Returns weighted ValidationScore

This design enables easy addition of new metrics (just add MetricSpec + field) and new scenarios (just create new scenario file).
