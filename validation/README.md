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

# Buffer-stock scenario
from validation import run_buffer_stock_validation

result = run_buffer_stock_validation(seed=42)
```

## Scenario Visualization

Run scenarios with detailed visualization including target bounds, statistical annotations, and validation status indicators:

```bash
# Baseline scenario (Section 3.9.1)
python -m validation.scenarios.baseline

# Growth+ scenario (Section 3.9.2)
python -m validation.scenarios.growth_plus

# Buffer-stock scenario (Section 3.9.3)
python -m validation.scenarios.buffer_stock
```

Or use programmatically:

```python
from validation import (
    run_baseline_scenario,
    run_growth_plus_scenario,
    run_buffer_stock_scenario,
)

# Run with visualization
run_baseline_scenario(seed=0, show_plot=True)
run_growth_plus_scenario(seed=0, show_plot=True)
run_buffer_stock_scenario(seed=0, show_plot=True)

# Run without visualization (returns metrics)
metrics = run_baseline_scenario(seed=0, show_plot=False)
```

## Scenarios

### Baseline (Section 3.9.1)

Standard BAM model behavior — 25 metrics across 3 categories (TIME_SERIES, CURVES, DISTRIBUTION). Validates unemployment, inflation, GDP, real wages, vacancy rates, Phillips/Okun/Beveridge curve correlations, and firm size distribution.

### Growth+ (Section 3.9.2)

Endogenous productivity growth via R&D investment — 65 metrics across 6 categories:

| Category         | Count | Key Metrics                                                                |
| ---------------- | ----- | -------------------------------------------------------------------------- |
| TIME_SERIES      | 14    | Unemployment, inflation, GDP trend/growth, vacancy rates                   |
| CURVES           | 6     | Phillips, Okun, Beveridge correlations                                     |
| DISTRIBUTION     | 4     | Firm size metrics (skewness, tail ratios)                                  |
| GROWTH           | 11    | Productivity/wage growth, co-movement, recession detection                 |
| FINANCIAL        | 20    | Interest rates, fragility, price ratio, dispersions, Minsky classification |
| GROWTH_RATE_DIST | 10    | Tent-shape R², bounds checks, outlier percentages                          |

Notable additions over baseline: GDP cyclicality correlations (Minsky hypothesis validation), coefficient of variation for financial variables, and Laplace distribution fit (tent-shape R²) for growth rate distributions.

### Buffer-Stock (Section 3.9.3)

Buffer-stock consumption extension replacing the baseline mean-field MPC with an individual adaptive rule — ~30 metrics across 4 categories:

| Category     | Count | Key Metrics                                                      |
| ------------ | ----- | ---------------------------------------------------------------- |
| TIME_SERIES  | 10    | Unemployment, inflation, GDP trend/growth, vacancy rates         |
| CURVES       | 3     | Phillips, Okun, Beveridge correlations                           |
| DISTRIBUTION | 12    | Wealth CCDF fitting (Singh-Maddala, Dagum, GB2), Gini, MPC stats |
| FINANCIAL    | 5     | Interest rates, fragility, price ratio                           |

Notable additions over baseline: heavy-tailed wealth distribution fitting with R² on log-log CCDF (Figure 3.8 from the book), wealth Gini coefficient, MPC distribution statistics, and dissaving rate.

## API Reference

### Validation Functions

- `run_validation(**kwargs) -> ValidationScore` - Single run baseline validation
- `run_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed baseline testing
- `run_growth_plus_validation(**kwargs) -> ValidationScore` - Single run Growth+ validation
- `run_growth_plus_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed Growth+ testing
- `run_buffer_stock_validation(**kwargs) -> ValidationScore` - Single run buffer-stock validation
- `run_buffer_stock_stability_test(seeds, **kwargs) -> StabilityResult` - Multi-seed buffer-stock testing

### Scenario Functions

- `run_baseline_scenario(seed, n_periods, burn_in, show_plot) -> BaselineMetrics` - Run baseline with visualization
- `run_growth_plus_scenario(seed, n_periods, burn_in, show_plot) -> GrowthPlusMetrics` - Run Growth+ with visualization
- `run_buffer_stock_scenario(seed, n_periods, burn_in, show_plot) -> BufferStockMetrics` - Run buffer-stock with visualization

### Report Functions

- `print_validation_report(result)` - Formatted baseline report
- `print_baseline_stability_report(result)` - Formatted baseline stability report
- `print_growth_plus_report(result)` - Formatted Growth+ report
- `print_growth_plus_stability_report(result)` - Formatted Growth+ stability report
- `print_buffer_stock_report(result)` - Formatted buffer-stock report
- `print_buffer_stock_stability_report(result)` - Formatted buffer-stock stability report
- `print_report(result)` - Generic report (auto-detects scenario)
- `print_stability_report(result)` - Generic stability report

### Growth+ Extension (RnD Role)

The R&D extension for Growth+ is in the `extensions/` package:

```python
from extensions.rnd import RnD, RND_EVENTS

# Use in custom simulations
sim = bam.Simulation.init(sigma_min=0.0, sigma_max=0.1, sigma_decay=-1.0)
sim.use_role(RnD)
sim.use_events(*RND_EVENTS)
```

### Buffer-Stock Extension

The buffer-stock consumption extension is in `extensions/buffer_stock/`:

```python
from extensions.buffer_stock import BufferStock, BUFFER_STOCK_EVENTS

# Use in custom simulations
sim = bam.Simulation.init(buffer_stock_h=1.0)
sim.use_role(BufferStock, n_agents=sim.n_households)
sim.use_events(*BUFFER_STOCK_EVENTS)
```

### Core Types

- `ValidationScore` - Single validation result with metrics and total score
- `StabilityResult` - Multi-seed result with mean/std/pass_rate + per-metric `MetricStats`
- `MetricResult` - Individual metric validation result (name, status, actual, score, weight)
- `MetricStats` - Per-metric stability: mean_value, std_value, mean_score, std_score, pass_rate
- `MetricSpec` - Declarative specification: name, field, check_type, target_path, weight, group
- `Scenario` - Bundles metric_specs, collect_config, compute_metrics, setup_hook
- `BaselineMetrics` - Computed metrics for baseline scenario (25 fields)
- `GrowthPlusMetrics` - Computed metrics for Growth+ scenario (70+ fields)
- `BufferStockMetrics` - Computed metrics for buffer-stock scenario (40+ fields)
- `CheckType` - Enum: MEAN_TOLERANCE, RANGE, PCT_WITHIN, OUTLIER, BOOLEAN
- `MetricGroup` - Enum: TIME_SERIES, CURVES, DISTRIBUTION, GROWTH, FINANCIAL, GROWTH_RATE_DIST
- `Status` - Enum: PASS, WARN, FAIL

### Scoring Functions

- `score_mean_tolerance(actual, target, tolerance)` - Score based on distance from target (0-1)
- `score_range(actual, min_val, max_val)` - Score based on range position (0-1)
- `score_pct_within_target(actual, target, min_pct)` - Score for percentage checks (0-1)
- `score_outlier_penalty(outlier_pct, max_pct)` - Penalize excess outliers (0-1)
- `check_mean_tolerance(actual, target, tolerance)` - Status check (PASS/WARN/FAIL) for mean tolerance
- `check_range(actual, min_val, max_val)` - Status check for range
- `check_pct_within_target(actual, target, min_pct)` - Status check for percentage within target
- `check_outlier_penalty(outlier_pct, max_pct)` - Status check for outlier penalty
- `compute_combined_score(stability)` - Combined score: `mean_score * pass_rate * (1 - std_score)`

### Engine Functions

- `validate(scenario, seed, n_periods, **config)` - Generic validation engine
- `stability_test(scenario, seeds, n_periods, **config)` - Generic stability testing
- `evaluate_metric(spec, metrics, targets)` - Evaluate single metric
- `load_targets(scenario)` - Load YAML targets for scenario

### Calibration Support

- `get_validation_func(scenario)` - Get the validation function for a scenario
- `get_validation_funcs(scenario)` - Get `(validator, stability_runner, report_printer, stability_printer)` tuple

### Constants

- `DEFAULT_STABILITY_SEEDS = list(range(20))` - 20 seeds for stability testing
- `BASELINE_WEIGHTS` - Metric weights for baseline scenario
- `GROWTH_PLUS_WEIGHTS` - Metric weights for Growth+ scenario
- `BUFFER_STOCK_WEIGHTS` - Metric weights for buffer-stock scenario

## Target Files

Target values are defined in YAML files with standardized keys:

- `validation/targets/baseline.yaml` - Baseline scenario targets (25 metrics)
- `validation/targets/growth_plus.yaml` - Growth+ scenario targets (65 metrics)
- `validation/targets/buffer_stock.yaml` - Buffer-stock scenario targets (~30 metrics)

YAML structure uses standardized keys per check type:

| CheckType        | YAML Keys                       | Purpose                           |
| ---------------- | ------------------------------- | --------------------------------- |
| `MEAN_TOLERANCE` | `target`, `tolerance`           | Value within target +/- tolerance |
| `RANGE`          | `min`, `max`                    | Value within [min, max]           |
| `PCT_WITHIN`     | `target`, `min`                 | Percentage meeting threshold      |
| `OUTLIER`        | `max_outlier`, `penalty_weight` | Penalize excess outliers          |
| `BOOLEAN`        | `threshold` (in MetricSpec)     | Simple > or < check               |

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
│   ├── _utils.py            # Shared utilities: IQR filtering, burn-in adjustment
│   ├── baseline.py          # Baseline: metrics + computation + run_scenario()
│   ├── baseline_viz.py      # Baseline visualization (8-panel)
│   ├── growth_plus.py       # Growth+: metrics + computation + run_scenario()
│   ├── growth_plus_viz.py   # Growth+ visualization (16-panel + recession bands)
│   ├── buffer_stock.py      # Buffer-stock: metrics + computation + run_scenario()
│   ├── buffer_stock_viz.py  # Buffer-stock visualization (8-panel + CCDF)
│   └── output/              # Saved visualization panels
│       ├── baseline/        # Individual baseline scenario panels
│       ├── growth-plus/     # Individual growth+ scenario panels
│       └── buffer-stock/    # Individual buffer-stock scenario panels
└── targets/
    ├── baseline.yaml        # Baseline target values (25 metrics)
    ├── growth_plus.yaml     # Growth+ target values (65 metrics)
    └── buffer_stock.yaml    # Buffer-stock target values (~30 metrics)

extensions/                  # Separate package for model extensions
├── rnd/
│   ├── __init__.py          # Exports RnD role and events
│   ├── role.py              # RnD role definition
│   └── events.py            # R&D pipeline events
└── buffer_stock/
    ├── __init__.py          # Exports BufferStock role and events
    ├── role.py              # BufferStock role definition
    └── events.py            # Buffer-stock consumption events
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
