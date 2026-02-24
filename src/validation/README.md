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

# Buffer-stock scenario (Section 3.9.4)
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

### Buffer-Stock (Section 3.9.4)

Buffer-stock consumption extension replacing the baseline mean-field MPC with an individual adaptive rule — ~30 metrics across 4 categories:

| Category     | Count | Key Metrics                                                      |
| ------------ | ----- | ---------------------------------------------------------------- |
| TIME_SERIES  | 10    | Unemployment, inflation, GDP trend/growth, vacancy rates         |
| CURVES       | 3     | Phillips, Okun, Beveridge correlations                           |
| DISTRIBUTION | 12    | Wealth CCDF fitting (Singh-Maddala, Dagum, GB2), Gini, MPC stats |
| FINANCIAL    | 5     | Interest rates, fragility, price ratio                           |

Notable additions over baseline: heavy-tailed wealth distribution fitting with R² on log-log CCDF (Figure 3.8 from the book), wealth Gini coefficient, MPC distribution statistics, and dissaving rate. MPC metrics are adjusted using the core `Shareholder` role's per-period dividend data to remove the dividend artifact from the buffer-stock formula.

### Robustness Analysis (Section 3.10)

Internal validity and sensitivity analysis to verify model robustness across random seeds and parameter variations. See [`validation/robustness/README.md`](robustness/README.md) for full documentation.

```bash
# CLI
python -m validation.robustness                    # Full analysis
python -m validation.robustness --internal-only    # Internal validity only
python -m validation.robustness --sensitivity-only --experiments credit_market
```

```python
from validation.robustness import (
    run_internal_validity,
    run_sensitivity_analysis,
    print_internal_validity_report,
    print_sensitivity_report,
    plot_comovements,
)

# Internal validity (20 seeds, default params)
result = run_internal_validity(n_seeds=20, n_periods=1000)
print_internal_validity_report(result)
plot_comovements(result)

# Sensitivity analysis (5 experiments)
sa = run_sensitivity_analysis()
print_sensitivity_report(sa)
```

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

The R&D extension for Growth+ is in the `extensions` package (under `src/extensions/`):

```python
from extensions.rnd import RnD, RND_EVENTS

# Use in custom simulations
sim = bam.Simulation.init(sigma_min=0.0, sigma_max=0.1, sigma_decay=-1.0)
sim.use_role(RnD)
sim.use_events(*RND_EVENTS)
```

### Buffer-Stock Extension

The buffer-stock consumption extension is in `src/extensions/buffer_stock/`:

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

- `fail_escalation_multiplier(weight)` - Compute weight-based fail escalation multiplier (see below)
- `score_mean_tolerance(actual, target, tolerance)` - Score based on distance from target (0-1)
- `score_range(actual, min_val, max_val)` - Score based on range position (0-1)
- `score_pct_within_target(actual, target, min_pct)` - Score for percentage checks (0-1)
- `score_outlier_penalty(outlier_pct, max_pct)` - Penalize excess outliers (0-1)
- `check_mean_tolerance(actual, target, tolerance, escalation=1.0)` - Status check (PASS/WARN/FAIL) for mean tolerance
- `check_range(actual, min_val, max_val, escalation=1.0)` - Status check for range
- `check_pct_within_target(actual, target, min_pct, escalation=1.0)` - Status check for percentage within target
- `check_outlier_penalty(outlier_pct, max_pct, escalation=1.0)` - Status check for outlier penalty
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

- `src/validation/scenarios/baseline/targets.yaml` - Baseline scenario targets (25 metrics)
- `src/validation/scenarios/growth_plus/targets.yaml` - Growth+ scenario targets (65 metrics)
- `src/validation/scenarios/buffer_stock/targets.yaml` - Buffer-stock scenario targets (~30 metrics)

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
├── __init__.py              # Registry-driven package exports
├── types.py                 # Core types: MetricSpec, Scenario, CheckType, etc.
├── scoring.py               # Scoring and status check functions
├── engine.py                # Generic validate() and stability_test()
├── reporting.py             # Report printing functions
├── scenarios/
│   ├── __init__.py          # Scenario registry + get_scenario()
│   ├── _utils.py            # Shared utilities: IQR filtering, burn-in adjustment
│   ├── baseline/
│   │   ├── __init__.py      # Metrics + computation + run_scenario()
│   │   ├── viz.py           # Visualization (8-panel)
│   │   ├── targets.yaml     # Target values (25 metrics)
│   │   ├── output/          # Saved visualization panels
│   │   └── __main__.py      # python -m entry point
│   ├── growth_plus/
│   │   ├── __init__.py      # Metrics + computation + run_scenario()
│   │   ├── viz.py           # Visualization (16-panel + recession bands)
│   │   ├── targets.yaml     # Target values (65 metrics)
│   │   ├── output/          # Saved visualization panels
│   │   └── __main__.py      # python -m entry point
│   └── buffer_stock/
│       ├── __init__.py      # Metrics + computation + run_scenario()
│       ├── viz.py           # Visualization (8-panel + CCDF)
│       ├── targets.yaml     # Target values (~30 metrics)
│       ├── output/          # Saved visualization panels
│       └── __main__.py      # python -m entry point
├── robustness/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI: python -m validation.robustness
│   ├── stats.py             # HP filter, cross-correlation, AR fitting, IRF
│   ├── experiments.py       # 5 experiment definitions (Section 3.10.1)
│   ├── internal_validity.py # Multi-seed analysis pipeline
│   ├── sensitivity.py       # Univariate parameter sweep pipeline
│   ├── viz.py               # Co-movement plots (Figure 3.9), IRF, sensitivity
│   ├── reporting.py         # Text report formatting
│   └── output/              # Saved figures

../extensions/               # Sibling package for model extensions
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

### Weight-Based Fail Escalation

The status check functions use a weight-based escalation multiplier to adjust the WARN→FAIL boundary per metric. High-weight metrics fail more easily; low-weight metrics are more lenient:

| Weight | Multiplier | Effect (MEAN_TOLERANCE, normal FAIL at 2× tol) |
| ------ | ---------- | ---------------------------------------------- |
| 3.0    | 0.5        | FAIL at 1× tolerance (stricter)                |
| 2.0    | 1.0        | FAIL at 2× tolerance (normal)                  |
| 1.5    | 2.0        | FAIL at 4× tolerance                           |
| 1.0    | 3.0        | FAIL at 6× tolerance                           |
| 0.5    | 4.0        | FAIL at 8× tolerance (lenient)                 |

Formula: `clamp(5 - 2 × weight, 0.5, 5.0)`. BOOLEAN checks are exempt (always natural PASS/FAIL).

`evaluate_metric()` computes the multiplier from `MetricSpec.weight` and passes it to each check function via the `escalation` parameter. Scores are unaffected — only the status threshold changes.
