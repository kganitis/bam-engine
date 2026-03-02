# Validation Package

Tools for validating BAM simulation results against empirical targets from Delli Gatti et al. (2011).

> **Full documentation:** See the [Sphinx docs](https://kganitis.github.io/bam-engine/validation/index.html) for the complete validation reference including scenarios, scoring system, CLI, and robustness analysis.

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

```bash
# Baseline scenario (Section 3.9.1)
python -m validation.scenarios.baseline

# Growth+ scenario (Section 3.9.2)
python -m validation.scenarios.growth_plus

# Buffer-stock scenario (Section 3.9.4)
python -m validation.scenarios.buffer_stock
```

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
│   │   └── __main__.py      # python -m entry point
│   ├── growth_plus/
│   │   ├── __init__.py      # Metrics + computation + run_scenario()
│   │   ├── viz.py           # Visualization (16-panel + recession bands)
│   │   ├── targets.yaml     # Target values (65 metrics)
│   │   └── __main__.py      # python -m entry point
│   └── buffer_stock/
│       ├── __init__.py      # Metrics + computation + run_scenario()
│       ├── viz.py           # Visualization (8-panel + CCDF)
│       ├── targets.yaml     # Target values (~30 metrics)
│       └── __main__.py      # python -m entry point
├── robustness/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI: python -m validation.robustness
│   ├── stats.py             # HP filter, cross-correlation, AR fitting, IRF
│   ├── experiments.py       # Experiment definitions (Section 3.10.1)
│   ├── internal_validity.py # Multi-seed analysis pipeline
│   ├── sensitivity.py       # Univariate parameter sweep pipeline
│   ├── viz.py               # Co-movement plots (Figure 3.9), IRF, sensitivity
│   └── reporting.py         # Text report formatting

../extensions/               # Sibling package for model extensions
├── rnd/                     # R&D extension (Growth+ scenario)
└── buffer_stock/            # Buffer-stock consumption extension
```
