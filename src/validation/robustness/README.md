# Robustness Analysis (Section 3.10)

Comprehensive robustness checking for the BAM model, implementing the procedures described in
Section 3.10 of *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011).

The analysis consists of three parts:

1. **Internal Validity** (3.10.1, Part 1) — run multiple simulations with different random seeds to verify that results are robust to stochastic variation
1. **Sensitivity Analysis** (3.10.1, Part 2) — vary one parameter at a time to assess how changes in input parameters alter the output
1. **Structural Experiments** (3.10.2) — test model mechanisms: PA toggle (consumer loyalty) and entry neutrality (profit taxation)

> **Full documentation:** See the [Sphinx docs](https://kganitis.github.io/bam-engine/validation/robustness/index.html) for the complete robustness analysis reference.

## Quick Start

### CLI

```bash
# Full analysis (internal validity + sensitivity + structural experiments)
python -m validation.robustness

# Internal validity only
python -m validation.robustness --internal-only

# Sensitivity analysis only (specific experiments)
python -m validation.robustness --sensitivity-only --experiments credit_market,contract_length

# Structural experiments (Section 3.10.2)
python -m validation.robustness --structural-only
python -m validation.robustness --pa-experiment
python -m validation.robustness --entry-experiment

# Custom settings (Growth+ is the default; use --no-growth-plus for baseline)
python -m validation.robustness --seeds 10 --periods 500 --workers 4 --no-plots
```

### Python API

```python
from validation.robustness import (
    run_internal_validity,
    run_sensitivity_analysis,
    run_pa_experiment,
    run_entry_experiment,
    print_internal_validity_report,
    print_sensitivity_report,
    print_pa_report,
    print_entry_report,
    plot_comovements,
    plot_irf,
    plot_sensitivity_comovements,
    plot_pa_gdp_comparison,
    plot_entry_comparison,
)

# Part 1: Internal validity
result = run_internal_validity(n_seeds=20, n_periods=1000)
print_internal_validity_report(result)
plot_comovements(result)

# Part 2: Sensitivity analysis
sa = run_sensitivity_analysis(experiments=["credit_market", "contract_length"])
print_sensitivity_report(sa)

# Part 3: Structural experiments (Section 3.10.2)
pa = run_pa_experiment(n_seeds=20, n_periods=1000)
print_pa_report(pa)

entry = run_entry_experiment(n_seeds=20, n_periods=1000)
print_entry_report(entry)
```

### Growth+ (R&D Extension)

Growth+ is the default model (matching the book). Pass the setup hook to any analysis function:

```python
from validation.robustness import setup_growth_plus, GROWTH_PLUS_COLLECT_CONFIG

iv = run_internal_validity(
    setup_hook=setup_growth_plus,
    collect_config=GROWTH_PLUS_COLLECT_CONFIG,
)
sa = run_sensitivity_analysis(
    setup_hook=setup_growth_plus,
    collect_config=GROWTH_PLUS_COLLECT_CONFIG,
)
```

## Module Structure

```
validation/robustness/
├── __init__.py              # Public API exports
├── __main__.py              # CLI: python -m validation.robustness
├── stats.py                 # Pure statistical tools (HP filter, AR, IRF)
├── experiments.py           # 7 experiment definitions (5 parameter + 2 structural)
├── internal_validity.py     # Multi-seed analysis pipeline
├── sensitivity.py           # Univariate parameter sweep pipeline
├── structural.py            # Structural experiment orchestrators (Section 3.10.2)
├── viz.py                   # Matplotlib visualizations (Figure 3.9, 3.10, etc.)
├── reporting.py             # Text report formatting
├── REFERENCE.md             # Expected findings from the book (qualitative)
├── reference_values.yaml    # Quantitative benchmarks from book figures
├── README.md                # This file
└── output/                  # Saved figures
```
