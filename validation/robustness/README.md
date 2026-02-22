# Robustness Analysis (Section 3.10)

Comprehensive robustness checking for the BAM model, implementing the procedures described in
Section 3.10 of *Macroeconomics from the Bottom-up* (Delli Gatti et al., 2011).

The analysis consists of three parts:

1. **Internal Validity** (3.10.1, Part 1) — run multiple simulations with different random seeds to verify that results are robust to stochastic variation
1. **Sensitivity Analysis** (3.10.1, Part 2) — vary one parameter at a time to assess how changes in input parameters alter the output
1. **Structural Experiments** (3.10.2) — test model mechanisms: PA toggle (consumer loyalty) and entry neutrality (profit taxation)

## Quick Start

### CLI

```bash
# Full analysis (internal validity + sensitivity + structural experiments)
python -m validation.robustness

# Internal validity only
python -m validation.robustness --internal-only

# Sensitivity analysis only (specific experiments)
python -m validation.robustness --sensitivity-only --experiments credit_market,contract_length

# Structural experiments only (Section 3.10.2)
python -m validation.robustness --structural-only

# Individual structural experiments
python -m validation.robustness --pa-experiment
python -m validation.robustness --entry-experiment

# PA experiment without baseline comparison
python -m validation.robustness --pa-experiment --no-baseline

# Use baseline model instead of Growth+ (Growth+ is the default)
python -m validation.robustness --no-growth-plus

# Custom settings
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
    plot_pa_comovements,
    plot_entry_comparison,
)

# Part 1: Internal validity
result = run_internal_validity(n_seeds=20, n_periods=1000)
print_internal_validity_report(result)
plot_comovements(result)
plot_irf(result)

# Part 2: Sensitivity analysis
sa = run_sensitivity_analysis(experiments=["credit_market", "contract_length"])
print_sensitivity_report(sa)
for exp in sa.experiments.values():
    plot_sensitivity_comovements(exp)

# Part 3: Structural experiments (Section 3.10.2)
pa = run_pa_experiment(n_seeds=20, n_periods=1000)
print_pa_report(pa)
plot_pa_gdp_comparison(pa)
plot_pa_comovements(pa)

entry = run_entry_experiment(n_seeds=20, n_periods=1000)
print_entry_report(entry)
plot_entry_comparison(entry)
```

### Growth+ (R&D Extension)

```python
from validation.robustness import (
    run_internal_validity,
    run_sensitivity_analysis,
    setup_growth_plus,
    GROWTH_PLUS_COLLECT_CONFIG,
)

# Internal validity with Growth+ model
iv = run_internal_validity(
    setup_hook=setup_growth_plus,
    collect_config=GROWTH_PLUS_COLLECT_CONFIG,
)

# Sensitivity with Growth+ model
sa = run_sensitivity_analysis(
    setup_hook=setup_growth_plus,
    collect_config=GROWTH_PLUS_COLLECT_CONFIG,
)
```

## Internal Validity Analysis

### What It Does

Internal validity tests whether the model produces qualitatively similar results regardless of the random seed. It runs `n_seeds` simulations (default 20) with default parameters and checks:

| Check                     | Description                                                                                       | Book Reference                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Cross-simulation variance | Key macro statistics (GDP, unemployment, inflation, etc.) should have small variance across seeds | "the cross-simulation variance is remarkably small"                |
| Co-movement structure     | Cross-correlations between HP-filtered GDP and other variables at leads/lags should be consistent | Figure 3.9                                                         |
| AR model fit              | Individual seeds fit AR(2); cross-seed average fits AR(1) with parameter ~0.8                     | Section 3.10.1                                                     |
| Firm size distributions   | Should remain non-normal and positively skewed across all seeds                                   | "definitely invariant in its significant departure from normality" |
| Empirical curves          | Phillips, Okun, and Beveridge curves should emerge from each simulation                           | "continue to emerge from each simulation and on average"           |

### API

```text
def run_internal_validity(
    n_seeds: int = 20,         # Number of random seeds to test
    n_periods: int = 1000,     # Simulation periods per seed
    burn_in: int = 500,        # Burn-in periods to discard
    n_workers: int = 10,       # Parallel workers
    max_lag: int = 4,          # Maximum lead/lag for cross-correlations
    ar_order_single: int = 2,  # AR order for individual seeds
    ar_order_mean: int = 1,    # AR order for cross-seed average
    irf_periods: int = 20,     # IRF horizon
    baseline_seed: int = 0,    # Reference seed for baseline plots
    verbose: bool = True,
    setup_hook: Callable | None = None,     # Extension setup (e.g. setup_growth_plus)
    collect_config: dict | None = None,     # Custom collection config
    **config_overrides,        # Additional config overrides
) -> InternalValidityResult
```

### Result Structure

```text
InternalValidityResult:
    n_seeds: int
    n_periods: int
    burn_in: int
    seed_analyses: list[SeedAnalysis]       # Per-seed results
    baseline_comovements: dict[str, NDArray] # Baseline seed co-movements
    mean_comovements: dict[str, NDArray]     # Cross-seed mean co-movements
    std_comovements: dict[str, NDArray]      # Cross-seed std co-movements
    mean_ar_coeffs: NDArray                  # Mean AR coefficients
    mean_ar_order: int                       # AR order used for mean
    mean_ar_r_squared: float                 # Mean R²
    mean_irf: NDArray                        # Mean impulse-response function
    cross_sim_stats: dict[str, dict]         # {stat_name: {mean, std, min, max, cv}}
    n_collapsed: int                         # Number of collapsed simulations
    n_degenerate: int                        # Number of degenerate simulations
    collapse_rate: float                     # Property: n_collapsed / n_seeds
    degenerate_rate: float                   # Property: n_degenerate / n_seeds
```

Each `SeedAnalysis` contains:

- Co-movement cross-correlations (5 variables x 9 lags)
- AR model coefficients and R²
- Impulse-response function
- Summary statistics (unemployment, inflation, GDP growth, etc.)
- Empirical curve correlations (Phillips, Okun, Beveridge)
- Firm size distribution metrics (skewness, normality p-values for sales and net worth)

## Sensitivity Analysis

### What It Does

Univariate sensitivity analysis varies one parameter at a time while holding all others at baseline values. For each parameter value, it runs `n_seeds` simulations and computes the same statistics as the internal validity analysis.

### The Five Experiments

The book defines five parameter groups:

| #   | Experiment         | Parameter   | Values                 | Baseline   | Book Section |
| --- | ------------------ | ----------- | ---------------------- | ---------- | ------------ |
| i   | Credit market      | `max_H`     | 1, 2, 3, 4, 6          | 2          | 3.10.1(i)    |
| ii  | Goods market       | `max_Z`     | 2, 3, 4, 5, 6          | 2          | 3.10.1(ii)   |
| iii | Labor applications | `max_M`     | 2, 3, 4, 5, 6          | 4          | 3.10.1(iii)  |
| iv  | Contract length    | `theta`     | 1, 4, 6, 8, 10, 12, 14 | 8          | 3.10.1(iv)   |
| v   | Economy size       | multi-param | 7 configurations       | 100/500/10 | 3.10.1(v)    |

**Economy size configurations:**

- Proportional scaling: baseline (100F/500H/10B), 2x, 5x, 10x
- Class-specific doubling: 2x banks, 2x households, 2x firms

### Key Findings from the Book

| Experiment                  | Finding                                                                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Credit market (H)**       | General properties stable. H↑ makes price index coincident with output; net worth distribution becomes more Pareto-like. H=1 gives exponential-like distribution.    |
| **Goods market (Z)**        | Z↑ increases competition, smooths production, reduces firm size kurtosis. Real wages become lagging. Big firms cannot emerge → lower systemic risk.                  |
| **Labor applications (M)**  | M↓ makes prices pro-cyclical/lagging, increases instability. M↑ pushes wages above productivity, distribution becomes uniform/exponential.                           |
| **Contract length (theta)** | Extreme values cause degenerate dynamics. theta=1: frequent collapse (coordination failure). theta≥12: supply-side breakdown. theta=6-10: stable.                    |
| **Economy size**            | Proportional scaling preserves co-movements but smooths fluctuations. 2x banks: no effect. 2x households: faster growth. 2x firms: slower growth (more competition). |

### API

```text
def run_sensitivity_analysis(
    experiments: list[str] | None = None,  # None = all 5 experiments
    n_seeds: int = 20,                     # Seeds per parameter value
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 10,
    max_lag: int = 4,
    ar_order: int = 2,
    irf_periods: int = 20,
    verbose: bool = True,
    setup_hook: Callable | None = None,    # Extension setup (e.g. setup_growth_plus)
    collect_config: dict | None = None,    # Custom collection config
    **config_overrides,
) -> SensitivityResult
```

### Result Structure

```text
SensitivityResult:
    experiments: dict[str, ExperimentResult]  # Keyed by experiment name
    n_seeds_per_value: int
    n_periods: int
    burn_in: int

ExperimentResult:
    experiment: Experiment           # Experiment definition
    value_results: list[ValueResult] # One per parameter value
    baseline_idx: int                # Index of baseline in value_results
    baseline: ValueResult            # Property: value_results[baseline_idx]

ValueResult:
    label: str
    config_overrides: dict
    n_seeds: int
    n_collapsed: int
    mean_comovements: dict[str, NDArray]  # Mean co-movements across seeds
    std_comovements: dict[str, NDArray]
    mean_ar_coeffs: NDArray
    mean_ar_r_squared: float
    mean_irf: NDArray
    stats: dict[str, dict[str, float]]    # {stat_name: {mean, std, min, max, cv}}
    n_degenerate: int                     # Degenerate (superset of collapsed)
    collapse_rate: float                  # Property
    degenerate_rate: float                # Property
```

## Statistical Tools

The `stats` module provides pure statistical functions used by both analyses:

```python
from validation.robustness.stats import (
    hp_filter,
    cross_correlation,
    fit_ar,
    impulse_response,
)

# Hodrick-Prescott filter (lambda=1600 for quarterly data)
trend, cycle = hp_filter(series, lamb=1600.0)

# Cross-correlation at leads and lags
# Returns 2*max_lag+1 values: corr(x_t, y_{t+k}) for k = -max_lag..+max_lag
corrs = cross_correlation(gdp_cycle, unemployment_cycle, max_lag=4)

# Autoregressive model fitting via OLS
# Returns [constant, phi_1, ..., phi_p] and R²
coeffs, r_squared = fit_ar(gdp_cycle, order=2)

# Impulse-response function from AR coefficients
irf = impulse_response(coeffs, n_periods=20)
```

### HP Filter

Solves the penalized least-squares problem:

```
min_τ  Σ(y_t - τ_t)² + λ Σ(Δ²τ_t)²
```

via the sparse linear system `(I + λK'K)τ = y`, where K is the second-difference matrix. Uses `scipy.sparse.linalg.spsolve` for efficiency.

### Cross-Correlation

Computes `corr(x_t, y_{t+k})` for integer lags `k = -max_lag, ..., +max_lag`. At lag k=0, this is the contemporaneous correlation. Positive k means y *leads* x.

### AR Fitting

Fits `y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t` via ordinary least squares using `np.linalg.lstsq`.

### Impulse-Response Function

Simulates the response to a unit shock at t=0 through the AR recursion. For a stable AR(1) with coefficient φ, the IRF is simply φᵗ (exponential decay).

## Visualization

### Co-Movement Plot (Figure 3.9)

```python
from validation.robustness import plot_comovements

plot_comovements(iv_result, output_dir="path/to/output", show=True)
```

Creates a 3x2 grid of 5 panels showing cross-correlations at leads and lags (-4 to +4):

- **(a)** Unemployment
- **(b)** Productivity
- **(c)** Price index
- **(d)** Real interest rate
- **(e)** Real wage

Baseline run shown as '+' markers, cross-simulation mean as 'o' markers. Dotted lines at ±0.2 indicate the acyclicality band.

Saves individual panel PNGs plus a combined figure to `output/`.

### Impulse-Response Function Plot

```python
from validation.robustness import plot_irf

plot_irf(iv_result, show=True)
```

Compares baseline AR(2) IRF (dashed) with cross-simulation mean AR(1) IRF (solid).

### Sensitivity Co-Movement Comparison

```python
from validation.robustness import plot_sensitivity_comovements

for exp_result in sa.experiments.values():
    plot_sensitivity_comovements(exp_result, show=True)
```

Shows how co-movement structure changes across parameter values.

## Reporting

### Text Reports

```python
from validation.robustness import (
    print_internal_validity_report,
    print_sensitivity_report,
    format_internal_validity_report,
    format_sensitivity_report,
)

# Print to stdout
print_internal_validity_report(iv_result)
print_sensitivity_report(sa_result)

# Get as string
report_text = format_internal_validity_report(iv_result)
```

The internal validity report includes:

1. **Cross-simulation variance** — mean, std, CV for unemployment, inflation, GDP growth, etc.
1. **Co-movement structure** — contemporaneous correlations with cyclicality classification
1. **AR structure** — baseline AR(2) and mean AR(1) parameters
1. **Empirical curves** — Phillips, Okun, Beveridge correlation means with confirmation status
1. **Distribution invariance** — skewness and normality rejection rates

The sensitivity report includes per-experiment tables of:

- Key statistics (unemployment, inflation, GDP growth) across parameter values
- Contemporaneous co-movements across parameter values
- Collapse rates for extreme values

## Structural Experiments (Section 3.10.2)

### What They Do

Structural experiments test the model's **mechanisms** rather than just sweeping parameter values. Two experiments are implemented:

#### PA (Preferential Attachment) Experiment

Tests the effect of consumer loyalty on economic dynamics by disabling the "rich get richer" mechanism:

1. **Phase 1**: Run internal validity with `consumer_matching="random"` (PA off) — shows volatility drops and deep crises vanish
1. **Phase 2**: Run Z-sweep with PA off — shows minimal additional effect since the main driver of inequality is already disabled
1. **Optional baseline**: Run internal validity with PA on for side-by-side comparison

Expected findings (from the book):

- GDP volatility drops sharply
- Deep crises disappear
- Wages/prices become lagging or acyclical
- AR persistence drops from ~0.8 to ~0.4
- Firm size distribution becomes more uniform

#### Entry Neutrality Experiment

Tests whether automatic firm entry artificially drives recovery by imposing heavy profit taxation without redistribution:

- Sweeps `profit_tax_rate` from 0% to 90%
- Uses the `extensions/taxation/` package for profit taxation
- Expected finding: **monotonic degradation** of economic performance, confirming that the business cycle is genuinely endogenous

### API

```python
from validation.robustness import (
    run_pa_experiment,
    run_entry_experiment,
    PAExperimentResult,
    EntryExperimentResult,
)

# PA experiment (Phases 1-3)
pa = run_pa_experiment(n_seeds=20, n_periods=1000, include_baseline=True)

# Entry neutrality experiment
entry = run_entry_experiment(n_seeds=20, n_periods=1000)
```

### Result Structures

```text
PAExperimentResult:
    pa_off_validity: InternalValidityResult   # Phase 1: PA-off internal validity
    pa_off_z_sweep: SensitivityResult         # Phase 2: Z-sweep with PA off
    baseline_validity: InternalValidityResult | None  # Phase 3: optional PA-on baseline

EntryExperimentResult:
    tax_sweep: SensitivityResult              # Tax rate sweep results
```

### Visualization

```python
from validation.robustness import (
    plot_pa_gdp_comparison,
    plot_pa_comovements,
    plot_entry_comparison,
)

# PA: GDP time series comparison (Figure 3.10)
plot_pa_gdp_comparison(pa, show=True)

# PA: Co-movement comparison (PA-on vs PA-off)
plot_pa_comovements(pa, show=True)

# Entry: GDP growth and bankruptcy across tax rates
plot_entry_comparison(entry, show=True)
```

### Reporting

```python
from validation.robustness import (
    print_pa_report,
    print_entry_report,
    format_pa_report,
    format_entry_report,
)

print_pa_report(pa)  # PA-off validity + baseline comparison + Z-sweep
print_entry_report(entry)  # Tax sweep table + monotonicity assessment
```

## Custom Experiments

You can define custom experiments using the `Experiment` dataclass:

```python
from validation.robustness.experiments import Experiment
from validation.robustness.sensitivity import run_sensitivity_analysis

# Define a custom experiment
my_experiment = Experiment(
    name="custom_delta",
    description="Sensitivity to depreciation rate (delta)",
    param="delta",
    values=[0.05, 0.10, 0.15, 0.20],
    baseline_value=0.10,
)

# Register it temporarily
from validation.robustness.experiments import EXPERIMENTS

EXPERIMENTS["custom_delta"] = my_experiment

# Run it
result = run_sensitivity_analysis(experiments=["custom_delta"], n_seeds=10)
```

## Reference Materials

The expected results from the book are documented in two files within this package:

- **[`REFERENCE.md`](REFERENCE.md)** — Qualitative findings with direct quotes from Section 3.10.1, organized by analysis component. Includes co-movement classifications, AR structure expectations, sensitivity experiment findings, and a figure reference table.

- **[`reference_values.yaml`](reference_values.yaml)** — Quantitative benchmarks: approximate co-movement correlation values read from Figure 3.9, expected signs and classifications, AR parameters, and practical tolerance guidelines for comparing results.

Book figures are in `notes/BAM/figures/robustness/` (Figure 3.9 panels, Figure 3.10) and `notes/BAM/figures/dsge-comparison/` (Figure 3.7f IRF reference).

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

## Implementation Notes

### Parallelism

Both analyses use `ProcessPoolExecutor` for parallel simulation execution. Each seed runs as an independent worker process. The worker function (`_run_seed`) is defined at module level to ensure it's picklable.

### Collapsed and Degenerate Simulations

Extreme parameter values (e.g., `theta=1`) may cause economic collapse or degenerate dynamics. These are tracked separately:

- **Collapsed** — detected via `sim.ec.collapsed` (all firms or banks bankrupt)
- **Degenerate** — detected via heuristics: near-zero unemployment (\<0.5%), extreme unemployment (>50%), zero GDP variance, or NaN in co-movements. Degenerate is a superset of collapsed.

Both return NaN-filled `SeedAnalysis` objects, are excluded from cross-simulation averages, and are reported as counts/rates in the output.

### Real Interest Rate

Computed as the loan-principal-weighted average nominal interest rate minus inflation, matching the growth+ scenario calculation:

```
real_interest_rate_t = Σ(rate_i × principal_i) / Σ(principal_i) - inflation_t
```

where `principal_i` and `rate_i` are drawn from `LoanBook` relationship data (actual outstanding loans), not bank-level `credit_supply` (lending capacity).

### HP Filter

Uses the standard quarterly smoothing parameter λ=1600 (Hodrick & Prescott, 1997). The implementation uses a sparse matrix formulation with `scipy.sparse.linalg.spsolve`, which is efficient for typical time series lengths (500-5000 periods).

### Burn-In

The first `burn_in` periods (default 500) are discarded from all statistical analyses to avoid transient initialization effects. The `adjust_burn_in` utility from `validation.scenarios._utils` ensures burn-in doesn't exceed half the simulation length.

## Performance

Typical execution times (10-core machine, Apple M2):

| Configuration                             | Seeds        | Periods | Time    |
| ----------------------------------------- | ------------ | ------- | ------- |
| Internal validity (20 seeds)              | 20           | 1000    | ~2 min  |
| Single sensitivity experiment             | 20×5 values  | 1000    | ~10 min |
| Full sensitivity analysis (5 experiments) | 20×28 values | 1000    | ~50 min |
| Economy size experiment (10x)             | 20×7 values  | 1000    | ~30 min |

The economy size experiment takes longer because larger economies (5x, 10x) are proportionally slower to simulate.

## Testing

```bash
# Unit tests (statistical functions + experiment definitions)
pytest tests/unit/validation/test_robustness_stats.py -v

# Integration tests (full pipelines with small configurations)
pytest tests/validation/test_robustness.py -v -m slow

# All robustness tests
pytest tests/unit/validation/test_robustness_stats.py tests/validation/test_robustness.py -v
```
