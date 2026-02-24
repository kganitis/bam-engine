# Calibration Package

Tools for finding optimal parameter values through sensitivity analysis, focused grid search, and tiered stability testing.

## Quick Start

### Command Line

```bash
# Run all phases (baseline, Morris default)
python -m calibration --scenario baseline --workers 10

# Run individual phases
python -m calibration --phase sensitivity --scenario baseline
python -m calibration --phase morris --scenario baseline
python -m calibration --phase grid --scenario baseline
python -m calibration --phase stability --scenario baseline

# Use OAT instead of Morris for sensitivity
python -m calibration --phase sensitivity --method oat --scenario baseline

# Morris with custom number of trajectories
python -m calibration --phase sensitivity --morris-trajectories 20

# Pairwise interaction analysis
python -m calibration --phase pairwise --scenario baseline

# Resume interrupted grid search
python -m calibration --phase grid --scenario baseline --resume

# Custom stability tiers
python -m calibration --phase stability --stability-tiers "50:10,20:30,5:100"
```

### Programmatic

```python
from calibration import (
    run_morris_screening,
    print_morris_report,
    build_focused_grid,
    run_focused_calibration,
    export_best_config,
    compare_configs,
)

# Phase 1: Morris screening (default, recommended)
morris = run_morris_screening(scenario="baseline", n_workers=10, n_seeds=3)
print_morris_report(morris)
sensitivity = morris.to_sensitivity_result()

# Phase 2: Build focused grid
grid, fixed = build_focused_grid(sensitivity)

# Phases 3-4: Grid search + tiered stability (default tiers: 100:10, 50:20, 10:100)
results = run_focused_calibration(grid, fixed)

# Export best config
export_best_config(results[0], "baseline")

# Before/after comparison
comparison = compare_configs({}, results[0].params, "baseline")
```

## Calibration Process

### Phase 1: Sensitivity Analysis

Two methods are available for Phase 1:

#### Morris Method (Default)

Morris Method (Morris 1991) runs multiple OAT trajectories from random starting points across the parameter space, producing two measures per parameter:

- **mu\*** (mean absolute elementary effect): Average importance across different parameter contexts
- **sigma** (std of elementary effects): Interaction/nonlinearity indicator — how much the effect varies depending on other parameters

Classification uses a **dual threshold**:

- `INCLUDE: mu* > threshold OR sigma > threshold` — important OR interaction-prone
- `FIX: mu* <= threshold AND sigma <= threshold` — truly unimportant

This catches parameters whose sensitivity depends on other parameter values (interaction effects) that standard OAT would miss.

Output: `output/{scenario}_morris.json` + `output/{scenario}_sensitivity.json`

#### OAT (One-at-a-Time)

Traditional OAT testing of each parameter while holding others at defaults. Faster but sensitive to baseline choice — parameter rankings can change depending on which defaults are used. Use `--method oat` to select.

Output: `output/{scenario}_sensitivity.json`

#### Morris vs OAT: When to Use Which

| Aspect                   | Morris                   | OAT                    |
| ------------------------ | ------------------------ | ---------------------- |
| Interaction detection    | Yes (via sigma)          | No                     |
| Baseline dependency      | Minimal (random starts)  | High (single baseline) |
| Cost (baseline, 3 seeds) | ~660 sim runs (~2.5 min) | ~216 sim runs (~50s)   |
| Recommended for          | Production calibration   | Quick exploration      |

### Phase 2: Grid Screening

Categorizes parameters by sensitivity and builds a focused grid:

- **INCLUDE (Δ > threshold)**: Include in grid search with all values
- **FIX (Δ ≤ threshold)**: Fix at best value from sensitivity analysis

**Value pruning** (optional): Drops grid values whose OAT score is more than `pruning_threshold` below the best value for that parameter. Default: `auto` (2× sensitivity threshold). Use `--pruning-threshold none` to disable.

Screens all combinations with single seed. Includes parameter pattern analysis (which values appear most often in top configs) and checkpointing for resumability.

Output: `output/{scenario}_screening.json`

### Phase 3: Tiered Stability Testing

Incremental tiered tournament to efficiently find the most stable config:

```
Default tiers: [(100, 10), (50, 20), (10, 100)]
  Tier 1: top 100 configs × 10 seeds → rank → keep top 50
  Tier 2: top 50 × +10 new seeds (20 total) → rank → keep top 10
  Tier 3: top 10 × +80 new seeds (100 total) → final ranking
```

Each tier only runs NEW seeds (incremental), accumulating all prior results. Total: 100×10 + 50×10 + 10×80 = 2,300 vs naive 100×100 = 10,000.

Output: `output/{scenario}_calibration_results.json` + `output/{scenario}_best_config.yml`

### Pairwise Interaction Analysis (Optional)

After OAT identifies HIGH-sensitivity params, tests all 2-param combinations to find synergies and conflicts:

```bash
python -m calibration --phase pairwise --scenario baseline
```

Output: `output/{scenario}_pairwise.json`

## Scenarios

| Scenario       | Parameters            | Description                              |
| -------------- | --------------------- | ---------------------------------------- |
| `baseline`     | 14 common             | Standard BAM model (Section 3.9.1)       |
| `growth_plus`  | 14 + 2 R&D            | Endogenous R&D growth (Section 3.9.2)    |
| `buffer_stock` | 14 + 3 (R&D + buffer) | Buffer-stock consumption (Section 3.9.4) |

## Parameter Grid

The 14 common parameters cover:

- **Initial conditions**: `price_init`, `min_wage_ratio`, `net_worth_ratio`, `equity_base_init`, `savings_init`
- **New firm entry**: `new_firm_size_factor`, `new_firm_production_factor`, `new_firm_wage_factor`, `new_firm_price_markup`
- **Economy-wide**: `beta`
- **Search frictions**: `max_M`
- **Implementation variants**: `max_loan_to_net_worth`, `max_leverage`, `job_search_method`

Extension-specific:

- **Growth+**: `sigma_decay`, `sigma_max`
- **Buffer-stock**: `buffer_stock_h` (+ R&D params)

## API Reference

### Morris Method Screening

- `run_morris_screening(scenario, grid, n_trajectories, seed, n_seeds, n_periods, n_workers) -> MorrisResult`
- `print_morris_report(result, mu_star_threshold, sigma_threshold)` — Formatted report with dual classification
- `MorrisResult` — Full result with `effects`, `to_sensitivity_result()` for downstream compatibility
- `MorrisParameterEffect` — Per-parameter mu\*, sigma, elementary effects, value scores

### OAT Sensitivity Analysis

- `run_sensitivity_analysis(scenario, grid, baseline, seed, n_seeds, n_periods, n_workers) -> SensitivityResult`
- `print_sensitivity_report(result)` — Formatted report with score decomposition
- `SensitivityResult` — Full result with rankings, `avg_time_per_run`, `n_seeds`
- `ParameterSensitivity` — Per-parameter data with `group_scores`

### Pairwise Interaction

- `run_pairwise_analysis(params, grid, best_values, scenario, ...) -> PairwiseResult`
- `print_pairwise_report(result)` — Formatted synergy/conflict report
- `PairwiseResult` — Interactions ranked by strength
- `PairInteraction` — Single pair interaction data

### Optimization

- `build_focused_grid(sensitivity, ..., sensitivity_threshold, pruning_threshold) -> tuple[grid, fixed]`
- `run_screening(combinations, scenario, ...) -> list[CalibrationResult]` — With progress/ETA/checkpointing
- `run_tiered_stability(candidates, scenario, tiers, ...) -> list[CalibrationResult]`
- `run_focused_calibration(grid, fixed, ...) -> list[CalibrationResult]` — Orchestrates screening + stability
- `screen_single_seed(params, scenario, seed, n_periods) -> CalibrationResult`
- `evaluate_stability(params, scenario, seeds, n_periods) -> CalibrationResult`
- `analyze_parameter_patterns(results, top_n) -> dict` — Value frequency analysis
- `export_best_config(result, scenario) -> Path` — Export as YAML
- `compare_configs(default, calibrated, scenario) -> ComparisonResult` — Side-by-side comparison
- `parse_stability_tiers(tiers_str) -> list[tuple]` — Parse "100:10,50:20" format

### Progress Helpers

- `format_eta(remaining, avg_time, n_workers) -> str`
- `format_progress(completed, total, remaining, eta) -> str`

### Parameter Space

- `PARAMETER_GRID` — Alias for baseline grid (backwards compat)
- `DEFAULT_VALUES` — Alias for baseline overrides (backwards compat)
- `get_parameter_grid(scenario)` — Get grid for specific scenario
- `get_default_values(scenario)` — Get overrides for specific scenario
- `generate_combinations(grid, scenario)` — Generate all parameter combinations
- `count_combinations(grid, scenario)` — Count total combinations

## CLI Options

| Option                    | Default               | Description                                                         |
| ------------------------- | --------------------- | ------------------------------------------------------------------- |
| `--scenario`              | baseline              | Scenario to calibrate (baseline, growth_plus, buffer_stock)         |
| `--phase`                 | all                   | Run a single phase (sensitivity, morris, grid, stability, pairwise) |
| `--method`                | morris                | Sensitivity method: "morris" or "oat"                               |
| `--morris-trajectories`   | 10                    | Number of Morris trajectories                                       |
| `--workers`               | 10                    | Number of parallel workers                                          |
| `--periods`               | 1000                  | Simulation periods                                                  |
| `--output`                | auto                  | Output file for results                                             |
| `--sensitivity-threshold` | 0.02                  | Minimum Δ for INCLUDE in grid search                                |
| `--pruning-threshold`     | "auto"                | Max score gap for keeping values: "auto", "none", or float          |
| `--sensitivity-seeds`     | 3                     | Seeds per sensitivity evaluation                                    |
| `--stability-tiers`       | "100:10,50:20,10:100" | Tiers as "configs:seeds,..."                                        |
| `--resume`                | false                 | Resume from checkpoint                                              |

## Output Files

Results are saved to `calibration/output/`:

- `{scenario}_morris.json` — Morris method detailed results (mu\*, sigma, elementary effects)
- `{scenario}_sensitivity.json` — Sensitivity analysis (from Morris or OAT)
- `{scenario}_screening.json` — Grid screening results with patterns
- `{scenario}_calibration_results.json` — Final stability-tested results
- `{scenario}_best_config.yml` — Best config as ready-to-use YAML
- `{scenario}_pairwise.json` — Pairwise interaction analysis
- `{scenario}_*_checkpoint.json` — Intermediate checkpoints (auto-deleted)

## Module Structure

```
calibration/
├── __init__.py         # Package exports
├── __main__.py         # CLI entry point
├── cli.py              # Command-line interface (phase orchestration)
├── morris.py           # Morris Method screening (elementary effects)
├── sensitivity.py      # OAT sensitivity + pairwise interaction
├── optimizer.py        # Grid search, tiered stability, patterns, export, comparison
├── parameter_space.py  # Parameter grids (14 common + extensions)
├── run_grid_search.py  # Alternative standalone grid search script
└── output/             # Results JSON/YAML files
```
