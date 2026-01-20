# Calibration Package

Tools for finding optimal parameter values through sensitivity analysis and focused grid search.

## Quick Start

### Command Line

```bash
# Run sensitivity analysis only
python -m calibration --sensitivity-only --workers 10

# Full calibration (baseline scenario)
python -m calibration --workers 10 --periods 1000

# Calibrate Growth+ scenario
python -m calibration --scenario growth_plus --workers 10

# Custom sensitivity thresholds
python -m calibration --high-threshold 0.08 --medium-threshold 0.04
```

### Programmatic

```python
from calibration import (
    run_sensitivity_analysis,
    build_focused_grid,
    run_focused_calibration,
    print_sensitivity_report,
)

# Phase 1: Sensitivity analysis
sensitivity = run_sensitivity_analysis(scenario="baseline", n_workers=10)
print_sensitivity_report(sensitivity)

# Phase 2: Build focused grid
grid, fixed = build_focused_grid(sensitivity)

# Phases 3-4: Grid search + stability testing
results = run_focused_calibration(grid, fixed, top_k=20)
print(f"Best: {results[0].combined_score:.4f}")
```

## Calibration Process

### Phase 1: Sensitivity Analysis

One-at-a-time (OAT) testing of each parameter while holding others at defaults. Identifies which parameters have the most impact on validation scores.

### Phase 2: Build Focused Grid

Categorizes parameters by sensitivity:

- **HIGH (Δ > 0.05)**: Include in grid search with full value range
- **MEDIUM (0.02-0.05)**: Fix at best value from sensitivity analysis
- **LOW (Δ ≤ 0.02)**: Fix at default value

### Phase 3: Grid Search Screening

Tests all combinations in the focused grid using a single seed. Parallel processing with `ProcessPoolExecutor`.

### Phase 4: Stability Testing

Multi-seed validation of top candidates from Phase 3. Ranks by combined score:

```
combined_score = mean_score * pass_rate * (1 - std_score)
```

## API Reference

### Sensitivity Analysis

- `run_sensitivity_analysis(scenario, grid, baseline, seed, n_periods, n_workers) -> SensitivityResult`
- `print_sensitivity_report(result)` - Formatted sensitivity report
- `SensitivityResult` - Full sensitivity analysis result with rankings
- `ParameterSensitivity` - Single parameter sensitivity data

### Optimization

- `build_focused_grid(sensitivity, scenario, high_threshold, medium_threshold) -> tuple[grid, fixed]`
- `run_focused_calibration(grid, fixed_params, scenario, top_k, n_workers, n_periods) -> list[CalibrationResult]`
- `screen_single_seed(params, scenario, seed, n_periods) -> float`
- `evaluate_stability(params, scenario, seeds, n_periods) -> StabilityResult`
- `CalibrationResult` - Calibration result with params and scores

### Parameter Space

- `PARAMETER_GRID` - Full parameter grid for all scenarios
- `DEFAULT_VALUES` - Default parameter values for all scenarios
- `get_parameter_grid(scenario)` - Get grid for specific scenario
- `get_default_values(scenario)` - Get defaults for specific scenario
- `generate_combinations(grid)` - Generate all parameter combinations
- `count_combinations(grid)` - Count total combinations

## CLI Options

| Option               | Default                  | Description                                   |
| -------------------- | ------------------------ | --------------------------------------------- |
| `--scenario`         | baseline                 | Scenario to calibrate (baseline, growth_plus) |
| `--sensitivity-only` | false                    | Run only sensitivity analysis                 |
| `--top-k`            | 20                       | Number of top configs for stability testing   |
| `--workers`          | 10                       | Number of parallel workers                    |
| `--periods`          | 1000                     | Simulation periods                            |
| `--output`           | calibration_results.json | Output file for results                       |
| `--high-threshold`   | 0.05                     | Sensitivity threshold for HIGH importance     |
| `--medium-threshold` | 0.02                     | Sensitivity threshold for MEDIUM importance   |

## Output Files

Results are saved to `calibration/output/`:

- `{scenario}_sensitivity.json` - Sensitivity analysis results (with `--sensitivity-only`)
- `{scenario}_calibration_results.json` - Full calibration results

## Module Structure

```
calibration/
├── __init__.py         # Package exports
├── __main__.py         # CLI entry point
├── cli.py              # Command-line interface
├── sensitivity.py      # OAT sensitivity analysis
├── optimizer.py        # Grid search and stability testing
├── parameter_space.py  # Parameter grids and defaults
└── output/             # Results JSON files
```
