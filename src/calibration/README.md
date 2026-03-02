# Calibration Package

Tools for finding optimal parameter values through sensitivity analysis, focused grid search, and tiered stability testing.

> **Full documentation:** See the [Sphinx docs](https://kganitis.github.io/bam-engine/calibration/index.html) for the complete calibration reference including sensitivity analysis, grid search, stability testing, CLI options, and full API.

## Quick Start

### Command Line

```bash
# Run all phases (baseline, Morris default)
python -m calibration --scenario baseline --workers 10

# Run individual phases
python -m calibration --phase sensitivity --scenario baseline
python -m calibration --phase grid --scenario baseline
python -m calibration --phase stability --scenario baseline

# Use OAT instead of Morris for sensitivity
python -m calibration --phase sensitivity --method oat --scenario baseline

# Resume interrupted grid search / custom grid
python -m calibration --phase grid --scenario baseline --resume
python -m calibration --phase grid --grid my_grid.yaml --fixed beta=1.0

# Ranking strategies and custom stability tiers
python -m calibration --phase stability --rank-by stability --k-factor 1.5
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
results = run_focused_calibration(grid, fixed, rank_by="combined", k_factor=1.0)

# Export best config
export_best_config(results[0], "baseline")

# Before/after comparison
comparison = compare_configs({}, results[0].params, "baseline")
```

## Calibration Process

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Phase 1          │     │ Phase 2          │     │ Phase 3          │
│ Sensitivity      │────>│ Grid Screening   │────>│ Tiered Stability │
│ (Morris or OAT)  │     │ (single seed)    │     │ (multi-seed)     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
 Identifies important      Evaluates all           Tournament-style
 parameters, fixes         combinations,           multi-seed testing
 unimportant ones          prunes poor values       with ranking
```

## Module Structure

```
calibration/
├── __init__.py         # Public API exports
├── __main__.py         # CLI entry point (delegates to cli.py)
├── cli.py              # Argument parsing + phase dispatch
├── analysis.py         # Types (CalibrationResult), patterns, export, comparison
├── grid.py             # Grid building, YAML loading, validation, combinations
├── screening.py        # Single-seed grid screening + checkpointing
├── stability.py        # Tiered stability testing + ranking strategies
├── io.py               # Save/load for all result types + timestamped dirs
├── reporting.py        # Auto-generated markdown reports
├── morris.py           # Morris Method screening (elementary effects)
├── sensitivity.py      # OAT sensitivity + pairwise interaction
├── parameter_space.py  # Parameter grids (14 common + extensions)
└── output/             # Timestamped run directories (gitignored)
```
