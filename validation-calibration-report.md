# Report: How `validation/`, `calibration/`, and `tests/validation/` Work Together

## 1. High-Level Architecture

These three packages form a **closed-loop calibration and validation system** for the BAM macroeconomic simulation:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        tests/validation/                             │
│  Pytest test suite — regression guard ensuring metrics don't break   │
│  Calls into validation/ and asserts on results                       │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ imports & calls
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          validation/                                 │
│  Core scoring engine — runs simulations, computes metrics,           │
│  validates against YAML targets, returns scored results              │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▲ imports & calls
                           │
┌──────────────────────────┴───────────────────────────────────────────┐
│                          calibration/                                │
│  Parameter optimizer — sensitivity analysis + grid search            │
│  Uses validation/ as its objective function                          │
└──────────────────────────────────────────────────────────────────────┘
```

The **validation package** is the central hub. Both the test suite and the calibration package depend on it, but they never depend on each other.

______________________________________________________________________

## 2. Package-by-Package Breakdown

### 2.1 `validation/` — The Scoring Engine

```
validation/
├── __init__.py              # Public API (thin wrappers for backwards compat)
├── types.py                 # Core types: MetricSpec, Scenario, CheckType, etc.
├── scoring.py               # Scoring: score_mean_tolerance(), score_range(), etc.
├── engine.py                # Generic validate(), stability_test()
├── reporting.py             # Report printing functions
├── scenarios/
│   ├── __init__.py          # Re-exports scenarios
│   ├── baseline.py          # BaselineMetrics dataclass (23 fields), compute, run_scenario
│   ├── baseline_viz.py      # 8-panel visualization
│   ├── growth_plus.py       # GrowthPlusMetrics dataclass (50+ fields), compute, run_scenario
│   └── growth_plus_viz.py   # 16-panel visualization + recession bands
└── targets/
    ├── baseline.yaml        # Section 3.9.1 targets (22 metrics)
    └── growth_plus.yaml     # Section 3.9.2 targets (39 metrics)

extensions/                  # Separate package for model extensions
└── rnd/
    ├── __init__.py          # Exports RnD role and events
    ├── role.py              # RnD role definition
    └── events.py            # 3 custom pipeline events for R&D
```

**Core data flow for a single validation run:**

```
validate(BASELINE_SCENARIO, seed=42, n_periods=1000, **config_overrides)
  │
  ├─ Load targets from YAML
  ├─ Build simulation config from scenario.default_config
  ├─ Run setup_hook if provided (e.g., import RnD for Growth+)
  ├─ sim = bam.Simulation.init(**config)
  ├─ results = sim.run(collect=scenario.collect_config)
  ├─ metrics = scenario.compute_metrics(sim, results, burn_in)
  ├─ For each MetricSpec in scenario.metric_specs:
  │    ├─ actual = getattr(metrics, spec.field)
  │    ├─ target_section = get_nested_value(targets, spec.target_path)
  │    ├─ status = check_*(actual, ...) → PASS/WARN/FAIL
  │    ├─ score  = score_*(actual, ...) → float [0, 1]
  │    └─ Append MetricResult(name, status, actual, score, weight)
  └─ Return ValidationScore(metric_results, total_score, n_pass, n_warn, n_fail)
```

**Key data structures:**

| Type              | Purpose                                                                     |
| ----------------- | --------------------------------------------------------------------------- |
| `MetricSpec`      | Declarative specification: name, field, check_type, target_path, weight     |
| `Scenario`        | Bundles metric_specs, collect_config, compute_metrics, setup_hook           |
| `MetricResult`    | Single metric: name, status, actual value, score (0-1), weight              |
| `ValidationScore` | One run: list of MetricResult + total_score + pass/warn/fail counts         |
| `StabilityResult` | Multi-seed: list of ValidationScore + mean/std/pass_rate + per-metric stats |

**Check types and standardized YAML keys:**

| CheckType        | YAML Keys                       | Purpose                         |
| ---------------- | ------------------------------- | ------------------------------- |
| `MEAN_TOLERANCE` | `target`, `tolerance`           | Value within target ± tolerance |
| `RANGE`          | `min`, `max`                    | Value within [min, max]         |
| `PCT_WITHIN`     | `target`, `min`                 | Percentage meeting threshold    |
| `OUTLIER`        | `max_outlier`, `penalty_weight` | Penalize excess outliers        |
| `BOOLEAN`        | `threshold` (in MetricSpec)     | Simple > or < check             |

**Scoring functions** map actual values to a 0–1 scale:

- `score_mean_tolerance()` — linear decay from target mean
- `score_range()` — 0.75–1.0 inside range, decays outside
- `score_pct_within_target()` — for tiered percentage checks
- `score_outlier_penalty()` — exponential decay for excess outliers

**Two scenarios** are supported:

|                    | Baseline (Section 3.9.1) | Growth+ (Section 3.9.2)              |
| ------------------ | ------------------------ | ------------------------------------ |
| Firms / Households | 300 / 3000 (scaled 6×)   | 100 / 500 (book values)              |
| Metrics validated  | 22                       | 39                                   |
| Extensions         | None                     | RnD role + 3 events (in extensions/) |
| Key feature        | Stationary equilibrium   | Endogenous productivity growth       |

**Design notes:**

- The **MetricSpec abstraction** enables declarative metric validation. Adding a new metric requires only: (1) add field to Metrics dataclass, (2) add MetricSpec to scenario, (3) add target to YAML.
- The Growth+ scenario uses a **lazy import pattern** via `setup_hook`: the `RnD` role and its 3 custom events are imported before `Simulation.init()` so the `@event(after=...)` hooks register in the pipeline.
- The `compute_combined_score()` function balances three qualities: `mean_score × pass_rate × (1 - std_score)` — accuracy, reliability, and consistency in a single number.

______________________________________________________________________

### 2.2 `calibration/` — The Parameter Optimizer

```
calibration/
├── __init__.py            # Public API re-exports
├── __main__.py            # python -m calibration entry point
├── cli.py                 # 4-phase pipeline orchestrator
├── parameter_space.py     # Grids and defaults per scenario
├── sensitivity.py         # OAT sensitivity analysis
├── optimizer.py           # Grid search + stability evaluation
├── run_grid_search.py     # Standalone alternative (skips sensitivity)
└── output/                # JSON result files
```

**The calibration pipeline has 4 phases:**

```
Phase 1: SENSITIVITY ANALYSIS (sensitivity.py)
  │  One-At-a-Time: vary each parameter alone, measure score change
  │  → Categorize: HIGH (Δ>0.05), MEDIUM (0.02<Δ≤0.05), LOW (Δ≤0.02)
  │
Phase 2: BUILD FOCUSED GRID (optimizer.py)
  │  HIGH params → full range in grid
  │  MEDIUM params → min/max only
  │  LOW params → fixed at best value
  │
Phase 3: PARALLEL SCREENING (optimizer.py)
  │  All combinations × single seed=0
  │  → Keep top_k by single_score
  │
Phase 4: STABILITY TESTING (optimizer.py)
  │  top_k configs × multiple seeds
  │  → Final ranking by combined_score
  │
  └─→ JSON output in calibration/output/
```

**How calibration uses validation** — these are the only validation imports:

```python
from validation import (
    get_validation_func,  # Returns scenario-specific validator
    get_validation_funcs,  # Returns (validator, stability_runner, ...) tuple
    compute_combined_score,  # mean * pass_rate * (1 - std)
    StabilityResult,
    DEFAULT_STABILITY_SEEDS,
)
```

The calibration package treats validation as a **black-box objective function**: pass in parameters, get back a score. It never touches simulation internals, metrics computation, or YAML targets directly.

______________________________________________________________________

### 2.3 `tests/validation/` — The Regression Guard

```
tests/validation/
├── __init__.py                      # Empty
├── test_baseline_scenario.py        # 2 tests
└── test_growth_plus_scenario.py     # 2 tests
```

**4 total tests, all marked `@pytest.mark.slow` and `@pytest.mark.validation`:**

| Test                                   | What It Does                             | Pass Criteria               |
| -------------------------------------- | ---------------------------------------- | --------------------------- |
| `test_baseline_scenario_validation`    | Runs 1 simulation (seed=0, 1000 periods) | Zero FAIL metrics (WARN ok) |
| `test_baseline_seed_stability`         | Runs 20 seeds × 1000 periods             | pass_rate ≥ 95%, std ≤ 0.15 |
| `test_growth_plus_scenario_validation` | Runs 1 Growth+ simulation (seed=0)       | Zero FAIL metrics           |
| `test_growth_plus_seed_stability`      | Runs 20 Growth+ seeds                    | pass_rate ≥ 95%, std ≤ 0.15 |

The tests are thin wrappers — all complexity lives in the validation package. They serve as **regression guards** to catch when model changes break expected behavior.

______________________________________________________________________

## 3. How the Three Packages Interact

### 3.1 Dependency Graph

```
tests/validation/
    │
    │ imports: run_validation, run_stability_test,
    │          run_growth_plus_validation, run_growth_plus_stability_test,
    │          print_*_report, DEFAULT_STABILITY_SEEDS
    │
    └──────▶ validation/
                 │
                 │ exports: get_validation_func, get_validation_funcs,
                 │          compute_combined_score, StabilityResult,
                 │          DEFAULT_STABILITY_SEEDS
                 │
    ┌────────────┘
    │
calibration/
    imports: get_validation_func, get_validation_funcs,
             compute_combined_score, StabilityResult,
             DEFAULT_STABILITY_SEEDS
```

No circular dependencies. The validation package is the **shared foundation** that both consumers depend on independently.

### 3.2 Shared Concepts

All three packages operate on the same conceptual pipeline:

```
Parameters → Simulation → Metrics → Scoring → Decision
```

| Stage      | validation/                 | calibration/             | tests/                     |
| ---------- | --------------------------- | ------------------------ | -------------------------- |
| Parameters | Defaults or overrides       | Grid search combinations | Fixed (seed=0 or 20 seeds) |
| Simulation | Runs bam.Simulation         | Delegates to validation  | Delegates to validation    |
| Metrics    | Computes from results       | N/A (uses validation)    | N/A (uses validation)      |
| Scoring    | score\_*/check\_* functions | Reads total_score        | Reads status (PASS/FAIL)   |
| Decision   | Returns ValidationScore     | Ranks by combined_score  | assert / pytest.fail       |

### 3.3 The Feedback Loop

These packages support an iterative development workflow:

```
1. DEVELOP    → Change model code
2. TEST       → pytest tests/validation/ (catches regressions)
3. CALIBRATE  → python -m calibration (finds better parameters)
4. VALIDATE   → run_validation() with new params (verify improvement)
5. UPDATE     → Apply best params to defaults
6. REPEAT
```

______________________________________________________________________

## 4. YAML Targets — The Source of Truth

Both validation and tests ultimately validate against two YAML files:

| File                                  | Scenario      | Source                    | Metrics |
| ------------------------------------- | ------------- | ------------------------- | ------- |
| `validation/targets/baseline.yaml`    | Section 3.9.1 | Delli Gatti et al. (2011) | 22      |
| `validation/targets/growth_plus.yaml` | Section 3.9.2 | Delli Gatti et al. (2011) | 39      |

Each YAML file contains:

- **Metadata**: Book setup, our setup, burn-in period, params for compute_metrics
- **Metrics section**: Standardized keys (`target`/`tolerance`, `min`/`max`, etc.)
- **Legacy section**: Nested structure for visualization compatibility

______________________________________________________________________

## 5. Summary

| Package             | Role                | Lines\* | Key Abstraction                                    |
| ------------------- | ------------------- | ------- | -------------------------------------------------- |
| `validation/`       | Scoring engine      | ~1,500  | `MetricSpec` + `Scenario` — declarative validation |
| `extensions/rnd/`   | R&D model extension | ~200    | `RnD` role + pipeline events                       |
| `calibration/`      | Parameter optimizer | ~1,400  | `CalibrationResult` — params + combined score      |
| `tests/validation/` | Regression guard    | ~170    | pytest assertions on PASS/FAIL status              |

*\*Approximate, excluding YAML and output files.*

The design cleanly separates concerns: **validation** defines *what correct looks like*, **calibration** searches for *parameters that achieve it*, and **tests** ensure *it stays correct over time*.
