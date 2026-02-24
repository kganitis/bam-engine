"""
=======================
Calibration Walkthrough
=======================

This example demonstrates the calibration package's programmatic API
for parameter optimization of the BAM model.

The calibration pipeline has three phases:

1. **Sensitivity analysis** (Morris Method) — identify which parameters
   matter and which can be fixed at defaults.
2. **Grid screening** — evaluate all combinations of important parameters
   with a single seed.
3. **Tiered stability testing** — progressively test top candidates with
   increasing numbers of seeds.

For the full calibration (all params, 100+ seeds), use the CLI::

    python -m calibration --scenario baseline --workers 10

This example uses tiny settings for a quick demonstration (~30 seconds).
"""

# %%
# Phase 1: Morris Method Sensitivity
# -----------------------------------
#
# The Morris Method runs multiple OAT trajectories from random starting
# points, producing mu* (importance) and sigma (interaction strength) per
# parameter.  We use a minimal setup here: 3 trajectories, 1 seed,
# 50-period simulations.

from calibration import (
    build_focused_grid,
    count_combinations,
    print_morris_report,
    run_morris_screening,
)

morris = run_morris_screening(
    scenario="baseline",
    n_trajectories=3,
    n_seeds=1,
    n_periods=50,
    n_workers=4,
)

print_morris_report(morris, mu_star_threshold=0.02, sigma_threshold=0.02)

# Convert to a SensitivityResult for downstream use
sensitivity = morris.to_sensitivity_result()

# %%
# Phase 2: Build Focused Grid
# ----------------------------
#
# Sensitivity results split parameters into two groups:
#
# - **INCLUDE**: parameters with sensitivity > threshold, searched in grid
# - **FIX**: low-sensitivity parameters, fixed at their best value
#
# Value pruning drops grid values that scored poorly in sensitivity.

grid, fixed = build_focused_grid(
    sensitivity,
    sensitivity_threshold=0.02,
    pruning_threshold=0.04,
)

n_combos = count_combinations(grid)
print(f"\nGrid: {len(grid)} params, {n_combos} combinations")
print(f"Fixed: {len(fixed)} params at optimal values")

for name, values in grid.items():
    print(f"  {name}: {values}")
for name, value in fixed.items():
    print(f"  {name} = {value} (fixed)")

# %%
# Phase 3: Screening + Stability
# --------------------------------
#
# ``run_focused_calibration`` orchestrates both phases:
#
# 1. Screen all combinations with a single seed
# 2. Run tiered stability on top candidates
#
# We use tiny tiers here: keep top 3 after 2 seeds, then top 2 after
# 4 seeds total.

from calibration import run_focused_calibration

results = run_focused_calibration(
    grid,
    fixed,
    scenario="baseline",
    n_workers=4,
    n_periods=50,
    stability_tiers=[(3, 2), (2, 4)],
    rank_by="combined",
    k_factor=1.0,
)

# %%
# Results
# -------
#
# Each result has a ``params`` dict, scores, and stability metrics.

from calibration import compare_configs, export_best_config

for i, r in enumerate(results):
    print(f"\n--- Rank {i + 1} ---")
    print(f"  Params: {r.params}")
    print(f"  Mean score: {r.mean_score:.4f}")
    print(f"  Std score:  {r.std_score:.4f}")
    print(f"  Pass rate:  {r.pass_rate:.1%}")
    print(f"  Combined:   {r.combined_score:.4f}")

if results:
    best = results[0]

    # Export as YAML
    path = export_best_config(best, "baseline")
    print(f"\nBest config exported to: {path}")

    # Before/after comparison
    comparison = compare_configs({}, best.params, "baseline", n_periods=50)
    print(f"\nDefault score: {comparison.default_score:.4f}")
    print(f"Best score:    {comparison.calibrated_score:.4f}")

# %%
# Custom Grid from YAML
# ----------------------
#
# You can also load a custom grid from a YAML or JSON file,
# bypassing sensitivity analysis entirely.

import tempfile
from pathlib import Path

import yaml

from calibration import load_grid

custom = {"beta": [1.0, 2.0, 3.0], "max_M": [2, 3, 4]}
grid_path = Path(tempfile.mkdtemp()) / "custom_grid.yaml"
with open(grid_path, "w") as f:
    yaml.dump(custom, f)

loaded = load_grid(grid_path)
print(f"\nLoaded custom grid: {loaded}")
print(f"Combinations: {count_combinations(loaded)}")
