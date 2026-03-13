# Calibration Package Redesign — Design Spec

- **Date:** 2026-03-13
- **Status:** Approved
- **Scope:** Extend the existing calibration toolkit with 4 new composable tools + a Claude skill orchestrator

---

## 1. The Calibration Story — What Actually Happened

Four calibration campaigns were conducted on the BAM Engine model between February 22-23, 2026. Each campaign revealed patterns that the existing framework didn't encode — patterns that became the ad-hoc scripts and manual processes documented in the `src/calibration/output/` reports. This redesign captures those patterns as first-class tools.

### Act 1: Baseline Calibration (Feb 22, 2026)

**Phase 1 — OAT Sensitivity (1,440 runs, ~7 min)**
Started with single-baseline OAT: vary each of 21 parameters one at a time from defaults. Found 7 "important" parameters (Delta > 0.02), led by `new_firm_price_markup` (Delta=0.065). Classified 14 parameters as insensitive and would have fixed them.

**Phase 2 — Morris Method Screening (2,200 runs, ~10 min)**
Ran Morris Method as a sanity check — and it completely overturned OAT. Morris classified **20 of 21** parameters as important (vs OAT's 7). The most dramatic miss: `inflation_method`, which OAT ranked as insensitive (Delta=0.017) but Morris revealed as the 3rd most important parameter (mu*=0.042, sigma=0.063). One trajectory showed a 23 percentage-point score swing from this single parameter.

The key metric: **mean sigma/mu* = 1.58** across all parameters — interaction effects exceeded main effects by 58%. This meant single-baseline OAT was fundamentally broken for this model. Most critically, OAT recommended `min_wage_ratio=0.5` as optimal, while Morris found `min_wage_ratio=1.0`. The OAT recommendation was later confirmed to be **catastrophic** (75% pass rate at 0.67 vs 100% at 1.0).

*Lesson: Always use Morris, never trust OAT alone. The BAM model has pervasive parameter interactions.*

**Phase 3 — Grid Search (25,600 runs, ~1.6 hr)**
Built a grid from the top 11 Morris parameters (mu* > 0.023), fixing the bottom 10 at their Morris best values. The grid produced a right-skewed score distribution — 57% scored below 0.80, but the top 0.3% scored above 0.88. Two parameters locked at 100% across the top 100: `max_M=4` and `job_search_method=all_firms`.

*Lesson: The parameter space has a narrow sweet spot. Broad exploration is necessary — you can't hill-climb to it.*

**Phase 4 — Tiered Stability (3,000 runs, ~15 min)**
Tournament-style filtering: 100 configs x 10 seeds → top 50 x 20 seeds → top 10 x 100 seeds. The top 4 were statistically indistinguishable (combined scores spanning only 0.0004). They shared 18 parameters and differed only in `new_firm_production_factor` — confirming it was insensitive in the optimal region.

Key finding: configs with `price_cut_allow_increase=False` + `min_wage_ratchet=True` had 2x the variance (std 0.014-0.016 vs 0.007-0.008). The screening winner (#1 by single seed) did not survive stability — a classic single-seed overfitting case.

*Lesson: Single-seed screening over-fits. Always validate with tiered multi-seed stability. The combined score formula `mean × (1 - std)` effectively penalizes variance.*

**Phase 5 — Targeted Cost Analysis (1,624 runs, ~65 min)**
The grid winner had `price_init=3.0` and `beta=10.0` — values the user found undesirable. Instead of accepting them, a targeted analysis measured the "cost" of swapping each preferred value individually (29 swaps × 20 seeds), then grid-searched the cheap combinations (24 combos × 100 seeds).

Result: `price_init=2.0` cost +0.004 (CHEAP), `beta=5.0` cost +0.001 (FREE). The final config incorporated both preferences at **near-zero cost** (combined 0.8715 vs 0.8713).

Most important finding: **initial conditions are irrelevant** — `net_worth_ratio`, `equity_base_init`, `savings_init` can be set to any reasonable value (1-10) with negligible effect. The Kalecki attractor erases starting conditions in ~50 periods. The one exception: `min_wage_ratio < 1.0` is catastrophic because it affects income *flow*, not balance-sheet *stock*.

*Lesson: Targeted cost analysis is invaluable for incorporating user preferences post-optimization. Initial conditions mostly don't matter — save them for last or skip them.*

**Baseline Final:** Score improved from **0.778 → 0.877** (+12.7%), 100% pass rate across 100 seeds. Total: ~33,900 runs, ~3.2 hours.

---

### Act 2: Growth+ Calibration (Feb 23, 2026)

**Phase 1 — Morris Screening (2,300 runs, ~19 min)**
22 parameters (21 common + `sigma_decay`). Morris classified 21/22 as INCLUDE — too many for a manageable grid. The ranking was fundamentally different from baseline: new-firm entry parameters dominated the top 5 (vs labor market params in baseline). `price_init` jumped from rank 20 to rank 4.

**Phase 2 — Grid Search (11,200 runs, ~53 min)**
Fixed 13 lower-sensitivity params at **current defaults** (the known-good 0.855 center point) instead of Morris best values. This was a strategic decision — Morris "best" values come from random contexts where most other params are suboptimal.

Three parameters locked: `new_firm_wage_factor=0.5` (100%), `min_wage_ratchet=False` (100%), `price_cut_allow_increase=True` (92%). `min_wage_ratchet=True` produced **zero** zero-fail configs across all 5,600 combinations — strictly dominated.

**Phase 3 — Stability (2,590 runs, ~15 min)**
The screening winner (nfpm=1.25, pca=False) dropped out entirely under multi-seed testing — another single-seed overfitting case. Only 2 parameters differed from defaults: `job_search_method` and `max_M`. The improvement was a modest +1.5%.

**Phase 4 — Second-Pass Morris (900 runs, ~5 min)**
This was the novel step. With the 9 behavioral params locked at optimal values, re-screened the 8 structural/initial-condition params. Result: **all 8 classified as FIX** — mu* values dropped by 71-85% compared to Phase 1.

This proved that Phase 1's apparent sensitivity of structural params was entirely due to interaction with behavioral params. Once behavioral params were optimized, structural params became irrelevant. **The order of calibration matters** — behavioral first, structural second.

*Lesson: Second-pass Morris after locking optimized params reveals which remaining params are truly sensitive vs. interaction artifacts. This should be a standard step.*

**Growth+ Final:** Defaults confirmed near-optimal. Only 2 param changes. Total: ~17,000 runs, ~90 min.

---

### Act 3: Unified Entry Search (Feb 23, 2026)

The baseline and Growth+ calibrations produced **incompatible entry parameters** — baseline wanted large, competitive entrants (nfsf=0.9, nfwf=1.0, nfpm=1.5) while Growth+ wanted small, cheap ones (nfsf=0.5, nfwf=0.5, nfpm=1.0). Since both scenarios share defaults.yml, a compromise was needed.

**Phase 1 — Cross-Scenario Grid (5,040 runs, ~25 min)**
252 entry-parameter combos × 2 scenarios × 10 seeds. Ranked by `min(BL_combined, GP_combined)` — the bottleneck scenario determines the config's quality. Found a fundamental tension: `nfsf` and `nfpm` pull in opposite directions between scenarios. The compromise zone was nfsf=0.4-0.5, nfpm=1.0-1.15.

**Phase 2 — Cross-Scenario Stability (6,500 runs, ~35 min)**
Tiered: 45 configs × 20 seeds → 45 × 50 seeds → 20 × 100 seeds.

**Winner:** nfsf=0.5, nfpf=1.0, nfwf=0.5, nfpm=1.15 — 100% BL pass, 94% GP pass (6 fails).

*Lesson: Cross-scenario calibration needs a different ranking metric — `min(pass_rates)` across scenarios, then total fails, then min(combined). Score-optimal configs often have terrible stability on one scenario.*

---

### Act 4: Growth+ Stability Search (Feb 23, 2026)

The 6% GP failure rate motivated a final push. Instead of re-running the whole calibration, a structured 4-phase sweep tested non-entry parameters in order of intrusiveness:

- **Phase A (entry production):** `nfpf` sweep. `nfpf=0.5` achieved 96% GP pass.
- **Phase B (behavioral):** `beta × max_loan_to_net_worth × max_leverage` (64 combos). `beta=1.0, mlnw=2, ml=10` achieved **100% GP / 0 total fails**.
- **Phase C (initial conditions):** 720 combos across 5 params. Improved scores but not pass rate.
- **Phase D (market structure):** `max_M × job_search_method`. Current defaults already optimal.

Then re-ran Phase C with `beta=2.5` (default) as base → found initial conditions that compensated for the 1 extra fail.

**Winner (Option C):** `max_leverage=10` + specific initial conditions → 100% GP, 99% BL.

*Lesson: Structured sweep by parameter category (entry → behavioral → initial conditions → market) is more efficient than another full grid search. `max_leverage` was the single most impactful lever — doubling the fragility cap from 5 to 10.*

---

## 2. Lessons Learned — Distilled Principles

From the four campaigns, these are the actionable lessons that shape the framework:

### L1: Morris > OAT, always
OAT missed 13 of 20 important parameters in baseline. The BAM model has pervasive interactions (sigma/mu* = 1.4-1.6). OAT should exist as a fast sanity check but never as the sole screening method.

### L2: Single-seed screening over-fits
In both baseline and Growth+ campaigns, the single-seed screening winner did NOT survive multi-seed stability. Tiered stability is not optional — it's the only way to get reliable rankings.

### L3: The order of calibration matters
Growth+ Phase 4 proved that structural params' apparent Phase 1 sensitivity was 78% interaction artifact. **Behavioral params first, structural/initial-conditions second.** Second-pass Morris after locking the first group is the right way to determine if the second group needs grid search at all.

### L4: Cross-scenario calibration requires different ranking
Per-scenario ranking uses `combined = mean × (1 - std)`. Cross-scenario ranking must use `min(pass_rates)` across scenarios as the primary criterion, then total fails, then `min(combined)`. Score-optimal configs often catastrophically fail one scenario.

### L5: Targeted cost analysis is the right endgame
Grid search finds the mathematical optimum. Targeted cost analysis lets you incorporate user preferences, economic reasoning, and book compliance at known, quantified cost. The cost is usually negligible (FREE or CHEAP).

### L6: Initial conditions are mostly irrelevant
The Kalecki attractor erases balance-sheet initial conditions in ~50 periods. Only flow parameters (`min_wage_ratio`) have persistent effects. Save initial conditions for last or skip them.

### L7: Structured sweep beats full re-grid
When pushing for marginal stability improvements (Act 4), a structured sweep by parameter category (entry → behavioral → initial conditions → market structure) was more efficient and more interpretable than another full grid search.

### L8: The parameter space has a narrow sweet spot
In baseline, 57% of the 25,600 grid configs scored below 0.80, while only 0.3% scored above 0.88. Broad exploration (Morris + grid) is necessary — you can't hill-climb into this sweet spot.

### L9: Fix at known-good defaults, not Morris-best
Growth+ fixed lower-sensitivity params at **current defaults** (scoring 0.855) rather than Morris "best" values (which come from random, mostly-bad contexts). This preserves the proven baseline while optimizing the impactful params around it.

### L10: Parameter coupling must be respected
`job_search_method` and `max_M` were perfectly correlated across all top Growth+ configs — always `vac+mM=2` or `all+mM=4`, never mixed. The framework should support coupled parameters in grids.

---

## 3. Tool Design — The Composable Toolkit

### 3.1 Existing Tools (Keep & Refine)

| Tool | CLI Command | What it does | Refinements needed |
|------|------------|--------------|-------------------|
| **morris** | `calibration morris` | Morris Method screening | Add `--fix` flag to lock params (for second-pass). Add `--seeds` alias for `--sensitivity-seeds` |
| **oat** | `calibration oat` | OAT sensitivity analysis | Keep as-is, rarely used |
| **grid** | `calibration grid` | Single-seed grid screening | Add `--constraint` for coupled params (e.g., `nfpf>=nfsf`). Already supports `--grid` YAML and `--fixed` |
| **stability** | `calibration stability` | Tiered multi-seed tournament | Already supports `--stability-tiers`, `--rank-by`, `--k-factor` |
| **pairwise** | `calibration pairwise` | Pairwise interaction analysis | Keep as-is |

### 3.2 New Tools

#### Tool: `calibration rescreen`

**Purpose:** Second-pass Morris screening after locking optimized params (Growth+ Phase 4).

**Usage:**
```bash
calibration rescreen \
  --scenario growth_plus \
  --fix-from stability_result.json \
  --params structural \
  --trajectories 20 --seeds 5
```

**What it does:**
1. Loads the stability winner's params as fixed values
2. Runs Morris on only the specified parameter group (`structural`, `behavioral`, `initial_conditions`, `entry`, or explicit list)
3. Reports which params are still sensitive vs. now-FIX
4. Outputs the sensitivity collapse comparison (Phase 1 mu* vs Phase 2 mu*)

**Key design:** The `--params` flag accepts predefined groups OR comma-separated param names. Groups are defined in `parameter_space.py`:
```python
PARAM_GROUPS = {
    "entry": [
        "new_firm_size_factor",
        "new_firm_production_factor",
        "new_firm_wage_factor",
        "new_firm_price_markup",
    ],
    "behavioral": ["beta", "max_M", "job_search_method", "consumer_matching"],
    "initial_conditions": [
        "price_init",
        "min_wage_ratio",
        "net_worth_ratio",
        "equity_base_init",
        "savings_init",
    ],
    "credit": ["max_loan_to_net_worth", "max_leverage"],
}
```

---

#### Tool: `calibration cost`

**Purpose:** Targeted cost analysis — measure the cost of swapping individual values into a base config (baseline Phase 5).

**Usage:**
```bash
calibration cost \
  --scenario baseline \
  --base stability_result.json \
  --swaps "price_init=2.0,1.5,1.0" "beta=5.0,2.5" "min_wage_ratio=0.5,0.67" \
  --seeds 20
```

**What it does:**
1. Loads the base config (stability winner or explicit YAML)
2. For each swap value, substitutes it into the base and runs N seeds
3. Computes the cost (Delta mean) and classifies: FREE (<0.002), CHEAP (<0.005), MODERATE (<0.010), EXPENSIVE (>=0.010)
4. Reports a cost table with pass rate impact
5. Optionally runs a combination grid of the cheap swaps (`--combo-grid`)

**Output:** `cost_result.json` with per-swap costs and classifications.

---

#### Tool: `calibration cross-eval`

**Purpose:** Evaluate a set of parameter configs across multiple scenarios simultaneously (unified entry search ranking).

**Usage:**
```bash
calibration cross-eval \
  --scenarios baseline,growth_plus \
  --configs screening_result.json \
  --seeds 100 \
  --rank-by stability-first
```

**What it does:**
1. Takes configs from a screening/stability result or a YAML grid
2. Runs each config on ALL specified scenarios with N seeds
3. Ranks using cross-scenario criteria:
   - `stability-first`: min(pass_rates) → total fails → min(combined)
   - `score-first`: min(combined) → total fails
   - `balanced`: geometric mean of combined scores
4. Reports per-scenario breakdown and scenario tension analysis

**Key design:** This is the tool that was missing for the unified entry search and the stability search Phase D. It answers: "does this config work for ALL my scenarios?"

---

#### Tool: `calibration sweep`

**Purpose:** Structured parameter sweep by category, carrying forward winners (stability search Act 4).

**Usage:**
```bash
calibration sweep \
  --scenario growth_plus \
  --base defaults.yml \
  --stages "entry:nfpf=0.5,0.6,0.7,0.8,0.9" \
           "behavioral:beta=0.5,1.0,2.5,5.0 max_leverage=5,10,20" \
           "initial_conditions:price_init=0.5,1.0,1.5,2.0 min_wage_ratio=0.5,1.0" \
  --seeds 100 \
  --cross-scenario baseline
```

**What it does:**
1. Takes a base config and a sequence of stages
2. For each stage: runs a grid of that stage's params (holding all else fixed from base + prior winners), does tiered stability, selects winner
3. Carries the winner forward as the base for the next stage
4. Optionally cross-evaluates against other scenarios at each stage (`--cross-scenario`)
5. Reports per-stage results and cumulative improvement

**Key design:** This is the structured A→B→C→D process from the stability search. Each stage is independently interpretable — you can stop after any stage and inspect results before continuing. The `--cross-scenario` flag ensures you don't break one scenario while optimizing another.

---

### 3.3 Data Format

All tools read/write the same JSON format (already mostly in place). The key additions:

```python
@dataclass
class CalibrationResult:
    params: dict[str, Any]
    single_score: float
    mean_score: float | None = None
    std_score: float | None = None
    combined_score: float | None = None
    pass_rate: float | None = None
    n_fails: int | None = None  # NEW: explicit fail count
    seed_scores: list[float] | None = None
    scenario_results: dict[str, ScenarioResult] | None = None  # NEW: cross-scenario


@dataclass
class ScenarioResult:
    mean_score: float
    std_score: float
    combined_score: float
    pass_rate: float
    n_fails: int
    seed_scores: list[float]
```

### 3.4 CLI Structure

The CLI already uses `--phase` dispatch. Extend it with the new tools:

```bash
# Existing
calibration morris --scenario baseline --trajectories 20 --seeds 5
calibration grid --scenario baseline
calibration stability --scenario baseline

# New
calibration rescreen --scenario growth_plus --fix-from result.json --params structural
calibration cost --scenario baseline --base result.json --swaps "beta=5.0,2.5"
calibration cross-eval --scenarios baseline,growth_plus --configs result.json
calibration sweep --scenario growth_plus --stages "..." --cross-scenario baseline
```

---

## 4. The Claude Skill — Calibration Orchestrator

The skill is the "recipe intelligence" — it knows the optimal calibration process, reads intermediate results, and issues the right CLI commands. It replaces the need for custom scripts by encoding the lessons learned into a decision framework.

### 4.1 Skill Trigger

```
name: calibrate
description: Run model calibration using the composable toolkit.
  Triggers on: "calibrate", "run calibration", "calibrate the model",
  "recalibrate", "tune parameters", "optimize parameters".
```

### 4.2 Skill Process Flow

```
1. ASSESS
   - Which scenarios need calibrating? (baseline, growth_plus, buffer_stock)
   - Are there existing calibration results to build on?
   - What's the current parameter count? (determines grid feasibility)
   - What's the user's time budget? (quick=1hr, normal=3hr, thorough=6hr)

2. PHASE 1: SCREENING
   IF param_count <= 12:
     → Skip Morris, go straight to full grid (feasible)
   ELSE:
     → Run Morris with 20 trajectories, 5 seeds
     → Report rankings, classify INCLUDE/FIX
     → Ask user: "Morris found N important params. Proceed with grid?"

3. PHASE 2: GRID SEARCH
   - Build grid from Morris results (or full param space if skipped)
   - Report grid size, estimated time
   - Run single-seed screening
   - Report: score distribution, parameter convergence, top configs
   - Ask user: "Top config scores X. Proceed to stability?"

4. PHASE 3: STABILITY
   - Run tiered stability (100:10 → 50:20 → 10:100)
   - Report: winner, statistical significance, screening #1 fate
   - Check: did the screening winner survive? (warn if not)

5. PHASE 4: SECOND-PASS SCREENING (Lesson L3)
   IF there are params that were fixed in Phase 2:
     → Run `calibration rescreen` on fixed params
     → Report sensitivity collapse
     → IF any params are still sensitive: run targeted grid + stability
     → ELSE: confirm fixed values are fine

6. PHASE 5: TARGETED COST ANALYSIS (Lesson L5)
   → Ask user: "Are there any parameter values you prefer?"
   → Run `calibration cost` with user-preferred swaps
   → Report cost table
   → IF cheap combos exist: run combo grid + stability
   → Present final recommended config

7. PHASE 6: CROSS-SCENARIO (if multiple scenarios)
   IF calibrating multiple scenarios:
     → Run `calibration cross-eval` on winner across all scenarios
     → IF pass rate < 90% on any scenario:
       → Run `calibration sweep` to find cross-scenario compromise
       → Report per-stage winners and cross-scenario impact
     → ELSE: confirm winner works across scenarios

8. FINALIZE
   - Present before/after comparison
   - Show parameter changes from current defaults
   - Ask user: "Apply these as new defaults?"
   - If yes: update defaults.yml and commit
```

### 4.3 Key Skill Behaviors

**Checkpoint after every phase.** The skill should present results and ask for confirmation before proceeding. This is where human judgment enters — the user might say "skip second-pass Morris" or "I want beta=2.5, check the cost."

**Read intermediate JSON files.** After each CLI command, the skill reads the output JSON to make decisions. For example, after Morris:
- If only 1 param is FIX → note it
- If sigma/mu* > 1.5 mean → warn about pervasive interactions
- If >15 params are INCLUDE → suggest fixing lower-sensitivity ones at defaults (Lesson L9)

**Apply lessons automatically:**
- L1: Default to Morris, never OAT-only
- L2: Always run stability, warn if user tries to skip
- L4: Use `stability-first` ranking for cross-scenario
- L6: Put initial conditions last, mention Kalecki attractor
- L9: Fix at defaults, not Morris-best, for lower-sensitivity params

**Time estimation.** Based on past campaigns (~2.5-3.1 sec/run with 10 workers), the skill can estimate wall time for each phase before running it:
```
Morris (20 trajectories, 12 params, 5 seeds): ~1,300 runs → ~6 min
Grid (8 params, ~3,200 combos): ~3,200 runs → ~15 min
Stability (100:10, 50:20, 10:100): ~3,000 runs → ~14 min
```

### 4.4 Skill Template (Abbreviated)

```markdown
---
name: calibrate
description: Orchestrate model calibration using the composable toolkit.
  Trigger on "calibrate", "run calibration", "recalibrate",
  "tune parameters", "optimize parameters"
---

## Context
You are orchestrating a BAM Engine calibration using the composable
calibration toolkit. Each phase is a CLI command that produces JSON
results. You read the results, apply lessons learned from past
campaigns, and decide the next step.

## Tools Available
- `calibration morris` — Morris Method screening
- `calibration grid` — Single-seed grid search
- `calibration stability` — Tiered multi-seed tournament
- `calibration rescreen` — Second-pass Morris on subsets
- `calibration cost` — Targeted cost analysis
- `calibration cross-eval` — Cross-scenario evaluation
- `calibration sweep` — Structured parameter sweep

## Process
[The full decision tree from 4.2 above]

## Lessons (Hard Rules)
[The 10 lessons from Section 2, encoded as rules]

## Reporting
After each phase, present a concise summary:
- What was tested (params, grid size, seeds)
- Key findings (rankings, convergence, surprises)
- Recommendation for next step
- Estimated time for next phase
Ask for user confirmation before proceeding.
```

---

## 5. Implementation Plan

### 5.1 Existing Code Changes (Minimal)

The existing calibration package is well-structured. Most changes are additions, not rewrites.

**`parameter_space.py` — Add parameter groups**
```python
PARAM_GROUPS = {
    "entry": [
        "new_firm_size_factor",
        "new_firm_production_factor",
        "new_firm_wage_factor",
        "new_firm_price_markup",
    ],
    "behavioral": ["beta", "max_M", "job_search_method", "consumer_matching"],
    "initial_conditions": [
        "price_init",
        "min_wage_ratio",
        "net_worth_ratio",
        "equity_base_init",
        "savings_init",
    ],
    "credit": ["max_loan_to_net_worth", "max_leverage"],
}
```
Also: update the parameter grid to reflect removed params (remove the 9 deleted ones). This is the most important cleanup — the current `get_parameter_grid()` likely still lists removed params.

**`analysis.py` — Add `ScenarioResult` and `n_fails`**
Add the `ScenarioResult` dataclass and `n_fails` field to `CalibrationResult`. Add a `from_cross_eval()` factory for creating results with per-scenario data.

**`morris.py` — Add `--fix` support**
The existing `run_morris_screening()` takes a `scenario` and builds params internally. Add a `fixed_params: dict` argument that locks specified params at given values, running Morris only on the remaining params. This enables second-pass Morris without a new module.

**`grid.py` — Add constraint support**
Add a `constraints` parameter to `generate_combinations()` that filters combinations based on expressions like `new_firm_production_factor >= new_firm_size_factor`. Simple lambda-based filtering on generated combos.

**`stability.py` — Add `stability-first` ranking**
Add a ranking strategy that sorts by `min(pass_rates)` → `total_fails` → `min(combined)` for cross-scenario use. The existing `rank_by` choices are `combined`, `stability`, `mean` — add `cross_scenario`.

**`cli.py` — Extend dispatch**
Add the 4 new subcommands. The existing `--phase` dispatch pattern extends naturally.

**`io.py` — Add cross-scenario serialization**
Extend `save_stability()` / `load_stability()` to handle `ScenarioResult` data.

### 5.2 New Modules

| Module | Size (est.) | What it does |
|--------|------------|-------------|
| `rescreen.py` | ~150 lines | Wraps `morris.run_morris_screening()` with `fixed_params` from a loaded result file. Computes sensitivity collapse table (Phase 1 mu* vs Phase 2 mu*). |
| `cost.py` | ~200 lines | Implements targeted cost analysis: univariate swaps, cost classification (FREE/CHEAP/MODERATE/EXPENSIVE), combo grid generation. |
| `cross_eval.py` | ~250 lines | Runs configs across multiple scenarios in parallel. Implements cross-scenario ranking strategies. Computes scenario tension analysis. |
| `sweep.py` | ~200 lines | Structured multi-stage sweep. Parses stage definitions, runs grid+stability per stage, carries winners forward. Optional cross-scenario check at each stage. |

### 5.3 Implementation Sequence

Build order matters — each tool can be tested independently before composing them:

```
Step 1: Cleanup & Foundation
  - Update parameter_space.py (remove deleted params, add groups)
  - Add n_fails and ScenarioResult to analysis.py
  - Add fixed_params to morris.py
  - Add constraints to grid.py
  - Tests for all the above

Step 2: rescreen tool
  - Implement rescreen.py
  - Add CLI command
  - Test with a synthetic Morris result

Step 3: cost tool
  - Implement cost.py
  - Add CLI command
  - Test with a known base config

Step 4: cross-eval tool
  - Implement cross_eval.py
  - Add stability-first ranking to stability.py
  - Add CLI command
  - Test with both scenarios

Step 5: sweep tool
  - Implement sweep.py (depends on grid + stability + optionally cross_eval)
  - Add CLI command
  - Test with a 2-stage sweep

Step 6: Claude skill
  - Write the calibrate skill
  - Test with a dry run (skill reads mock results)
  - Test with a real short calibration (few params, few seeds)

Step 7: Documentation
  - Update src/calibration/README.md
  - Update Sphinx docs (docs/calibration/)
  - Write the optimal recipe as a standalone doc
```

### 5.4 What NOT to Change

- **The core Morris/OAT/grid/stability modules** — they work well, just need minor extensions
- **The JSON data format** — extend it, don't break backward compatibility with existing result files in `output/`
- **The reporting module** — extend with new report types for the new tools, don't rewrite existing reports
- **The `output/` directory** — existing calibration results are valuable historical data

### 5.5 Estimated Scope

| Component | New/Modified files | Est. lines changed |
|-----------|-------------------|-------------------|
| Foundation changes | 5 existing files | ~150 lines |
| 4 new tools | 4 new modules | ~800 lines |
| CLI extensions | 1 file (cli.py) | ~200 lines |
| Tests | ~6 test files | ~600 lines |
| Skill | 1 skill file | ~200 lines |
| Docs | 3 files | ~200 lines |
| **Total** | ~19 files | **~2,150 lines** |

---

**Summary:** Four new composable tools (rescreen, cost, cross-eval, sweep) that encode the ad-hoc patterns from past calibration campaigns, composed by a Claude skill that knows the optimal process and checkpoints for human judgment between phases.
