# comparison: Cross-Framework Benchmark Harness

This sub-package benchmarks `bamengine` against alternative agent-based modelling
frameworks that implement the same BAM (Bottom-Up Adaptive Macroeconomics) model
(Delli Gatti et al., 2011).

## Competitor matrix

| Framework   | Runner module                    | Status            |
| ----------- | -------------------------------- | ----------------- |
| bamengine   | `comparison/runners/bamengine/`  | complete          |
| Mesa        | `comparison/runners/mesa/`       | Plan B (deferred) |
| mesa-frames | `comparison/runners/mesaframes/` | Plan C (deferred) |
| Agents.jl   | `comparison/runners/agentsjl/`   | Plan D (deferred) |
| NetLogo     | `comparison/runners/netlogo/`    | Plan E (deferred) |

Each runner plugs into the harness by registering an entry in `RUNNER_CMD`
(in `comparison/orchestrator/run.py`) and emitting a `RunResult` JSON to stdout.
A runner counts in the timing results only after it passes the behavioral
equivalence gate (Phase A).

## Install

```bash
pip install -e ".[comparison]"
```

This installs `bamengine` plus the benchmark dependencies (psutil, pandas,
matplotlib, scipy, tabulate).

## Running the benchmark

```bash
python -m comparison.orchestrator.run --frameworks bamengine
```

Optional flags:

```
--frameworks   Comma-separated list of framework names (default: bamengine)
--gate-workers Number of parallel workers for Phase A (default: 10)
--budget       Per-job wall-clock budget in seconds (default: 120)
--quick        Shrink periods/seeds/sizes for a fast smoke-test run
--sizes        Comma-separated firm counts to restrict Phase B (e.g. 100,1000)
--results-dir  Output directory (default: comparison/results)
```

## Two-phase pipeline

### Phase A: Behavioral equivalence gate (parallel)

Each framework runs a fixed set of gate jobs (multiple seeds at the canonical
BAM population). Output time-series are compared against the `bamengine`
reference using five metrics (GDP autocorrelation, unemployment autocorrelation,
inflation autocorrelation, firm-size Pareto shape, cross-correlation structure).
A framework must pass all metrics within pre-defined half-width tolerances before
its timing results are counted.

The gate runs the jobs in parallel via `ProcessPoolExecutor`.

### Phase B: Scaling timing (serial, single-thread, adaptive cap)

Gate-passing frameworks (and `bamengine` unconditionally) are timed across a
range of population sizes. Jobs run **serially** so that RSS and wall-time
measurements are not inflated by CPU contention. The first timeout for a given
framework caps all larger sizes for that framework (adaptive cap); skipped runs
are recorded in `results_dir/skips.json`.

## Single-thread requirement

All timing jobs pin worker threads to one via environment variables
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`POLARS_MAX_THREADS=1`, `JULIA_NUM_THREADS=1`). This makes results comparable
across frameworks that use different BLAS or parallelism backends.

## Where results land

```
comparison/results/          (default --results-dir)
    env_<hash>.json          environment snapshot (CPU, Python version, ...)
    skips.json               sizes dropped per framework due to timeout
    report.md                headline comparison table (auto-generated)
    figures/
        scaling.png          steady-state time vs population (log-log)
        memory.png           peak RSS vs population (log-log)
    raw/
        <run_id>.json        one file per job (gate and timing runs)
```

Committed `comparison/results/` snapshots serve as the source for the headline
speed table in `paper/paper.md`.

## Julia and NetLogo runners

The Agents.jl and NetLogo runners are deferred to Plans D and E respectively.
They require additional toolchain detection (Julia installation, NetLogo
headless mode) and are implemented as separate plan items to keep each plan
self-contained and testable on its own.
