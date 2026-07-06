# comparison: Cross-Framework Benchmark Harness

This sub-package benchmarks `bamengine` against alternative agent-based modelling
frameworks that implement the same BAM (Bottom-Up Adaptive Macroeconomics) model
(Delli Gatti et al., 2011).

## Competitor matrix

| Framework   | Runner module                     | Status   |
| ----------- | --------------------------------- | -------- |
| bamengine   | `comparison/runners/bamengine/`   | complete |
| Mesa        | `comparison/runners/mesa/`        | complete |
| mesa-frames | `comparison/runners/mesa_frames/` | complete |
| Agents.jl   | `comparison/runners/agentsjl/`    | complete |
| NetLogo     | `comparison/runners/netlogo/`     | complete |

Each runner plugs into the harness by registering an entry in `RUNNER_CMD`
(in `comparison/orchestrator/run.py`) and emitting a `RunResult` JSON to stdout.
A runner counts in the timing results only after it passes the behavioral
equivalence gate (Phase A).

### Port fairness

For a fair comparison, the competitor ports are written competently rather than
naively, then profiled to their inherent performance floor. The firm-selection
sampling uses a vectorized sparse k-subset draw in mesa-frames (`O(n_rows * k)`,
not a dense `O(n_rows * n_firms)` priority matrix) and an `O(k)` without-
replacement draw in Agents.jl (not an `O(n_firms)` shuffle-the-whole-pool per
household). Both match the uniform-k-subset distribution exactly, so the ports
still pass the equivalence gate.

Each port was then profiled to confirm its dominant cost is INHERENT rather than
an avoidable inefficiency (see `.claude/docs/analysis/2026-06-25-port-audit-*.md`):

- **mesa-frames**: additionally optimized with lazy `collect()` fusion, batch numpy
  sort/argmax, early-exit guards, and a vectorized `group_by` bank-equity update.
  Its remaining cost is inherent (the sequential market loops plus the per-event
  agent-DataFrame write-backs that `AgentSetPolars` requires by design).
- **Agents.jl**: additionally made allocation-free (no per-call `Set`) with
  id-range household iteration. Its remaining cost is inherent (the sequential
  goods loop plus `@multiagent` union dispatch). One residual (~7% at large N) is
  in RNG-order-sensitive loops kept faithful for exact per-seed reproducibility.

The ports are "competently written, profiled to their inherent floor": their caps
reflect a competent implementation in each framework, not a naive one, but also
not a maximally hand-tuned one.

## Install

```bash
pip install -e ".[comparison]"
```

This installs `bamengine`, the Mesa runner (`mesa`), and the benchmark
dependencies (psutil, pandas, matplotlib, scipy, tabulate). The bamengine and
Mesa runners work in this environment directly; the mesa-frames, Agents.jl, and
NetLogo runners need the dedicated setup described below.

### mesa-frames runner: dedicated environment

mesa-frames pins `numpy<2`, which conflicts with bamengine's `numpy>=2`. Because
each runner is a separate subprocess, the `mesa_frames` runner uses its own
virtual environment instead of the main one:

```bash
bash comparison/runners/mesa_frames/setup_env.sh
```

This creates `comparison/runners/mesa_frames/.venv-mf/` (Python 3.12) from
`requirements-mesa-frames.txt` and installs mesa-frames + Polars. The
orchestrator's `RUNNER_CMD["mesa_frames"]` points at that interpreter, so
bamengine's environment is never touched. The env is gitignored; rebuild it on a
fresh checkout before running the `mesa_frames` framework. CI runs the harness
unit tests only; the multi-framework benchmark (and the mesa-frames env) is run
locally.

### Agents.jl runner: Julia toolchain

The `agentsjl` runner is a Julia process (not Python). It needs Julia (1.12+,
e.g. via juliaup) and a one-time package install into a dedicated Julia project:

```bash
bash comparison/runners/agentsjl/setup_env.sh
```

This instantiates `comparison/runners/agentsjl/` from its committed
`Project.toml` + `Manifest.toml` (Agents.jl + JSON3) into the shared `~/.julia`
depot (nothing repo-local to gitignore). The orchestrator's
`RUNNER_CMD["agentsjl"]` invokes `julia --project=... run.jl`. Julia compiles on
first run, so the runner warms up untimed before the timed loop and the
compilation cost lands in `init_seconds` (timed separately). The benchmark is
run locally.

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

## NetLogo runner

The NetLogo runner drives the existing third-party Platas `DelliBAM_.nlogo` model
(Platas-Lopez and Guerra-Hernandez 2020, GPL-2.0) via pyNetLogo as a
**non-blocking cross-language reference**. Its gate result is informational: an
independent NetLogo BAM reproduces the baseline levels (unemployment ~8%,
inflation ~5%, Phillips-curve slope) but diverges on some co-movement structure
(Okun, Beveridge) and firm-size skew, so it is reported beside bamengine with
deviations noted rather than held to the blocking gate.

The third-party model and the NetLogo/Java toolchain are installed locally only
to run the comparison and removed afterward; neither is committed. See
`comparison/runners/netlogo/README.md` for the full setup, the parameter and
series mapping, and teardown.
