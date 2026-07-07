# Agents.jl BAM Runner

Julia implementation of the BAM (Bottom-Up Adaptive Macroeconomics) model
using [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) v7,
part of the cross-framework benchmark in `comparison/`.

## Status

Complete. The port implements all 37 events of the baseline BAM model and
passes the behavioral equivalence gate against the bamengine reference
(see `comparison/equivalence/`).

## Prerequisites

- Julia 1.10+ (tested with 1.12.x via juliaup)
- Agents.jl v7.x and JSON3 (resolved via `setup_env.sh`)

## Environment Setup

Run once to download, pin, and precompile all dependencies:

```bash
bash comparison/runners/agentsjl/setup_env.sh
```

This runs `Pkg.instantiate()` + `Pkg.precompile()` and verifies that
`using Agents, JSON3` succeeds. The generated `Manifest.toml` is committed
to pin exact package versions for reproducibility.

## Running Standalone

```bash
# Create a minimal RunRequest JSON
echo '{"run_id":"test","framework":"agentsjl","model_params":{},"population":{"n_firms":100,"n_households":500,"n_banks":10},"n_periods":100,"warmup_periods":0,"seed":42,"collect_outputs":false,"outputs_requested":[]}' > /tmp/req.json

# Run the runner directly (prints a RunResult JSON as the last line of stdout)
julia --project=comparison/runners/agentsjl --startup-file=no \
    comparison/runners/agentsjl/run.jl /tmp/req.json
```

## Orchestrator Integration

The runner is registered in `comparison/orchestrator/run.py`:

```python
RUNNER_CMD["agentsjl"] = [
    "julia",
    "--project=<abs path to comparison/runners/agentsjl>",
    "--startup-file=no",
    "<abs path to run.jl>",
]
```

## Julia Tests

```bash
julia --project=comparison/runners/agentsjl test/runtests.jl
```

## Protocol

The runner reads a `RunRequest` JSON file path from `ARGS[1]` and prints a
`RunResult`-shaped JSON as the **last line of stdout**. The orchestrator
captures the last stdout line and calls `RunResult.from_json()` on it.
See `comparison/orchestrator/contract.py` for the full schema.
