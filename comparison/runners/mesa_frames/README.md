# mesa-frames Runner

mesa-frames (Polars backend) implementation of the baseline BAM model for the
cross-framework comparison harness.

## Why a dedicated virtual environment?

mesa-frames pins `numpy<2`, which conflicts with the main bamengine venv
(which uses numpy 2.x). The runner therefore operates inside its own venv at
`.venv-mf/`, and the orchestrator invokes it via a dedicated `RUNNER_CMD`
entry pointing to `.venv-mf/bin/python`.

## Setup

```bash
bash comparison/runners/mesa_frames/setup_env.sh
```

Requires Python 3.12 at `/opt/homebrew/bin/python3.12`. Install via:

```bash
brew install python@3.12
```

Verify the environment:

```bash
comparison/runners/mesa_frames/.venv-mf/bin/python -c "import mesa_frames, polars; print('ok')"
```

## Importing `comparison.*`

The runner imports `comparison.orchestrator.contract` for the JSON contract.
The `comparison` package lives at the repo root and is NOT installed into
`.venv-mf`. The orchestrator sets `PYTHONPATH=<repo root>` in the subprocess
environment so that `comparison.*` is importable without installing bamengine.

## Dependencies

See `requirements-mesa-frames.txt`:

- `numpy<2` (mesa-frames constraint)
- `polars`
- `pyarrow`
- `mesa-frames==0.1.0a0`
- `psutil` (used by `comparison.orchestrator.environment` and `subprocess_runner`)
- `pyyaml` (used by `comparison.orchestrator.params` to load `defaults.yml`)
- `scipy` (used by `comparison.equivalence.metrics` at module load via `scipy.stats.skew`)
- `pytest` (to run the smoke test suite)

## Running manually

```bash
# Write a request JSON, then:
comparison/runners/mesa_frames/.venv-mf/bin/python \
    -m comparison.runners.mesa_frames.run request.json
```

(Set `PYTHONPATH=<repo root>` if invoking outside the orchestrator.)
