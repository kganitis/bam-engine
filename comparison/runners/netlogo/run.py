"""NetLogo runner for the cross-framework comparison harness.

Drives the third-party Platas ``DelliBAM_.nlogo`` model through pyNetLogo and
prints a RunResult JSON as the FINAL line of stdout. Reads a RunRequest JSON
file whose path is ``sys.argv[1]`` (contract in
``comparison/orchestrator/contract.py``).

The pyNetLogo import is guarded: when the toolchain is absent (for example in
CI), the runner emits a ``status = "skipped"`` RunResult and the rest of the
harness proceeds. The real run path is added in later tasks.
"""

from __future__ import annotations

import sys
import traceback

from comparison.orchestrator.contract import (
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_SKIPPED,
    RunRequest,
    RunResult,
)


def _skeleton(req: RunRequest, status: str, error: object) -> RunResult:
    """Build a RunResult carrying only request echo fields (no outputs/timing)."""
    return RunResult(
        schema_version=SCHEMA_VERSION,
        run_id=req.run_id,
        framework="netlogo",
        framework_version="unknown",
        language="netlogo",
        language_version="unknown",
        status=status,
        error=error,
        population=req.population,
        n_periods=req.n_periods,
        warmup_periods=req.warmup_periods,
        seed=req.seed,
        timing={},
        outputs=None,
    )


def main(request_path: str) -> None:
    """Run one request and print a RunResult JSON to stdout."""
    with open(request_path) as fh:
        req = RunRequest.from_json(fh.read())

    try:
        import pynetlogo  # noqa: F401
    except ImportError:
        print(
            _skeleton(
                req, STATUS_SKIPPED, "pynetlogo/NetLogo toolchain not available"
            ).to_json()
        )
        return

    # Real run path is implemented in Task 4. Until then, emit a clear error so a
    # partially-installed environment does not look like a silent success.
    try:
        raise NotImplementedError("NetLogo run path not yet implemented (Task 4)")
    except Exception:
        print(_skeleton(req, STATUS_ERROR, traceback.format_exc()).to_json())


if __name__ == "__main__":
    main(sys.argv[1])
