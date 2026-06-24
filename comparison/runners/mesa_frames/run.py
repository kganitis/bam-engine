"""mesa-frames runner stub for the comparison harness.

This is the mesa-frames subprocess runner: reads a :class:`RunRequest` JSON
file (path in ``sys.argv[1]``), and raises :class:`NotImplementedError` until
the actual BAM model is implemented in Task 9.

Usage
-----
    python -m comparison.runners.mesa_frames.run <request.json>

This runner runs inside a dedicated virtual environment
(``comparison/runners/mesa_frames/.venv-mf/``) that pins ``numpy<2``, which
is required by mesa-frames.  The orchestrator invokes it via
``RUNNER_CMD["mesa_frames"]`` with ``PYTHONPATH`` set to the repo root so that
``comparison.*`` is importable without installing bamengine into this venv.
"""

from __future__ import annotations

import sys

from comparison.orchestrator.contract import RunRequest


def main(request_path: str) -> None:
    """Read a RunRequest and raise NotImplementedError (model not yet implemented)."""
    with open(request_path) as fh:
        _req = RunRequest.from_json(fh.read())

    raise NotImplementedError("mesa_frames model not yet implemented")


if __name__ == "__main__":
    main(sys.argv[1])
