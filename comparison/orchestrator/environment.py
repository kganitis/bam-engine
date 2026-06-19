from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC

import psutil

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "JULIA_NUM_THREADS",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _bamengine_version() -> str:
    try:
        import bamengine

        return getattr(bamengine, "__version__", "unknown")
    except Exception:
        return "unknown"


def capture_environment() -> dict:
    from datetime import datetime

    return {
        "cpu": platform.processor() or platform.machine(),
        "n_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "ram_bytes": psutil.virtual_memory().total,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "bamengine_version": _bamengine_version(),
        "bamengine_commit": _git_commit(),
        "thread_env": {v: os.environ.get(v) for v in _THREAD_VARS},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def environment_id(env: dict) -> str:
    stable = {k: env[k] for k in sorted(env) if k != "timestamp"}
    digest = hashlib.sha1(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return digest[:12]
