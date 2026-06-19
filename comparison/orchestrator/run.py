"""Orchestrator CLI: phases A (equivalence) + B (timing).

Phase A: Run equivalence (gate) jobs in parallel using ProcessPoolExecutor.
Phase B: Run timing jobs serially with an adaptive wall-clock cap -- larger
         sizes are skipped after the first timeout for a given framework.

Usage
-----
    python -m comparison.orchestrator.run [--frameworks bamengine,...] \\
        [--gate-workers 10] [--budget 120] [--sizes ...] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from comparison.equivalence.gate import evaluate_gate
from comparison.equivalence.metrics import compute_metrics
from comparison.orchestrator import matrix
from comparison.orchestrator.contract import (
    STATUS_ERROR,
    STATUS_TIMEOUT,
    RunRequest,
    RunResult,
)
from comparison.orchestrator.environment import capture_environment, environment_id
from comparison.orchestrator.subprocess_runner import run_subprocess

RUNNER_CMD: dict[str, list[str]] = {
    "bamengine": [sys.executable, "-m", "comparison.runners.bamengine.run"],
}

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
}


def _pinned_env() -> dict:
    """Return os.environ merged with single-thread pinning vars."""
    env = dict(os.environ)
    env.update(_THREAD_ENV)
    return env


def _execute(req: RunRequest, budget_s: float) -> dict:
    """Execute one RunRequest in a subprocess, returning a raw result dict."""
    if req.framework not in RUNNER_CMD:
        return {
            "run_id": req.run_id,
            "framework": req.framework,
            "status": "skipped",
            "error": "no runner registered",
        }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(req.to_json())
        path = f.name
    try:
        outcome = run_subprocess(
            RUNNER_CMD[req.framework] + [path],
            budget_s=budget_s,
            env=_pinned_env(),
        )
    finally:
        os.unlink(path)

    if outcome.timed_out:
        return {
            "run_id": req.run_id,
            "framework": req.framework,
            "status": STATUS_TIMEOUT,
            "error": "wall budget exceeded",
            "process": {"wall_seconds": outcome.wall_seconds},
        }

    try:
        last_line = outcome.stdout.strip().splitlines()[-1]
        res = RunResult.from_json(last_line)
        rec = json.loads(res.to_json())
    except Exception as exc:
        rec = {
            "run_id": req.run_id,
            "framework": req.framework,
            "status": STATUS_ERROR,
            "error": f"unparsable runner output: {exc}",
            "stderr": outcome.stderr[-2000:],
        }

    rec["process"] = {
        "wall_seconds": outcome.wall_seconds,
        "peak_rss_bytes": outcome.peak_rss_bytes,
        "startup_seconds": max(
            0.0,
            outcome.wall_seconds
            - (
                rec.get("timing", {}).get("init_seconds", 0.0)
                + rec.get("timing", {}).get("run_seconds", 0.0)
            ),
        ),
    }
    return rec


def _execute_gate(arg: tuple) -> dict:
    """Module-level wrapper for ProcessPoolExecutor (must be picklable).

    Parameters
    ----------
    arg : tuple[RunRequest, float]
        (request, budget_s) pair.
    """
    req, budget = arg
    return _execute(req, budget)


def _write(rec: dict, env_id: str, raw_dir: Path) -> None:
    """Write a raw result record to disk, tagging it with the environment ID."""
    rec["environment_id"] = env_id
    (raw_dir / f"{rec['run_id']}.json").write_text(json.dumps(rec, indent=2))


def run_benchmark(
    frameworks: list[str],
    results_dir: str | Path,
    quick: bool = False,
    gate_workers: int = 10,
    budget_s: float = 120.0,
    sizes: list[int] | None = None,
) -> dict:
    """Run the full benchmark pipeline (phases A and B).

    Phase A (equivalence gate, parallel): runs gate jobs for all frameworks in
    parallel, computes behavioral metrics, and evaluates whether each framework
    passes the equivalence gate.

    Phase B (timing, serial, adaptive): runs timing jobs for gate-passing
    frameworks in strict serial order. Skips sizes larger than the first
    timeout encountered for a given framework.

    Parameters
    ----------
    frameworks : list[str]
        Framework names to benchmark (e.g. ``["bamengine"]``).
    results_dir : str or Path
        Root directory for output. Raw results go to ``<results_dir>/raw/``.
    quick : bool
        If True, shrink periods/seeds/sizes for a fast smoke-test run.
    gate_workers : int
        Number of parallel workers for Phase A.
    budget_s : float
        Per-job wall-clock budget in seconds.
    sizes : list[int] or None
        If provided, restrict Phase B timing to these firm counts (intersected
        with ``matrix.SCALE_FIRMS``). When ``None``, all sizes are used.

    Returns
    -------
    dict
        ``{"gate": <gate dict>, "env_id": <str>, "skips": <dict>}``
    """
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    env_id = environment_id(env)
    (results_dir / f"env_{env_id}.json").write_text(json.dumps(env, indent=2))

    # ── Phase A: equivalence gate (parallel) ────────────────────────────────
    gate_jobs = matrix.equivalence_jobs(frameworks)
    if quick:
        # Keep only seeds 0 and 1, shrink to 60 periods for fast testing.
        gate_jobs = [
            RunRequest(
                run_id=j.run_id,
                framework=j.framework,
                model_params=j.model_params,
                population=j.population,
                n_periods=60,
                warmup_periods=0,
                seed=j.seed,
                collect_outputs=j.collect_outputs,
                outputs_requested=j.outputs_requested,
            )
            for j in gate_jobs
            if j.seed in (0, 1)
        ]

    by_fw: dict[str, list] = {fw: [] for fw in frameworks}

    with ProcessPoolExecutor(max_workers=gate_workers) as ex:
        recs = list(ex.map(_execute_gate, [(j, budget_s) for j in gate_jobs]))

    for req, rec in zip(gate_jobs, recs, strict=True):
        _write(rec, env_id, raw)
        if rec.get("status") == "ok" and rec.get("outputs"):
            burn_in = 0 if quick else 500
            by_fw[req.framework].append(
                compute_metrics(rec["outputs"], burn_in=burn_in)
            )

    if by_fw.get("bamengine"):
        gate = evaluate_gate(by_fw)
    else:
        gate = {"frameworks": {fw: {} for fw in frameworks}}

    # ── Phase B: timing (serial, adaptive cap) ──────────────────────────────
    passing = [
        fw
        for fw in frameworks
        if fw == "bamengine"
        or gate["frameworks"].get(fw, {}).get("passed")
        or not gate["frameworks"].get(fw, {}).get("blocking", True)
    ]

    if quick:
        allowed_sizes = set(matrix.SCALE_FIRMS[:2])
    elif sizes is not None:
        allowed_sizes = set(matrix.SCALE_FIRMS) & set(sizes)
    else:
        allowed_sizes = set(matrix.SCALE_FIRMS)
    skips: dict[str, str] = {}

    for fw in passing:
        capped_at: int | None = None
        for req in matrix.timing_jobs([fw]):
            if req.population["n_firms"] not in allowed_sizes:
                continue
            if capped_at is not None and req.population["n_firms"] >= capped_at:
                skips[req.run_id] = f"skipped: {fw} capped at {capped_at} firms"
                continue
            rec = _execute(req, budget_s)
            _write(rec, env_id, raw)
            if rec.get("status") == STATUS_TIMEOUT:
                capped_at = req.population["n_firms"]
                skips[req.run_id] = (
                    f"timeout at {capped_at} firms; larger sizes skipped"
                )

    (results_dir / "skips.json").write_text(json.dumps(skips, indent=2))

    return {"gate": gate, "env_id": env_id, "skips": skips}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Run the BAM benchmark matrix (phases A + B)."
    )
    p.add_argument(
        "--frameworks",
        default="bamengine",
        help="Comma-separated list of frameworks (default: bamengine)",
    )
    p.add_argument(
        "--gate-workers",
        type=int,
        default=10,
        help="Parallel workers for Phase A (default: 10)",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=120.0,
        help="Per-job wall-clock budget in seconds (default: 120)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Shrink periods/seeds/sizes for a fast smoke-test run",
    )
    p.add_argument(
        "--sizes",
        default=None,
        help=(
            "Comma-separated firm counts to restrict Phase B timing "
            "(e.g. --sizes 100,1000); must be values in matrix.SCALE_FIRMS"
        ),
    )
    p.add_argument(
        "--results-dir",
        default="comparison/results",
        help="Root directory for output (default: comparison/results)",
    )
    a = p.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else None
    run_benchmark(
        frameworks=a.frameworks.split(","),
        results_dir=a.results_dir,
        quick=a.quick,
        gate_workers=a.gate_workers,
        budget_s=a.budget,
        sizes=sizes,
    )


if __name__ == "__main__":
    main()
