"""Subprocess runner: spawn one benchmark process, measure wall time + peak RSS, enforce timeout."""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ProcessOutcome:
    wall_seconds: float
    peak_rss_bytes: int
    exit_code: int
    timed_out: bool
    stdout: str
    stderr: str


def _rss_total(proc: psutil.Process) -> int:
    """Return RSS of *proc* plus all its current children (best-effort)."""
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            rss += child.memory_info().rss
    return rss


_SPIN_WINDOW_S = 0.2  # spin without sleep for the first 200 ms to catch short bursts


def _sample_peak(
    proc: psutil.Process,
    stop: threading.Event,
    out: dict,
    interval: float,
) -> None:
    """Monitor thread: sample RSS of proc + children until stop is set.

    Spins tightly for the first ``_SPIN_WINDOW_S`` seconds so short-lived
    processes (< 200 ms) are still measured accurately, then falls back to
    the interval-based polling to avoid burning CPU for long runs.
    """
    peak = out.get("peak", 0)  # inherit any pre-start sample
    spin_deadline = time.monotonic() + _SPIN_WINDOW_S
    spinning = True
    while not stop.is_set():
        try:
            peak = max(peak, _rss_total(proc))
        except psutil.Error:
            break
        if spinning:
            if time.monotonic() >= spin_deadline:
                spinning = False
        else:
            time.sleep(interval)
    out["peak"] = peak


def run_subprocess(
    cmd: list[str],
    budget_s: float,
    sample_interval_s: float = 0.05,
    env: dict | None = None,
) -> ProcessOutcome:
    """Spawn *cmd* as a subprocess, measure wall time and peak RSS, kill on timeout.

    Parameters
    ----------
    cmd:
        Command and arguments to execute.
    budget_s:
        Wall-clock budget in seconds. The process is killed if it exceeds this.
    sample_interval_s:
        Polling interval for RSS sampling (seconds).
    env:
        Environment variables for the subprocess. ``None`` inherits the current
        process environment.

    Returns
    -------
    ProcessOutcome
        Wall time, peak RSS, exit code, timeout flag, stdout, and stderr.
    """
    start = time.perf_counter()
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        ps: psutil.Process | None = psutil.Process(popen.pid)
    except psutil.Error:
        ps = None

    # Take an immediate sample so very short-lived processes aren't missed.
    initial_rss = 0
    if ps is not None:
        with contextlib.suppress(psutil.Error):
            initial_rss = _rss_total(ps)

    peak_out: dict[str, int] = {"peak": initial_rss}
    stop = threading.Event()
    monitor: threading.Thread | None = None

    if ps is not None:
        monitor = threading.Thread(
            target=_sample_peak,
            args=(ps, stop, peak_out, sample_interval_s),
            daemon=True,
        )
        monitor.start()

    timed_out = False
    try:
        stdout, stderr = popen.communicate(timeout=budget_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        popen.kill()
        stdout, stderr = popen.communicate()
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()

    return ProcessOutcome(
        wall_seconds=time.perf_counter() - start,
        peak_rss_bytes=int(peak_out["peak"]),
        exit_code=popen.returncode if popen.returncode is not None else -1,
        timed_out=timed_out,
        stdout=stdout or "",
        stderr=stderr or "",
    )
