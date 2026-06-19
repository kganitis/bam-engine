import sys

from comparison.orchestrator.subprocess_runner import run_subprocess


def test_captures_stdout_and_exit():
    out = run_subprocess([sys.executable, "-c", "print('hello')"], budget_s=30)
    assert out.exit_code == 0
    assert "hello" in out.stdout
    assert not out.timed_out
    assert out.wall_seconds >= 0
    assert out.peak_rss_bytes > 0


def test_times_out_and_kills():
    out = run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"], budget_s=1
    )
    assert out.timed_out
    assert out.exit_code != 0


def test_measures_allocation_peak():
    code = (
        "x = bytearray(160_000_000); x[::4096] = b'\\x01' * len(x[::4096]);"
        " import sys, time; sys.stdout.write('ok'); sys.stdout.flush(); time.sleep(0.3)"
    )
    out = run_subprocess([sys.executable, "-c", code], budget_s=30)
    assert out.peak_rss_bytes > 80_000_000
