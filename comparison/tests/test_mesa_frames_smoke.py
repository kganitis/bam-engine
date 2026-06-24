"""Smoke tests for the mesa-frames runner registration and requirements.

These tests do NOT require the .venv-mf virtual environment to exist, so they
remain fast in CI without triggering a full environment build.
"""

from __future__ import annotations

from pathlib import Path

from comparison.orchestrator.run import RUNNER_CMD

_RUNNERS_DIR = Path(__file__).resolve().parent.parent / "runners" / "mesa_frames"
_REQS_FILE = _RUNNERS_DIR / "requirements-mesa-frames.txt"


def test_runner_cmd_has_mesa_frames_entry():
    """RUNNER_CMD must register a mesa_frames entry."""
    assert "mesa_frames" in RUNNER_CMD, (
        "RUNNER_CMD does not contain a 'mesa_frames' key"
    )


def test_mesa_frames_runner_cmd_points_to_venv_python():
    """The mesa_frames RUNNER_CMD must point to .venv-mf/bin/python."""
    cmd = RUNNER_CMD["mesa_frames"]
    assert isinstance(cmd, list), "RUNNER_CMD['mesa_frames'] must be a list"
    assert len(cmd) >= 1, "RUNNER_CMD['mesa_frames'] must be non-empty"
    python_path = cmd[0]
    assert python_path.endswith(".venv-mf/bin/python"), (
        f"Expected path ending with '.venv-mf/bin/python', got: {python_path!r}"
    )


def test_mesa_frames_runner_cmd_uses_module_flag():
    """The mesa_frames RUNNER_CMD must invoke the runner via -m."""
    cmd = RUNNER_CMD["mesa_frames"]
    assert len(cmd) >= 3, "RUNNER_CMD['mesa_frames'] must have at least 3 elements"
    assert cmd[1] == "-m", f"Expected '-m' as second element, got: {cmd[1]!r}"
    assert cmd[2] == "comparison.runners.mesa_frames.run", (
        f"Expected module path 'comparison.runners.mesa_frames.run', got: {cmd[2]!r}"
    )


def test_requirements_pins_numpy_lt_2():
    """requirements-mesa-frames.txt must pin numpy<2."""
    assert _REQS_FILE.exists(), f"Requirements file not found: {_REQS_FILE}"
    content = _REQS_FILE.read_text()
    assert "numpy<2" in content, (
        f"'numpy<2' not found in {_REQS_FILE}; content:\n{content}"
    )


def test_requirements_pins_mesa_frames_version():
    """requirements-mesa-frames.txt must pin mesa-frames==0.1.0a0."""
    assert _REQS_FILE.exists(), f"Requirements file not found: {_REQS_FILE}"
    content = _REQS_FILE.read_text()
    assert "mesa-frames==0.1.0a0" in content, (
        f"'mesa-frames==0.1.0a0' not found in {_REQS_FILE}; content:\n{content}"
    )
