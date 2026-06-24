"""Smoke tests for the Agents.jl runner registration and project layout.

These tests do NOT require Julia to be installed; they only inspect the
RUNNER_CMD registration and the presence of Project.toml with the expected
dependency declarations.

Run with:
    .venv/bin/python -m pytest comparison/tests/test_agentsjl_smoke.py --no-cov -v
"""

from __future__ import annotations

from pathlib import Path

from comparison.orchestrator.run import RUNNER_CMD

_AGENTSJL_DIR = Path(__file__).resolve().parent.parent / "runners" / "agentsjl"
_PROJECT_TOML = _AGENTSJL_DIR / "Project.toml"


def test_runner_cmd_has_agentsjl_entry():
    """RUNNER_CMD must register an agentsjl entry."""
    assert "agentsjl" in RUNNER_CMD, "RUNNER_CMD does not contain an 'agentsjl' key"


def test_agentsjl_runner_cmd_starts_with_julia():
    """The agentsjl RUNNER_CMD must invoke julia as the executable."""
    cmd = RUNNER_CMD["agentsjl"]
    assert isinstance(cmd, list), "RUNNER_CMD['agentsjl'] must be a list"
    assert len(cmd) >= 1, "RUNNER_CMD['agentsjl'] must be non-empty"
    assert cmd[0] == "julia", f"Expected 'julia' as first element, got: {cmd[0]!r}"


def test_agentsjl_runner_cmd_project_arg_ends_with_runners_agentsjl():
    """The --project arg must point to a path ending with runners/agentsjl."""
    cmd = RUNNER_CMD["agentsjl"]
    project_args = [a for a in cmd if a.startswith("--project=")]
    assert project_args, "No --project= argument found in agentsjl RUNNER_CMD"
    project_path = project_args[0].removeprefix("--project=")
    assert project_path.endswith("runners/agentsjl"), (
        f"Expected --project path ending with 'runners/agentsjl', got: {project_path!r}"
    )


def test_agentsjl_runner_cmd_script_ends_with_run_jl():
    """The last positional arg (script path) must end with run.jl."""
    cmd = RUNNER_CMD["agentsjl"]
    # The script path is the last element (after the flags).
    script_path = cmd[-1]
    assert script_path.endswith("run.jl"), (
        f"Expected script path ending with 'run.jl', got: {script_path!r}"
    )


def test_agentsjl_runner_cmd_has_startup_file_no():
    """The agentsjl RUNNER_CMD must include --startup-file=no."""
    cmd = RUNNER_CMD["agentsjl"]
    assert "--startup-file=no" in cmd, (
        "'--startup-file=no' not found in agentsjl RUNNER_CMD"
    )


def test_project_toml_exists():
    """Project.toml must exist at runners/agentsjl/Project.toml."""
    assert _PROJECT_TOML.exists(), f"Project.toml not found: {_PROJECT_TOML}"


def test_project_toml_declares_agents():
    """Project.toml must declare Agents as a dependency."""
    assert _PROJECT_TOML.exists(), f"Project.toml not found: {_PROJECT_TOML}"
    content = _PROJECT_TOML.read_text()
    assert "Agents" in content, (
        f"'Agents' not found in {_PROJECT_TOML}; content:\n{content}"
    )


def test_project_toml_declares_json3():
    """Project.toml must declare JSON3 as a dependency."""
    assert _PROJECT_TOML.exists(), f"Project.toml not found: {_PROJECT_TOML}"
    content = _PROJECT_TOML.read_text()
    assert "JSON3" in content, (
        f"'JSON3' not found in {_PROJECT_TOML}; content:\n{content}"
    )
