#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///
"""Nox sessions for BAM Engine development tasks.

This module defines automated development workflows using Nox.
Run `nox -l` to list all available sessions.
"""

from __future__ import annotations

import nox

nox.needs_version = ">=2024.3.2"

# Use uv for faster dependency resolution if available
nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]
DEFAULT_PYTHON = "3.12"


@nox.session(python=DEFAULT_PYTHON)
def lint(session: nox.Session) -> None:
    """Run linting checks (ruff, mypy)."""
    session.install("-e", ".[lint]")
    session.run("ruff", "format", "--check", ".")
    session.run("ruff", "check", ".")
    session.run("mypy")


@nox.session(python=DEFAULT_PYTHON)
def format(session: nox.Session) -> None:
    """Format code with ruff."""
    session.install("-e", ".[lint]")
    session.run("ruff", "format", ".")
    session.run("ruff", "check", "--fix", ".")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e", ".[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=DEFAULT_PYTHON)
def tests_quick(session: nox.Session) -> None:
    """Run quick tests only (skip slow, regression, invariants)."""
    session.install("-e", ".[test]")
    session.run(
        "pytest",
        "-m",
        "not slow and not regression and not invariants",
        *session.posargs,
    )


@nox.session(python=DEFAULT_PYTHON)
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-b", "html", ".", "_build/html")


@nox.session(python=DEFAULT_PYTHON)
def docs_live(session: nox.Session) -> None:
    """Build documentation with live reload."""
    session.install("-e", ".[docs]", "sphinx-autobuild")
    session.chdir("docs")
    session.run(
        "sphinx-autobuild",
        ".",
        "_build/html",
        "--open-browser",
        "--watch",
        "../src",
    )


@nox.session(python=DEFAULT_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run tests with coverage report."""
    session.install("-e", ".[test]")
    session.run(
        "pytest",
        "--cov=src/bamengine",
        "--cov-report=term-missing",
        "--cov-report=html",
        *session.posargs,
    )


@nox.session(python=DEFAULT_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("-e", ".[lint]")
    session.run("mypy", *session.posargs)


if __name__ == "__main__":
    nox.main()
