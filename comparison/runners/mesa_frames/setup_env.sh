#!/usr/bin/env bash
# setup_env.sh — Build the dedicated mesa-frames virtual environment.
#
# Creates comparison/runners/mesa_frames/.venv-mf/ using Python 3.12
# (mesa-frames requires numpy<2, which has Python 3.12 wheels available).
# Installs only the packages listed in requirements-mesa-frames.txt.
# The comparison.* package is NOT installed here; the orchestrator sets
# PYTHONPATH=<repo root> when invoking this runner subprocess.
#
# Usage (from any directory):
#   bash comparison/runners/mesa_frames/setup_env.sh

set -euo pipefail

PYTHON="/opt/homebrew/bin/python3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-mf"
REQS="$SCRIPT_DIR/requirements-mesa-frames.txt"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found."
    echo "mesa-frames requires numpy<2 with Python 3.11 or 3.12 wheels."
    echo "Install Python 3.12 via Homebrew: brew install python@3.12"
    exit 1
fi

echo "Using interpreter: $("$PYTHON" --version)"
echo "Creating venv at: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

echo "Installing requirements from $REQS ..."
"$VENV_DIR/bin/pip" install --quiet -r "$REQS"

echo ""
echo "Done. Verify with:"
echo "  $VENV_DIR/bin/python -c \"import mesa_frames, polars; print('ok')\""
