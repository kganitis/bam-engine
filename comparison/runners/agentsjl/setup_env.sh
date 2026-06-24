#!/usr/bin/env bash
# setup_env.sh - Instantiate and precompile the Agents.jl Julia environment.
#
# Run from any directory:
#   bash comparison/runners/agentsjl/setup_env.sh
#
# What it does:
#   1. Runs Pkg.instantiate() to resolve and download dependencies from
#      Project.toml / Manifest.toml (pins exact versions for reproducibility).
#   2. Runs Pkg.precompile() to compile all packages (speeds up first run.jl
#      invocation significantly).
#   3. Verifies that `using Agents, JSON3` succeeds (smoke check).
#
# After the first successful run, Manifest.toml is generated. Commit it to
# pin exact resolved versions so future installs are reproducible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-/opt/homebrew/bin/julia}"

echo "==> Julia binary: $JULIA"
echo "==> Project dir:  $SCRIPT_DIR"

echo ""
echo "==> Step 1: Instantiate + precompile ..."
"$JULIA" --project="$SCRIPT_DIR" --startup-file=no -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
    println("instantiate+precompile: ok")
'

echo ""
echo "==> Step 2: Verify using Agents, JSON3 ..."
"$JULIA" --project="$SCRIPT_DIR" --startup-file=no -e '
    using Agents, JSON3
    println("using Agents, JSON3: ok")
'

echo ""
echo "==> setup_env.sh complete. Commit Manifest.toml to pin exact versions."
