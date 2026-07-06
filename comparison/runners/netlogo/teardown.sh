#!/usr/bin/env bash
# Remove all local NetLogo benchmark artifacts so the repo and machine are left
# as they were before the run. The third-party model and the NetLogo/Java
# toolchain are temporary by design (install, run the comparison, delete).
#
# This script removes the repo-local artifacts (the fetched model and the bridge
# venv). The NetLogo install and the JDK are removed with the manual commands
# printed at the end, so teardown never deletes a JDK/NetLogo you use elsewhere.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Removing local model and bridge venv ..."
rm -rf "$HERE/model" "$HERE/.venv-nl"
echo "Removed: $HERE/model  $HERE/.venv-nl"

cat <<'NOTE'

Local model and bridge venv removed.

To remove the temporary toolchain entirely, run the commands that match how you
installed it:

  # openjdk (Homebrew formula, installed for the arm64-matching JVM):
  brew uninstall openjdk@17

  # NetLogo 6.4.0: delete the extracted directory you created, e.g.
  #   rm -rf "<path>/NetLogo-6.4.0-64"
  # and the curated extensions dir you built for palette + array, e.g.
  #   rm -rf "<path>/nlext"

These are left as manual steps so teardown never uninstalls a JDK or a NetLogo
copy you may rely on for other work.
NOTE
