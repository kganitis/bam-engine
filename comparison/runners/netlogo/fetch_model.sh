#!/usr/bin/env bash
# Fetch the third-party Platas BAM model (GPL-2.0) into a local, gitignored dir.
# The model is NEVER committed or pushed; this script is how a developer obtains
# it locally to run the NetLogo benchmark, and removes it again via teardown.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/DelliBAM_.nlogo"
URL="https://raw.githubusercontent.com/alexplatasl/BAMmodel/master/DelliBAM_.nlogo"

# Pinned checksum. If empty, the script prints the downloaded file's SHA-256 and
# exits 3 so the developer can pin it on first fetch (Task 3 records the value).
EXPECTED_SHA256="${NETLOGO_MODEL_SHA256:-}"

mkdir -p "$MODEL_DIR"
echo "Downloading Platas DelliBAM_.nlogo ..."
curl -fsSL "$URL" -o "$MODEL_FILE"

ACTUAL="$(shasum -a 256 "$MODEL_FILE" | awk '{print $1}')"
if [[ -z "$EXPECTED_SHA256" ]]; then
  echo "Downloaded SHA-256: $ACTUAL"
  echo "Pin this value in fetch_model.sh (EXPECTED_SHA256) or export NETLOGO_MODEL_SHA256."
  exit 3
fi
if [[ "$ACTUAL" != "$EXPECTED_SHA256" ]]; then
  echo "CHECKSUM MISMATCH: expected $EXPECTED_SHA256, got $ACTUAL" >&2
  rm -f "$MODEL_FILE"
  exit 4
fi
echo "OK: $MODEL_FILE ($ACTUAL)"
