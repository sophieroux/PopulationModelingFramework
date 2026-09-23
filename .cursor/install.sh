#!/usr/bin/env bash
# Idempotent environment bootstrap for the AGN neutrino population modeling notebooks.
#
# Dependencies are installed into the per-user site (~/.local), NOT into a venv
# inside /workspace. Environment builds re-check-out /workspace on every boot and
# skip the install step, so anything placed under /workspace would be lost; the
# user site lives in $HOME and is baked into the build image, so plain `python3`
# and `jupyter` work on every boot without activating anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure a C toolchain / Python headers exist for any source builds.
# Harmless if already present (the base image normally provides these).
if command -v sudo >/dev/null 2>&1 && ! dpkg -s build-essential >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-dev build-essential
fi

PIP="python3 -m pip install --user --break-system-packages"
$PIP --upgrade pip
$PIP -r requirements.txt

echo "Dependencies installed into the user site (~/.local). Use python3 / jupyter directly."
