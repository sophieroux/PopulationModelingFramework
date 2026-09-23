#!/usr/bin/env bash
# Idempotent environment bootstrap for the AGN neutrino population modeling notebooks.
# Creates a Python virtual environment and installs pinned scientific/ML dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure Python venv support and a C toolchain exist. Harmless if already present
# (the base image / snapshot normally already provides these).
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-dev build-essential
  fi
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
