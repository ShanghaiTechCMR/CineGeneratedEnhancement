#!/usr/bin/env bash
# Portable CGE training launcher. Activate the environment from environment.yaml
# before running this script, then pass the public NPZ shard directories below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPOSITORY_DIR}"

python scripts/train.py "$@"
