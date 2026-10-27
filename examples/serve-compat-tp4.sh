#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export IMAGE=${IMAGE:-local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.1}
# Use a separate cache from the baseline experiment.
export CACHE_DIR=${CACHE_DIR:-/srv/cache/sglang-deepseek-v41-compat-v0.4.4-rc.1-tp4}
export PORT=${PORT:-18001}
export CONTAINER_NAME=${CONTAINER_NAME:-deepseek-v41-flash-sm120-compat-tp4}
exec bash "$root/examples/serve-deepseek-v41-flash-tp4.sh" "$@"
