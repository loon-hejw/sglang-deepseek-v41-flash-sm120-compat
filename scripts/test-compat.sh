#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${COMPAT_PYTHON:-python3}
source_dir=${SGLANG_COMPAT_SOURCE:-$root/.cache/compat-sources/sglang}
if [[ ! -d "$source_dir" ]]; then
  "$python" "$root/scripts/verify-source-archives.py" --dest "$(dirname "$source_dir")" --sources sglang
fi
mkdir -p "$root/artifacts"
# An explicit local pytest.ini avoids collecting unrelated parent directories.
cd "$source_dir/test/registered/unit/compat_0731"
SGLANG_COMPAT_SOURCE="$source_dir" "$python" -m pytest -p no:cacheprovider --noconftest -c pytest.ini -q test_compat.py \
  --junitxml="$root/artifacts/compat-cpu-tests.xml" "$@"
