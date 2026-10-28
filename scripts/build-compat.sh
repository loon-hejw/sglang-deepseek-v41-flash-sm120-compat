#!/usr/bin/env bash
# Build locally on Linux amd64; never push or deploy.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
[[ $# == 0 || ( $# == 1 && "$1" == --dry-run ) ]] || { echo 'usage: scripts/build-compat.sh [--dry-run]' >&2; exit 2; }
jobs=${COMPAT_MAX_JOBS:-4}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'COMPAT_MAX_JOBS must be a positive integer' >&2; exit 2; }
network=${COMPAT_BUILD_NETWORK:-default}
[[ "$network" == default || "$network" == host || "$network" == none ]] || { echo 'Invalid build network' >&2; exit 2; }
bash scripts/validate-release.sh
read -r stack_key flashinfer_key < <(bash build/stack-key.sh "$root")
tag=$(jq -r .candidate_tag release.json)
image="local/sglang-deepseek-v41-flash-sm120-compat:${tag}"
stack="local/sglang-deepseek-v41-flash-sm120-compat-stack:${stack_key}"
base=${COMPAT_BASE_IMAGE:-$(jq -r .base.image stack.lock.json)}
receipt=${COMPAT_BASE_RECEIPT:-$root/artifacts/base-receipt.json}
revision=$(git rev-parse HEAD)
source_cache=${COMPAT_SOURCE_CACHE:-}
context_args=()
if [[ -n "$source_cache" ]]; then
  [[ -f "$source_cache/git-env.sh" && -f "$source_cache/receipt.json" ]] || { echo "Missing source cache receipt" >&2; exit 2; }
  context_args=(--build-context "locked_sources=$source_cache")
fi
recipes="$root/artifacts/build-recipes"
mkdir -p "$recipes"
# Match upstream CI: cap jobs in temporary recipes, while COPY retains the
# canonical Containerfile.stack used by the content-addressed stack key.
python3 - "$jobs" "$recipes" "$source_cache" <<'PY'
from pathlib import Path
import sys
jobs, destination, source_cache = sys.argv[1:]
for name, original in [('Containerfile.stack', '&& bash /opt/dsv41-platform/scripts/build-base.sh'), ('Containerfile', 'RUN bash /opt/dsv41-platform/scripts/build-stack.sh')]:
    source = Path(name).read_text()
    assert source.count(original) == 1, name
    modified = source.replace(original, original.replace('bash ', f'MAX_JOBS={jobs} CARGO_BUILD_JOBS={jobs} bash ', 1))
    if source_cache:
        needle = 'RUN test -n' if name == 'Containerfile.stack' else f'RUN MAX_JOBS={jobs}'
        prefix = 'RUN --mount=type=bind,from=locked_sources,target=/opt/dsv41-platform/git-mirrors . /opt/dsv41-platform/git-mirrors/git-env.sh && '
        assert modified.count(needle) == 1
        modified = modified.replace(needle, prefix + needle.removeprefix('RUN '), 1)
    (Path(destination) / name).write_text(modified)
PY
base_cmd=(docker buildx build --platform linux/amd64 --load --pull=false --progress plain --network "$network" ${context_args[@]+"${context_args[@]}"} -f "$recipes/Containerfile.stack" --build-arg "BASE_IMAGE=$base" --build-arg "FLASHINFER_KEY=$flashinfer_key" -t "$stack" .)
serve_cmd=(docker buildx build --platform linux/amd64 --load --pull=false --progress plain --network "$network" ${context_args[@]+"${context_args[@]}"} -f "$recipes/Containerfile" --build-arg "STACK_IMAGE=$stack" --build-arg IMAGE_SOURCE=local/sglang-deepseek-v41-flash-sm120-compat --build-arg "IMAGE_SOURCE_REVISION=$revision" -t "$image" .)
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${base_cmd[@]}"; printf '\n'
  printf '%q ' "${serve_cmd[@]}"; printf '\n'
  exit 0
fi
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit image inputs first so the image revision identifies the exact source.' >&2; exit 1; }
platform=$(docker info --format '{{.OSType}}/{{.Architecture}}')
[[ "$platform" == linux/x86_64 || "$platform" == linux/amd64 ]] || { echo "A Linux amd64 builder is required (found $platform)." >&2; exit 1; }
if [[ ! -f "$receipt" ]]; then
  python3 scripts/verify-base-image.py --source "${COMPAT_BASE_SOURCE:-$base}" --receipt "$receipt"
fi
if ! docker image inspect "$base" >/dev/null 2>&1; then
  docker pull --platform linux/amd64 "$base"
fi
python3 scripts/verify-base-image.py --receipt "$receipt" --image "$base"
"${base_cmd[@]}"
"${serve_cmd[@]}"
docker image inspect --format '{{.Id}}' "$image" > artifacts/compat-image-id.txt
printf 'Built %s; installed-source, dependency and CPU gates passed. GPU validation remains separate.\n' "$image"
