#!/usr/bin/env bash
# Build only on a Linux amd64 Docker builder. Does not push or deploy.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
bash scripts/validate-release.sh
read -r stack_key flashinfer_key < <(bash build/stack-key.sh "$root")
tag=$(jq -r .candidate_tag release.json)
image="local/sglang-deepseek-v41-flash-sm120-compat:${tag}"
stack="local/sglang-deepseek-v41-flash-sm120-compat-stack:${stack_key}"
revision=$(git rev-parse HEAD)
base_cmd=(docker buildx build --platform linux/amd64 --load -f Containerfile.stack --build-arg "FLASHINFER_KEY=$flashinfer_key" -t "$stack" .)
serve_cmd=(docker buildx build --platform linux/amd64 --load -f Containerfile --build-arg "STACK_IMAGE=$stack" --build-arg IMAGE_SOURCE=local/sglang-deepseek-v41-flash-sm120-compat --build-arg "IMAGE_SOURCE_REVISION=$revision" -t "$image" .)
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${base_cmd[@]}"; printf '\n'
  printf '%q ' "${serve_cmd[@]}"; printf '\n'
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: scripts/build-compat.sh [--dry-run]' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit image inputs first so the image revision identifies the exact source.' >&2; exit 1; }
platform=$(docker info --format '{{.OSType}}/{{.Architecture}}')
[[ "$platform" == linux/x86_64 || "$platform" == linux/amd64 ]] || { echo "A Linux amd64 builder is required (found $platform)." >&2; exit 1; }
"${base_cmd[@]}"
"${serve_cmd[@]}"
mkdir -p artifacts
docker image inspect --format '{{.Id}}' "$image" > artifacts/compat-image-id.txt
printf 'Built %s; image build includes installed-source and CPU gates. GPU validation remains separate.\n' "$image"
