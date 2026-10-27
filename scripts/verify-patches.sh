#!/usr/bin/env bash
# Reproduce every source tree of the image from its official upstream commits
# and verify the pinned vendor base. For SGLang, FlashInfer, DeepGEMM and
# Triton this fetches the locked commit, checks its commit and tree, applies the
# hash-pinned patch series in lock order and checks the effective tree and the
# packaging edits: the same build/prepare-sources.py the image build runs. No
# private source credentials are involved. Needs bash, git, jq, curl, python3
# and network access.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${COMPAT_PYTHON:-python3}
for tool in jq git curl "$python"; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
"$repo/scripts/validate-release.sh"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$repo/stack.lock.json" "$work/"
cp -R "$repo/patches" "$work/patches"

echo "== source trees =="
if [[ "${COMPAT_SOURCE_FETCH:-git}" == archive ]]; then
  "$python" "$repo/scripts/verify-source-archives.py" --dest "$work/sources"
else
  "$python" "$repo/build/prepare-sources.py" "$work" stack.lock.json sglang flashinfer deepgemm triton
fi

echo "== base image manifest =="
base=$(jq -er '.base.image' "$repo/stack.lock.json")
name=${base#docker.io/}
name=${name%@*}
digest=${base##*@}
token=$(curl --connect-timeout 10 --max-time 60 -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${name}:pull" | jq -er .token)
accept='application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'
curl --connect-timeout 10 --max-time 60 -fsSL -o /dev/null -H "Authorization: Bearer ${token}" -H "Accept: ${accept}" \
  "https://registry-1.docker.io/v2/${name}/manifests/${digest}"
printf '  %s@%s\n' "$name" "$digest"

if [[ "${COMPAT_SOURCE_FETCH:-git}" == archive ]]; then
  mkdir -p "$repo/artifacts"
  cp "$work/sources/archive-source-receipt.json" "$repo/artifacts/archive-source-receipt.json"
fi
echo "release bundle verified: official SGLang, FlashInfer, DeepGEMM and Triton commits plus the locked series reproduce every effective tree"
