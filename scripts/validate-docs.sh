#!/usr/bin/env bash
# Check the fork's public documentation without confusing it with upstream releases.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tag=$(jq -er '.candidate_tag' "$repo/release.json")
published_tag=$(jq -er '.published.tag' "$repo/release.json")
model=$(jq -er '.model.repository' "$repo/stack.lock.json")
revision=$(jq -er '.model.revision' "$repo/stack.lock.json")
image="local/sglang-deepseek-v41-flash-sm120-compat:${tag}"
upstream_image="ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:${published_tag}"
launcher="$repo/examples/serve-compat-tp4.sh"
require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} missing: $expected" >&2; exit 1;
  }
}
for name in README.md README.zh-CN.md RUN.md BUILDING.md; do
  require_text "$repo/$name" "$image"
  require_text "$repo/$name" 'compat/validation.json'
done
for name in README.md README.zh-CN.md RUN.md; do
  require_text "$repo/$name" "$model"
  require_text "$repo/$name" "$revision"
  require_text "$repo/$name" 'examples/serve-compat-tp4.sh'
  require_text "$repo/$name" "/srv/cache/sglang-deepseek-v41-compat-${tag}-tp4"
  require_text "$repo/$name" "$upstream_image"
done
if [[ -f "$repo/compat/publication.json" ]]; then
  registry_image=$(jq -er '.target' "$repo/compat/publication.json")
  registry_digest=$(jq -er '.manifest_digest' "$repo/compat/publication.json")
  for name in README.md README.zh-CN.md RUN.md BUILDING.md; do
    require_text "$repo/$name" "$registry_image"
    require_text "$repo/$name" "$registry_digest"
    require_text "$repo/$name" 'compat/publication.json'
  done
else
  require_text "$repo/README.md" 'does not currently publish a downloadable container image'
fi
require_text "$repo/README.md" 'not a throughput benchmark or proof of complete sampling equivalence'
require_text "$repo/README.md" 'https://github.com/loon-hejw/sglang-deepseek-v41-flash-sm120-compat.git'
require_text "$repo/README.md" 'TESTING.md'
require_text "$repo/README.zh-CN.md" 'TESTING.zh-CN.md'
for name in TESTING.md TESTING.zh-CN.md; do
  require_text "$repo/$name" 'compat/validation.json'
  require_text "$repo/$name" 'compat/topology-comparison-20261103.json'
  require_text "$repo/$name" 'compat/deployment-comparisons-20261103.json'
done
grep -Fx "IMAGE=\${IMAGE:-${image}}" "$launcher" >/dev/null
for value in '--tp-size 4 --ep-size 2' '--speculative-algorithm DSPARK' \
  '--speculative-dspark-block-size 5' '--speculative-dspark-sps-autotune auto' \
  '--reasoning-parser deepseek-v41' '--tool-call-parser deepseekv41' \
  '--sampling-defaults openai' 'MEM_FRACTION=${MEM_FRACTION:-0.85}' \
  'GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}' '--dry-run'; do
  require_text "$launcher" "$value"
done
require_text "$repo/examples/serve-deepseek-v41-flash-tp4.sh" "$upstream_image"
require_text "$repo/CHANGELOG.md" "## ${published_tag} (stable;"

# Report file/line only, not possible secret values.
python3 - "$repo" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
files = list(root.glob('*.md')) + list((root/'examples').glob('*.sh')) + [root/'compat/validation.json']
if (root/'compat/publication.json').exists():
    files.append(root/'compat/publication.json')
if (root/'compat/topology-comparison-20261103.json').exists():
    files.append(root/'compat/topology-comparison-20261103.json')
files.append(root/'compat/deployment-comparisons-20261103.json')
patterns = [r'10\.216\.', r'harbor\.bhidi\.com', r'/Users/', r'~/git/',
            r'home\.[a-z0-9.-]+\.(?:com|net|org)', r'\u2014',
            r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
            r'-----BEGIN [A-Z ]*PRIVATE KEY-----']
errors = []
for path in files:
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append(f'{path.relative_to(root)}:{line}: private reference, credential pattern or em dash')
for name in ['README.md','README.zh-CN.md','RUN.md','BUILDING.md','TESTING.md','TESTING.zh-CN.md']:
    path=root/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
        if '://' in target or target.startswith(('#','mailto:')): continue
        if not (path.parent/target.split('#',1)[0]).exists():
            errors.append(f'{name}: missing relative link {target}')
if errors: raise SystemExit('\n'.join(errors))
PY
printf 'documentation contract valid: compatibility %s, upstream %s, model %s\n' "$tag" "$published_tag" "$model"
