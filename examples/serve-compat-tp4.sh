#!/usr/bin/env bash
# Validated TP4/EP2 profile for four 96 GiB SM120 GPUs over PCIe.
set +x
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: MODEL_DIR=/path/to/checkpoint CACHE_DIR=/path/to/cache \
       [GPU_DEVICES=0,1,2,3] [API_KEY=...] bash examples/serve-compat-tp4.sh [--dry-run]

Starts a detached container; --dry-run prints a redacted command without writes.
GPU_DEVICES must select exactly four distinct GPU indices or GPU UUIDs.
Defaults: PORT=18001, SERVED_MODEL_NAME=deepseek-v4.1-flash-1017,
CONTAINER_NAME=deepseek-v41-flash-sm120-compat-tp4, MEM_FRACTION=0.85,
CONTEXT_LENGTH=1048576, MAX_RUNNING_REQUESTS=16, ENABLE_HICACHE=1,
HICACHE_RATIO=2, MAX_NEW_TOKENS_LIMIT=393216. IMAGE may override the local image.
API_KEY is optional; setting it enables bearer authentication. Never commit it.
EOF
}
if [[ "${1:-}" == --help && $# == 1 ]]; then usage; exit 0; fi
if [[ $# != 0 && ! ( $# == 1 && "$1" == --dry-run ) ]]; then usage >&2; exit 2; fi
dry_run=${1:-}
: "${MODEL_DIR:?set MODEL_DIR to the pinned DeepSeek-V4.1-Flash-1017 checkpoint}"
: "${CACHE_DIR:?set CACHE_DIR to a version-specific persistent cache directory}"
IMAGE=${IMAGE:-local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
PORT=${PORT:-18001}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4.1-flash-1017}
CONTAINER_NAME=${CONTAINER_NAME:-deepseek-v41-flash-sm120-compat-tp4}
MEM_FRACTION=${MEM_FRACTION:-0.85}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
MAX_NEW_TOKENS_LIMIT=${MAX_NEW_TOKENS_LIMIT:-393216}
ENABLE_HICACHE=${ENABLE_HICACHE:-1}
HICACHE_RATIO=${HICACHE_RATIO:-2}

fail() { echo "$*" >&2; exit 2; }
[[ "${TP_SIZE:-4}" == 4 ]] || fail 'This profile requires TP_SIZE=4'
[[ -f "$MODEL_DIR/config.json" ]] || fail 'MODEL_DIR must contain config.json'
[[ ! -e "$CACHE_DIR" || -d "$CACHE_DIR" ]] || fail 'CACHE_DIR must be a directory'
[[ "$ENABLE_HICACHE" == 0 || "$ENABLE_HICACHE" == 1 ]] || fail 'ENABLE_HICACHE must be 0 or 1'
for key in PORT CONTEXT_LENGTH MAX_RUNNING_REQUESTS MAX_NEW_TOKENS_LIMIT HICACHE_RATIO; do
  value=${!key}
  [[ "$value" =~ ^[1-9][0-9]{0,8}$ ]] || fail "$key must be a positive integer"
done
(( PORT <= 65535 )) || fail 'PORT exceeds 65535'
(( CONTEXT_LENGTH <= 1048576 )) || fail 'CONTEXT_LENGTH exceeds 1048576'
(( MAX_RUNNING_REQUESTS <= 16 )) || fail 'This profile supports at most 16 running requests'
[[ "$MEM_FRACTION" =~ ^0\.[0-9]+$ ]] || fail 'MEM_FRACTION must be between 0 and 1'
awk -v value="$MEM_FRACTION" 'BEGIN { exit !(value > 0 && value < 1) }' || fail 'Invalid MEM_FRACTION'
[[ "$GPU_DEVICES" != ,* && "$GPU_DEVICES" != *, && "$GPU_DEVICES" != *,,* ]] || fail 'GPU_DEVICES contains an empty entry'
IFS=, read -r -a devices <<< "$GPU_DEVICES"
[[ ${#devices[@]} == 4 ]] || fail 'GPU_DEVICES must select exactly four GPUs'
seen=,
for device in "${devices[@]}"; do
  [[ "$device" =~ ^(0|[1-9][0-9]*)$ || "$device" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || fail 'GPU_DEVICES must contain GPU indices or full GPU UUIDs'
  [[ "$seen" != *",${device},"* ]] || fail 'GPU_DEVICES contains a duplicate'
  seen+="${device},"
done
model_dir=$(cd "$MODEL_DIR" && pwd -P)
if [[ "$dry_run" != --dry-run ]]; then mkdir -p "$CACHE_DIR"; fi
if [[ -d "$CACHE_DIR" ]]; then
  cache_dir=$(cd "$CACHE_DIR" && pwd -P)
elif [[ "$CACHE_DIR" == /* ]]; then
  cache_dir=$CACHE_DIR
else
  cache_dir=$PWD/$CACHE_DIR
fi
container_model_path=/models/deepseek-v41-flash
cmd=(docker run --detach --restart no
  --name "$CONTAINER_NAME" --entrypoint sglang
  --gpus "\"device=${GPU_DEVICES}\"" --ipc host --ulimit memlock=-1
  --publish "${PORT}:8000"
  --volume "${model_dir}:${container_model_path}:ro"
  --volume "${cache_dir}:/root/.cache"
  --env CUDA_VISIBLE_DEVICES=0,1,2,3
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
  --env SGLANG_RAGGED_VERIFY_MODE=compact
  --env SGLANG_RAGGED_VERIFY_SUB_WIDTH_TIERS=1
  --env SGLANG_DSPARK_BUDGET_SURVIVAL_EMA=0.15
  --env SGLANG_DSPARK_SPS_AUTOTUNE_TEMPERATURE=1.0
  --env SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P=1.0
  --env SGLANG_SPEC_TP_SYNC=rng
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1
  --env SGLANG_PCIE_IPC_MAX_NUMEL=41943040
  --env SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
  --env SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
  --env SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
  --env SGLANG_ENABLE_SM120_MXFP8_SKINNY_GEMM=1
  --env SGLANG_ENABLE_BF16_SKINNY_LM_HEAD=1
  --env SGLANG_DSV41_WO_A_FP8_COPY=1
  --env SGLANG_NUMA_BIND_V2=0
  --env SGLANG_MAX_NEW_TOKENS_LIMIT="$MAX_NEW_TOKENS_LIMIT"
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
  --env MAX_JOBS=4
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
  --env TRITON_CACHE_DIR=/root/.cache/triton
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang
  "$IMAGE" serve --model-path "$container_model_path"
  --served-model-name "$SERVED_MODEL_NAME"
  --tp-size 4 --ep-size 2 --moe-a2a-backend none
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --speculative-dspark-sps-autotune auto
  --speculative-dspark-align-verify-tokens-to-graph-tier
  --max-running-requests "$MAX_RUNNING_REQUESTS" --cuda-graph-max-bs-decode 16
  --chunked-prefill-size 8192 --max-prefill-tokens 8192
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION"
  --kv-cache-dtype fp8_e4m3 --attention-backend dsv4
  --moe-runner-backend flashinfer_mxfp4 --fp8-gemm-backend flashinfer_cutlass
  --load-format instanttensor --warmups serving_coverage
  --enable-encoder-swa-bounded-replay --enable-decoder-swa-bounded-replay
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41
  --sampling-defaults openai --enable-multimodal
  --min-free-slots-delay 1 --disable-custom-all-reduce
  --enable-deepseek-v4-fp4-indexer --limit-mm-data-per-request '{"image":128}'
  --enable-metrics --enable-cache-report --host 0.0.0.0 --port 8000)
if [[ "$ENABLE_HICACHE" == 1 ]]; then cmd+=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO"); fi
if [[ -n "${API_KEY:-}" ]]; then cmd+=(--api-key "$API_KEY"); fi
if [[ "$dry_run" == --dry-run ]]; then
  redact=0
  for arg in "${cmd[@]}"; do
    if [[ "$redact" == 1 ]]; then printf '%q ' '[REDACTED]'; redact=0
    else printf '%q ' "$arg"; fi
    if [[ "$arg" == --api-key ]]; then redact=1; fi
  done
  printf '\n'
  exit 0
fi
exec "${cmd[@]}"
