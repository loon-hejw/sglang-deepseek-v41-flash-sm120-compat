# Running the validated compatibility TP4 profile

Use `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Build `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` as described in [BUILDING.md](BUILDING.md); this repository does not publish a pullable fork image yet.

## Launch and lifecycle

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-compat-v0.4.4-rc.4-tp4
export GPU_DEVICES=0,1,2,3
export IMAGE=local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4
# Supply API_KEY in your environment if authentication is required.
bash examples/serve-compat-tp4.sh --dry-run
bash examples/serve-compat-tp4.sh
```

`GPU_DEVICES` selects exactly four distinct indices or full GPU UUIDs through Docker's device request. CUDA sees the selected devices as indices 0–3. The launcher never requests all host GPUs. The dry run needs an existing model directory containing `config.json`, but creates no cache directory and invokes no Docker command. Its printed command masks the API key.

The container is detached, keeps its name after exit, and uses restart policy `no`. Follow startup logs and wait for readiness:

```bash
docker logs --follow deepseek-v41-flash-sm120-compat-tp4
curl --fail http://127.0.0.1:18001/health
# In a shell with API_KEY set (omit the header for an unauthenticated instance):
curl --fail -H "Authorization: Bearer ${API_KEY}" http://127.0.0.1:18001/v1/models
```

Restart a stopped container with its saved configuration using `docker start deepseek-v41-flash-sm120-compat-tp4`. Stop it with `docker stop --time 120 deepseek-v41-flash-sm120-compat-tp4`. Running the launcher again does not replace a container with the same name; choose a different `CONTAINER_NAME` for a separate instance, with separate GPUs, port and cache. A configuration change requires creating a new container after freeing its selected devices. The image and cache are not deleted when a container stops.

## Settings

| Environment variable | Default / behavior |
| --- | --- |
| `MODEL_DIR` | Required, checkpoint directory; mounted read-only |
| `CACHE_DIR` | Required, version-specific persistent directory mounted at `/root/.cache` |
| `IMAGE` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` |
| `GPU_DEVICES` | `0,1,2,3`; exactly four GPU indices or full GPU UUIDs |
| `PORT` | `18001`, mapped to container port 8000 |
| `CONTAINER_NAME` | `deepseek-v41-flash-sm120-compat-tp4` |
| `SERVED_MODEL_NAME` | `deepseek-v4.1-flash-1017` |
| `API_KEY` | Optional bearer token, supplied at runtime |
| `CONTEXT_LENGTH` | `1048576`, maximum per-request context |
| `MAX_RUNNING_REQUESTS` | `16`, supported range 1–16 |
| `MEM_FRACTION` | `0.85`, validated on the stated hardware |
| `MAX_NEW_TOKENS_LIMIT` | `393216`, cap when a client omits `max_tokens` |
| `ENABLE_HICACHE` / `HICACHE_RATIO` | `1` / `2`; use `ENABLE_HICACHE=0` to disable the host cache |
| `NCCL_P2P_LEVEL` | `SYS`; requires working peer access on the selected GPUs |

The fixed profile is TP4/EP2 with `moe-a2a-backend=none`. It uses FP8 E4M3 KV, `dsv4` attention, FlashInfer MXFP4 MoE and CUTLASS FP8 GEMM; prefill chunk and token limits are both 8192. Decoder CUDA Graphs cover up to 16 requests. Encoder/decoder SWA bounded replay, HiCache, metrics and cache reporting are enabled.

DSPARK uses block size 5, automatic SPS profiling, compact ragged verification, sub-width tiers, graph-tier alignment and survival EMA 0.15. SPS sampling uses temperature/top_p 1.0. SM120 MXFP8 skinny GEMM, BF16 skinny LM head, WO-A FP8 copy and PCIe-IPC all-reduce are explicit. Engram host tables are sharded per rank, taking about 189 GiB of host memory in total. Compilation is limited to four jobs and caches use persistent TorchInductor, Triton and TileLang paths.

The static memory fraction 0.80 failed to leave a KV pool after loading this model and its draft weights on the qualified four-card host; 0.85 passed. Do not interpret the static fraction as a cap on every CUDA allocation or assume a smaller number is always viable. Plan host memory for Engram, HiCache, runtime and other services; two model replicas duplicate the Engram allocation.

## API and cache behavior

Use the OpenAI-compatible `/v1/chat/completions` endpoint. Defaults are `sampling-defaults=openai`, temperature 1.0 and top_p 1.0; explicit request values take precedence. The reasoning parser is `deepseek-v41` and the tool parser is `deepseekv41`. Reasoning counts against `max_tokens`; allocate enough output budget for the final answer. Image input supports at most 128 images per request.

### Opt-in coding requests

Use [examples/code-request-profile.json](examples/code-request-profile.json) for an explicit coding request: temperature 1, reasoning_effort `low`, top_p 1 and max_tokens 12,288. This is a request preset, not a server default. The reasoning effort is a soft model instruction, not a hard reservation of tokens for final code.

Save a request such as this as `request.json`:

```json
{
  "model": "deepseek-v4.1-flash-1017",
  "messages": [{"role": "user", "content": "Implement Python valid_brackets(text) for matching ()[]{} while ignoring other characters. Return only Python code."}]
}
```

```bash
INFERENCE_BASE_URL=http://127.0.0.1:18001/v1 python3 examples/code-request.py --request request.json
```

Set `INFERENCE_API_KEY` in the process environment if authentication is enabled. Explicit request fields override the profile, including temperature 0. The example accepts one nonstream completion and prints final content only. A length finish, abnormal finish or empty final body is an error; no partial output is accepted, reasoning is not substituted, and there is no automatic retry. Validate generated code before using it.

The profile was selected from eight small coding tasks repeated five times in each DSPARK mode. It is not a guarantee for harder tasks or immunity from token exhaustion. Tool argument parsing now preserves literal thinking markers inside V4.1 tool bodies, but the model/decoder can still generate the wrong string. Do not treat parser fidelity as proof that a tool argument is semantically correct.

The coding profile is not a literal-tool preservation policy. A separate 100-request literal diagnostic per DSPARK mode using `reasoning_effort=low` passed only 17 requests with DSPARK disabled and 19 enabled. Keep exact-preservation requirements explicit in the application; this repository does not silently reconstruct arguments from prompts or enable retries that replace the original result.

### Context and cache behavior

The 1M limit is per request; all requests share the device KV pool. The candidate workload comparison included inputs up to 160,054 tokens and concurrency 1/4, not full 1M stress or a capacity benchmark. See the [validation record](compat/validation.json) for the exact artifact, configuration and limitations.

For the earlier `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3` artifact, the empty-cache first start took 2343 seconds on the qualified host, including compilation, CUDA Graph capture and automatic SPS profiling. Later starts can reuse compatible compilation artifacts and SPS tables, but still load weights and capture process-local CUDA Graphs. No timed warm-restart measurement has been made. Preserve the cache bind mount; a different image, GPU architecture or key configuration may invalidate entries. On a new host, measure SPS again rather than assuming identical GPU models imply identical communication costs.

`examples/serve-deepseek-v41-flash-tp4.sh` is retained as the upstream baseline recipe and defaults to `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3`. Use the compatibility launcher above for this fork.
