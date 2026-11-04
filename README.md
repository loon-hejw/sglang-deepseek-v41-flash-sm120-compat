# DeepSeek-V4.1-Flash TP4 on SM120

A compatibility fork of [ormandj/sglang-deepseek-v41-flash-sm120](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120), with applicable correctness fixes from the [0731 project](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) and a validated four-GPU serving profile. [中文说明](README.zh-CN.md) · [Run guide](RUN.md) · [Test comparisons](TESTING.md) · [Validation record](compat/validation.json).

## Acknowledgments and provenance

This work builds on two projects. We thank their maintainers and contributors for sharing the implementations, deployment recipes and validation work that made this fork possible:

- **[ormandj/sglang-deepseek-v41-flash-sm120](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120)** provides the primary V4.1/1017 baseline: the pinned source and build stack, SM120 optimization patch series, TP4/EP2 serving recipe, DSPARK/SPS integration and documentation. This fork starts from commit [`11c525d`](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120/commit/11c525d0234dc25e167e4d6bb0fe9fe757762f5f) and preserves its Git history.
- **[Ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000)** provides the 0731 four-GPU deployment experience and correctness patches used in the compatibility migration, including work by the SGLang contributors cited there. We evaluated its patch set at commit [`79bbc00`](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000/commit/79bbc001db042fc85dc2b15e4175210eaec3e78e); the [migration record](compat/migration.json) accounts for all 33 patches, with 7 ported, 22 already equivalent and 4 not applicable.

This fork's contributions are the V4.1 compatibility adaptations, regression tests, build integration fixes, configurable TP4 launcher and validation on the documented hardware. Credit for the underlying framework, algorithms, kernels and original patches belongs to their respective authors. Patch authorship and upstream pull-request links are retained below and in the [original 0731 patches](compat/upstream-0731/patches/). See [NOTICE.md](NOTICE.md), the [Apache-2.0 license](LICENSE) and the preserved Ombori [NOTICE](compat/upstream-0731/NOTICE) and [LICENSE](compat/upstream-0731/LICENSE) for attribution and license details.

## Published scope

This repository publishes source, configuration, validation summaries and the Linux amd64 candidate `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`. **This is a candidate, not a stable fork release; end-to-end literal generation remains unreliable.** Its registry manifest digest is `sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92`; see the [publication receipt](compat/publication.json). The upstream stable image `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` does not include this fork's compatibility fixes.

| Item | Profile |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Hardware | Four 96 GB RTX PRO 6000 Blackwell GPUs (SM120), working PCIe P2P, Linux amd64 |
| Parallelism | TP4 / attention DP1 / EP2; no MoE all-to-all backend |
| Speculation | DSPARK block 5, automatic SPS profiling, compact ragged verification |
| Context / concurrency | 1,048,576 tokens per request / at most 16 running requests |
| KV / static memory fraction | FP8 E4M3 / 0.85 |
| Host memory | Engram tables use about 189 GiB plus HiCache and runtime allocations; the qualified host has 503 GiB RAM |
| Storage | About 480 GiB for the checkpoint, additional image/build space, and a persistent compilation cache |
| API | OpenAI chat completions; default model `deepseek-v4.1-flash-1017`, host port `18001` |

For the retained shared-host configuration, set `HOST_PROFILE=shared` and an explicit `NUMA_MEMORY_NODE` matching the selected GPUs. This applies a 224 GiB memory cap, no swap, a 16 CPU quota, process-local THP disable and HiCache ratio 0.1 while keeping TP4 / attention DP1 / EP2. The standalone launcher does not reserve cluster GPUs or continuously supervise host resources. See the [shared-host instructions](RUN.md#shared-host-profile) and the [preliminary topology comparison](compat/topology-comparison-20261103.json). That experiment completed 204 quality requests and 18 of 48 performance batches before being deliberately ended; it does not qualify the unfinished workloads or a stable release.

## Quick start

Use a Linux amd64 Docker host meeting the hardware requirements above. The GHCR package is currently private: sign in with `docker login ghcr.io` using an account with package read access before pulling. Anonymous pull access has not been enabled. Pull the candidate and download the pinned model before starting the service. To build `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` yourself, follow [BUILDING.md](BUILDING.md) and set `IMAGE` to that local image instead.

```bash
git clone https://github.com/loon-hejw/sglang-deepseek-v41-flash-sm120-compat.git
cd sglang-deepseek-v41-flash-sm120-compat
export IMAGE=ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4
docker pull --platform linux/amd64 "$IMAGE"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /srv/models/DeepSeek-V4.1-Flash-1017

export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-compat-v0.4.4-rc.4-tp4
export GPU_DEVICES=0,1,2,3
# Set API_KEY in your environment to require bearer authentication.
bash examples/serve-compat-tp4.sh --dry-run
bash examples/serve-compat-tp4.sh
```

The launcher selects only the four requested devices, mounts the model read-only and persists caches. It starts a detached container with restart policy `no` and retains it after exit. The dry run does not create directories or start Docker and redacts the API key. See [RUN.md](RUN.md) for health checks, lifecycle commands and configurable settings.

## Validation and limits

The separately built `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` was tested on 2026-11-03. Its local OCI index identity is `sha256:bb27bdd8172e86db037840528b971fa5e2ec8d055236f49a3930d73be4ac58e7`, from source commit `b808761fbef06cec27ce7b3afc55c372cefa78c1`. Publication reused its Linux amd64 runtime manifest, configuration and all 79 layers byte for byte, without rebuilding. The registry digest identifies that platform manifest; the local index additionally contained a build attestation omitted from publication. A local rebuild is a different artifact and needs its own validation.

- Installed-image CPU: 1,198 cases and 691 subtests passed; one CUDA-only compact-verify test was skipped.
- Native CUDA: eight installed Triton Markov combine / CUDA Graph replay cases passed, with no skips. This is a separate suite from the skipped CPU-build case.
- The same four RTX PRO 6000 Blackwell GPUs ran three sequential workload modes. `rc3_on` uses `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3` with DSPARK; `rc4_on` and `rc4_off` use `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` with DSPARK enabled and disabled respectively. The dependency version lists were identical.

| Mode | Quality suite | Literal-tool suite |
| --- | --- | --- |
| `rc3_on` | 99/102 | 0/100 |
| `rc4_on` | 100/102 | 6/100 |
| `rc4_off` | 100/102 | 3/100 |

The 51-question quality set ran at temperatures 0 and 1. Literal-tool checks used both streaming modes, concurrency 1 and 4, and five rounds per combination. Correctness was scored against fixed answers; failed quality requests were repeated separately without replacing primary scores. Candidate API tool parameters matched the directly decoded raw tool arguments in both DSPARK modes. End-to-end literal generation still fails, so this candidate is not evidence that the complete problem is solved. Publication did not replace the production service.

A subsequent focused diagnostic run on the same candidate used 442 requests, with private observation hooks outside the image. Twenty captured prompts matched the checkpoint's official encoder token for token. At reproduced failure positions, the target model favored `</think>` over the required `<think>` before grammar masking, and the mask left both scores unchanged; native generation without tool grammar reproduced the error. The original literal matrix passed 5/100 with DSPARK disabled and 6/100 enabled. An explicit `reasoning_effort=low` diagnostic passed only 17/100 and 19/100 respectively, so it does not guarantee literal preservation. No additional runtime defect was established, and no new patch, image or default change was made. These findings describe the tested checkpoint and numerical configuration, not every backend or precision.

The [opt-in coding profile](examples/code-request-profile.json) uses temperature 1, reasoning effort `low`, top_p 1 and max_tokens 12,288. On eight small coding tasks repeated five times at concurrency 1, it passed 40/40 requests with DSPARK and 40/40 without it. The [example client](examples/code-request.py) preserves explicit request overrides and rejects truncated or empty final answers; it never promotes reasoning to code or retries automatically. See [RUN.md](RUN.md#opt-in-coding-requests).

These are **not a throughput benchmark or proof of complete sampling equivalence**. API seed values were recorded, but deterministic inference was disabled and server seeds differed; these are repeated stochastic observations, not paired deterministic trials. The shared-host test used a 224 GiB memory limit, 16 CPU quota, process-local THP disable and HiCache ratio 0.1, with inputs up to 160,054 tokens. It does not qualify the full 1M limit, TP8, broad coding ability or production throughput. Full conditions, per-profile token accounting including reasoning, prior serving qualification and remaining checks are in the [validation record](compat/validation.json). Upstream performance history remains attributed in [CHANGELOG.md](CHANGELOG.md).

## Source and patches

The image is built from four upstream commits, recorded in [`stack.lock.json`](stack.lock.json) with their trees and every patch's hash: SGLang `main`, FlashInfer `main`, sgl-project DeepGEMM `release/v0.2.3` and Triton `release/3.8.x`, on the `lmsysorg/sglang` CUDA 13 nightly, which supplies CUDA, PyTorch and runtime libraries only. SGLang, FlashInfer, DeepGEMM and Triton are rebuilt from source. [`scripts/verify-patches.sh`](scripts/verify-patches.sh) fetches the official commits, applies the series and checks every resulting tree.

SGLang series, applied in order to the pinned SGLang commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| 01 | `01-instanttensor-host-resident.patch` | [sgl-project/sglang#42919](https://github.com/sgl-project/sglang/pull/42919) | Host-resident weights (Engram tables) stay off the InstantTensor GPU stream |
| 02 | `02-engram-positioned-reads.patch` | [sgl-project/sglang#42920](https://github.com/sgl-project/sglang/pull/42920), on 01 | Parallel positioned reads fill each rank's host Engram shard |
| 03 | `03-sm12x-candidate-indexer.patch` | [sgl-project/sglang#42371](https://github.com/sgl-project/sglang/pull/42371) | V4.1 candidate indexer on SM12x through the paged decode path |
| 04 | `04-dsv41-ragged-verify-engram.patch` | [sgl-project/sglang#42718](https://github.com/sgl-project/sglang/pull/42718) | Packed compact-verify tokens mapped to their requests (replaces #40567) |
| 05 | `05-dspark-forced-budget-verify-all.patch` | [sgl-project/sglang#42711](https://github.com/sgl-project/sglang/pull/42711) | Correct SPS cost tables when profiling without a table |
| 06 | `06-flash-mla-sm120-split-buffers.patch` | [sgl-project/sglang#42714](https://github.com/sgl-project/sglang/pull/42714) | SM120 FlashMLA page-split buffers allocated once per size (captured graphs never write freed memory) |
| 07 | `07-dspark-confidence-relay-sync.patch` | [sgl-project/sglang#42717](https://github.com/sgl-project/sglang/pull/42717) | Every TP rank resolves the same relayed DSpark confidence |
| 08 | `08-forced-uniform-capture-token-key.patch` | [sgl-project/sglang#42713](https://github.com/sgl-project/sglang/pull/42713) | Forced-uniform compact captures keyed by token count, with a full-width layout |
| 09 | `09-dspark-sts-accept-labels.patch` | [sgl-project/sglang#42712](https://github.com/sgl-project/sglang/pull/42712) | STS calibration labels from the actual acceptance, unverified positions masked |
| 10 | `10-dsv4-fp4-indexer-exp2-free.patch` | [sgl-project/sglang#42861](https://github.com/sgl-project/sglang/pull/42861) | FP4 index keys decoded without `exp2` (bit-identical) |
| 11 | `11-sm120-mxfp8-skinny.patch` | [sgl-project/sglang#42865](https://github.com/sgl-project/sglang/pull/42865) | Split-K MXFP8 GEMM for tuned decode shapes at up to 16 rows |
| 12 | `12-mhc-mix-vec-reduce.patch` | [sgl-project/sglang#42864](https://github.com/sgl-project/sglang/pull/42864), ported | Vectorized mHC reduce; 160 K slices at SM120 decode |
| 13 | `13-mhc-prefill-stats-overlap.patch` | [sgl-project/sglang#42985](https://github.com/sgl-project/sglang/pull/42985), ported | Wide prefill overlaps the mHC mix stats with the TP all-reduce again (#42245 removed it; measured prefill regression) |
| 14 | `14-pcie-ipc-gather.patch` | [sgl-project/sglang#42863](https://github.com/sgl-project/sglang/pull/42863) | TP4 logits and DSpark vocab gathers on FlashInfer PCIe-IPC |
| 15 | `15-dsv41-request-window-history-check.patch` | [sgl-project/sglang#42921](https://github.com/sgl-project/sglang/pull/42921), rebased | Request-window history checked once per layout with an async device assert (no prefill host sync) |
| 16 | `16-bf16-skinny-lm-head.patch` | [sgl-project/sglang#42867](https://github.com/sgl-project/sglang/pull/42867) | Opt-in weight-streaming BF16 LM head GEMM at <= 16 rows (`SGLANG_ENABLE_BF16_SKINNY_LM_HEAD`) |
| 17 | `17-serving-coverage-warmup.patch` | [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541) | `--warmups serving_coverage`: real requests through every serving path, including decodes at every power-of-two batch size up to the running-request limit |
| 18 | `18-dspark-sps-autotune.patch` | not yet upstream | `--speculative-dspark-sps-autotune off\|auto\|force`: fit the DSpark SPS cost table at startup from real decode steps per batch size and verify tier plus a per-verify-token context term, cached per configuration |
| 19 | `19-ragged-sub-width-tiers.patch` | [sgl-project/sglang#43011](https://github.com/sgl-project/sglang/pull/43011), rebased (closed upstream) | Opt-in ragged-verify tiers below one request's width (`SGLANG_RAGGED_VERIFY_SUB_WIDTH_TIERS`) |
| 20 | `20-dspark-verify-pad-last-row.patch` | [sgl-project/sglang#43374](https://github.com/sgl-project/sglang/pull/43374) | No top-p renormalization on compact-verify padding rows |
| 21 | `21-compact-verify-committed-seq-lens.patch` | [sgl-project/sglang#43225](https://github.com/sgl-project/sglang/pull/43225) | Compact verify passes committed CPU lengths to self-adding backends |
| 22 | `22-dsv41-candidate-scorer.patch` | [sgl-project/sglang#43401](https://github.com/sgl-project/sglang/pull/43401) | Paged decode consumer layers score only their candidate blocks |
| 23 | `23-dsv41-wo-a-fp8-weight.patch` | [sgl-project/sglang#43008](https://github.com/sgl-project/sglang/pull/43008) | SM120 decode/verify WO-A reads an exact FP8 weight copy (`SGLANG_DSV41_WO_A_FP8_COPY`, default on) |
| 24 | `24-encoder-swa-replay-multimodal.patch` | [sgl-project/sglang#43241](https://github.com/sgl-project/sglang/pull/43241) | Under encoder SWA bounded replay, the server warms up the image path on CUDA, where replay carries image inputs |
| 25 | `25-dsv41-indexer-prefill-tile-budget.patch` | [sgl-project/sglang#43376](https://github.com/sgl-project/sglang/pull/43376) | Prefill indexer score tiles sized from free memory, scratch counted, no padded copy (OOM fix) |
| 26 | `26-dspark-smoothed-budget-sizing.patch` | not yet upstream | Host verify-budget sizing from a per-request running average of lagged survival (opt-in) |
| 27 | `27-dspark-read-accept-no-alias.patch` | [sgl-project/sglang#42812](https://github.com/sgl-project/sglang/pull/42812), amended for #43790 | DSpark folded accept results are copied before the next verify replay, so TP ranks commit the same tokens under overlap scheduling with HiCache; the in-graph accept still writes the static buffers |
| 28 | `28-chat-usage-completion-tokens-details.patch` | [sgl-project/sglang#45131](https://github.com/sgl-project/sglang/pull/45131) | Usage also reports reasoning tokens at `usage.completion_tokens_details.reasoning_tokens`, where OpenAI clients and gateways read them |
| 29 | `29-openai-error-envelope.patch` | [sgl-project/sglang#45130](https://github.com/sgl-project/sglang/pull/45130) | OpenAI routes return errors in OpenAI's error format |
| 30 | `30-openai-abort-finish-reason.patch` | [sgl-project/sglang#45132](https://github.com/sgl-project/sglang/pull/45132) | Aborted requests are reported as an error or a stop, never `finish_reason: "abort"` |
| 31 | `31-openai-stream-exception-frame.patch` | [sgl-project/sglang#45133](https://github.com/sgl-project/sglang/pull/45133) | Failed chat and completion streams end with an error frame and `[DONE]` |
| 32 | `32-dsv41-effort-minimal-medium.patch` | [sgl-project/sglang#45134](https://github.com/sgl-project/sglang/pull/45134) | DeepSeek-V4.1 supports `reasoning_effort` `minimal` and `medium` |
| 33 | `33-compat-verify-penalties.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Restore additive and repetition penalties across DFLASH/DSPARK verify and skip gates |
| 34 | `34-compat-committed-penalty-state.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Track all committed speculative output tokens exactly once across batch reuse/filter/merge |
| 35 | `35-compat-tool-final-flush.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Flush all completed tool arguments once, including complete final name chunks |
| 36 | `36-compat-reasoning-prefix.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Hold trailing partial reasoning and tool markers in normal text |
| 37 | `37-compat-request-routing-key.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Preserve request routing keys when slicing batched generation requests |
| 38 | `38-compat-deepgemm-warmup-budget.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Skip DeepGEMM warmup when no supported shape fits available memory |
| 39 | `39-compat-draft-sentinel.patch` | [Ombori 0731](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) compatibility port; see [migration audit](compat/migration.json) | Keep degenerate Markov greedy draft rows within vocabulary bounds |
| 40 | `40-compat-cpu-regressions.patch` | Local regression tests for migrated fixes; see [migration audit](compat/migration.json) | CPU regression coverage for migrated behavior and retained V4.1 DSML parsing |
| 41 | `41-v41-tool-literal-reasoning.patch` | Local V4.1 parser regression fix | Preserve literal think markers inside tool parameters across streaming boundaries and non-streaming parsing |

FlashInfer series, applied to the pinned FlashInfer commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| fi-01 | `fi-01-sm120-mxfp8-default-tactic.patch` | [flashinfer-ai/flashinfer#6162](https://github.com/flashinfer-ai/flashinfer/pull/6162) | SM120 MXFP8 GEMM calls without a tuned tactic use the swap-AB 128x32 tile |

Triton series, applied to the pinned Triton commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| tr-01 | `tr-01-release-3.8-dot-operand-remat.patch` | [triton-lang/triton#11940](https://github.com/triton-lang/triton/pull/11940), backported to `release/3.8.x` | Restore dot-operand rematerialization reuse: Triton 3.8.0 (issue #11526) sends tf32x3 dot operands through extra shared-memory round trips |

Patches marked "not yet upstream" are inherited from the ormandj baseline; the label records their status in that baseline. [BUILDING.md](BUILDING.md) describes the build and release workflows.

## License

The repository's own files are licensed under the [Apache License 2.0](LICENSE). Dependencies and derived patches retain their applicable upstream licenses and notices; the DeepSeek-V4.1-Flash weights are distributed by DeepSeek under the MIT License. See [NOTICE.md](NOTICE.md).
