# 1017 tests and parameter comparisons

[中文完整说明](TESTING.zh-CN.md) · [README](README.md) · [Run guide](RUN.md)

This record covers experiments through 2026-11-03 and was compiled on 2026-11-04. The retained configuration is TP4 / attention DP1 / EP2 using `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`. It was selected from tested configurations, not an exhaustive optimum or stable-release qualification. Publishing these records changes no image, service or request defaults.

## Evidence and hardware

| Record | Coverage |
| --- | --- |
| [Validation](compat/validation.json) | Build, CPU/CUDA checks, DSPARK modes, coding profiles and literal-generation diagnosis |
| [Topology comparison](compat/topology-comparison-20261103.json) | Exact configurations, 204 quality requests, 18 completed speed batches and unfinished work |
| [Deployment comparisons](compat/deployment-comparisons-20261103.json) | Original-framework control, 0731 comparison, artifact identities, category scores and 24 speed batches |
| [Publication receipt](compat/publication.json) | Relationship between measured and published artifacts |

Each instance used four NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs, PCIe P2P and no NVLink. The host had two Intel Xeon Platinum 8480+ CPUs, 503 GiB RAM, driver 610.43.02 and 600 W GPU power limits. Side-by-side experiments used separate four-GPU groups and corresponding NUMA memory nodes on one eight-GPU host.

The model was `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Candidate source was `b808761fbef06cec27ce7b3afc55c372cefa78c1`; the published Linux amd64 manifest is `sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92`. Quality experiments measured the separately built `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`. Publication reused its runtime manifest, configuration and every layer without rebuilding. Its local OCI index also contained a build attestation, so the index digest differs from the published platform manifest; see the receipt.

Quality checks cover reasoning, coding, instructions, structured output, tools, long input and sampling penalties. After excluding one ambiguous question symmetrically, each main comparison used 51 valid cases at temperatures 0/1, or 102 requests per configuration. Truncation, empty final answers and wrong formats/types can fail a request. Failure repeats never replace primary scores. Each experiment is a separate run; do not pool its pass rate with another run.

API seeds were recorded, but deterministic inference was disabled. Identical seeds do not establish identical random streams. These small synthetic sets are not official benchmarks, general quality rankings or proof of sampling-distribution equivalence.

## Framework and DSPARK

The first control compared `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` with `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`, sequentially on the same four GPUs with identical weights and explicit server commands. Both used a 224 GiB host memory limit, 16 CPU quota, process-local THP disable and HiCache ratio 0.1. Scores were 98/102 and 99/102. This checked common parameters, not every historical deployment default, and did not establish a controlled throughput result. Identities and counts are in the deployment record.

The subsequent parser/DSPARK experiment ran three sequential modes on the same four GPUs:

| Mode | Artifact | Main quality | Literal-tool matrix |
| --- | --- | --- | --- |
| `rc3_on` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`, DSPARK on | 99/102 | 0/100 |
| `rc4_on` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`, DSPARK on | 100/102 | 6/100 |
| `rc4_off` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`, DSPARK off | 100/102 | 3/100 |

The literal matrix covered streaming/non-streaming, temperatures 0/1, concurrency 1/4 and five rounds per combination. No parser-caused argument changes were observed for the new candidate, but generated `<think>` / `</think>` text remained wrong in many responses. Disabling DSPARK did not eliminate the failure. Quality-suite timings include reasoning tokens and are diagnostic, not production throughput benchmarks.

## Coding request profiles

Each profile used eight small coding tasks repeated five times at concurrency 1, `top_p=1` and `max_tokens=12288`. Passing includes response completeness and code unit tests. Both DSPARK modes together account for 320 requests.

| Temperature | Reasoning effort | DSPARK on: passed / truncated | DSPARK off: passed / truncated |
| --- | --- | --- | --- |
| 0 | high | 37/40; 3 truncated | 38/40; 2 truncated |
| 1 | high | 39/40; 1 truncated | 40/40; 0 truncated |
| 1 | low | 40/40; 0 truncated | 40/40; 0 truncated |
| 1 | none | 38/40; 0 truncated | 39/40; 0 truncated |

The predefined rule chose the [temperature 1 / low preset](examples/code-request-profile.json). It is opt-in per request, not a global default. Low reasoning is a model instruction, not a hard final-code token reservation. All 80 low-profile requests passed, which does not establish correctness on larger workloads. No-thinking failures also show a possible correctness tradeoff.

## Further literal diagnosis

A separate 442-request diagnostic run on the same artifact passed 5/100 with DSPARK off and 6/100 with it on for the original matrix. Explicit low reasoning passed 17/100 and 19/100 respectively. These are new observations, not replacements for the earlier 3/100 and 6/100.

Twenty inputs matched the checkpoint's official encoder token for token. At reproduced positions, the target model favored the wrong closing marker before grammar masking; the mask allowed the correct marker, and native generation without tool grammar reproduced the error. Neither low reasoning nor disabling DSPARK guarantees preservation. Evidence did not support an additional transparent runtime repair. See `literal_generation_diagnosis` in the validation record for counts and numerical/backend limits.

## TP4 attention DP / EP

Both sides used the same published candidate and 1017 weights, four GPUs each. DP below means attention DP; ordinary `dp_size` remained 1 on the experimental side.

| Parameter | Retained | Experimental |
| --- | --- | --- |
| TP / attention DP / EP | 4 / 1 / 2 | 4 / 4 / 4 |
| DP LM head | Disabled | Enabled |
| Static memory fraction | 0.85 | 0.90 |
| DSPARK / SPS | Block 5 / auto | Block 5 / auto |
| Context / running requests | 1,048,576 / 16 | 1,048,576 / 16, four per DP rank |
| Effective chunked prefill | 8192 | 2048 after DP splitting |
| Scheduler conservativeness | 1.0 | Automatically adjusted to 0.3 |
| Cache / host limits | HiCache 0.1; 224 GiB; no swap; 16 CPUs; process-local THP disabled | Same |

Quality scores were 100/102 and 97/102. Each case ran once per temperature at temperatures 0/1, top_p 1 and concurrency 1 per side. Reasoning/coding used high thinking with a 12,288-token budget; tools used 4096, other tasks 2048. Maximum actual input was 160,054 tokens.

Completed speed tests used a Chinese explanation task with 40 input tokens, no thinking, temperature 1, top_p 1 and 1024 returned tokens per request. Each concurrency level had at least four warmup requests and three measured repetitions per side. Order alternated; repeated-prefix caches were not flushed. Throughput is total returned completion tokens / whole-batch wall time including TTFT. Observed reasoning-token counts were zero in these speed outputs.

| Concurrency | Retained token/s | Experimental token/s | Experimental change |
| --- | --- | --- | --- |
| 1 | 215.69 | 124.88 | -42.10% |
| 4 | 584.77 | 416.25 | -28.82% |
| 8 | 813.94 | 669.46 | -17.75% |

Values are medians of three repetitions. C4/C8 are aggregate rates, not per-request speeds or server decode-log rates. The operator ended the experiment after 18/48 planned batches and 78 timed requests. C16 was unfinished; coding speed tests and the additional low-thinking coding follow-up had not started.

The experimental side could not allocate KV at memory fraction 0.85 and started at 0.90. This was a startup feasibility check. DP, EP, LM head, memory fraction and GPU/NUMA placement differ together, so results cannot be attributed to DP alone. Retaining the original profile is supported only for completed workloads. Exact parameters and batches are in the topology record.

## 0731 versus 1017 deployments

0731 used TP4 / attention DP4 / EP4, DSPARK block 7, running-request limit 256 and decode graph limit 64, without the 1017 host CPU/memory caps. The 1017 side used the retained profile. The measured 0731 artifact is Docker image ID `sha256:2fceff11661eb240cb6c007bfbc505c95e549c76c1af437dd7839d0aea6cf6ef`, not attributed to an unverified upstream release.

Quality scores were 88/102 for 0731 and 98/102 for 1017. Each side ran 51 cases at temperatures 0/1, top_p 1 and concurrency 1. Reasoning/coding used high thinking with 12,288 output tokens, tools 4096 and other tasks 2048. Quality requests were paired concurrently. Category counts are in the deployment record.

Separate speed tests used no thinking, temperature 1, top_p 1 and 1024 output tokens per request. Explanation inputs had 40 tokens; the coding task requested a Python LRU cache with TTL, capacity limits, metrics, thread safety and unit tests, using 46 input tokens. Each workload/concurrency/model combination had three repetitions, totaling 24 batches and 60 requests. Four warmup requests were excluded; prefixes remained cached and servers were timed separately in alternating order. The table reports median aggregate completion tokens / batch wall time, including TTFT and reasoning tokens if any.

| Workload | Concurrency | 0731 token/s | 1017 token/s |
| --- | --- | --- | --- |
| Chinese explanation | 1 | 126.05 | 136.86 |
| Chinese explanation | 4 | 336.28 | 551.02 |
| Code generation | 1 | 188.64 | 252.31 |
| Code generation | 4 | 553.96 | 724.63 |

Models, frameworks, parallelism, DSPARK, caches and resource limits differ; this does not isolate any 1017 parameter or model-weight improvement. Tokenizers also limit cross-model token-rate comparisons. This and the topology experiment are independent runs; do not mix their speeds or quality scores.

## What has not been tuned

- Context limit 1M versus 256K under the same workload. Inputs of 20,054 / 40,054 / 80,054 / 160,054 tokens were checked with the limit kept at 1M.
- TP4 versus TP8. Two four-GPU instances do not constitute TP8.
- DSPARK block 5/7 or other values on the same 1017 model, or SPS auto versus manual policies in isolation.
- Independent performance sweeps of HiCache, FP8/BF16 KV, prefill, running-request limits, CPU/NUMA or THP.
- Full 1M input stress, long-context throughput, sustained load or broad real-world quality.
- The ended DP4 experiment's C16, coding speed and additional low-thinking code checks.

Frequency, presence and repetition penalties were checked for correctness, not tuned for throughput. Memory fractions 0.80/0.85 and DP4's 0.85/0.90 were startup feasibility checks, not isolated speed experiments. The retained side uses 0.85, experimental DP4 uses 0.90. HiCache's change from dedicated-host default 2 to shared-host 0.1 was resource allocation, without a separate throughput A/B.

## Reusing the profile

Follow the [shared-host run guide](RUN.md#shared-host-profile), choosing a NUMA memory node appropriate for the GPUs. The coding preset is a separate request option. New hardware, builds or workloads require their own validation.

Public records contain parameters, artifact identities, numerical summaries and evidence hashes, excluding private addresses, credentials, model weights and raw conversations. Summaries alone do not reproduce every request. The [API checker](scripts/validate-gpu.py), [CPU checks](scripts/test-compat.sh) and [native CUDA checks](compat/test_gpu_native.py) support independent validation, not automatic replay of all historical experiments. Attribution is preserved in [NOTICE.md](NOTICE.md).
