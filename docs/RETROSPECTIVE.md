# From SGLang compatibility work to DeepSeek 1017 validation

[中文](RETROSPECTIVE.zh-CN.md) · [Overview](../README.md) · [Engineering handbook](ENGINEERING.md) · [Test overview](../TESTING.md)

Compiled on 2026-11-05 to explain the problems, evidence, changes and remaining limits. Operational instructions belong to the handbook; per-experiment data belongs to TESTING. This work changed inference software and deployment policies, not the pinned model weights through training or fine-tuning.

## 1. Where the work started

[ormandj's V4.1/1017 project](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120) already supplied pinned sources, the build stack, SM120 patches, DSPARK/SPS and a TP4/EP2 recipe. [Ombori's 0731 project](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) supplied earlier deployment experience and correctness patches. This fork added compatibility assessment, applicable ports, regressions, build integration, deployment and validation. Upstream algorithms and kernels remain attributed to their authors in [NOTICE](../NOTICE.md), the original patches and PR references.

The objective was not wholesale transplantation of old code and parameters. New SGLang fields, layouts and interfaces already covered some fixes and made others inapplicable. All 33 decisions are accounted for: 7 ported, 22 already equivalent and 4 not applicable, with evidence in the [migration record](../compat/migration.json).

## 2. Identify what the current implementation still lacks

| Problem | Change and tradeoff | Evidence |
| --- | --- | --- |
| A speculative step can commit several tokens, making one-token penalty updates inaccurate | Track committed output through reuse/filter/merge and apply repetition scaling to actual logits signs | [Penalties](../patches/33-compat-verify-penalties.patch), [state](../patches/34-compat-committed-penalty-state.patch) |
| Final chunks can carry complete names and arguments; multiple tools need complete finalization | Flush every completed call once, including final name chunks | [Tool completion](../patches/35-compat-tool-final-flush.patch) |
| SSE boundaries can split markers in ordinary text | Add ordinary-text prefix holdback using existing final-flush behavior | [Marker prefixes](../patches/36-compat-reasoning-prefix.patch) |
| Batch slicing loses a routing field | Preserve routing_key rather than introducing a new routing policy | [Request field](../patches/37-compat-request-routing-key.patch) |
| Even the smallest warmup shape can exceed memory; degenerate scores can leave invalid ids | Choose supported shapes within budget and keep draft ids inside vocabulary bounds | [Warmup](../patches/38-compat-deepgemm-warmup-budget.patch), [sentinel](../patches/39-compat-draft-sentinel.patch) |

These primarily establish correctness boundaries. Passing regressions does not demonstrate throughput improvement, and all 41 current SGLang patches cannot be counted as new optimizations by this fork.

## 3. Build reliability is part of delivery

FlashInfer builds could encounter nested installer-lock waiting. The integration installs pinned compiler requirements first, then enables `FLASHINFER_BUILD_NO_PIP=1`. Triton's intended test-build option must reach CMake through its supported `TRITON_APPEND_CMAKE_ARGS` hook rather than an ineffective environment setting. Repetitive cubin output is filtered while retaining `pipefail`, so real compiler failures remain failures.

Mirror bases and source caches help restricted networks only while manifest/config/layer checks and commit/patch/effective-tree checks remain mandatory. Publication reused the tested runtime artifact and verified all 79 layers rather than testing one image and publishing a rebuild. The local index contained a build attestation while publication used the runtime platform manifest; the [receipt](../compat/publication.json) explains their different digests.

Qualification remained layered: 279 local source-body CPU cases; 1198 installed-image cases and 691 subtests with one CUDA-only skip; eight separate native CUDA cases. These cannot be summed into a model-quality score. A complete network source-verification command was historically interrupted by networking; separately verified source trees and base identity retain separate evidence, not a rewritten claim that every step passed in one run.

## 4. Tool arguments: parsing fidelity versus generation correctness

When a tool argument had to preserve `<think>literal</think>`, observed responses damaged the literal and one modified-framework response returned no tool call. Investigation followed input encoding, raw tokens, reasoning parsing, tool parsing and API output rather than treating HTTP 200 as success.

The candidate's [parser fix](../patches/41-v41-tool-literal-reasoning.patch) distinguishes tool-body literals from structural reasoning markers and checks chunk-invariant argument handling. Candidate checks observed zero parser-caused argument changes, but end-to-end literal pass rates stayed low. The supported conclusion therefore concerns parser fidelity only.

A subsequent 442-request diagnosis found 20 captured inputs token-identical to the official checkpoint encoder. At reproduced positions, the target model already preferred the wrong closing marker before grammar masking; the correct marker remained allowed, and native generation without tool grammar reproduced the error. Neither disabling DSPARK nor low reasoning guaranteed preservation. Evidence did not justify another transparent framework patch or reconstructing arguments from prompts to hide the original output. `literal_generation_diagnosis` in the [validation record](../compat/validation.json) retains counts and numerical/backend limits.

## 5. Coding: compare request policies first

A temperature-zero coding request exhausted 12,288 output tokens without final code. Reasoning consumes the same budget, so the symptom alone does not establish lost transmission or parsing.

The experiment fixed eight small tasks, five repeats, concurrency 1, top_p 1 and a 12,288-token budget, testing temperature/reasoning combinations with DSPARK both enabled and disabled. Temperature 1 / low passed 40/40 in each mode; temperature 0 / high still truncated, and no-thinking requests also produced coding errors. TESTING contains all four profiles.

The result is an [opt-in preset](../examples/code-request-profile.json) and [example client](../examples/code-request.py), not changed global defaults. Explicit overrides remain effective; truncation and empty final content fail, reasoning is not substituted for code, and no automatic retry replaces the failure. This is a policy choice for small coding tasks, not a guarantee of broad coding ability or literal preservation.

## 6. Test parallelism assumptions on the same model

A functioning TP4 / attention DP4 / EP4 deployment for 0731 does not imply faster 1017 inference. The later experiment compared the same candidate image and 1017 weights on four GPUs per instance. The original profile retained TP4 / attention DP1 / EP2, DSPARK block 5, SPS auto and memory fraction 0.85. The experimental side enabled a DP LM head and needed fraction 0.90 to allocate KV.

In completed short-input Chinese-explanation measurements, the original profile had higher aggregate throughput at C1/C4/C8. All 204 main quality requests completed. The operator ended the experiment after 18/48 speed batches; C16, coding speed and additional low-thinking coding checks remained incomplete. Exact values, three-repeat medians, TTFT and cache conditions are in the [topology record](../compat/topology-comparison-20261103.json).

DP, EP, LM head, memory fraction, physical GPU/NUMA placement and derived scheduler parameters changed together. The difference cannot be assigned to DP alone. Retaining the original profile identifies a better observed configuration for completed workloads, not an optimum across all concurrency, long contexts or eight-GPU topologies.

The independent 0731/1017 comparison also changed model, framework, DSPARK block size and resource limits. It describes two deployments, not an isolated parameter improvement. The [deployment record](../compat/deployment-comparisons-20261103.json) stays separate from the same-model experiment; speeds and quality scores are not mixed across runs.

## 7. Conclusions and limits for further work

- Delivered: compatibility ports, build integration, V4.1 tool-body parsing repair, opt-in coding policy, shared-host launcher profile and qualification/publication records.
- Known limits: generated literals can still be wrong; difficult code can exhaust its budget; supplied seeds do not establish strict determinism; small synthetic results do not establish production throughput or broad quality gains.
- Unfinished: matched 1M/256K workloads, TP8, same-model DSPARK block/SPS sweeps, full 1M stress, long-context throughput and sustained load.
- Maintenance rule: identify weights, source, image and runtime configuration separately; check cache behavior and actual requests separately; update conclusions from new evidence without relabeling unfinished work as verified.

The current candidate is `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`. Further changes follow [build and release rules](../BUILDING.md); operation follows the handbook. This retrospective summarizes recorded engineering work without publishing private addresses, credentials or raw conversations.
