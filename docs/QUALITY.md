# Quality-first request policies

This experiment evaluates request policies for the existing `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4` image and the locked 1017 weights. It does not train weights or change the serving image. See [the test overview](../TESTING.md) for earlier measurements and known tool-literal limitations.

## Measured result: 2026-11-05

**The full policy bundle did not qualify and remains experimental.** Held-out quality improved from 388/700 (55.43%) to 511/700 (73.00%), or +17.57 percentage points, with a family-bootstrap 95% interval of +5.00 to +30.57 points. The quality, speed, legacy and code-control gates passed, but the nonstream literal-tool control declined from 2/60 to 1/60. Its parameters were identical in both arms, so this difference does not establish a parameter-caused regression; it still fails the predeclared non-regression rule. No failure was replaced by a retry and no gate was relaxed.

The run completed 2,979 scored observations and 140 speed batches. Evidence includes [the summary and provenance](../compat/quality-policy-20261105.json), [per-request records](../compat/quality-policy-20261105-results.json), [speed batches](../compat/quality-policy-20261105-speed.json) and [the experimental profile bundle](../examples/quality-policy-profiles.json). All scored API requests completed without protocol errors; model failures and truncation still count against quality. The production service retained its original container, start time and healthy state. The test instance was stopped and 0731 remained stopped.

Both arms used the same four RTX PRO 6000 Blackwell GPUs and serving instance, TP4 / attention DP1 / EP2, DSPARK block 5 / SPS auto, FP8 KV, a 1M context limit, 224 GiB host memory and 16 CPUs. Model revision was `1bae0e1088993b9b82de08f59de5be5a34e42170`; the runtime image digest was `sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92`. Actual input reached 262,132 tokens.

Each category below contains 20 held-out instances repeated five times per arm. All profiles use top_p 1. Reasoning, code and tool profiles were unchanged between arms, so their score differences are stochastic observations, not parameter improvements.

| Category | Candidate effort / temperature / budget | Baseline passed | Candidate passed | Candidate truncated |
| --- | --- | --- | --- | --- |
| Reasoning | high / 1 / 12,288 | 100/100 | 100/100 | 0 |
| Coding | low / 1 / 12,288 | 98/100 | 96/100 | 3 |
| Instructions | low / 1 / 2,048 | 46/100 | 93/100 | 4 |
| Structured output | low / 1 / 2,048 | 55/100 | 69/100 | 7 |
| Literal-tool stress | none / 1 / 4,096 | 0/100 | 3/100 | 7 |
| Long context | low / 1 / 2,048 | 44/100 | 98/100 | 1 |
| Penalty fixtures | low / 0.3 / 2,048 | 45/100 | 52/100 | 44 |

The strongest observed gains were in instructions and long-context tasks when changing from no thinking to low effort. Structured output also improved, but remained far from universally correct. Penalty fixtures had 44 candidate truncations and only 52% success; that profile is not a general penalty recommendation. Every held-out tool task contained reserved `<think>` text, so its very low pass rate describes this exact-preservation stress test, not ordinary tool calling. The literal-generation problem remains unresolved.

The older 51-case temperature 0/1 suite passed 98/102 versus 99/102, with no category decline. The separate low-effort code control passed 40/40 in both arms. The nonstream tool control passed 2/60 versus 1/60. All 279 framework CPU checks passed; these checks are separate from model quality.

### Latency and token rate

Full-response latency below includes failed and truncated requests, repeated-prefix caching and the mixed concurrency described below. It is not latency conditional on obtaining a correct answer. No hard full-response latency gate was imposed.

| Category | Median seconds, baseline → candidate | p95 seconds, baseline → candidate |
| --- | --- | --- |
| Reasoning | 6.60 → 6.49 | 27.04 → 28.49 |
| Coding | 10.74 → 10.94 | 35.49 → 26.90 |
| Instructions | 0.37 → 1.67 | 0.80 → 8.39 |
| Structured output | 0.66 → 3.08 | 1.14 → 6.29 |
| Literal-tool stress | 0.97 → 0.94 | 12.02 → 11.62 |
| Long context | 1.50 → 2.29 | 13.67 → 14.44 |
| Penalty fixtures | 0.57 → 4.01 | 0.90 → 14.35 |

The separate speed test reports median aggregate completion token/s, including reasoning, over five batches per arm/concurrency. All fourteen comparisons retained at least 92.74% of baseline, passing the 80% threshold. Extra reasoning increases the token numerator and amortizes fixed overhead, so the large instruction and long-context ratios are not equivalent to faster final answers.

| Category | C1 token/s, baseline → candidate | C4 token/s, baseline → candidate |
| --- | --- | --- |
| Reasoning | 303.81 → 314.61 | 722.10 → 714.93 |
| Coding | 264.06 → 286.09 | 627.67 → 582.13 |
| Instructions | 73.51 → 282.04 | 142.64 → 605.42 |
| Structured output | 277.32 → 340.38 | 733.70 → 762.42 |
| Literal-tool stress | 243.81 → 241.12 | 563.29 → 571.43 |
| Long context | 111.79 → 265.61 | 263.99 → 547.96 |
| Penalty fixtures | 77.28 → 251.85 | 166.35 → 587.89 |

### Harness corrections and limits

An initial API probe wrongly treated an incorrect arithmetic answer as startup failure. Before selection began, the gate was corrected to check protocol completion and effective thinking mode; wrong answers remain scored. Later, after 217 held-out observations, the run stopped at a saved checkpoint to assign unique per-invocation CPU grading containers. Identical code could otherwise share a container name under concurrency, although no such conflict failure had occurred. Four concurrent identical reference programs passed after the fix. The first 1,392 observations, including those 217 holdout rows, were inherited unchanged in a versioned continuation; source hashes, parent commit and record digests are published. No completed request was retried, and model requests, profiles and scoring rules did not change.

Three test-instance startup observations were 292.93, 282.84 and 282.94 seconds with an initially copied compilation cache. These are operational observations, not a controlled warm-start benchmark or a promise for other hosts. The synthetic tasks, fourteen held-out families, cached prefixes and single hardware configuration limit generalization. The measured gains support further task-specific evaluation, not a blanket quality claim or a global default change.

## What changes

Reasoning effort is a model instruction, not a hard allocation of reasoning tokens or a reservation for the final answer. The pinned encoder maps `low` to 50, `medium` to 62, `high` and `xhigh` to 75, and `max` to 100. When thinking is enabled without an effort override, its default strength is `high`. Thinking activation is separate: with `SGLANG_DEFAULT_THINKING` unset, omitting both effort and the thinking toggle uses chat mode. Supplying a non-none `reasoning_effort` enables thinking unless the caller explicitly disables it through `chat_template_kwargs`. Tokenization on the measured instance confirmed that omitted controls equal explicit no-thinking, while explicit high effort equals thinking=true/high. A larger effort can help a difficult task, but can also increase latency or exhaust the completion budget. `xhigh` is not a separate stronger setting in this build.

The primary test baseline uses temperature 1 and top_p 1. Reasoning requests use high effort; code uses the previously selected low-effort profile; other categories use no thinking. Budgets are 12,288 for reasoning/code, 4,096 for tools, and 2,048 elsewhere. These explicit evaluation requests do not establish what an arbitrary production client currently sends.

The selection stage compares low/medium/high/max for reasoning and code, then a 24,576-token budget for the two best profiles, followed by temperature 0.6 for the best combination. Other categories compare none/low/medium followed by temperatures 0.3 and 0.6 for the best combination. Every candidate keeps top_p at 1. Category-specific penalty fixtures remain identical across policies. The complete effective settings and request hashes are recorded.

The [request client](../examples/quality-request.py) applies a named task profile to an ordinary JSON request containing `model` and `messages`. Explicit request fields take precedence, including temperature 0 and `chat_template_kwargs.thinking=false`; an explicit `max_completion_tokens` replaces the profile's legacy `max_tokens` field. `--dry-run` prints the effective body without sending it. Unqualified profiles require deliberate `--experimental` use. The client returns the original completion JSON, rejects truncated or reasoning-only responses, and neither executes tool calls nor retries requests. Applications must still validate tool semantics and generated code.

JSON schemas describe the output contract without embedding reference answers as const or enum values. Nullable fields remain nullable for every array item. Tool calls are checked for the function, multiplicity, types and exact decoded arguments; no tool is executed. Valid JSON alone does not count as a correct answer. Tool schemas and JSON response constraints are not combined in these requests.

To inspect an experimental request without sending it:

```bash
cat > /tmp/quality-request.json <<'JSON'
{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"Return only the JSON array of these numbers in ascending order: 12, 3, 8."}]}
JSON
python3 -B examples/quality-request.py \
  --request /tmp/quality-request.json --category instructions \
  --experimental --dry-run
```

Remove `--dry-run` and supply `--base-url` to evaluate it against your endpoint. The `penalties` profile only selects effort, temperature and budget; the benchmark's nondefault penalty fixtures are not applied to ordinary requests. The client does not infer a category or inject a new system prompt.

## Reproduce the experiment

The runner requires a dedicated, already running inference endpoint and Docker for isolated CPU execution of generated code. It does not manage services or change a server's configuration. Use an external resource monitor when sharing a host. The model image can serve as the CPU sandbox image because it already contains Python; pin its digest locally before running.

```bash
python3 -B scripts/test_quality_eval.py
python3 -B -m scripts.quality_eval.run \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-v4.1-flash-1017 \
  --sandbox-image ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4@sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92 \
  --output artifacts/quality-policy-local
```

Set `INFERENCE_API_KEY` in the environment when the endpoint requires authentication. Credentials are not stored in result records. The output directory contains raw answers and generated code and is private; do not publish it. Only the allowlisted `public-summary.json`, `public-results.json` and `public-speed.json` outputs are intended for review and sanitization before publication.

The runner freezes source hashes, cases and methodology before selection, saves each completed observation, and refuses to resume in place if its source or methodology changes. A resumed run keeps failed observations. Creating `STOP_AFTER_BATCH` in its output directory requests a checkpoint stop after active results are saved; remove that file before resuming. A separately versioned continuation directory can record parent source hashes, the parent result digest and inherited observation identities. This is for documented harness corrections, not retuning against held-out answers or replacing failures. Three API failures within a phase stop the run for inspection. Each request has a 900-second safety deadline; this is separate from the decision to report answer latency without a qualification threshold.

## Evidence and acceptance

The new set contains 210 task instances in 21 problem families: seven selection families with ten instances each, and fourteen held-out families with ten instances each. Family membership is disjoint across selection and holdout. These are correlated instances, not 210 independent abilities. Selection uses three repetitions, holdout five. The original 51 valid synthetic cases remain a separate temperature 0/1 regression test, with an additional low-effort code control and nonstream tool checks. No held-out result is used to retune the selected policies.

Short quality tasks run with up to four simultaneous requests on the same server; long-context tasks run singly. The tokenizer calibrates new long inputs to within 1% of 32,768, 131,072 or 262,144 tokens. Actual completion usage is retained. This does not qualify the full 1M context window. Supplied seeds are paired, but backend deterministic inference is not enabled.

The predeclared quality gate requires an increase of at least three percentage points in the category-equal mean pass rate, a positive lower bound for a stratified family-bootstrap 95% interval, and no category declining more than two percentage points. The bootstrap has only fourteen held-out family units and should not be interpreted as broad production confidence. Legacy categories, the low-effort code control and the nonstream tool control must also avoid regressions. The framework's CPU parser regression checks are recorded separately.

Speed checks use one fixed selection prompt per category, a shared 1,024-output-token cap, four warmup requests per arm, C1/C4, and five measured paired repetitions. Baseline/candidate order alternates. The measurement is all returned completion tokens, including reasoning, divided by batch wall time including first-token latency. Repeated prefixes stay cached. Every category/concurrency median must retain at least 80% of baseline throughput. First-token, first-answer and full-response times are recorded separately; generating more reasoning tokens is not proof of more useful answers. Short responses and reasoning changes can make these rates noisy, so the individual batches are retained.

Passing source tests or finding a selection-stage winner does not qualify a new policy. A recommendation requires the frozen holdout and performance gates. Unqualified experimental settings must not replace the existing defaults.
