# 1017 测试与参数对比

[English](TESTING.md) · [项目说明](README.zh-CN.md) · [启动方法](RUN.md) · [工程手册](docs/ENGINEERING.zh-CN.md) · [技术复盘](docs/RETROSPECTIVE.zh-CN.md)

本文汇总截至 2026-11-03 的实验，整理日期为 2026-11-04。保留配置为 TP4 / Attention DP1 / EP2，使用 `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`。这是已测方案中选定的部署配置，尚未遍历所有参数，也不代表稳定版验收。发布本记录不改变镜像、服务参数或请求默认值。

## 证据与共同条件

| 记录 | 内容 |
| --- | --- |
| [验证记录](compat/validation.json) | 构建、CPU/CUDA 检查、DSPARK 开关、代码请求策略及字面量诊断 |
| [同模型并行配置对比](compat/topology-comparison-20261103.json) | 完整配置、204 次质量请求摘要、18 组已完成测速及未完成项 |
| [框架与模型部署对比](compat/deployment-comparisons-20261103.json) | 原框架对照、0731 对照、镜像身份、分项得分及 24 组测速数据 |
| [镜像发布回执](compat/publication.json) | 已测镜像与发布镜像的内容对应关系 |

每实例使用四张 NVIDIA RTX PRO 6000 Blackwell 96 GB，PCIe P2P、无 NVLink；宿主机为双路 Intel Xeon Platinum 8480+、503 GiB RAM，驱动 610.43.02，单卡功率上限 600 W。并行部署实验使用同一八卡主机的两个四卡分组及对应 NUMA 内存节点。

1017 权重为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，修订 `1bae0e1088993b9b82de08f59de5be5a34e42170`。rc.4 构建源码为 `b808761fbef06cec27ce7b3afc55c372cefa78c1`，发布的 Linux amd64 清单摘要为 `sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92`。质量实验测量的是独立构建的 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`；发布复用了其运行清单、配置和全部层，没有重新构建。本地 OCI 索引还含构建证明，因此索引摘要与发布清单摘要不同，详见回执。

主质量集包含推理、代码、指令遵循、结构化输出、工具调用、长输入和惩罚参数。一道歧义题在两侧同时排除后，每轮为 51 道有效题 × 温度 0/1，共 102 次请求。截断、空最终答案、格式或类型不合要求均可判失败；失败复测不替换主得分。各轮独立记录，不合并不同轮次的通过率。

API seed 有记录，但未启用确定性推理，相同 seed 不保证相同随机序列。这些小型合成测试不是官方能力榜单，也不证明采样分布等价或普遍质量提升。

## 已完成的范围

| 对比项 | 取值或模式 | 完成情况 |
| --- | --- | --- |
| 框架 | 原框架与兼容修改版；解析修复前后 | 质量测试完成 |
| DSPARK | 开启 / 关闭 | 质量、代码策略、工具字面量诊断完成 |
| 代码请求 | temperature 0/1；reasoning_effort high/low/none | 下列四种组合完成，共 320 次 |
| 并行配置 | TP4 / Attention DP1 / EP2 与 TP4 / Attention DP4 / EP4 | 质量完成，测速完成 18/48 组后结束 |
| 工具请求 | 流式 / 非流式，温度 0/1，并发 1/4 | 字面量矩阵完成，每组合 5 轮 |
| 输入长度 | 20,054 / 40,054 / 80,054 / 160,054 tokens | 正确性检查完成；上下文上限一直为 1M |
| 惩罚参数 | frequency、presence、repetition penalty | 正确性检查完成，未做吞吐调优 |
| 不同模型部署 | 0731 与 1017 | 质量与 C1/C4 两种短输入测速完成 |

## 框架与 DSPARK

第一轮在相同权重、相同显式启动命令、相同四卡上顺序对比 `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` 与 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`。两侧限制 224 GiB 主机内存、16 CPU，进程级关闭 THP，HiCache ratio 为 0.1。有效质量得分为 98/102 与 99/102；这不复现所有历史部署默认值，也没有受控吞吐结论。镜像身份及分温度计数见[部署对比记录](compat/deployment-comparisons-20261103.json)。

后续解析修复与 DSPARK 实验使用同一组四卡顺序运行三种模式：

| 模式 | 镜像 | 主质量集 | 工具字面量专项 |
| --- | --- | --- | --- |
| `rc3_on` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`，DSPARK 开启 | 99/102 | 0/100 |
| `rc4_on` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`，DSPARK 开启 | 100/102 | 6/100 |
| `rc4_off` | `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`，DSPARK 关闭 | 100/102 | 3/100 |

rc.4 在这批检查中没有发生解析器造成的参数损坏，但模型本身仍可能写错 `<think>` / `</think>`。关闭 DSPARK 没有消除问题。质量实验包含推理 token 的请求耗时仅用于诊断，不作为生产吞吐基准。

## 代码请求策略

每种组合均为 8 道小型代码题 × 5 次、并发 1，固定 `top_p=1`、`max_tokens=12288`。通过要求包含代码单元测试及响应完整性。

| temperature | reasoning_effort | DSPARK 开启：通过 / 截断 | DSPARK 关闭：通过 / 截断 |
| --- | --- | --- | --- |
| 0 | high | 37/40；截断 3 | 38/40；截断 2 |
| 1 | high | 39/40；截断 1 | 40/40；截断 0 |
| 1 | low | 40/40；截断 0 | 40/40；截断 0 |
| 1 | none | 38/40；截断 0 | 39/40；截断 0 |

预先确定的选择规则选出 [temperature 1 / low 请求预设](examples/code-request-profile.json)。它是调用方可选配置，不是全局默认值；low 是模型指令，不保证为最终代码预留固定 token 数。`none` 的错误说明关闭思考也可能损失正确性。80 次 low 请求全部通过，不代表复杂代码任务都能通过。

## 工具字面量追加诊断

同一 rc.4 产物另做 442 次诊断请求。原始字面量矩阵在 DSPARK 关闭 / 开启时分别通过 5/100、6/100；显式 `reasoning_effort=low` 分别通过 17/100、19/100。这是独立复测，不能替换上表的 3/100、6/100。

20 条实际输入与权重附带的官方编码器逐 token 一致。复现位置在语法掩码之前，目标模型已偏向错误闭合标记；掩码仍允许正确标记，去掉工具语法的原生生成也复现错误。证据不支持仅靠关闭 DSPARK 或改用 low 解决字面量保真，也未建立新的透明运行时修复依据。计数及数值配置边界见[验证记录](compat/validation.json)的 `literal_generation_diagnosis`。

## TP4 下的 Attention DP / EP

两侧使用同一个已发布 rc.4 镜像和相同 1017 权重，每实例四卡。以下 DP 指 Attention DP；实验侧普通 `dp_size` 仍为 1。

| 参数 | 保留配置 | 实验配置 |
| --- | --- | --- |
| TP / Attention DP / EP | 4 / 1 / 2 | 4 / 4 / 4 |
| DP LM Head | 关闭 | 开启 |
| 静态显存比例 | 0.85 | 0.90 |
| DSPARK / SPS | block 5 / auto | block 5 / auto |
| 上下文 / 最大运行请求数 | 1,048,576 / 16 | 1,048,576 / 16，DP rank 各 4 |
| 运行时 chunked prefill | 8192 | 按 DP 拆分为 2048 |
| 调度保守系数 | 1.0 | 自动调整为 0.3 |
| 缓存 / 主机限制 | HiCache 0.1，224 GiB，无 swap，16 CPU，进程级 THP 关闭 | 相同 |

主质量得分为 100/102 与 97/102。温度 0/1、top_p 1，每题每温度一次；推理和代码 high 思考、12,288 输出 token 预算，工具预算 4096，其他 2048，每侧并发 1。最长实际输入为 160,054 tokens。

完成的测速是中文解释任务：40 个输入 tokens、关闭思考、temperature 1、top_p 1，每请求返回 1024 tokens。每个并发档先预热至少 4 个请求，每侧各测 3 轮，轮换测量顺序，保留重复前缀缓存。单位为总返回 completion tokens / 整批墙钟时间，包含首 token 等待时间；这批输出的推理 token 为 0。

| 并发数 | 保留配置 token/s | 实验配置 token/s | 实验配置变化 |
| --- | --- | --- | --- |
| 1 | 215.69 | 124.88 | -42.10% |
| 4 | 584.77 | 416.25 | -28.82% |
| 8 | 813.94 | 669.46 | -17.75% |

表格为三轮中位数；并发 4/8 是合计吞吐，不是单请求速度或服务端 decode 日志速率。完成 18/48 组、78 次计时请求后，由操作方结束实验。C16 未完成，代码生成测速和追加 low 思考代码复测尚未开始。

实验侧 0.85 无法分配 KV 池，调到 0.90 后启动。这属于启动可行性检查；DP、EP、LM Head、显存比例及 GPU/NUMA 分组同时不同，不能把差异单独归因于 DP。保留原配置的依据限于已完成的工作负载。[逐组数据与完整参数](compat/topology-comparison-20261103.json)。

## 0731 与 1017 部署对照

0731 使用 TP4 / Attention DP4 / EP4、DSPARK block 7、最大运行请求数 256、decode graph 上限 64，未设置与 1017 相同的主机 CPU/内存限额；1017 使用本页保留配置。0731 测量产物以 Docker 镜像 ID `sha256:2fceff11661eb240cb6c007bfbc505c95e549c76c1af437dd7839d0aea6cf6ef` 标识，不归为未经核实的上游发布版本。

主质量得分为 0731 的 88/102、1017 的 98/102。两侧各 51 题 × 温度 0/1、top_p 1、并发 1；推理/代码 high 思考且预算 12,288，工具预算 4096，其他 2048。两侧质量请求成对同时运行。分项得分见[部署对比记录](compat/deployment-comparisons-20261103.json)。

另做关闭思考、temperature 1、top_p 1、每请求 1024 输出 tokens 的短输入测速。解释任务输入 40 tokens；代码任务要求实现支持 TTL、容量限制、统计指标、线程安全和单元测试的 Python LRU 缓存，输入 46 tokens。每组 3 轮，C1/C4，各模型分别计时并交替顺序，排除 4 次预热请求，不清空前缀缓存。共 24 组、60 次计时请求。以下为包含首 token 等待时间的端到端总吞吐中位数，completion tokens 计数包含推理（如有）。

| 任务 | 并发数 | 0731 token/s | 1017 token/s |
| --- | --- | --- | --- |
| 中文解释 | 1 | 126.05 | 136.86 |
| 中文解释 | 4 | 336.28 | 551.02 |
| 代码生成 | 1 | 188.64 | 252.31 |
| 代码生成 | 4 | 553.96 | 724.63 |

模型、框架、并行、DSPARK、缓存和资源限制都不同，不能归因于单个 1017 参数。分词器差异也影响跨模型 token/s 的含义。这一轮与同模型 DP 实验是独立测量，不横向混用速度或通过率。

## 尚未完成的参数调优

- 上下文上限 1M 与 256K 的同负载对比；不同输入长度不等于调整上下文上限。
- TP4 与 TP8；八卡主机上的两个四卡实例不等于 TP8。
- 同一 1017 上 DSPARK block 5/7 等取值扫描，以及 SPS auto 与手工策略的独立对比。
- HiCache、FP8/BF16 KV、prefill、最大运行请求数、CPU/NUMA/THP 的逐项性能扫描。
- 完整 1M 输入压力、长上下文吞吐、持续负载与更大真实业务质量集。
- 已结束 DP4 实验的 C16、代码生成测速、追加 low 思考代码复测。

显存比例 0.80/0.85，以及 DP4 下 0.85/0.90 的尝试用于判断能否分配 KV 池，不是独立速度实验。原配置采用 0.85，DP4 实验采用 0.90。HiCache 从专用主机默认 ratio 2 改为共享主机 0.1 是资源安排，没有完成单独吞吐 A/B。

## 使用这些结果

按 [RUN.md](RUN.md#shared-host-profile) 显式启用共享主机 profile，选择与 GPU 对应的 NUMA 内存节点；代码请求可另选 [low 预设](examples/code-request-profile.json)。新硬件、新构建或新负载需要重新验证。

公开记录只包含参数、镜像身份、数值摘要和证据文件哈希，不包含内部地址、凭据、模型权重或原始对话。摘要不能完整复现每条请求。现有 [API 检查](scripts/validate-gpu.py)、[CPU 检查](scripts/test-compat.sh)及[原生 CUDA 检查](compat/test_gpu_native.py)可用于独立验证，但不会自动重放本文全部实验。上游项目及补丁作者归属见 [NOTICE.md](NOTICE.md)。
