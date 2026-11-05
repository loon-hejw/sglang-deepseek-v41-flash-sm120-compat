# DeepSeek V4.1 Flash SM120 TP4 兼容优化

基于 [ormandj 的 V4.1 项目](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120)，保留锁定的 SM120 优化和依赖栈，迁入 [0731 项目](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) 中适用的正确性修复，提供已经完成四卡部署验收的通用启动配方。

本仓库发布源码、配置、验证摘要和 Linux amd64 候选镜像 `ghcr.io/loon-hejw/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`。**这是候选版，尚未发布稳定版；端到端字面量生成仍不可靠。** 仓库清单摘要为 `sha256:0a32f40290db16a139ee2ac38ec4a8641b7310de07b79a249dd1c11d46a48a92`，详见[发布回执](compat/publication.json)。原版 `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` 不包含本项目新增的兼容修复。

## 按用途阅读

本项目调整推理框架、镜像工程和运行策略，模型权重保持固定修订，没有重新训练或微调。

| 用途 | 文档入口 |
| --- | --- |
| 首次使用 | 本页[拉取与启动](#拉取与启动)，完整操作见 [RUN](RUN.md) |
| 运维交接与后续开发 | [工程交付手册](docs/ENGINEERING.zh-CN.md)：来源、构建、资源、缓存、主机迁移、升级回滚 |
| 了解修改过程和取舍 | [技术复盘](docs/RETROSPECTIVE.zh-CN.md)：问题、证据、处理与结论边界 |
| 查看参数和质量／性能证据 | [测试总览](TESTING.zh-CN.md)：实验条件、结果和未完成项 |
| 调整请求质量 | [质量策略](docs/QUALITY.zh-CN.md)：思考开关、分任务预设、留出验证与速度门槛 |
| 构建与发布新产物 | [BUILDING](BUILDING.md)、[版本记录](CHANGELOG.md) |

工程手册和技术复盘均提供完整英文版。测试结果统一引用现有测试总览和 JSON 记录，避免形成多套数据口径。

## 致谢与项目来源

本项目建立在以下两个开源项目的工作之上。感谢两方维护者及贡献者公开实现、部署方案和验证经验，为本仓库的兼容适配提供基础：

- **[ormandj/sglang-deepseek-v41-flash-sm120](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120)**：本仓库的主要 V4.1/1017 基线，提供锁定源码与构建栈、SM120 优化补丁系列、TP4/EP2 部署方案、DSPARK/SPS 集成及文档。本仓库从其 [`11c525d`](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120/commit/11c525d0234dc25e167e4d6bb0fe9fe757762f5f) 提交继续开发，保留原有 Git 历史。
- **[Ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000)**：提供 0731 四卡部署经验及本次兼容迁移所参考的正确性修复，其中也包含其注明来源的 SGLang 贡献者工作。本仓库以其 [`79bbc00`](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000/commit/79bbc001db042fc85dc2b15e4175210eaec3e78e) 提交为迁移来源，对全部 33 项补丁逐项核对，记录为 7 项迁移、22 项已有等效实现、4 项不适用，详见[迁移清单](compat/migration.json)。

本仓库新增工作集中于 V4.1 兼容适配、回归测试、构建集成修复、可配置的 TP4 启动入口及所述硬件上的部署验证。底层框架、算法、内核和原始补丁的贡献归属于各自作者。[英文补丁表](README.md#source-and-patches)及[0731 原始补丁](compat/upstream-0731/patches/)保留作者信息与上游 PR 引用；项目归属和许可证见 [NOTICE.md](NOTICE.md)、[Apache-2.0](LICENSE)，以及保留的 Ombori [NOTICE](compat/upstream-0731/NOTICE) 和 [LICENSE](compat/upstream-0731/LICENSE)。

## 修复内容与来源

[迁移清单](compat/migration.json) 对旧项目 33 项补丁逐一给出来源、哈希和处理结果：7 项迁移、22 项已有等效实现、4 项不适用。当前 SGLang 系列共 41 个补丁，包含原有兼容迁移及新增的 V4.1 工具参数字面量保护；FlashInfer、DeepGEMM、Triton 的锁定版本保留。

- DSPARK/DFLASH 正确应用加性和重复惩罚，并按已提交输出累计 token 历史，覆盖批次复用、过滤与合并。
- 工具调用结束时补全所有调用的剩余参数，避免重复收尾及最终名称分片造成的参数遗漏。
- 普通正文暂存跨分片的推理标签，避免标签碎片直接泄漏。
- 批请求拆分保留 `routing_key`。
- DeepGEMM 只预热显存预算容纳得下的受支持形状。
- Markov greedy 合并遇到退化分数时，将无效 sentinel 限制在词表范围内。
- 修复 FlashInfer 嵌套安装锁等待及 Triton CMake 测试开关传递，保留来源、依赖和 CPU 构建门禁。

完整源码版本、补丁哈希及最终源码树由 [锁文件](stack.lock.json) 记录。保留上游历史、作者署名和许可证；旧补丁原文及许可证在 `compat/upstream-0731/`，仅作来源记录。不会恢复旧版 C4 布局、prefix-affinity 新功能，也不套用旧版 DP4、256 并发或 gamma=7 配方。

## 拉取与启动

需要 Linux amd64 Docker 主机、四张 96 GB RTX PRO 6000 Blackwell GPU，以及可用的 GPU P2P。已验证主机为双路 Xeon Platinum 8480+、503 GiB 内存、驱动 610.43.02，单卡功率上限 600 W。Engram 表约需 189 GiB 主机内存，此外还需 HiCache、运行时和系统内存。模型约占 480 GiB 磁盘，镜像构建还需额外空间。

GHCR 镜像当前为私有，需要先通过 `docker login ghcr.io` 登录具有该软件包读取权限的账号；尚未开放匿名拉取。

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
# 在当前运行环境中设置 API_KEY 可启用鉴权，不要提交真实密钥。
bash examples/serve-compat-tp4.sh --dry-run
bash examples/serve-compat-tp4.sh
```

如需自行构建 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`，按[构建说明](BUILDING.md)执行，并将 `IMAGE` 改为该本地镜像。模型权重单独下载，不包含在镜像中。

启动脚本只向容器暴露指定的四张卡，可用四个索引或完整 GPU UUID。模型只读挂载，缓存持久化；容器后台运行、退出后保留，重启策略为 `no`。`--dry-run` 不创建目录或调用 Docker，输出隐藏 API Key。默认宿主端口 `18001`，模型名 `deepseek-v4.1-flash-1017`，通过 OpenAI `/v1/chat/completions` 调用。容器已存在时不会被覆盖；停止后可用 `docker start deepseek-v41-flash-sm120-compat-tp4` 复用原配置启动。

## 已验证的 TP4 配置

| 项目 | 配置 |
| --- | --- |
| 并行 | TP4 / Attention DP1 / EP2，`moe-a2a-backend=none` |
| 推测解码 | DSPARK block 5，auto SPS，compact ragged verify，graph tier 对齐 |
| 上下文 / 并发 | 1,048,576 / 最多 16 路 |
| KV / 静态显存比例 | `fp8_e4m3` / **0.85** |
| Prefill | chunk 8192、max prefill tokens 8192 |
| 后端 | attention `dsv4`、MoE `flashinfer_mxfp4`、FP8 GEMM `flashinfer_cutlass` |
| 缓存 | 默认 HiCache ratio 2；共享主机配置为 0.1，Engram host table 按 rank 分片，encoder/decoder SWA bounded replay |
| 解析 | reasoning `deepseek-v41`、tool calls `deepseekv41` |
| 默认采样 | OpenAI 默认值，temperature 1.0、top_p 1.0；客户端显式值优先 |

显式启用 PCIe-IPC all-reduce、SM120 MXFP8 skinny GEMM、BF16 skinny LM head、WO-A FP8 copy；保留重叠调度，编译任务数限制为 4。完整参数及环境变量接口见 [RUN.md](RUN.md)。0.80 在已验证硬件上无法为模型和草稿权重之外的 KV 池留出空间，实际采用并验证了 0.85。

共享主机部署可显式设置 `HOST_PROFILE=shared` 和与所选 GPU 对应的 `NUMA_MEMORY_NODE`：采用 224 GiB 内存上限、无 swap、16 CPU 配额、进程级 THP 关闭和 HiCache ratio 0.1，保留 TP4 / Attention DP1 / EP2。启动器不负责集群 GPU 预留或持续内存监督，使用前需安排资源隔离。用法及运行边界见 [RUN.md 的共享主机配置](RUN.md#shared-host-profile)。

[初步并行配置对比](compat/topology-comparison-20261103.json)记录了保留该配置的依据：204 次主质量请求全部完成，原配置通过 100/102，DP4/EP4 实验配置通过 97/102；仅完成 18/48 组短输入测速后结束实验。DP4 另需显存比例 0.90，两边物理 GPU/NUMA 分组也不同，不能单独归因于 DP。C16、代码生成测速和追加 low 思考代码复测未完成。本次提交归档配置与脱敏摘要，不产生新镜像，也不代表稳定版或普遍性能结论。

## 验收证据与边界

[测试与参数对比总览](TESTING.zh-CN.md)集中记录框架、DSPARK 开关、代码请求四组策略、Attention DP/EP、0731 部署对照，以及已完成和未完成项目；每组结果均关联对应证据与适用边界。

2026-11-03 验证的是独立构建的 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`，本地 OCI 索引身份为 `sha256:bb27bdd8172e86db037840528b971fa5e2ec8d055236f49a3930d73be4ac58e7`，构建源码为 `b808761fbef06cec27ce7b3afc55c372cefa78c1`。发布时直接复用其 Linux amd64 运行清单、配置及全部 79 个层，逐字节一致，没有重新构建。仓库摘要对应平台清单；本地索引还包含本次未发布的构建证明，因此二者摘要不同。同标签的本地重新构建是另一份产物，不能自动沿用结论。

- 镜像内 CPU：1,198 项及 691 个子测试通过，1 项 CUDA 专用 compact-verify 测试跳过。
- 原生 GPU：实际安装的 Triton Markov combine 与 CUDA Graph replay，8 项通过、0 跳过；这是独立套件，不等于补跑上述跳过项。
- 同一组四张 RTX PRO 6000 Blackwell 顺序运行三组工作负载。`rc3_on` 使用 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3` 并开启 DSPARK；`rc4_on` 和 `rc4_off` 使用 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.4`，分别开启和关闭 DSPARK。两版安装包版本清单一致。

| 模式 | 主质量集 | 工具字面量专项 |
| --- | --- | --- |
| `rc3_on` | 99/102 | 0/100 |
| `rc4_on` | 100/102 | 6/100 |
| `rc4_off` | 100/102 | 3/100 |

主质量集是 51 道有效题 × 温度 0/1；工具字面量专项覆盖流式/非流式、并发 1/4，每组合 5 轮。失败另做复测，不替换原始得分。`rc4_on` 和 `rc4_off` 两种模式下，API 工具参数均保持原始生成参数，但生成序列本身仍会写错字面量，因此端到端问题尚未全面解决。本次发布未替换生产服务。

随后对同一候选进行了 442 次专项诊断请求，观察钩子仅挂载在镜像外。20 条实际输入与权重附带的官方编码器逐 token 一致；复现失败位置在语法掩码之前，目标模型就给错误的 `</think>` 更高分数，掩码没有改变这两个标记的分数。移除工具语法的原生生成也复现了错误。此次原始专项在 DSPARK 关闭/开启时分别通过 5/100、6/100；显式 `reasoning_effort=low` 诊断仅通过 17/100、19/100，不能保证字面量保留。未发现足以支持额外运行时修复的缺陷，因此没有新增补丁、镜像或默认参数变更。结论限于当前权重和数值配置，不代表所有后端或精度。

[代码请求预设](examples/code-request-profile.json) 显式设置 temperature 1、reasoning_effort `low`、top_p 1、max_tokens 12,288。8 道小型代码题各重复 5 次、并发 1，DSPARK 开启时通过 40/40，关闭时通过 40/40。[客户端示例](examples/code-request.py) 保留调用方显式覆盖，遇到截断或空正文报错，不把推理当代码、不自动重试；它不修改全局默认值。使用方法见 [RUN.md](RUN.md#opt-in-coding-requests)。

独立的[质量策略实验](docs/QUALITY.zh-CN.md)在合成留出集上从 55.43% 提高到 73.00%，主要收益来自指令和长输入开启 low 思考。C1/C4 全部测速保留至少 92.74% 的基线 token 速率（含思考），但调整后的预设增加了最终答案等待时间。非流式工具字面量“不退步”门槛未通过，因此整套预设保持实验状态，不替换请求或服务默认值。

请求 seed 已记录，但当前配置关闭确定性推理，服务器随机种子也不同；不能将这些结果解释为同随机序列的严格因果对照。测试与生产共享主机，测试副本限制 224 GiB 内存、16 CPU，采用进程级 THP 关闭与 HiCache ratio 0.1，最大实际输入 160,054 tokens。结果不证明完整采样等价、通用代码能力提升、生产吞吐或完整 1M/TP8 能力。详细条件、包含推理 token 的速度统计、此前服务验收及未完成项见[验证记录](compat/validation.json)。

## 开发验证

本地 CPU 检查使用 Python 3.12；准备环境后执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r compat/requirements-cpu.txt
export COMPAT_PYTHON="$PWD/.venv/bin/python"
bash scripts/test-compat.sh
.venv/bin/python scripts/check-migration.py --source .cache/compat-sources/sglang
python3 scripts/test_serve_compat_tp4.py
python3 scripts/test_code_request.py
bash scripts/validate-release.sh
bash scripts/validate-docs.sh
```

CPU 测试隔离加载锁定源码的真实函数体，Triton 使用张量模拟，不替代原生 CUDA 测试。API 对比入口 `scripts/validate-gpu.py` 需要两端同权重、GPU 配额和请求配置，先用 `--dry-run` 预览。原生 GPU 入口为 `compat/test_gpu_native.py`。完整来源验证和发布规则见 [BUILDING.md](BUILDING.md)。

公开仓库不包含模型权重、内部部署地址、生产凭据或对话正文。原始日志保留在本地，公开验证摘要只记录硬件、参数、版本与检查结果。原版稳定版历史见 [CHANGELOG.md](CHANGELOG.md)，本项目源码使用 [Apache-2.0](LICENSE)，第三方归属见 [NOTICE.md](NOTICE.md)。
