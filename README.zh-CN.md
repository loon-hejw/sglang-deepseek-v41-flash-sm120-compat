# DeepSeek V4.1 Flash SM120 TP4 兼容优化

基于 [ormandj 的 V4.1 项目](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120)，保留锁定的 SM120 优化和依赖栈，迁入 [0731 项目](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) 中适用的正确性修复，提供已经完成四卡部署验收的通用启动配方。

**本仓库发布源码、配置与验证摘要，尚未发布可公开拉取的兼容镜像或稳定版 Release。** 本地镜像完整名称为 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`。原版 `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` 不包含本项目新增的兼容修复。镜像构建输入及已验收版本保持不变。

## 致谢与项目来源

本项目建立在以下两个开源项目的工作之上。感谢两方维护者及贡献者公开实现、部署方案和验证经验，为本仓库的兼容适配提供基础：

- **[ormandj/sglang-deepseek-v41-flash-sm120](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120)**：本仓库的主要 V4.1/1017 基线，提供锁定源码与构建栈、SM120 优化补丁系列、TP4/EP2 部署方案、DSPARK/SPS 集成及文档。本仓库从其 [`11c525d`](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120/commit/11c525d0234dc25e167e4d6bb0fe9fe757762f5f) 提交继续开发，保留原有 Git 历史。
- **[Ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000)**：提供 0731 四卡部署经验及本次兼容迁移所参考的正确性修复，其中也包含其注明来源的 SGLang 贡献者工作。本仓库以其 [`79bbc00`](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000/commit/79bbc001db042fc85dc2b15e4175210eaec3e78e) 提交为迁移来源，对全部 33 项补丁逐项核对，记录为 7 项迁移、22 项已有等效实现、4 项不适用，详见[迁移清单](compat/migration.json)。

本仓库新增工作集中于 V4.1 兼容适配、回归测试、构建集成修复、可配置的 TP4 启动入口及所述硬件上的部署验证。底层框架、算法、内核和原始补丁的贡献归属于各自作者。[英文补丁表](README.md#source-and-patches)及[0731 原始补丁](compat/upstream-0731/patches/)保留作者信息与上游 PR 引用；项目归属和许可证见 [NOTICE.md](NOTICE.md)、[Apache-2.0](LICENSE)，以及保留的 Ombori [NOTICE](compat/upstream-0731/NOTICE) 和 [LICENSE](compat/upstream-0731/LICENSE)。

## 修复内容与来源

[迁移清单](compat/migration.json) 对旧项目 33 项补丁逐一给出来源、哈希和处理结果：7 项迁移、22 项已有等效实现、4 项不适用。最终 SGLang 系列共 40 个补丁，包括 7 个新增运行时修复及 1 个 CPU 测试补丁；FlashInfer、DeepGEMM、Triton 的锁定版本保留。

- DSPARK/DFLASH 正确应用加性和重复惩罚，并按已提交输出累计 token 历史，覆盖批次复用、过滤与合并。
- 工具调用结束时补全所有调用的剩余参数，避免重复收尾及最终名称分片造成的参数遗漏。
- 普通正文暂存跨分片的推理标签，避免标签碎片直接泄漏。
- 批请求拆分保留 `routing_key`。
- DeepGEMM 只预热显存预算容纳得下的受支持形状。
- Markov greedy 合并遇到退化分数时，将无效 sentinel 限制在词表范围内。
- 修复 FlashInfer 嵌套安装锁等待及 Triton CMake 测试开关传递，保留来源、依赖和 CPU 构建门禁。

完整源码版本、补丁哈希及最终源码树由 [锁文件](stack.lock.json) 记录。保留上游历史、作者署名和许可证；旧补丁原文及许可证在 `compat/upstream-0731/`，仅作来源记录。不会恢复旧版 C4 布局、prefix-affinity 新功能，也不套用旧版 DP4、256 并发或 gamma=7 配方。

## 构建与启动

需要 Linux amd64 Docker 主机、四张 96 GB RTX PRO 6000 Blackwell GPU，以及可用的 GPU P2P。已验证主机为双路 Xeon Platinum 8480+、503 GiB 内存、驱动 610.43.02，单卡功率上限 600 W。Engram 表约需 189 GiB 主机内存，此外还需 HiCache、运行时和系统内存。模型约占 480 GiB 磁盘，镜像构建还需额外空间。

```bash
git clone https://github.com/loon-hejw/sglang-deepseek-v41-flash-sm120-compat.git
cd sglang-deepseek-v41-flash-sm120-compat
bash scripts/build-compat.sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /srv/models/DeepSeek-V4.1-Flash-1017

export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-compat-v0.4.4-rc.3-tp4
export GPU_DEVICES=0,1,2,3
# 在当前运行环境中设置 API_KEY 可启用鉴权，不要提交真实密钥。
bash examples/serve-compat-tp4.sh --dry-run
bash examples/serve-compat-tp4.sh
```

构建命令只在本地构建镜像，不启动服务或推送镜像。首次构建需要下载锁定依赖，资源及网络选项见 [构建说明](BUILDING.md)。也可通过 `IMAGE` 指定已经获取的兼容镜像。

启动脚本只向容器暴露指定的四张卡，可用四个索引或完整 GPU UUID。模型只读挂载，缓存持久化；容器后台运行、退出后保留，重启策略为 `no`。`--dry-run` 不创建目录或调用 Docker，输出隐藏 API Key。默认宿主端口 `18001`，模型名 `deepseek-v4.1-flash-1017`，通过 OpenAI `/v1/chat/completions` 调用。容器已存在时不会被覆盖；停止后可用 `docker start deepseek-v41-flash-sm120-compat-tp4` 复用原配置启动。

## 已验证的 TP4 配置

| 项目 | 配置 |
| --- | --- |
| 并行 | TP4 / EP2，`moe-a2a-backend=none` |
| 推测解码 | DSPARK block 5，auto SPS，compact ragged verify，graph tier 对齐 |
| 上下文 / 并发 | 1,048,576 / 最多 16 路 |
| KV / 静态显存比例 | `fp8_e4m3` / **0.85** |
| Prefill | chunk 8192、max prefill tokens 8192 |
| 后端 | attention `dsv4`、MoE `flashinfer_mxfp4`、FP8 GEMM `flashinfer_cutlass` |
| 缓存 | HiCache ratio 2，Engram host table 按 rank 分片，encoder/decoder SWA bounded replay |
| 解析 | reasoning `deepseek-v41`、tool calls `deepseekv41` |
| 默认采样 | OpenAI 默认值，temperature 1.0、top_p 1.0；客户端显式值优先 |

显式启用 PCIe-IPC all-reduce、SM120 MXFP8 skinny GEMM、BF16 skinny LM head、WO-A FP8 copy；保留重叠调度，编译任务数限制为 4。完整参数及环境变量接口见 [RUN.md](RUN.md)。0.80 在已验证硬件上无法为模型和草稿权重之外的 KV 池留出空间，实际采用并验证了 0.85。

## 验收证据与边界

2026-11-02 验收的是单独构建的 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`，镜像身份为 `sha256:f124217b8a003822be10c11cb107739e7a2259c29b3b11c0aed18dea58bae356`，构建源码提交为 `d6528a773d1cda402de06844ba5270d94deb0d10`。本地重新构建产生的是另一份产物，需要独立验证，不能只凭同名标签沿用结论。

- 镜像内 CPU：895 项通过、480 个子测试通过、1 项 CUDA 专用测试跳过；132 项兼容回归已包含在 895 项中。
- 原生 GPU：实际 Triton Markov combine 与 CUDA Graph replay，8 项通过、0 跳过。
- 模型接口：15 项功能场景通过，包含 SSE、JSON schema、单/双工具调用、思考、图片、采样惩罚以及 1/4/16 并发。
- 长上下文：实际输入 34,220 prompt tokens，重复前缀命中 34,048 cached tokens。
- 稳定性：连续 1800.5 秒，61 次健康采样、31 次聊天探针全部成功；0 重启、0 OOM，未发现 CUDA/NCCL 运行错误。
- SPS：8 个 batch 探针、17 个 verify tiers，拟合 56 个 cell，无失败项；生效 KV 池为 2,772,736 tokens。

以上是功能和稳定性验收，不是吞吐基准、完整采样等价性或回答质量提升证明。尚未完成完整 1M 输入压力、请求取消/SWA 回收专项、关闭 DSPARK 的受控对照和热启动计时。短请求吞吐和上游性能结果不能作为本镜像的通用性能承诺，也没有 TP8 实测结论。详细状态见 [验证记录](compat/validation.json)。

独立空缓存首次启动耗时 2343 秒，包含内核编译、CUDA Graph 捕获和 SPS 调优。同镜像、硬件和参数可复用兼容编译缓存与 SPS 表，但仍需加载权重及捕获 CUDA Graph。换主机可尝试迁移兼容编译缓存，SPS 应按新机器重测；具体热启动耗时尚未测量。

## 开发验证

本地 CPU 检查使用 Python 3.12；准备环境后执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r compat/requirements-cpu.txt
export COMPAT_PYTHON="$PWD/.venv/bin/python"
bash scripts/test-compat.sh
.venv/bin/python scripts/check-migration.py --source .cache/compat-sources/sglang
python3 scripts/test_serve_compat_tp4.py
bash scripts/validate-release.sh
bash scripts/validate-docs.sh
```

CPU 测试隔离加载锁定源码的真实函数体，Triton 使用张量模拟，不替代原生 CUDA 测试。API 对比入口 `scripts/validate-gpu.py` 需要两端同权重、GPU 配额和请求配置，先用 `--dry-run` 预览。原生 GPU 入口为 `compat/test_gpu_native.py`。完整来源验证和发布规则见 [BUILDING.md](BUILDING.md)。

公开仓库不包含模型权重、内部部署地址、生产凭据或对话正文。原始日志保留在本地，公开验证摘要只记录硬件、参数、版本与检查结果。原版稳定版历史见 [CHANGELOG.md](CHANGELOG.md)，本项目源码使用 [Apache-2.0](LICENSE)，第三方归属见 [NOTICE.md](NOTICE.md)。
