# DeepSeek V4.1 Flash SM120 兼容修复项目

保留 V4.1 的模型实现、SM120 优化和依赖栈，将 0731 项目中仍适用而缺失的修复迁入当前代码。基线是 ormandj 项目的 `11c525d0234dc25e167e4d6bb0fe9fe757762f5f`，SGLang 固定为 `62cee1b15c22350b17a9256fc6ad5777feb2ec70`。旧补丁来自 ombori 项目的 `79bbc001db042fc85dc2b15e4175210eaec3e78e`。

本项目候选镜像名为 `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.1`。尚未构建、发布或进行 GPU 验收；不保证速度或回答质量提升。上游文档中的性能数字仅属于原版镜像。

## 已完成的迁移

[迁移清单](compat/migration.json) 对旧项目的 33 个补丁逐项给出源码位置、原始补丁 SHA256、判定及迁入补丁编号：7 项迁移、22 项已有等效实现、4 项不适用于当前 V4.1 配方。原补丁和许可证保存在 `compat/upstream-0731/`，仅作为来源记录，构建不会直接应用它们。

新增 7 个运行时修复补丁及 1 个 CPU 测试补丁，共 40 个 SGLang 补丁；FlashInfer、DeepGEMM、Triton 的锁定版本保持不变。

- 恢复 DSPARK/DFLASH 的加性惩罚、重复惩罚和两个快速跳过判断。按照真实 logits 的符号执行缩放，覆盖加性惩罚导致符号变化及 grammar/logit bias 组合。
- 按已经提交到请求历史的输出累计惩罚，处理每轮多个 token、重复准备、批次过滤、合并及重建。不会将尚未提交的草稿 token 纳入计数。
- 补发所有工具调用尚未发送的参数，记录最终完整名称分片，并避免重复收尾造成重复输出。
- 普通正文状态暂存跨分片的推理标签，避免标签碎片泄漏。
- 批请求切片保留 `routing_key`；`require_reasoning` 沿用新版已有实现。
- DeepGEMM 预热仅选择内存容纳得下的受支持形状，最小形状也放不下时跳过预热。
- 将退化 Markov greedy 合并结果的 sentinel 限制在词表范围内。

不恢复旧版的 C4 压缩布局或 prefix-affinity 新特性，不把旧版 256 并发、DP4 或 gamma=7 参数套到新版。V4.1 默认思考模式和启动参数保持上游行为。

## 本地 CPU 检查

需要 Python 3.12、Git、Bash、jq；首次获取源码需要访问 GitHub 官方 API/归档。独立环境示例：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r compat/requirements-cpu.txt
export COMPAT_PYTHON="$PWD/.venv/bin/python"
bash scripts/test-compat.sh
.venv/bin/python scripts/check-migration.py --source .cache/compat-sources/sglang
bash scripts/validate-release.sh
bash scripts/validate-docs.sh
COMPAT_SOURCE_FETCH=archive bash scripts/verify-patches.sh
```

测试读取锁定源码的实际函数和类，只隔离 GPU 导入依赖与传输对象，去除 `torch.compile` 包装；不复制一份业务实现作为被测对象。普通采样参照方法也来自同一源码。Triton 合并器的 CPU 测试执行其函数体的张量模拟，不能替代 CUDA 编译和 Graph replay 验证。

本地 CPU 环境使用 PyTorch 2.10.0；生产构建使用锁定底座的 PyTorch，镜像构建会再次运行该套测试及上游 CPU 检查。两者的执行记录必须分别标注。

本次 CPU 回归为 132 项通过；同套测试在原版上为 77 项通过、55 项失败，其中包含参数组合及新增辅助方法缺失，不代表 55 个独立缺陷。四个框架源码树已校验；`verify-patches.sh` 的 Docker Hub 基础镜像清单检查因认证端点连接超时尚未完成，联网恢复后需重跑。

`verify-source-archives.py` 验证官方归档的 Git tree、全部补丁哈希和最终 tree。对于子模块，它验证父树中的 gitlink，不下载子模块工作树；实际镜像构建继续使用上游递归 checkout。不能把源码树检查写成镜像构建成功。

## 构建和隔离启动

在具有足够内存和磁盘的 Linux amd64 Docker 构建机上运行：

```bash
bash scripts/build-compat.sh --dry-run
bash scripts/build-compat.sh
```

构建检查工作区干净，先构建锁定依赖栈，再构建服务镜像；执行来源、依赖和 CPU 门禁，不推送镜像、不部署服务。底座重建成本较高，首次构建需要联网下载锁定依赖及子模块。

在独立的四卡 SM120 测试环境准备锁定模型后：

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-compat-v0.4.4-rc.1-tp4
bash examples/serve-compat-tp4.sh
```

此启动命令会启动 GPU 服务，仅在测试资源就绪后手动执行。默认端口为 `18001`、容器名为 `deepseek-v41-flash-sm120-compat-tp4`，可用 `PORT` 和 `CONTAINER_NAME` 覆盖。使用独立的缓存、端口和容器名称；本次交付未执行启动命令。

## GPU 验证入口

`scripts/validate-gpu.py` 接收明确指定的两个 OpenAI 测试端点，检查流结束、工具参数、结构化 JSON、Markdown 基本结构、思考、惩罚、最小生成长度与 logprobs，并保存每条延迟、token 数和结果摘要。默认并发为 1、4、16，每组 3 次；先用 `--dry-run` 查看请求数量。密钥仅从 `V41_BASELINE_API_KEY`、`V41_CANDIDATE_API_KEY` 环境变量读取，不写入结果。

```bash
python3 scripts/validate-gpu.py --baseline-url http://127.0.0.1:18000/v1 \
  --candidate-url http://127.0.0.1:18001/v1 --dry-run
```

去掉 `--dry-run` 才发送请求。两个端点必须使用同一权重、GPU 配额、请求参数和启动配置；额外以关闭 DSPARK 的独立实例作参照。脚本不启动、不停止服务。合成协议检查不能替代原始格式异常案例的人工判读。

`compat/test_gpu_native.py` 在构建镜像内通过 pytest 手动运行，检查实际 Triton 合并内核和 CUDA Graph replay。发布前还需覆盖填充批次、长上下文、缓存命中、请求取消、SWA 回收及真实并发；这部分尚未执行。

惩罚状态的 CPU 测试只证明已提交请求历史的累计行为。重叠调度下历史可见时机，以及一个 verify block 内各 token 的惩罚语义，仍需与关闭 DSPARK 的实例实测对比，不能由本地测试推断为完整采样等价。

## 来源和版本

保留上游签名提交身份、源码树及 `upstream` remote。当前本地 Git 以该提交作为浅历史边界；网络恢复 Git 传输后可用 `git fetch --unshallow upstream` 获取更早历史。完整来源和每个本地补丁的父提交、tree、哈希均记录在锁文件及迁移清单中。

本项目不包含模型权重、生产密钥或对话正文。候选版本不代表已经发布的稳定版本。运行验证摘要见 `compat/validation.json`；本地测试详细结果位于被 Git 忽略的 `artifacts/`。
