# Notices

The container images do not include model weights. The vendor base supplies the pinned CUDA/PyTorch dependency stack; `stack.lock.json` records the SGLang, FlashInfer, DeepGEMM and Triton source inputs and patches.

`deepseek-ai/DeepSeek-V4.1-Flash` is released by DeepSeek under the **MIT License**.

SGLang and FlashInfer are Apache-2.0. DeepGEMM and Triton are MIT. The `instanttensor` wheel installed by the image carries its own license. The `lmsysorg/sglang` base image carries its own upstream notices.

## 0731 compatibility ports

The additional compatibility patches derive from the Apache-2.0 project https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000 at commit 79bbc001db042fc85dc2b15e4175210eaec3e78e and its cited SGLang pull requests. Original patch headers, copyright notices and the license are retained under compat/upstream-0731. Per-patch authorship and links remain in those originals and compat/migration.json. Local adaptations and tests are distributed under this repository's Apache-2.0 license.
