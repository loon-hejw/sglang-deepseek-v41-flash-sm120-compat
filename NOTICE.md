# Notices

## Project origins and acknowledgments

This repository is a compatibility fork of [ormandj/sglang-deepseek-v41-flash-sm120](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120), based on commit [`11c525d0234dc25e167e4d6bb0fe9fe757762f5f`](https://github.com/ormandj/sglang-deepseek-v41-flash-sm120/commit/11c525d0234dc25e167e4d6bb0fe9fe757762f5f). We thank ormandj and the project's contributors for the V4.1/1017 source and build baseline, SM120 patch integration, TP4/EP2 serving recipe, DSPARK/SPS integration and documentation. Its Git history, Apache-2.0 license and upstream patch references are retained.

We also thank [Ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) and its contributors for the 0731 four-GPU serving work and correctness patches used in the compatibility migration described below. The preserved [original NOTICE](compat/upstream-0731/NOTICE) identifies **Copyright 2026 Ombori** and credits the respective SGLang patch authors; its [Apache-2.0 license](compat/upstream-0731/LICENSE) is retained.

This fork adds compatibility adaptations, regression tests, build integration fixes, a configurable TP4 launcher and deployment validation. Original framework, algorithm, kernel and patch authorship remains with the respective upstream contributors, including those named in the patch headers and linked pull requests.

## Dependency notices

The container images do not include model weights. The vendor base supplies the pinned CUDA/PyTorch dependency stack; `stack.lock.json` records the SGLang, FlashInfer, DeepGEMM and Triton source inputs and patches.

`deepseek-ai/DeepSeek-V4.1-Flash` is released by DeepSeek under the **MIT License**.

SGLang and FlashInfer are Apache-2.0. DeepGEMM and Triton are MIT. The `instanttensor` wheel installed by the image carries its own license. The `lmsysorg/sglang` base image carries its own upstream notices.

## 0731 compatibility ports

The additional compatibility patches derive from the Apache-2.0 project [ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000) at commit [`79bbc001db042fc85dc2b15e4175210eaec3e78e`](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000/commit/79bbc001db042fc85dc2b15e4175210eaec3e78e) and its cited SGLang pull requests. Original patch headers, copyright notices and the license are retained under [compat/upstream-0731](compat/upstream-0731/). Per-patch authorship and links remain in those originals and [compat/migration.json](compat/migration.json). Local adaptations and tests are distributed under this repository's Apache-2.0 license.
