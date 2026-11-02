# Building and publishing this compatibility fork

This repository currently publishes source, the TP4 launcher and validation summaries. It does not publish a public fork container image or stable GitHub Release. The qualified private artifact is `local/sglang-deepseek-v41-flash-sm120-compat:v0.4.4-rc.3`; a local rebuild is a separate artifact and must be qualified separately.

## Local build

Use a Linux amd64 Docker host with sufficient RAM and free disk. The optional monitored builder aborts below 96 GiB available RAM or 200 GiB free disk. The ordinary build script does not enforce those host-level thresholds. Build jobs default to four; `COMPAT_MAX_JOBS` overrides the limit.

```bash
bash scripts/build-compat.sh --dry-run
bash scripts/build-compat.sh
```

The build dry run prints the commands and writes ignored build recipes; unlike the serving dry run, it can update build artifacts. The real build requires a clean checkout. It builds the locked stack first and then the service image, running source identity, dependency and installed CPU gates. It does not publish or deploy. `python3 scripts/run-build-monitored.py` adds resource monitoring on hosts with existing workloads.

The vendor base supplies CUDA 13.0.3, PyTorch and runtime libraries. `stack.lock.json` pins SGLang, FlashInfer, DeepGEMM and Triton commits, recursive source dependencies, patch hashes and resulting trees. Models are not baked into the image. The lock's model revision must be downloaded separately.

Restricted-network builds may use `COMPAT_BASE_IMAGE` for a mirror of the same locked base, `COMPAT_BASE_SOURCE` for its verification source, or a `COMPAT_BASE_RECEIPT` produced by `scripts/verify-base-image.py`. The verifier checks raw manifests, configuration digests and ordered RootFS layers; a matching tag is insufficient. `COMPAT_SOURCE_CACHE` can mount checksum-verified Git objects and recursive submodules read-only; normal source/patch/tree verification remains mandatory.

FlashInfer compiler dependencies are installed before enabling `FLASHINFER_BUILD_NO_PIP=1`, avoiding nested installer locks. Triton's intended test-build setting is passed through `TRITON_APPEND_CMAKE_ARGS`; installed SGLang CPU and compatibility gates remain enabled.

## Validation

```bash
bash -n scripts/*.sh build/*.sh examples/*.sh
bash scripts/validate-release.sh
bash scripts/validate-docs.sh
python3 scripts/test_serve_compat_tp4.py
python3 scripts/test_validate_publish_release.py
python3 scripts/test_verify_release_target.py
```

For compatibility CPU checks, use the Python 3.12 environment described in [README.zh-CN.md](README.zh-CN.md), then run `scripts/test-compat.sh` and `scripts/check-migration.py` against prepared locked sources. `scripts/verify-patches.sh` verifies official sources and the pinned base image; it requires network access. Native CUDA, image build and real model serving are separate qualification stages, recorded in [compat/validation.json](compat/validation.json).

## GitHub workflows and release integrity

| Workflow | Trigger / purpose |
| --- | --- |
| `validate.yml` | Pushes to `main`, pull requests and manual runs; checks source, documentation and launcher behavior without building or publishing an image |
| `build-image.yml` | Manual only; builds and publishes a new immutable candidate when explicitly requested |
| `promote-release.yml` | Manual only; promotes a qualified candidate with digest equality |
| `publish-release.yml` | Manual only; verifies a stable image and publishes its source tag and GitHub Release |

Initial source publication must not dispatch any of the image/release workflows. Changing `Containerfile`, `Containerfile.stack`, `release.json`, `stack.lock.json`, `patches/` or `build/` requires a new candidate version and matching labels/lock entries. Published tags are immutable. Documentation, launcher and validation-tool changes do not rebuild the existing candidate.

Before any future image publication, qualify the new artifact on the declared hardware and configuration, verify the registry digest and anonymous pull access, and update the reviewed release documentation. Stable promotion and source Release publication remain separate operations. Do not use the upstream stable tag or upstream benchmark results as evidence for this fork.
