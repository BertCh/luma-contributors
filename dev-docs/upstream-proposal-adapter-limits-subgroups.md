# Proposal for luma.gl: adapter limits, subgroups with a fallback, and buffer_view (PLAT-05)

**Status: a draft kept in this repository. Nothing has been sent upstream: no issue, no pull request, no message to the luma.gl maintainers.** Under [AGENTS.md](../AGENTS.md) the luma.gl half belongs upstream, so it is not implemented here.

## 1. Request the adapter's real limits

Spec defaults are 8 storage buffers per stage, 256 MiB buffers and 128 MiB binding size. The public web3dsurvey figures in the review (accessed 2026-11-09, no collection date shown) show most adapters offer far more: at least 10 storage buffers on 98%, at least 16 on 80%, `maxBufferSize` of 2 GiB or more on 77%, 512 invocations per workgroup on 93%. Chrome 146 exposes `maxStorageBuffersPerShaderStage` up to 16.

Proposal: `createDevice` on WebGPU takes a limits policy (for example `requiredLimits: 'adapter-max'` or a list of names) that requests the adapter's value for each named limit, and reports the granted values in `device.limits`. Kernels then tier by `device.limits`, as the contributor utilities already validate (`validateRequirement`).

Measured here: the headless test device (`getWebGPUTestDevice()`, Chromium, Apple GPU) reports `maxStorageBufferBindingSize` and `maxBufferSize` of 4,294,967,292, `maxStorageBuffersPerShaderStage` 10 and `maxComputeInvocationsPerWorkgroup` 1,024.

## 2. Subgroups with a fallback

`subgroups` is available on about 68% of reports (Chrome 134 and later; no Firefox or Safari support found through Safari 27.2 and Firefox 162). Published `GPUReduction` already chooses a `subgroups` or `portable` strategy (`getGPUReductionStrategy`). Proposal: do the same for `GPUScan` and `GPUSort`, with the portable path as the tested default and the subgroup path selected only when the feature and the WGSL language extension are both present. Gridwise reports subgroup emulation 11 to 28 times slower on an M4, so the fallback should be a hierarchical workgroup-memory scan rather than an emulation. Linebender notes put a portable radix sort near 1 G keys/s on an M1 Max and about 3 G/s with real subgroups on Metal. The test matrix doubles, which the proposal should accept explicitly.

## 3. buffer_view

`bufferView<T>()`, `bufferArrayView<T>()` and `bufferLength()` (Chrome 153 to 154) pack several logical arrays into one binding. That relieves storage-binding count pressure for kernels such as geographically weighted regression. Chrome only for now, so it needs the same fallback discipline as subgroups. Suggest waiting for a second browser before API work.

## Repository side

The review's PLAT-05 reads "luma.gl or host", and this repository's only host-side device creation already does the host half: `modules/lenses/src/computation-views/hosts/device-profile.ts` requests up to 16 storage buffers per stage and the adapter's maximum `maxStorageBufferBindingSize` and `maxBufferSize` through `getAnalysisDeviceLimits`. The review names no other repository change under PLAT-05, so none was made. The related repository item is PLAT-02 (guard storage-binding size in the single-view temporal and cell contributors), which is separate and not done here.

## Retired inference

The landscape report guessed that the "about 4.2M rows" limit is the 65,535 by 64 one-dimensional dispatch product. PLAT-01 shows the published primitives are correct to 33.5M rows, so that limit does not apply to them.
