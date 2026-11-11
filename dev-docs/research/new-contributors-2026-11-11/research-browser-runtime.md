# Small neural networks on browser WebGPU: limits, runtimes and a size budget

Research date 2026-11-11. Every external fact carries a source and the date of that source (or "undated" where the page gives none). Figures are labelled **measured** (someone ran it and published the number), **local** (this repository's probe, `probe-results.json` in this folder, run 2026-11-11 on an Apple M3 Pro with 18 GPU cores in headless Chromium) or **estimate** (my arithmetic, shown).

## Summary

- WebGPU is on by default in Chrome and Edge on desktop and Android 12+, Safari 26 on macOS and iOS, and Firefox on Windows and macOS. Firefox on Linux and Android is not shipped. WebNN is not shipped anywhere stable.
- The default device limits (128 MiB per storage binding, 256 MiB per buffer, 8 storage buffers per stage, 16 KiB workgroup memory) are enough for the small models this library should carry. Most desktop adapters allow far more if the device asks. Android is the long tail.
- `shader-f16` is close to universal on Apple and recent Android GPUs. `subgroups` is Chrome-only. Packed int8 dot products exist in WGSL but have no cross-browser shipping record that I could verify.
- ONNX Runtime Web 1.31.1 (2026-10-26) **does** accept an externally created `GPUDevice` and GPU-buffer tensors. luma.gl 10.0.0-alpha.12 can also `attach()` to an external `GPUDevice`. Sharing the device both ways is documented in the APIs. I did not run it end to end.
- Plain WGSL layers on the test machine reach 90 to 610 billion multiply-accumulates per second (local). The low end is a naive multi-channel convolution, the high end a fused per-point MLP. Published browser results put integrated GPUs at roughly 0.1 to 1.4 trillion per second (estimate, below). For planning across devices, assume **0.1 trillion multiply-accumulates per second**.
- **Recommendation: the hybrid (c).** Write per-pixel and per-point networks, and anything trained in the browser, as WGSL contributors. Put the generic layers (multi-channel convolution, matmul with bias and activation, normalisation) up as luma.gl pull requests. Keep ONNX Runtime Web out of `@luma-contributors/analysis`. If larger pretrained encoders are wanted, wrap it in a separate optional package and pass it the luma device.

## 1. WebGPU limits that bound model size

### Which browsers ship WebGPU (gpuweb implementation-status wiki, last edited 2026-10-02)

| Browser | Shipped by default | Not shipped |
|---|---|---|
| Chrome / Edge | macOS, Windows x64, ChromeOS since 113; Android 12+ on ARM/Qualcomm/Intel since 121, Imagination on Android 16+ since 139; Linux Intel Gen12+ since 144, NVIDIA on Wayland since 147 | Windows ARM64 (flag), other Linux setups |
| Safari 26 | macOS, iOS, iPadOS, visionOS | — |
| Firefox | Windows since 141 (2025-07-15); macOS 26+ since 145, all macOS since 147 | Linux (Nightly only), Android (flag) |

Source: https://github.com/gpuweb/gpuweb/wiki/Implementation-Status

**WebNN** is not shipped in any stable browser. Chrome's origin trial was planned for M146 to M148. It was postponed, restarted in M147, then disabled again over release-blocking issues, and Android was excluded on 2026-02-13 (blink-dev thread, December 2025 to March 2026: https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98). Edge's trial page shows it closed, expiring 2026-08-11 (https://developer.microsoft.com/microsoft-edge/origin-trials/trials/19857284-cf52-484b-8b09-8ca50ac9dccb). The W3C specification reached Candidate Recommendation Draft on 2026-08-13 (https://www.w3.org/groups/wg/webmachinelearning/activity/2026/). I found no Intent to Ship.

### Default limits and what adapters give

| Limit | Spec default (MDN, modified 2026-11-02) | Test machine adapter, `featureLevel: 'max'` (local) | Field data (web3dsurvey, undated) |
|---|---|---|---|
| `maxStorageBufferBindingSize` | 128 MiB | 4 GiB − 4 | ≥1 GB: Windows 100%, macOS 80–85%, iOS 82%, Android 40% |
| `maxBufferSize` | 256 MiB | 4 GiB − 4 | — |
| `maxStorageBuffersPerShaderStage` | 8 | 10 | — |
| `maxComputeInvocationsPerWorkgroup` | 256 | — | — |
| `maxComputeWorkgroupSizeX/Y/Z` | 256 / 256 / 64 | — | — |
| `maxComputeWorkgroupStorageSize` | 16 KiB | — | — |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 | — |
| `maxUniformBufferBindingSize` | 64 KiB | — | — |
| `maxTextureDimension2D` / `3D` | 8,192 / 2,048 (W3C CRD 2026-09-15) | — | — |

Sources: https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits, https://www.w3.org/TR/webgpu/#limits, https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize. Chrome's January 2025 notes said 4 GiB `maxBufferSize` was then possible only behind the "Unsafe WebGPU Support" flag (https://developer.chrome.com/blog/new-in-webgpu-133). The local probe shows the Apple adapter now offers 4 GiB without a flag. A device gets the defaults unless it requests more. luma.gl `DeviceProps` offers `featureLevel: 'max'` and `requiredLimits` for that (`node_modules/@luma.gl/core/dist/api/adapter/device.d.ts`).

**What this means for model size.** A 2M-parameter float32 model is 8 MB and fits in one binding many times over. Activations are the real constraint. A 512 × 512 × 64-channel float32 feature map is 64 MiB, half the default binding. At the defaults, a contributor should either tile above roughly 1 MiP × 32 channels or request the adapter's limits. The 8-storage-buffer default limits fusion. Input, weights, bias, output, plus a scale or normalisation buffer is already five or six bindings. Packing weights and biases into one buffer per layer keeps a fused two-layer kernel under 8.

**Total memory on integrated GPUs and phones.** Unified-memory devices have no separate VRAM figure, and WebGPU does not expose a memory budget. I found no authoritative per-device numbers for browser GPU memory caps, so this is **unverified**. The Android binding-size spread above (only 40% at ≥1 GB) is the best available proxy.

### Optional features

| Feature | Status (source, date) |
|---|---|
| `shader-f16` | Shipped in Chrome 120. Field support: iOS 100%, macOS 99.8% (Apple 100%), Windows 89.8%, Android 88.3% (Qualcomm Adreno 6xx only 31%, 7xx 97.5%), Linux 73% (NVIDIA 6.6%). web3dsurvey, undated: https://web3dsurvey.com/webgpu/features/shader-f16/platform/iOS, …/Android, …/Windows. Present on the test machine (local). Firefox status not verified. |
| `subgroups` | Chrome announced it in 134 (2025-02-26, https://developer.chrome.com/blog/new-in-webgpu-134). caniuse lists Chrome/Edge 144+ as supported, Safari and Firefox not supported in any version (https://caniuse.com/wf-webgpu-subgroups, data October 2026). iOS field support 0.02% (web3dsurvey). Present on the test machine in Chromium (local, subgroup size 32). |
| `packed_4x8_integer_dot_product` (`dot4I8Packed`, `dot4U8Packed`) | A WGSL language feature, not a device feature. Shipped in Chrome 123 (https://developer.chrome.com/blog/new-in-webgpu-123, 2024). Safari and Firefox versions **not verified**. The MDN-derived compatibility data I found was unreadable. |
| Cooperative / subgroup matrix | ONNX Runtime has a `subgroup_matrix_conv` kernel (repository listing, 2026-11), which implies a Chromium experimental feature. I found no public shipping or origin-trial record. **Unverified**. |

luma.gl 10.0.0-alpha.12 knows `shader-f16` and `subgroups` as device features and has subgroup helpers (`gpu-subgroup-utils.d.ts`). None of its dense operations has an f16 or packed-int8 path.

## 2. What published luma.gl already provides, and what a layer still needs

From `node_modules/@luma.gl/graph/dist/gpu-core/*.d.ts` (10.0.0-alpha.12):

| Operation | Accepts | Missing for neural layers |
|---|---|---|
| `GPUMatMul` | float32 row-major M×K · K×N, chunked views | batch dimension, transpose flags, bias, fused activation, f16, int8 |
| `GPUMatVec` | float32 row-major matrix · vector | the same |
| `GPUConvolution` | **single-channel** same-size centred 2D, float32, zero or wrap boundary, direct or FFT | input/output channels, stride, dilation, padding modes, bias, depthwise/grouped, batch |
| `GPUElementwise` | copy/add/subtract/multiply/multiply-add/min/max over equal-length vectors | broadcast (bias per channel), nonlinearities. `GPUVectorScalarMADD` broadcasts one scalar; `GPUExpressionOperation` can express per-element formulas |
| `GPUTranspose` | 16×16 tiled 2D transpose, u32/i32/f32 | — |
| `GPUKMeans`, `GPUSimilaritySearch`, `GPUIVFFlatIndex` | float32 embedding matrices | — (useful after an encoder) |

Not present at all: softmax, layer, batch or group normalisation, attention, pooling, upsampling, any backward pass. Each would be a generic layer. Under the upstream-first rule (AGENTS.md) these belong in luma.gl, not here.

## 3. Runtimes

Package and asset sizes come from the npm registry and jsDelivr on 2026-11-11. The ONNX Runtime wasm compression figures are my own `gzip -9` and `brotli` runs on the published files.

| Runtime | Latest (date) | Licence | Download | Accepts external GPUDevice | GPU-resident I/O | Training | Notes |
|---|---|---|---|---|---|---|---|
| ONNX Runtime Web | 1.31.1 (2026-10-26) | MIT | `ort.webgpu.bundle.min.mjs` 122 KB plus wasm: asyncify 28.7 MB raw / 7.1 MB gzip / 4.2 MB brotli; JSPI 17.9 / 4.4 / 2.8 MB | **Yes**: `executionProviders: [{name: 'webgpu', device}]` (`WebGpuExecutionProviderOption.device`, onnxruntime-common 1.31.1), or set `env.webgpu.device` before the first session | Yes: `Tensor.fromGpuBuffer`, `preferredOutputLocation: 'gpu-buffer'`, pre-allocated outputs | No (inference package) | 1.29 deprecated JSEP and WebGL; native WebGPU EP recommended (https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.29.0). Graph capture for static shapes. |
| Transformers.js | 4.3.2 (2026-10-27) | Apache-2.0 | 456 KB min plus ORT Web | Through ORT | Through ORT | No | Pins onnxruntime-web 1.31.0-dev |
| TensorFlow.js WebGPU | 4.22.0 (**2024-10-21**, no release since) | Apache-2.0 | core 37 MB + backend 18 MB unpacked | Yes: `new WebGPUBackend(device)` | Yes: `createTensorFromGPUData`, `readToGPU` | Yes (autodiff) | Has `conv2d_mm`, depthwise and backprop kernels; FP16/DP4A "will be brought in soon" (README) |
| LiteRT.js | `@litertjs/core` 2.6.0 (2026-10-22) | Apache-2.0 | about 11 MB wasm per variant | Shares a device with TF.js through `getWebGpuDevice` (docs undated) | Via TF.js interop | No | int32 and float32 I/O only; `.tflite` models |
| WebLLM | 0.2.88 (2026-11-10) | Apache-2.0 | 14 MB unpacked | Not documented | — | No | LLM only |
| jax-js | `@jax-js/jax` 0.1.27 (2026-10-18) | MIT | 1.7 MB unpacked, zero dependencies | Not verified | Not verified | **Yes** (`grad`, `vmap`, `jit`, Adam/SGD in `@jax-js/optax`) | conv2d "slow" (author, 2025-12-18) |
| Burn (wgpu) | active, pushed 2026-11-11 | MIT / Apache-2.0 | Rust to wasm | Not verified | — | Yes | Brush trains Gaussian splats in the browser on Burn |
| tinygrad | active | MIT | Python; exports WebGPU code | — | — | Not in browser | — |
| wonnx | **archived**, last push 2024-07-21 | — | — | — | — | — | — |
| ratchet | pushed 2026-09-24 | MIT | `@ratchet-ml/ratchet-web` 0.4.0 (2024-04-29) | — | — | No | — |
| webgpu-torch | 0.3.5 (2023-05-11), last push 2024-07-25 | MIT | 1.3 MB | — | — | Partial autodiff | Dormant |

Sources: ORT tutorial https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html (undated); `env.ts` doc comment https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/common/lib/env.ts ("Set a value before the first WebGPU inference session is created"); ORT 1.31.1 `lib/wasm/session-options.ts` on jsDelivr (`webgpuRegisterDevice(customDevice)`); TF.js https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/backend_webgpu.d.ts; LiteRT.js https://developers.google.com/edge/litert/web/get_started; jax-js https://ss.ekzhang.com/p/jax-js-an-ml-library-for-the-web (2025-12-18); repository states from the GitHub API on 2026-11-11.

**ONNX Runtime Web device sharing, in detail.** The older JSEP path built its own device and requested the adapter's maximum limits and `shader-f16` and `subgroups` when available (`lib/wasm/jsep/backend-webgpu.ts`, 1.31.1). The native EP path takes the user's device through `webgpuRegisterDevice`. The JS layer does not check the external device's limits or features. Whether the native EP degrades or fails on a default-limits device is **not verified**. The safe pairing is to create the luma device with `featureLevel: 'max'` and hand `device.handle` to ORT. Alternatively, let ORT create the device and call luma's `WebGPUAdapter.attach(gpuDevice)`, which is declared in `@luma.gl/core/dist/webgpu/adapter/webgpu-adapter.d.ts` and does not destroy a device it did not create. Constraints from the ORT tutorial: GPU tensors need buffers with `STORAGE | COPY_SRC | COPY_DST` usage and 16-byte-aligned size; zero-sized tensors are always CPU tensors; a destroyed buffer fails the run; a wrong-shaped pre-allocated output fails the run.

**ONNX Runtime WebGPU operator coverage.** The (JSEP-era) operator table lists about 108 operators, including Conv, ConvTranspose, MatMul, Gemm, the batch, instance and layer norms, Softmax, Attention, MultiHeadAttention, Resize, the pools, Gelu, Sigmoid and Relu. Its notes include "need perf optimization" for Conv and pooling, "conv3d is not supported" and Attention lacking masks (https://raw.githubusercontent.com/microsoft/onnxruntime/main/js/web/docs/webgpu-operators.md, undated). The native EP source has `conv2d_mm`, `im2col_matmul`, `grouped_conv`, `subgroup_matrix_conv`, layer, RMS, instance and batch norm, and pooling (https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/webgpu/nn, 2026-11). That covers small CNNs and ViTs.

## 4. Measured performance and a throughput figure

### Published measurements

| Workload | Device, browser | Result | Source (date) |
|---|---|---|---|
| WebSR 2× super-resolution, 7 conv layers, ~10k / ~30k / ~100k params | Apple M4 MacBook Pro, WebGPU, "720p" | 120 / 40 / 15 fps | https://free.upscaler.video/technical/architecture/ (2025-01-20, stats to 2026-01) |
| Same, ~10k params | $200 Chromebook, integrated | 40 fps | same |
| Same, ~10k params | recent Android/iOS | 15–30 fps | same |
| all-MiniLM-L6-v2, sequence 512, batch 1 / 32 | M1 MacBook Air, Chrome 129, WebGPU fp16 | 38 ms / 840 ms (fp32 47 / 1,592 ms; WASM about 382 ms at batch 1) | https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/104 (2024-09-30) |
| float32 4096² matmul | M4 Pro, jax-js | ~3 TFLOP/s | jax-js post (2025-12-18) |
| MobileCLIP text encoder, jit | M1 Pro, jax-js | ~500 GFLOP/s | same |
| Per-dispatch overhead, sequential | Safari M2 31.7 µs; Chrome 32.8–66.5 µs; Firefox ~1,040 µs (all vendors, "consistent with rate-limiting") | — | https://arxiv.org/html/2604.02344v1 (2026-02-09) |
| Fusing a tiled MLP from 7 to 3 dispatches | Metal | 2.01× | same |
| Segment Anything on ORT Web | RTX 3060 laptop | WebGPU 19× faster than WASM (encoder), 3.8× (decoder); no absolute ms | https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/ (2024-02-29) |

**Not found:** credible published millisecond figures for MobileNet, small U-Nets, ViT-tiny, Whisper tiny or SAM decoders on WebGPU with a named device and browser. Blog tables I found (SitePoint, Medium) do not name framework or hardware, so I left them out. Anime4K-WebGPU publishes no fps figures.

### Local measurements (this repository, 2026-11-11, M3 Pro, headless Chromium, float32, naive kernels with one invocation per output or per row)

| Case | Billion MACs | Median ms | Billion MAC/s | Compile ms |
|---|---|---|---|---|
| Fused per-point MLP 32→64→64→1, 1M points | 6.51 | 10.6 | 614 | 36 |
| Fused per-point MLP 16→32→32→1, 4M points | 6.58 | 11.9 | 553 | 29 |
| Dense 64→64, 262k rows, one invocation per row | 1.07 | 2.2 | 488 | 13 |
| Dense 64→64, one invocation per output | 1.07 | 8.8 | 122 | 14 |
| 3×3 conv 32→32 channels, 512² | 2.42 | 24.8 | 97 | 20 |
| 3×3 conv 8→8 channels, 1024² | 0.60 | 6.8 | 89 | 21 |

Maximum relative error against a CPU reference was ≤ 6.3e-7. In-browser MLP training (24→64→64→1, 5,825 parameters, 65,536 samples) ran at 3.98 ms per step. Relative test error fell from 2.75 to 0.098 in 4.0 s (1,000 steps) and 0.061 in 8.0 s. The step time scales linearly: 14.6 ms per step at 262,144 samples.

### Derived throughput (estimates; arithmetic shown)

For a conv-only network at constant resolution, MACs per input pixel ≈ parameter count.

- **WebSR, assuming "720p" means a 1280×720 input (921,600 px):** 10k × 0.92 M = 9.2 GMAC per frame × 120 fps = **1.1 TMAC/s**; 30k: 27.6 G × 40 = 1.1; 100k: 92 G × 15 = 1.4 (M4). Chromebook: 9.2 G × 40 = 0.37. Phones: 0.14–0.28. If "720p" is the *output*, divide all of these by 4 (M4 ≈ 0.28–0.35 TMAC/s). The page does not say which, and its parameter counts are approximate.
- **MiniLM-L6 on M1 Air:** 6 layers × (4·384² + 2·384·1536) = 10.6 M MAC per token × 512 = 5.4 G, plus attention 6 × 2 × 512² × 384 = 1.2 G, about 6.6 GMAC per sequence. Batch 1 fp16: 6.6 / 0.038 s = **0.17 TMAC/s**. Batch 32 fp16: 213 G / 0.84 s = 0.25. fp32: 0.13–0.14. Apple quoted the M1 GPU at 2.6 TFLOPS (https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/, 2020-11-10), i.e. 1.3 TMAC/s, so this is 10–20% of peak.
- **jax-js matmul on M4 Pro:** 3 TFLOP/s = 1.5 TMAC/s for one large matmul; 0.25 TMAC/s end to end for a transformer.

**Spread:** about 0.04 to 1.4 TMAC/s across integrated GPUs, a 35× range. A generic graph runtime on a laptop lands at 0.13–0.25. Hand-tuned conv on an M4 or fused per-point MLPs on an M3 Pro land at 0.5–1.4. Phones and cheap Chromebooks land at 0.04–0.37. Naive multi-channel convolution sits at 0.09–0.1 even on the fast test GPU (local). **Planning figure: 0.1 TMAC/s** on the weakest supported device class. That is also what a naive convolution achieves on the strong one.

## 5. How hand-written WGSL layers are organised

- **Convolution.** ONNX Runtime and TF.js both use implicit-GEMM convolution (`conv2d_mm`), with `im2col` plus matmul as an alternative, a separate depthwise kernel, and in ORT a subgroup-matrix kernel. TF.js additionally has `depthwise_conv2d_vec4` and an NCHW shared-memory variant. Neither has a Winograd kernel in its file listing (repository listings, 2026-11). WebSR packs 4 channels per RGBA texel and stores weights as `mat4` uniforms mapping 4 input to 4 output channels. It keeps channel counts multiples of 4 and uses conv2d only (2025-01-20 page). The local probe's naive conv reaches only 15–20% of the fused MLP's throughput, so tiling over output channels with workgroup memory is the first optimisation that matters.
- **Dispatch count matters as much as arithmetic.** At 30–70 µs per sequential dispatch in Chrome and Safari, and about 1 ms in Firefox (arXiv 2604.02344, 2026-02-09), a 20-layer network at 3 dispatches per layer costs 2–4 ms in Chrome but about 60 ms in Firefox before any arithmetic. Fusing a whole per-point MLP into one kernel (the local 614 GMAC/s case) avoids this, and so does fusing bias, activation and normalisation into the conv kernel.
- **Bindings.** The 8-storage-buffer default limits a fused kernel. WebSR sidesteps it with uniforms. A buffer-based design should pack each layer's weights and bias into one buffer and keep all layers' weights in one buffer with offsets.
- **Compile time.** Local compile times are 9–36 ms per inference kernel and 61–99 ms for a training kernel. A Mozilla bug records about 15 s for one FXC compile on Windows (https://bugzilla.mozilla.org/show_bug.cgi?id=1880415, 2024). Cold and warm compile times differ greatly (https://hugodaniel.com/posts/webgpu-diagnostics/, 2025-10-06, no numbers published). Use `createComputePipelineAsync` and specialise shaders per architecture, not per input size.
- **Precision.** fp16 gave 1.2–1.9× over fp32 in the MiniLM runs (M1, 2024-09-30). The contributor contract's Metal fast-math finding applies to every reduction in a layer. Accumulate in f32 even when storing f16, as ORT's `enableMatmulFp32Accumulation` option does for 4-bit matmuls. Brush notes WebGPU has no float atomics and uses a compare-and-swap loop (https://github.com/ArthurBrussee/brush, 2026). luma's `GPUKMeans` avoids float atomics by ordered per-component accumulation, which is the pattern a backward pass should copy.
- **Quantization in practice.** ORT ships `MatMulNBits` (4-bit weights) and WebLLM runs q4f16 models. For CNNs, 8-bit dequantise-in-shader is the realistic option. `dot4I8Packed` is Chrome-only as far as I could verify.

## 6. Weight delivery

- **Formats.** ONNX for ORT, `.tflite` for LiteRT.js, safetensors or raw little-endian float arrays for hand-written WGSL. Raw arrays need no parser. WebSR ships weights as JSON (2025-01-20 page).
- **Compression that holds accuracy.** LiteRT's table (undated) gives top-1 accuracy, float → post-training int8 → quantization-aware training: MobileNet v1 0.709 → 0.657 → 0.70; MobileNet v2 0.719 → 0.637 → 0.709; Inception v3 0.78 → 0.772 → 0.775; ResNet v2 101 0.770 → 0.768. Sizes shrink about 4×, for example MobileNet v2 from 14 to 3.6 MB (https://developers.google.com/edge/litert/models/model_optimization). In short, **f16 (2×) is safe**. **int8 (4×) is safe for large CNNs but costs small depthwise networks 5–8 points without quantization-aware training.** 4-bit is established for transformer matmuls (ORT MatMulNBits, WebLLM q4f16), but I found no accuracy data for small CNNs at 4 bits.
- **Caching.** Chrome's guidance recommends the Cache API, with OPFS and IndexedDB as alternatives (https://developer.chrome.com/docs/ai/cache-models, undated). Quotas (MDN, 2026-01-05): Chrome up to 60% of disk per origin; Firefox best-effort min(10% of disk, 10 GiB per site); Safari about 60% of disk for browser apps, but **Safari deletes script-written storage after seven days without user interaction** when tracking prevention is on (https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria).
- **Acceptable first load.** web.dev treats about 5 MB as the threshold for "large" and cites DistilBERT at 67 MB (https://web.dev/articles/client-side-ai-performance, undated). The ORT WebGPU runtime alone is 2.8–4.2 MB brotli before any model (measured above).
- **Hosting.** The Hugging Face Hub serves model files cross-origin and Transformers.js loads from it by default. I did not verify the Hub's current CORS headers or rate limits. **Licensing applies here too.** `THIRD-PARTY-DATA.md` already restricts bundled datasets, and pretrained weights carry their own licences.

## 7. Training in the browser

| Precedent | Measured | Source (date) |
|---|---|---|
| This repository, MLP 24→64→64→1 (5,825 params), 65,536 samples, fused WGSL forward and backward | 3.98 ms per step; relative error 0.098 after 4 s | local, 2026-11-11 |
| jax-js MNIST | ">99% accuracy in seconds"; no time or architecture published | 2025-12-18 |
| Brush Gaussian-splat training on Burn/wgpu, in browser | runs in Chrome/Edge; author calls it a proof of concept; no throughput found | GitHub, 2026 |
| TF.js training | longstanding, but the WebGPU backend has had no release since 2024-10-21 | npm |

What breaks:

- **No autodiff.** In WGSL every backward kernel is hand-written. That is reasonable for dense and activation layers and costly for convolution (TF.js needed three backprop kernels).
- **No float atomics.** Gradient reduction needs ordered or fixed-point sums. That is the contract's existing rule.
- **Metal fast math.** It removes compensated summation (contributor contract), so loss sums over many samples need the contract's exact methods.
- **Compile cost per architecture.** 61–99 ms locally for a training kernel. Fine for one model per session, not for an architecture search.
- **Memory.** Row state was 70 MB at 65k samples and 281 MB at 262k (local). Mini-batching is needed beyond the default 128 MiB binding.

## 8. The decision

| | (a) WGSL contributors on luma graphs | (b) ONNX Runtime Web peer dependency | (c) Hybrid |
|---|---|---|---|
| New dependencies | none | onnxruntime-web, about 3–4 MB brotli wasm, a C++ runtime | (a) none; (b) only where chosen |
| Fits contributor contract | yes: graph nodes, construction-time validation, luma buffers | no: an opaque session, own buffer cache, async `run()`, not a command node | yes for (a) |
| Data stays on GPU | yes | yes, with shared device and `fromGpuBuffer`. Each run is a separate submit outside the luma graph | yes |
| Training | yes for small MLPs (measured) | no | (a) only |
| Operator coverage | must be written (convolution channels, norms, attention) | ~108 ops, ViTs and CNNs | both |
| Performance | fused MLP 0.55–0.61 TMAC/s locally; naive conv 0.09–0.1 | tuned kernels, subgroup matrix, f16, 4-bit | — |
| Determinism and testability | full control; float behaviour per contract | opaque kernels, its own precision choices | (a) under contract |
| Upstream-first | generic layers belong in luma.gl as pull requests; domain-named fused kernels are allowed as a last resort | AGENTS.md says everything non-geospatial comes from luma.gl. A third-party ML runtime is arguably outside that rule's spirit and needs an explicit maintainer decision | — |

**Recommendation: (c), with (b) kept outside `@luma-contributors/analysis`.**

1. **Per-pixel and per-point networks, and anything trained in the browser, use (a).** This means a fused-MLP contributor on raster embeddings or point features, a coordinate network and a small conv head. These are single-kernel or few-kernel workloads where (a) is already fast (local), deterministic and trainable.
2. **Generic layers go upstream.** Multi-channel conv2d with bias and activation, batched matmul with bias, softmax and normalisation should be proposed as luma.gl pull requests. Until they land, a contributor may carry one fused kernel named for its domain (for example, a raster pixel classifier). It should not carry a layer library.
3. **(b) only for large pretrained encoders** (SAM-class, ViT-S-class), and only in a separate optional package or an example. Create the luma device with `featureLevel: 'max'` and pass `device.handle` as the ORT WebGPU EP `device`, with GPU tensors in and out. Run a spike first to confirm the native EP works on a luma-created device and that outputs can be wrapped as luma buffers without copying.

### Size budgets (estimates from the 0.1 TMAC/s planning figure and the limits above)

| Tier | Path | Max parameters | Max weights (float32 / f16 / int8) | Max MACs per interactive run | Runtime target |
|---|---|---|---|---|---|
| 0: per-pixel / per-point, trainable | (a) | 100 k | 0.4 MB / 0.2 MB / — | 5 G (e.g. 1 M elements × 5 k MAC) | ≤ 50 ms at 0.1 TMAC/s; about 10 ms on the test GPU |
| 1: small CNN / U-Net head, pretrained | (a), conv upstream | 2 M | 8 MB / 4 MB / 2 MB | 10 G | ≤ 100 ms; not trained in browser |
| 2: pretrained encoder | (b), optional package | 25 M | 100 MB / 50 MB / 25 MB (ship f16 or int8 only) | 50 G per run, not per frame | ≤ 0.5 s; cached, explicit download consent |
| Out of scope | — | > 25 M | — | — | LLMs, SAM-H-class |

For a per-frame interactive operation (16 ms), divide the per-run MAC budget by 6. Keep every single storage binding under 128 MiB unless the contributor requests limits and validates them at construction. Keep dispatches per run in the tens, because Firefox charges about 1 ms each.

## What I could not verify

- Absolute WebGPU latencies for MobileNet, U-Net, ViT-tiny, Whisper tiny and SAM decoders with a named device and browser.
- Whether ORT's native WebGPU EP runs correctly on an external device created with default limits or without `shader-f16`.
- Safari and Firefox support for `packed_4x8_integer_dot_product`, and Firefox support for `shader-f16`.
- Browser GPU memory caps on phones and integrated GPUs.
- Whether WebSR's "720p" is input or output resolution, which changes the derived figure by 4×.
- The status of the cooperative or subgroup-matrix feature in Chrome.
- The current CORS and rate-limit policy of the Hugging Face Hub CDN.
