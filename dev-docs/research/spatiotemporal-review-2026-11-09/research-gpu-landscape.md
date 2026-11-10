# GPU and WebGPU landscape for geospatial and spatiotemporal computing (late 2026)

Research date: 2026-11-09. Every URL below was opened (fetched or returned by search with the cited excerpt) during this session. Items marked **UNVERIFIED** were not confirmed from a primary source in this session. The web-search budget ran out before sections 4 and 5 were fully covered, so several ecosystem items are marked **NOT RESEARCHED**.

---

## 1. WebGPU platform status 2025-2026

### 1.1 Browser availability

| Browser / platform | Status | Source (date) |
|---|---|---|
| Chrome/Edge: Windows, macOS, ChromeOS | Since Chrome 113 | gpuweb implementation-status wiki (edited 2026-10-02) https://github.com/gpuweb/gpuweb/wiki/implementation-status |
| Chrome Android | Android 12+ on ARM/Qualcomm/Intel GPUs since 121; Imagination on Android 16+ since 139; Samsung Xclipse "TBD" | same |
| Chrome Linux | Intel Gen12+ since 144 (Chrome blog 2026-01-07, https://developer.chrome.com/blog/new-in-webgpu-144); NVIDIA (driver 535.183.01+) under Wayland since 147 (status wiki; Chrome 147-148 post 2026-04-22 https://developer.chrome.com/blog/new-in-webgpu-147-148). AMD and others still behind a flag | wiki |
| Chrome Windows ARM64 | Behind a flag | wiki |
| Firefox Windows | Since 141 (July 2025) | wiki; https://simonwillison.net/2025/Jul/16/webgpu-firefox |
| Firefox macOS | macOS 26+ since 145; all macOS versions since 147 (released 2026-01-13) | wiki; https://alternativeto.net/news/2026/1/firefox-147-brings-apple-silicon-webgpu-support-improved-video-playback-and-pip-and-more |
| Firefox Linux | Nightly only; "expected 2026" | wiki (2026-10-02) |
| Firefox Android | Off by default; Beta/Nightly via about:config | wiki |
| Safari macOS / iOS / iPadOS / visionOS | Enabled by default in Safari 26 (macOS Tahoe 26, iOS 26), shipped Sept 2025 | wiki; web.dev 2025-11-25 https://web.dev/blog/webgpu-supported-major-browsers |
| Safari 27 | Released ~2026-09-14; WebGPU additions include WGSL clip distances (secondary source) | https://app.cinevva.com/news/2026-09-17-safari-27-model-element-webgpu-clip-planes (secondary; Apple changelog not opened, **UNVERIFIED** detail) |

Implication: WebGPU is now on every major desktop engine and on iOS. The remaining gaps are Firefox on Linux and Android, Chrome on Linux AMD, and parts of Android (Samsung Xclipse, Android below 12).

### 1.2 Real-world capability distribution (web3dsurvey, accessed 2026-11-09, no collection date shown)

https://web3dsurvey.com/webgpu reports a working WebGPU adapter in 81.5% of reports. Feature and limit shares:

| Feature | Share | | Limit | Distribution |
|---|---|---|---|---|
| timestamp-query | 98.9% | | maxStorageBuffersPerShaderStage | ≥8: 100%, ≥10: 98%, ≥16: 80%, ≥31: 19% |
| shader-f16 | 94.0% | | maxBufferSize | ≥256 MiB: 100%, ≥1.25 GB: 87%, ≥2 GiB: 77% |
| texture-formats-tier1 | 92.7% | | maxStorageBufferBindingSize | ≥128 MiB: 100%, ≥1.25 GB: 79%, ≥2 GiB: 16% |
| float32-filterable | 91.3% | | maxComputeWorkgroupStorageSize | ≥16 KiB: 100%, ≥32 KiB: 99%, ≥64 KiB: 5% |
| subgroups | 68.4% | | maxComputeInvocationsPerWorkgroup | ≥256: 100%, ≥512: 93% |
| subgroup-size-control | 58.9% | | | |

Spec defaults (W3C CRD 2026-09-15, https://www.w3.org/TR/webgpu/): maxStorageBuffersPerShaderStage 8, maxBufferSize 256 MiB, maxStorageBufferBindingSize 128 MiB, maxComputeWorkgroupStorageSize 16 KiB, maxComputeInvocationsPerWorkgroup 256, **maxComputeWorkgroupsPerDimension 65535**, maxImmediateSize 64 bytes. Compatibility-mode defaults lower invocations per workgroup to 128 and storage buffers in the vertex stage to 0.

Inference (mine): the library's "about 4.2M rows per pass" limit equals 65535 workgroups × 64 invocations (4,194,240). That is a 1-D dispatch artefact, not a hardware limit. A 2-D or 3-D dispatch grid with a linearised index removes it (see `linear_indexing` below).

Chrome 146 (2026-02-25, https://developer.chrome.com/blog/new-in-webgpu-146) added Dawn limit tiers. `maxStorageBuffersPerShaderStage` is now exposed up to 16.

### 1.3 Optional features and WGSL extensions (Chrome "What's New in WebGPU" series)

Index: https://developer.chrome.com/docs/web-platform/webgpu/news (latest entry Chrome 155-156).

| Feature | Shipped in | Date / URL | Other engines |
|---|---|---|---|
| shader-f16 | Chrome 120 (f16 on Linux NVIDIA in 155-156) | https://developer.chrome.com/blog/new-in-webgpu-120 ; https://developer.chrome.com/blog/new-in-webgpu-155-156 | 94% of reports have it |
| timestamp-query (inside passes) | Chrome 121 | news index | 99% |
| **subgroups** | Chrome 134 ("now available") | 2025-02-26 https://developer.chrome.com/blog/new-in-webgpu-134 | caniuse lists **no Firefox or Safari support** through Safari 27.2 / Firefox 162: https://caniuse.com/wf-webgpu-subgroups |
| subgroup_id built-ins | Chrome 144 | 2026-01-07 (blog 144) | Chrome |
| uniform_buffer_standard_layout | Chrome 144 | same | Chrome |
| subgroup_uniformity | Chrome 145 | 2026-01-28 https://developer.chrome.com/blog/new-in-webgpu-145 | Chrome |
| `mapSync()` on GPUBuffer in workers (experimental, flag `WebGPUMapSyncOnWorkers`) | Chrome 145, flag only | same | not standardised |
| compatibility mode (featureLevel "compatibility", OpenGL ES 3.1, opt-in) | Chrome 146 | 2026-02-25 (blog 146) | spec has compat limits |
| linear_indexing (`global_invocation_index`, `workgroup_index`) | Chrome 147-148 | 2026-04-22 (blog 147-148) | Chrome |
| **immediates** (push constants, `var<immediate>`, `setImmediates()`, 64 bytes) | Chrome 149-150 | 2026-06-17 https://developer.chrome.com/blog/new-in-webgpu-149-150 | in spec (maxImmediateSize) |
| subgroup-size-control (`@subgroup_size`) | Chrome 151-152 | 2026-08-12 https://developer.chrome.com/blog/new-in-webgpu-151-152 | 59% of reports |
| **buffer_view** (`bufferView<T>()`, `bufferArrayView<T>()`, `bufferLength()`: several logical arrays in one binding) | Chrome 153-154 | 2026-09-15 https://developer.chrome.com/blog/new-in-webgpu-153-154 | proposal in gpuweb |
| swizzle_assignment | Chrome 153-154 | same | |
| Dawn experimental `BufferMapWriteExtendedUsages` (map-write storage buffers, no staging copy) | Dawn only | same | native only |
| texture-formats-tier1/tier2 | Chrome 142 | 2025-10-22 https://developer.chrome.com/blog/new-in-webgpu-142 | 93% tier1 |
| fragment_depth, texture-compression-unaligned | Chrome 155-156 (fragment_depth temporarily disabled) | blog 155-156 | rendering only |
| experimental multi-draw indirect | Chrome 131 (experimental) | news index | **UNVERIFIED** whether shipped since |
| Bindless | "available for early experimentation on Android, Linux, Windows" (155-156 post). gpuweb `GPUResourceTable` (#5464) and bindless WGSL issues (#5582) | blog 155-156; https://www.w3.org/groups/wg/gpu/activity/2026/ | not shipped |
| Device requests consume adapter; lost device returned by `requestDevice()` | Chrome 140 / 116 | news index | spec behaviour |

**64-bit:** WGSL has no `i64`, `u64` or `f64` type, and I found no active proposal for f64. The gpuweb `proposals/` directory (https://github.com/gpuweb/gpuweb/tree/main/proposals) holds 26 proposals. The only 64-bit one is `atomic-64-min-max.md`, which is **merged** (created 2025-09-15). It defines feature `atomic-vec2u-min-max`: `atomic<vec2u>` maps to u64, with `atomicStoreMin` / `atomicStoreMax` (no return value) on storage buffers. Hardware: Metal Apple9 family, Vulkan `shaderBufferInt64Atomics` (about 88% of Linux, about 31% of Android), D3D12 SM 6.6. The feature name appears in the W3C spec's `GPUFeatureName` list (CRD 2026-09-15). **UNVERIFIED:** which browser releases expose it by default. Dawn landed vec2u 64-bit atomics for Metal, Vulkan and D3D12 (https://dawn.googlesource.com/dawn/+/e2f0828f7c6919dba0521a47c19fc44df88dbadc).

Other draft proposals relevant to compute (all **not shipped**):
- `texel-buffers.md`: draft, 2024-10-07; the group decided to make it core. https://github.com/gpuweb/gpuweb/blob/main/proposals/texel-buffers.md
- `subgroup-matrix.md`: draft, 2025-10-02; cooperative-matrix multiply for f16/f32/i8; based on a Chromium experimental feature. https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md
- `sized-binding-arrays.md`, `bindless.md`.

**Ray tracing / ray query:** the proposals directory has no ray-tracing proposal. RT cores, which drive the 2024-2026 spatial-join literature (section 3), are therefore not reachable from WebGPU.

**Workers and device loss:** WebGPU works in dedicated, shared and service workers (Chrome 124). Device loss is reported through `device.lost`, and `requestDevice()` returns an already-lost device on failure (Chrome 116). There is still one adapter/device model per page or worker; `mapAsync` stays asynchronous outside the experimental worker `mapSync`.

### 1.4 What is not available
- No f64 and no i64 arithmetic.
- No GPU-side forward-progress guarantee (relevant for single-pass scans).
- No ray queries.
- No bindless in release builds.
- No read-write aliasing of one buffer in a pass (WGLog paper below).
- Readback latency: WGLog measured about **2.5 ms for a 4-byte mapAsync readback**, about 285× a CUDA 4-byte copy, and 0.18 ms for an empty fence (arXiv 2607.17571, 2026-07-20).

---

## 2. WebNN, ONNX Runtime Web, transformers.js

- **WebNN:** Chrome origin trial planned for M146-M148, delayed, re-enabled for M147-M149 on 2026-03-10, then **disabled again 2026-03-11** for release-blocking issues; Android excluded. Sources: https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98 and https://developer.chrome.com/blog/chrome-146-beta. Edge's WebNN trial expired 2026-08-11 (https://developer.microsoft.com/microsoft-edge/origin-trials/trials/19857284-cf52-484b-8b09-8ca50ac9dccb). **UNVERIFIED:** any status after March 2026. Treat WebNN as not shippable.
- **transformers.js v4:** announced 2026-02-09 (https://huggingface.co/blog/transformersjs-v4). It has a new WebGPU runtime rewritten in C++ with the ONNX Runtime team, about 4× faster BERT-style embeddings from the MultiHeadAttention op, and runs on Node, Bun and Deno with WebGPU. Secondary sources say 4.2.0 was the latest stable on 2026-04-23 (**UNVERIFIED** current version).
- **ONNX Runtime Web WebGPU EP:** `ort.Tensor.fromGpuBuffer()` wraps an existing `GPUBuffer` as model input, and `preferredOutputLocation: 'gpu-buffer'` keeps outputs on the GPU (https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html). The documented device is ORT's own (`ort.env.webgpu.device`). Injecting your own device is **UNVERIFIED**. GPUBuffers cannot cross devices, so the analysis library must allocate its interchange buffers on ORT's device or copy through the CPU.
- **Practicality:** running small embedding or similarity models next to analysis kernels is practical. Embedding throughput is good, and the zero-copy path works if both share ORT's `GPUDevice`. The remaining friction is device ownership, since only one runtime can own the device.

---

## 3. Research literature 2022-2026 (spatial indexing, joins, primitives)

### 3.1 Ray-tracing-core spatial databases (the dominant 2024-2026 thread)
- **RayJoin** (ICS 2024): LSI and PIP via RT cores; 1.6M × 303K polygons joined in about 460 ms. https://github.com/pwrliang/RayJoin
- **LibRTS** (PPoPP 2025): up to 85× for point queries, 94× for range-contains and 11× for range-intersects versus CPU and GPU libraries. Introduces "Ray Multicast" static load balancing: split N primitives into k subspaces and cast k rays per query. https://ppopp25.sigplan.org/details/PPoPP-2025-Main-Conference-1/2/LibRTS-A-Spatial-Indexing-Library-by-Ray-Tracing
- **X-HD** (ICS 2026): Hausdorff distance computed with RT cores. A uniform grid groups points into cell MBRs, upper and lower bounds prune whole cells, and heavy cells go to a CUDA kernel.
- **RayBooster** (PVLDB 19, 2026): RT engine inside SedonaDB, using DE-9IM through one `RelateEngine` and a single monolithic BVH with feature IDs encoded on Z. Up to 5.93× and 59% lower cloud cost. Source for these three: PVLDB tutorial "Hardware Acceleration for Spatial Databases" (vol. 19, 2026) https://www.vldb.org/pvldb/vol19/p4862-lee.pdf
- **Pierce** (arXiv 2609.24513, 2026-09-21): RT-core 3D polyhedral joins, more than 100× on pathology data. https://arxiv.org/abs/2609.24513
- **3DPipe** (arXiv 2604.19982, 2026): pipelined GPU 3D join. Seen in search only, **UNVERIFIED** detail.
- **Relevance:** WebGPU cannot use RT cores. The portable lessons are the load-balancing (multicast and heavy-cell offload), cell-MBR pruning and padded-AABB plus exact-refine designs, which transfer to compute-shader BVH and grid joins.

### 3.2 Filter-and-refine for polygons (non-RT)
- GCMF / PolySketch++ (2025): adaptive hierarchical GPU filters cut the filter workload by 92% on average; more than 600K polygons joined in under 8 s, with an upgraded version reported under 1 s. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5304748 and https://www.researchgate.net/publication/393627681_GPU-based_Filter_and_Refine_Algorithms_for_Polygon_Intersection
- GPU Foster/Hormann-style clipping with degenerate cases (CCGrid 2023): up to 40×, and filters remove 80-99% of edge pairs. https://www.aminer.org/pub/64ac88f83fda6d7f0688cc96/
- GPU polygon reconstruction for Boolean operations with hash tables and memory pools (Appl. Sci. 2025, 15(3):1111). https://mdpi.com/2076-3417/15/3/1111

### 3.3 Trees, kNN, clustering
- **ArborX 2.0** (released 2025-04-16; ACM TOMS 51(4) 2025): BVH with API v2, brute force, ray tracing, DistributedTree, MST/EMST, DBSCAN, HDBSCAN*, moving least squares. https://arxiv.org/abs/2507.23700
- **FDBSCAN / FDBSCAN-DenseBox** fuse neighbour search with union-find; DenseBox uses grid cells to skip distance tests in dense regions. https://arxiv.org/pdf/2103.05162 and "Advances in ArborX" (IJHPCA, Jan 2025) https://impact.ornl.gov/en/publications/advances-in-arborx-to-support-exascale-applications/
- **cuML HDBSCAN** is GPU end-to-end. https://developer.nvidia.com/blog/gpu-accelerated-hierarchical-dbscan-with-rapids-cuml-lets-get-back-to-the-future
- **LBVH on WebGPU:** only community implementations (Babylon.js forum), with a warning that atomics-based builds may misbehave on Apple. https://forum.babylonjs.com/t/i-am-writing-webgpu-ray-tracer-is-anyone-interested-in-this-topic/42188. A search snippet says luma.gl has an experimental GPU BVH pass sized to the 8 core storage bindings (https://luma.gl/next/docs/api-reference/experimental/gpu-primitives/gpu-bvh); the page returned 404 when fetched, **UNVERIFIED**.

### 3.4 Delaunay
- gCDT (draft dated April 2026, venue **UNVERIFIED**): GPU constrained Delaunay about 25× faster than gDel2D-based CDT, with constraint insertion about 1000× faster. https://min-tang.github.io/home/gCDT/files/gcdt.pdf
- No 2024-2026 successor to gDel2D for unconstrained 2D Delaunay was found.

### 3.5 Graphs
- **ADDS** (PPoPP 2021) remains the reference GPU SSSP: 2.9× over Near-Far on road networks. https://www.cs.utexas.edu/~lin/papers/ppopp21.pdf. No 2025-2026 successor was found.
- **Connected components:** ECL-CC (intermediate pointer jumping; https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/) and Afforest with sampling. GConn reports 2.8× over ECL-CC without sampling and 10× with sampling. https://arxiv.org/pdf/2008.11839
- **WGLog** (arXiv 2607.17571, 2026-07-20) is the most relevant paper found. It is a WebGPU-only engine for recursive (Datalog) queries:
  - **Atomic-free sorted-array joins:** binary search into a sorted base relation, count then emit into contiguous slabs, fused dedup and difference, Merge Path merge, 8-bit radix sort.
  - **Indirect dispatch for device-side sizing:** a small kernel writes the next stage's dispatch arguments.
  - **Batching:** K iterations are recorded into one command buffer (K=30, halving adaptively), double buffered, and the host checks the fixpoint only at batch boundaries.
  - Results: 1.5-4.7× faster than native CUDA Datalog engines (GPULog, mnmgJOIN) on an RTX 3060 laptop GPU. 14-24× faster than DuckDB-Wasm and 21-28× faster than Ascent-Wasm (the lead reviewer's re-read of the HTML found 4.13 s against 50.72 s, 12.3×, for the 12-graph transitive-closure suite on an M5 Pro; treat the DuckDB-Wasm factor as about 12× to 24× depending on the suite). Indirect dispatch alone gave 4.7×; with batching the total was 15.4× over the naive readback-per-iteration port.
  - Ran unchanged in Chrome on an M5 Pro Mac.
  - https://arxiv.org/abs/2607.17571 (HTML v1 read)
- Map matching: no 2024-2026 GPU HMM map-matching paper was found. The newest GPU work is Auer et al. 2017 (https://portal.fis.tum.de/en/publications/boosting-performance-of-map-matching-algorithms-by-parallelizatio/). This is an open niche.
- Trajectory similarity joins on GPU: the work found was all pre-2020 (https://arxiv.org/pdf/1410.2698). Another open niche.

### 3.6 Sort and scan on WebGPU
- **Gridwise** (WebGPU primitives; scan paper at SPAA 2025): DLDF chained scan with no forward-progress requirement, running at full memory bandwidth. Subgroup emulation is 11-28× slower on M4, so a hierarchical scan is needed when subgroups are absent. https://gridwise-webgpu.github.io/gridwise/scan-and-reduce
- **Decoupled Fallback** single-pass scan: near-memcpy speed on Apple M1/M3 under Dawn. https://escholarship.org/uc/item/0bk9z4bt (abstract seen in search; full text not loaded)
- **Linebender sorting notes:** Onesweep hardcodes 32-wide subgroups and needs forward progress. A portable WebGPU hybrid radix sort reached about 1 G keys/s on M1 Max; with real subgroups on Metal, about 3 G/s. https://linebender.org/wiki/gpu/sorting/
- **PlayCanvas** ships a WebGPU OneSweep radix sort for NVIDIA only, with a portable fallback (https://app.unpkg.com/playcanvas@2.21.1/files/build/playcanvas/src/scene/graphics/radix-sort/compute-radix-sort.d.ts).
- **WebSplatter** (arXiv 2602.03207, 2026-02-03): wait-free hierarchical radix sort that avoids global atomics, giving deterministic cross-device behaviour. https://arxiv.org/abs/2602.03207
- **Sparse linear algebra on WebGPU:** only crate-level CSR SpMV (numr). No browser CG paper was found. **Gap.**

---

## 4. Native GPU geospatial ecosystem

- **cuSpatial: archived.** RAPIDS stopped publishing packages from v25.06 (2025-06-05). It was removed from the rapids metapackage and containers, and no alternative is named. https://docs.nvidia.com/datascience/notices/rsn0045/ (last updated 2025-07-17). The GitHub repository was archived read-only on 2025-07-28 and names v25.04 as the final release (https://github.com/rapidsai/cuspatial, checked by the lead reviewer 2026-11-09).
- **SedonaDB:**
  - Launched 2025-09-24.
  - 0.4.0 (2026-06-19) added GPU spatial joins through RayBooster on RT cores (blog 2026-06-26, https://sedona.apache.org/1.9.1/blog/2026/06/26/sedonadb-04-gpu-accelerated-spatial-joins/). SpatialBench Q11 ran 7.51 s on CPU versus 1.61 s on an RTX 3090. Q10 ran faster on an RTX 3090 (1.26 s) than on an H100 (1.77 s), which has no RT cores.
  - GPU predicates: Intersects, Contains, Within, Covers, CoveredBy, Touches, Equals. Not on GPU: DWithin, KNN, GeometryCollection.
  - Off by default; Docker only, no GPU wheel. https://sedona.apache.org/sedonadb/latest/gpu-acceleration/
  - A 0.5.0 release exists (details **UNVERIFIED**).
- **DuckDB:** 1.5.0 (2026-03-09) made GEOMETRY a core type (https://duckdb.org/2026/03/09/announcing-duckdb-150). 2.0 docs exist; one source lists a 2.0.0 release on 2026-10-21 (**UNVERIFIED**). DuckDB-Wasm lists `spatial` as a loadable extension (https://duckdb.org/docs/2.0/clients/wasm/extensions).
- **Sirius GPU DuckDB:** the extension page still says "coming soon"; no tagged release found. https://duckdb.org/docs/2.0/core_extensions/sirius ; CIDR 2026 paper https://www.vldb.org/cidrdb/papers/2026/p12-yogatama.pdf (not opened)
- **Polars GPU engine:** open beta on cuDF, multi-GPU from cudf-polars 26.06, no geometry support found. https://docs.pola.rs/user-guide/gpu-support/
- **deck.gl:**
  - 9.2 (2025-10-07): WebGPU preview.
  - 9.3 (2026-04-13).
  - 9.4 (2026-09-05): all catalogue layers support WebGPU, but "experimental, not yet recommended for production". A `WebGPUAggregator` is on the roadmap but not shipped.
  - v10 will move to luma.gl v10.
  - https://deck.gl/docs/whats-new
- **kepler.gl:** 3.2.6 (2026-03-16); 3.3.0-alpha.3 (2026-07-12) upgrades to deck.gl 9.3.1. No WebGPU. Has DuckDB tables and Arrow containers. https://docs.kepler.gl/changelog
- **lonboard:** 0.15 (2026-03-13), 0.16 (2026-04-02), 0.17 reported for 2026-10 (**UNVERIFIED** content). https://developmentseed.org/lonboard/latest/CHANGELOG/
- **geoarrow-rs:** 0.8.0 (2026-03-27) (https://lib.rs/crates/geoarrow). geoarrow-js last pushed 2025-02 (v0.3.3). geoarrow-wasm 0.1.0 is dormant.
- **wgpu (Rust):** 28.x targets browser WebGPU or WebGL2 on wasm (https://docs.rs/crate/wgpu/28.0.0/source/src/lib.rs). A viable alternative runtime, but the same WebGPU limits apply.
- **ArcGIS Pro Spatial Analyst GPU tools:** Aspect, Feature Preserving Smoothing, Feature Solar Radiation, Geodesic Viewshed, and three Multiscale Surface tools; one GPU only. https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/gpu-processing-with-spatial-analyst.htm
- **Google:** Next '26 (2026-04-22) put Earth AI imagery analysis and models in BigQuery; Earth Engine in BigQuery reached GA earlier. https://cloud.google.com/blog/products/data-analytics/google-earth-ai-models-and-datasets-in-bigquery
- **Mosaic/vgplot:** DuckDB-Wasm plus Arrow, about 10M rows in-browser; the authors note Wasm parallelism limits. https://idl.uw.edu/mosaic/why-mosaic
- **WebAssembly 3.0:** standard complete 2025-09-17 (GC, memory64, relaxed SIMD, exception handling). https://webassembly.org/news/2025-09-17-wasm-3.0/. Per-browser memory64 status is inconsistent across secondary sources (**UNVERIFIED**).
- **NOT RESEARCHED** (search budget exhausted): cuGraph 2026 releases, HeavyDB status, Wherobots GPU, Felt, Fused, GEOS-WASM versions, flatgeobuf, PMTiles, zarrita versions, Arrow JS version, H3/S2 GPU work, FAISS-on-WebGPU.

---

## 5. Browser-side engines and zero-copy exchange

- DuckDB-Wasm (spatial loadable), Mosaic, kepler.gl and lonboard all exchange data as **Arrow**. A WebGPU library can take Arrow column buffers (typed-array views over Arrow's contiguous value buffers, including GeoArrow coordinate buffers, ring and geometry offsets) and upload them with `queue.writeBuffer`. That is one memcpy into the GPU, with no reformatting, which is the realistic "zero-copy" boundary.
- Results go back through `mapAsync` into an ArrayBuffer wrapped as Arrow vectors. Budget about 2.5 ms of latency per readback (WGLog), so batch readbacks.
- GeoArrow's separated coordinates and offsets match the struct-of-arrays layout that RayBooster also chose for GPU joins (Sedona blog), which is evidence the layout is right.
- ORT Web: `Tensor.fromGpuBuffer` gives true GPU zero-copy if devices are shared.
- Remaining details for GEOS-WASM, flatgeobuf, PMTiles and zarrita were not verified (see section 4).

---

## 6. Float precision without f64

- **luma.gl precision guide** (https://luma.gl/docs/api-guide/shaders/gpu-floating-point-precision):
  - WGSL allows reassociation and fusion, so classic double-single error-free transforms (TwoSum, TwoProduct) are not guaranteed.
  - `let`/`var` intermediates, runtime zero, identity bitcast and even WGSL `fma` are **not portable precision barriers**.
  - luma.gl switches to an **integer-controlled double-single path** automatically on Apple WebGPU adapters, overridable with `LUMA_FP64_INTEGER_ARITHMETIC`. It gives about 48 bits and is not binary64.
  - Recommendations: relative-to-centre (subtract the origin in exact binary64 on the CPU, then round once to f32), fixed point for bounded domains, and multi-vendor numeric validation.
- Chrome 131 listed a "strict math shader option" (news index). Its current status in the API is **UNVERIFIED**; it is not in the W3C feature list.
- Third-party `df64` crate has WGSL double-word kernels validated against MPFR (https://github.com/jesedv/df64, README only).
- The 64-bit atomic min/max (`atomic-vec2u-min-max`) lets a library pack (fixed-point distance, id) into one 64-bit key for argmin reductions without f64.

---

## 7. Benchmarks
- **WebGPU vs Wasm vs native:** WGLog (above) is the only rigorous three-way comparison found. On an RTX 3060 laptop GPU, WebGPU beat native CUDA engines by 1.5-4.7× (algorithmic and overhead differences), and beat DuckDB-Wasm by 14-24× and sql.js by 132-309× on recursive joins. Browser fixed overhead loses on small inputs (0.40× on the smallest dataset).
- **SpatialBench** (Apache Sedona, announced 2025-12-11; https://sedona.apache.org/spatialbench/): 12 SQL queries over synthetic trip, zone, building and customer data at SF 0.1/1/10/100, distributed on Hugging Face. Query list: https://sedona.apache.org/spatialbench/queries/
  - Covered: distance filters, PIP counts, PIP joins, convex hull aggregation, line length, DWithin distance join, polygon self-join IoU, LEFT PIP join, double PIP join, and a KNN join with k=5.
  - Published single-node results compare SedonaDB, DuckDB and GeoPandas (run 2025-09-22).
  - The obvious public suite to adopt: SF0.1 and SF1 are browser-sized, and GPU numbers already exist for Q10 and Q11.

---

## Ranked list A: shipped platform features to adopt

| # | Feature (availability) | Constraint removed |
|---|---|---|
| 1 | **Request adapter limits at device creation** (core API; ≥10 storage buffers on 98% of reports, ≥16 on 80%; maxBufferSize ≥2 GiB on 77%; binding ≥1.25 GB on 79%; 32 KiB workgroup memory on 99%; 512 invocations on 93%) | The 8-storage-buffer cap and 128/256 MiB caps. Tier kernels by limit. |
| 2 | **2-D/3-D dispatch plus `linear_indexing`** (core 2-D dispatch everywhere; `global_invocation_index` in Chrome 147+) | The about 4.2M-rows-per-pass bound (65535 workgroups × 64 per dimension). |
| 3 | **Indirect dispatch with GPU-written arguments and K-iteration batching** (core) | "No GPU-gated continuation": converged iterations dispatch zero workgroups, the host checks only per batch, and data-dependent sizes stay on the GPU (WGLog: 4.7× from indirect dispatch, 15× with batching). |
| 4 | **Subgroups** (Chrome 134+, 68% of reports, not Safari or Firefox) plus subgroup_id and size control | Workgroup-memory emulation of scan, reduce, ballot and multisplit (2-3× on sort and scan). Needs a non-subgroup fallback. |
| 5 | **buffer_view** (Chrome 153-154) | Storage-binding count pressure: pack several logical arrays (header, offsets, coordinates) into one binding. Chrome only; keep a fallback. |
| 6 | **Immediates** (Chrome 149-150, 64 bytes) | Per-dispatch uniform buffers and bind-group churn in unrolled graph loops (iteration index, bucket, offsets). |
| 7 | **atomic-vec2u-min-max** (merged spec; Apple9, Vulkan int64 atomics, SM 6.6; shipping status **UNVERIFIED**) | No 64-bit atomics: single-atomic argmin for nearest-feature, kNN-1, SSSP relax and Voronoi labelling. |
| 8 | **timestamp-query** (99%) | Blind performance tuning; enables per-node graph profiling. |
| 9 | **shader-f16** (94%) | Bandwidth for weights, attributes and raster values (never coordinates). |
| 10 | **Compatibility mode** (Chrome 146 opt-in) | Older Android/GLES devices: a reduced-capability tier. |
| 11 | **WebGPU in workers** (core) and experimental `mapSync` in workers (flag) | Main-thread stalls. Keep in mind that a GPUDevice is per realm. |

Not shipped, watch: bindless (resource tables), texel buffers, subgroup-matrix, i64/f64 (no proposal), ray queries (no proposal), WebNN (origin trial stalled).

## Ranked list B: algorithmic directions

| # | Direction | Parallel formulation | Expected gain | Effort |
|---|---|---|---|---|
| 1 | **GPU-resident iteration** (WGLog pattern) for SSSP, connected components, label propagation, k-means, PageRank-style centrality, DBSCAN expansion | Frontier size, then a tiny kernel writes `dispatchWorkgroupsIndirect` args; record K iterations per submit, zero-size dispatch after convergence, adaptive K, fixpoint readback per batch | 4-15× on iterative operations versus readback per iteration; removes the fixed unroll ceiling | M |
| 2 | **Count-then-emit with device-side sizing** for joins, buffers and overlay output | Per-item count → exclusive scan → contiguous slab emit; dispatch sizes from the scan total via indirect; a single readback at the end | Removes intermediate readbacks (2.5 ms each); enables data-dependent outputs inside one graph | S-M |
| 3 | **Portable single-pass scan and radix sort** (DLDF / Decoupled Fallback scan; subgroup radix with hierarchical fallback) | Chained scan with fallback work-stealing; 8-bit LSD radix with subgroup multisplit | Scan near memcpy; sort about 1→3 Gkeys/s with subgroups (Apple data). Everything sorted-cell-key benefits | M |
| 4 | **Sorted-array (binary search / Merge Path) joins** in place of hash or atomic joins | Sort the build side once by cell key; each probe binary-searches its range; contiguous emit | Skew-robust, atomic-free and deterministic (WGLog, WebSplatter) | M |
| 5 | **Skew handling for grid joins** (LibRTS multicast and X-HD heavy-cell ideas) | Detect heavy cells by histogram; split heavy cells into k sub-tasks or a cooperative workgroup kernel; cell-MBR upper/lower-bound pruning | Removes tail latency on clustered data; X-HD-style pruning for Hausdorff and nearest | S-M |
| 6 | **LBVH (Karras) over Morton/Hilbert keys** for polygons and segments; stackless traversal | Morton sort → radix-tree build (one pass) → bottom-up AABB with atomics flag → per-query stackless traversal | Polygon/segment joins and kNN on skewed extents where grids degrade | M-L |
| 7 | **Hierarchical sketch filters for polygon intersection** (PolySketch++) and Foster-style GPU clipping with degeneracy handling | Tile and edge-bucket sketches filter edge pairs; refine only survivors | 80-99% fewer edge pair tests; overlay speedups up to about 40× versus sequential | M |
| 8 | **Union-find connected components** (ECL-CC intermediate pointer jumping, Afforest sampling) reused for FDBSCAN and DenseBox | Hook with atomicMin on parents and pointer jumping; DBSCAN fuses neighbour search with union | 3-10× over label propagation; one primitive serves CC, DBSCAN and network components | M |
| 9 | **64-bit packed atomic argmin** (where available) | `atomicStoreMin(vec2u(distanceFixed, id))` | One pass instead of a two-pass min+select | S |
| 10 | **Integer-controlled double-single plus relative-to-centre** as the precision contract | Already documented upstream; adopt the luma.gl contract and add multi-vendor tests | Correctness on Apple and Metal; no fast-math surprises | S |
| 11 | **ADDS/near-far SSSP with GPU worklists** | Bucketed frontier, indirect dispatch per bucket | 2-3× over basic Bellman-Ford/near-far on road graphs | L |
| 12 | **EMST-based HDBSCAN** (ArborX approach) and GPU constrained Delaunay (gCDT) | Borůvka on a BVH; flip-based CDT | New capability; large engineering | L |

## Threats
1. **Native GPU spatial SQL is real and RT-core based** (SedonaDB 0.4+, RayBooster). It uses hardware WebGPU cannot reach, so for large server-side joins a browser library cannot compete.
2. **Server-side analytics plus AI** (BigQuery with Earth Engine and Earth AI, DuckDB 2.x, Sirius when released) pull workloads away from the client.
3. **Fragmentation:** subgroups, buffer_view, immediates and linear_indexing are Chrome-first. Safari and Firefox lack subgroups, and Firefox has no Linux or Android WebGPU yet, so every fast path needs a fallback, which doubles test matrices.
4. **deck.gl WebGPU is still experimental** (9.4), so the visual front-end for WebGPU-resident results is not production-ready. Results may need a copy to WebGL.
5. **No f64 or i64 roadmap** in gpuweb, and no strict-math guarantee.
6. **WebNN instability** and ORT's ownership of its GPUDevice complicate ML co-location.

## Opportunities
1. **cuSpatial is archived** (final release v25.04) and geoarrow-wasm and geoarrow-js are quiet. There is a gap for a maintained GPU vector-geometry library in the GeoArrow layout.
2. **WGLog shows WebGPU can beat native CUDA engines** with the right design (indirect dispatch, batching, sorted joins). That is evidence for the command-graph approach.
3. **SpatialBench** gives an off-the-shelf, browser-sized (SF0.1/1) benchmark with published CPU and RT-GPU reference numbers.
4. **Open research niches with no recent GPU papers:** map matching, trajectory similarity joins, sparse CG on WebGPU.
5. **Arrow/GeoArrow as the interchange** with DuckDB-Wasm, Mosaic and kepler.gl gives a one-memcpy boundary.
6. **transformers.js v4 and the ORT WebGPU EP** make embedding-based similarity co-located on GPU feasible today.
7. **Larger limits are widely available** (16 storage buffers on 80% of reports, 2 GiB buffers on 77%). Most of the library's current workarounds can be made tier-conditional.
