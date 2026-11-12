# Stream `shaders-core`: WGSL assembly and device use in `modules/analysis/src` against `@luma.gl/shaders` and `@luma.gl/core` 10.0.0-alpha.13

Reviewer: read-only. Nothing in the repository was edited. The only thing run was two node probes kept in
`review/shaders-core/` (`assemble-probe.mjs`, `refs.mjs`). They call the public `WGSLShaderAssembler` on
kernel-shaped WGSL. No GPU was used.

## 1. Coverage

Read in full:
- `analysis-core/kernels/wgsl-kernel-nodes.ts` (945 lines), `wgsl-common.ts`.
- `addition-rounding-error-wgsl.ts`, the header of `uint64-wgsl.ts`, `computation-record/submit-timed-command-graph.ts`,
  and `test/analysis-core/shader-fp64-module.spec.ts`.
- `dev-docs/upstream-workarounds.md`, roadmap 13.7 and `research/extended-precision-2026-11-11/README.md`.

Published code read: `@luma.gl/graph` `execution/compute/kernel.{d.ts,js}`, `geospatial/geospatial-utils.js`
(lines 1-90 and 295-428: luma.gl's own fp64 kernel route), `utils/assemble-compute-kernel.js`,
`execution/utils/get-platform-info.js` and `gpu-core/gpu-dispatch-utils.{d.ts,js}`. From `@luma.gl/shaders`:
`index.d.ts`, `wgsl.d.ts`, `lib/wgsl-shader-assembler.js`, `lib/shader-assembly/assemble-wgsl-shaders.js` (lines
1-200 and the binding allocator), `shader-assembly-shared.js` `getShaderModuleSource`, `preprocessor/preprocessor.js`,
`shader-module-dependencies.js`, `modules/math/fp64/*` (the classic path in full; the integer path's function list
and `fp64_two_sum_integer_bits`), the `fp32` tail, `random`, `valueNoise`, `color/float-colors` (function names only)
and `geospatial/dggs` (all 159 function names, plus the neighbor, boundary and web-mercator bodies). From
`@luma.gl/core`: `index.d.ts`, `uniform-store.d.ts`, `factories/bind-group-factory.js`,
`webgpu/adapter/resources/webgpu-compute-pass.js`, `webgpu-compute-pipeline.js`, `helpers/get-bind-group.js`,
`device-defaults.js` (the cache flags), `test-utils/index.d.ts` and the resource tracker.

Local code read in part: the H3, S2 and quadkey WGSL helpers in `cell-indexing/` and `cell-topology/` (headers and
function lists), `h3-index-extended-wgsl.ts` lines 1-56 and 136-320, `cpu-cell-measures-operation.ts` lines 55-110,
`cell-cover-kernels.ts:469`, `cell-cover-scanline.ts:187`, `gpu-contributor-utils.ts` lines 40-100 and 255-330, the
8 files that call `device.createBuffer` (the creation and destroy sites), `gpu-transfer-mask.ts` lines 120-170, and
`testing/gpu-contributor-test-utils.ts` lines 55-100.

Not covered:
- The bodies of `sphere-distance-wgsl.ts`, `philox-random.ts`, `symmetric-eigen-wgsl.ts`, `fixed-point-accumulator.ts`,
  `row-access-wgsl.ts`, `binary-search-wgsl.ts` and `cell-key-wgsl.ts`. I only checked their function names
  against the published module lists, and no published module function matches any of them.
- `healpix-cells`, `hilbert-keys`, `global-coordinates` (function names only; no published counterpart in
  `@luma.gl/shaders`).
- How the graph turns an `indirect` condition into a dispatch.
- The kepler `new Kernel(` sites (9, another stream).
- `@luma.gl/core/dynamic` beyond the one `DynamicBuffer` use.
- I did not measure any result on a GPU. Every performance claim below is unmeasured.

## 2. Findings

### SC-01: every `createWGSLKernelNode` dispatch creates a new GPU bind group and error scope; the identity cache in the code does nothing

- Class: `misuse` (with `stale-claim`). Evidence: read both.
- Local code:
  - `analysis-core/kernels/wgsl-kernel-nodes.ts:475-506`. It keeps `resolvedBindings` stable and passes them to
    `kernel.dispatch(computePass, {bindings, ...})`.
  - The props comment at `:274-279` says "the bind group is rebuilt only when a resolved buffer or texture view
    changes".
  - The comment at `:863-867` says "the WebGPU pipeline compares bindings by identity ... lets the pipeline keep its
    cached bind group".
  - `createChangedBindingResolver` and `createCachedBindingResolver` (`:878-945`) have **no caller in any `src`**.
    Only two specs call them (`test/utils/wgsl-kernel-nodes.node.spec.ts`,
    `test/geospatial/gpu-point-spatial-query-bindings.node.spec.ts`).
- Published code, step by step:
  - `Kernel.dispatch` calls `_setPipeline` (`@luma.gl/graph/dist/execution/compute/kernel.js:81-99`).
  - That calls `computePass.setPipeline(pipeline)` and then `computePass.setBindings(bindings)`.
  - `WebGPUComputePass.setBindings` (`@luma.gl/core/dist/webgpu/adapter/resources/webgpu-compute-pass.js:57-66`)
    calls `BindGroupFactory.getBindGroups(pipeline, bindings)` **without cache keys**.
  - Without keys, `bind-group-factory.js:27-41` always calls `_createBindGroupWebGPU`, which runs
    `pushErrorScope` + `createBindGroup` + `popErrorScope` (`webgpu/adapter/helpers/get-bind-group.js:19-31`).
  - The identity comparison the comment describes exists only on the **pipeline-held** bindings path:
    `WebGPUComputePipeline.setBindings` (`webgpu-compute-pipeline.js:62-78`) resets a per-group cache key only when a
    binding object changes. `WebGPUComputePass.setPipeline` (`webgpu-compute-pass.js:42-52`) reuses that bind group.
- Scale: 2,538 call expressions in 573 files of `modules/analysis/src` build kernels through this one function. So
  every encode of every graph creates one bind group and one error scope per dispatch. The same happens in
  `computation-record/gpu-transfer-mask.ts:154-160`, which builds its own `Kernel`.
- Change (one function, public API only), applied at `wgsl-kernel-nodes.ts:485-506`:
  - Keep the identity check.
  - On every encode, call `kernel.pipeline.setBindings(resolvedBindings)`. It is cheap: per-name identity
    comparison, and no key reset when nothing changed.
  - Then call `kernel.dispatch(computePass, {x, y, z})` without `bindings`.
  - Why this works: `Kernel._setPipeline` then binds the pipeline's cached group. Its `setBindings({})` sets nothing,
    because group 0 has layout bindings and no supplied entries (`bind-group-factory.js:21-25`).
  - Repeat the same change at `gpu-transfer-mask.ts:154-160`.
  - Then delete the two unused resolvers, or fix their comments, and fix the `:274-279` doc.
- Verify:
  - Headless: wrap `device.handle.createBindGroup` with a counter. Encode one compiled graph three times and expect
    one creation per node on the first encode and none afterwards.
  - Result parity is covered by the existing contributor specs (no numeric change).
- Risk:
  - `PipelineFactory` shares one pipeline across nodes with the same source, layout and constants. Because the
    bindings are set again before every `setPipeline`, results stay correct. Shared pipelines only re-create the bind
    group when nodes alternate, which is no worse than today.
  - The pipeline object keeps references to the last bound buffers until it is re-bound. With the device default
    `_destroyPipelines: false`, that can last the device's lifetime.
  - This works against `Kernel`'s documented intent ("does not retain mutable resource binding state",
    `kernel.d.ts:28-31`). The clean upstream fix is the cache-key option already requested in register row
    "Bind-group reuse".

### SC-02: FP64-01 can be built today with public calls only; luma.gl's own geospatial kernels are the template

- Class: `adopt-now`. Evidence: read both.
- Local code: `wgsl-kernel-nodes.ts:387-408` (source) and `:442-474` (`new Kernel`).
- Published code:
  - `@luma.gl/shaders/dist/lib/wgsl-shader-assembler.js:19-81`.
  - `@luma.gl/graph/dist/geospatial/geospatial-utils.js:305-372`: assemble, then `new Kernel`. Classic gets a
    `UniformStore` and a uniform row in the shader layout.
  - `getPlatformInfo` is a public export of `@luma.gl/graph` (`dist/index.d.ts:11`).
- See the design note in section 5. One file changes, plus a prop type. No caller changes until FP64-02.
- Verify: extend `shader-fp64-module.spec.ts` to go through `createWGSLKernelNode` (section 5 lists what it does not
  prove yet).
- Risk: none for existing kernels, because the assembler is skipped when `modules` is empty.

### SC-03: the claim "the assembler chooses the fp64 variant per device" holds only for Apple; elsewhere it picks classic with no barrier

- Class: `stale-claim` (register row "Double-float WGSL"; extended-precision README section 4). Evidence: read both.
- Published code:
  - `WGSLShaderAssembler.getPlatformPreprocessorDefines` (`wgsl-shader-assembler.js:69-81`) sets
    `LUMA_FP64_INTEGER_ARITHMETIC` only when `gpu === 'apple'`, and never sets `LUMA_FP64_CODE_ELIMINATION_WORKAROUND`.
  - So on NVIDIA, Intel or an unidentified GPU, the default assembly is classic with no barrier. The README measured
    that variant at 22 bits on Metal, which is float32.
  - luma.gl's own kernels add vendor defines through `getClassicFP64Defines` (`geospatial-utils.js:63-81`). That
    function is `@internal` and not exported from any index (checked).
- Consequences for a local caller:
  - It must pass `defines` explicitly. Caller defines override platform defines; they are spread last at
    `wgsl-shader-assembler.js:59-67`.
  - Classic also needs the module's uniform bound (SC-02, section 5).
  - The register row "can be replaced" is true only with explicit defines. The spec proves Apple only.
- Change: correct the register row and README section 4 (wording in section 4 of this report).
- Risk: if documentation only. If the defines are not passed, results on non-Apple GPUs silently drop to float32.

### SC-04: helper WGSL is joined by hand under "concatenate after" rules; the published module system does the ordering and de-duplication

- Class: `adopt-with-cost`. Evidence: read both. The probe ran the assembler on kernel-shaped source.
- Local code, counted:
  - 152 exported `*_WGSL` snippet constants and 153 `get*WGSL` generators in `modules/analysis/src`, with 2,151
    `${...WGSL}` interpolations in 523 files.
  - `dggs.source` is interpolated 20 times in 15 files.
  - `GUARDED_ATAN2_WGSL` is interpolated 19 times in 15 files.
  - 16 doc comments carry hand-kept ordering or uniqueness rules. Examples:
    - `wgsl-common.ts:9-10`: "kernels that concatenate several declaration strings must include each constant once".
    - `cell-indexing/h3-boundary-wgsl.ts:17` and `:34-35`: "must not add it again".
    - `cell-topology/s2-analysis-wgsl.ts:8`.
    - `h3-measures-wgsl.ts:19`, `h3-neighbor-wgsl.ts:49`, `h3-local-ij-wgsl.ts:46`,
      `cell-topology-wgsl.ts:8,138`, `cell-index-wgsl.ts:39,61,98`.
  - A kernel that needs both `H3_BOUNDARY_WGSL` and `S2_ANALYSIS_WGSL` gets `guardedAtan2` twice and fails to compile.
- Published code: `getShaderModuleDependencies` (`shader-module-dependencies.js:18-…`) sorts modules
  topologically, emits each module object once, and throws on two different objects with the same name. The probe
  shows two dependent modules that share `guardedAtan2` and `dggs` are emitted with one copy each.
- Change, once SC-02 lands (it takes `modules`):
  - Turn the dependency-carrying helpers into `{name, source, dependencies}` objects: `GUARDED_ATAN2`, `dggs` users,
    `H3_BOUNDARY`, `H3_NEIGHBOR`, `H3_LOCAL_IJ`, `H3_MEASURES`, `S2_ANALYSIS`, `CELL_KEY`, `CELL_INDEX`,
    `CELL_TOPOLOGY`, `UINT64`, `GET_ORDERED_KEY`.
  - Callers then pass `modules` instead of interpolating text.
  - Do not convert one-line, dependency-free snippets.
- Cost:
  - CPU: assembly costs 5-17 ms per kernel in node (probe; 116 KB of source with `dggs` + `fp64`). That is graph-build
    time on every kernel that uses modules.
  - Modules are emitted before the application source.
  - The node record (`WGSL_KERNEL_NODE_RECORDS`) must hold the assembled text.
  - Helpers that read caller bindings by name (for example `NODE_OVERLAPS_WGSL` reads `nodeMinima`) still compile
    after joining, but they do not become independent modules.
- Verify: `test/analysis-core/kernel-audit.node.spec.ts` and the per-family node specs that parse kernel sources.
- Risk: compile-time only. WGSL output is identical apart from module header comments and order.

### SC-05: register row "H3 face tables as data" misses the second parser of `dggs.source`

- Class: `stale-claim` (register). Evidence: read one (the local code; the published tables are inside
  `dggs-wgsl.js`).
- Local code: `gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation.ts:74-96`. `readIntegerTable` parses
  `dggs_h3_get_base_cell_home` (122 x 4) and `dggs_h3_get_face_neighbor` (20 x 3) out of the module text. The same
  file also calls `parseDggsFaceTable` for two float tables. The register row lists only
  `h3-index-extended-wgsl.ts:25-47`.
- Change: add the site to the row. The upstream request grows: base-cell homes and face-neighbor tables exported as
  data as well.
- Verify: the shape check throws at module load (`:84-90`).
- Risk: none today. A `dggs` text change would break module load, loudly.

### SC-06: the integer-mode `twoSum`/`twoProd` compute what `getAdditionRoundingError`/`getMultiplicationRoundingError` compute (FP64-04)

- Class: `adopt-with-cost`. Recommendation: **do not** adopt. Evidence: read one (local in full; published
  `fp64_two_sum_integer_bits` partly).
- Local code: `analysis-core/kernels/addition-rounding-error-wgsl.ts` (13 importing files),
  `multiplication-rounding-error-wgsl.ts` (2 files).
- Published code: `fp64-arithmetic-wgsl-integer.js:106-177` (`fp64_two_sum_integer_bits`, the `twoSum` at `:349`)
  and `:200-223`/`:364` (`twoProd`). These are only error-free under `LUMA_FP64_INTEGER_ARITHMETIC`. Under classic
  without the workaround they are the float forms that Metal removes.
- What would be lost:
  - Each kernel would pull in the whole integer arithmetic module, 46 KB of WGSL (probe), for a 60-line helper.
    Compile time grows; luma.gl's source says the integer path compiles impractically slowly on D3D12.
  - The edge rules differ. Locally, an error below 2^-103 is reported as 0 and the caller's own `sum` is used.
    Published code canonicalises infinities and NaN its own way.
- Change: close FP64-04 as "reviewed, keep local" with this reason.
- Verify: if it is ever tried, use `test/utils/addition-rounding-error-wgsl.spec.ts` as the oracle.

### SC-07: hand-written float64 high/low split; published `fp64ify` does the same thing

- Class: `adopt-now` (minor). Evidence: read both.
- Local code:
  - `dataframe-analysis/time-window-filter/time-window-parameters.ts:86-89` (`splitTime`).
  - `cell-indexing/h3-index-extended-wgsl.ts:50-53` (`formatDoubleFloat`, which also formats the text).
- Published code: `fp64ify` / `fp64LowPart`, from the `@luma.gl/shaders` root (`modules/math/fp64/fp64-utils.js:11-25`).
- Change: `splitTime` becomes `fp64ify(value)`, giving the same pair once stored in a `Float32Array`. In the
  formatter, keep the text formatting and use `fp64ify` for the split. Touches 2 sites.
- Verify: the existing time-window specs; `h3-index-extended.node.spec.ts`.
- Risk: the low part is unrounded until it is stored. Both sites store or print through float32, so the result is the same.

## 3. Checked and fine

- **`Kernel` itself.**
  - It comes from the `@luma.gl/graph` root (`execution/compute/kernel.d.ts`). It offers `createAsync`, `dispatch`,
    `dispatchIndirect`, and optional `pipelineFactory`, `shaderFactory` and `debugShaders`. It has no cache and no
    uniform helper of its own.
  - It uses the default `ShaderFactory` and `PipelineFactory`. Their caches are on by default
    (`device-defaults.js:41-43`), so the override-constant design (`wgsl-kernel-nodes.ts:383-386`) does make
    same-shaped kernels share shaders and pipelines.
  - `createAsync` is not needed: `GPUCommandGraph.compileAsync` opens `PipelineFactory.beginAsyncCompilation`
    (`gpu-command-graph-compiler.js:96`), and the `Kernel` constructor joins it (`kernel.js:65-73`).
- **Explicit `shaderLayout` plus phony uses** (`wgsl-kernel-nodes.ts:375-382`, `:451-473`). This is required: core's
  WebGPU compute pipeline always uses `layout: 'auto'` (`webgpu-compute-pipeline.js:44-52`). luma.gl does the same
  phony trick for its fp64 uniform (`geospatial-utils.js:401-428`). `scanWGSLInterface` would also list unused module
  bindings, so it cannot replace the hand-built layout.
- **Dispatch limits.**
  - `getBoundedDispatchLayout` is called with `device.limits.maxComputeWorkgroupsPerDimension` at
    `wgsl-kernel-nodes.ts:75-80`, `:185-192` and `:355-360`.
  - The local index arithmetic (`:401-405`) equals published `getBoundedInvocationIndexSource`
    (`gpu-dispatch-utils.js:26-31`). The published helper bakes the layout into source text; the local code keeps it
    in overrides, which is better for caching. Keep local.
- **Device limits** are read through `device.limits`:
  - storage buffers (`:328`), binding size (`:337`) and sampled textures (`:348`);
  - `maxComputeWorkgroupStorageSize` (5 sites) and `maxComputeInvocationsPerWorkgroup` (`pointcloud-kernel-nodes.ts:95`);
  - `getStorageBindingElementLimit` (`gpu-contributor-utils.ts:315-330`).
- **Hard-coded limit numbers** are documented portable maxima, not misread limits:
  - `GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE` (`:270`), the two GWR constants of 134217728, and the four "more than
    8 storage bindings" checks (`wgsl-kernel-nodes.ts:793`, `gpu-pair-gather.ts:501`, `gpu-zone-events.ts:1091`,
    `gpu-point-horizon-profile.ts:237`) follow contract rule 16.
  - All 35 occurrences of `65535` are domain limits (u16 quantization, zone and seed counts).
- **Uniforms and struct layout.**
  - Analysis source declares 0 `var<uniform>` and binds only `array<u32|i32|f32>` storage.
  - Parameters are flat 32-bit arrays (`GPUParameterBuffer`, `gpu-contributor-utils.ts:46-98`).
  - No hand-computed offsets, padding or alignment for GPU structs. The 6 `DataView` sites are CPU twins and a
    readback.
  - So `makeShaderBlockLayout`, `ShaderBlockWriter` and `UniformStore` have nothing to replace. `UniformStore` is
    needed only for classic fp64 (section 5).
- **`wgsl_reflect`** is used only in test code (`src/testing/gpu-contributor-test-utils.ts:14,140` and tests) as a full
  WGSL parser. `scanWGSLInterface` is a binding scanner that returns `null` on syntax it does not support
  (`wgsl-interface-scan.d.ts`), not a parser, so it cannot replace it.
- **`dggs` use.** It enters only as `dggs.source` text. The local H3 code is justified for these reasons:
  - The published boundary is single-face only: `dggs_h3_get_boundary_point` returns `vec2f(0)` for a multi-face cell.
    That justifies `H3_BOUNDARY_WGSL`.
  - `dggs_h3_neighbor` steps IJK coordinates, not cells. That justifies `H3_NEIGHBOR_WGSL`.
  - `dggs` has no encoders (longitude/latitude to cell), so `cell-index-wgsl.ts`, `h3-index-wgsl.ts` and the HEALPix
    indexing are new code.
  - `cellIndexH3UpAp7` is the counter-clockwise twin (`h3-index-wgsl.ts:148-149`).
- **`coverTileLatitude`** (2 copies: `cell-cover-kernels.ts:469`, `cell-cover-scanline.ts:187`) is the same formula
  as `dggs_web_mercator_tile_y_to_latitude`. Delegating would pull in 70 KB of `dggs` for two lines. Not worth it.
- **Published math modules against local helpers:**
  - `random` is already a register row.
  - `valueNoise` hashes floats and is not equivalent to `hashUint32`.
  - `colors`, `floatColors` and `storageColors` normalise colours; locally only the builtin `pack4x8unorm` is used
    (2 sites).
  - `fp32.tan_fp32` exists for the imprecise native `tan` on non-NVIDIA/AMD WebGPU. The local `tan` uses are
    `rhumb-wgsl.ts:141` (argument at most 22.5 degrees) and `gpu-geomorphons.ts:428` (one constant per kernel). This
    is a lead only; there is no measurement to act on.
  - No published module has haversine, ordered-float keys, Philox or eigen solvers.
  - The local copies of the ordered key are local hygiene, not a luma.gl matter: `getOrderedKey` in 3 private copies
    beside the shared one, `getOrderedFloatKey` ×2, `orderedKey` ×1, and the pinned `encodeOrderedFloat`. So are the
    two Philox implementations (`philox-random.ts:142`, `permutation-random.ts:97`) and `getInfinity` ×5.
- **Device resources.**
  - The 10 `device.createBuffer` lines in 8 files are each released: `GPUParameterBuffer.destroy`, the time-upload
    `GPUData` (`ownsBuffer`), snapshot arena `:194`, `ownedBuffers` in conditional simulation, prepared join and study
    window, spike repair `:961-968`, and test utilities.
  - No published persistent graph-owned view exists that would replace them (`gpu-core` exports only transient views).
  - `submit-timed-command-graph.ts` uses the public `timeProfilingQuerySet` and frees in `finally`.
  - The snapshot arena uses `MAP_READ` + `readAsync`.
  - `GPURegionStatisticsReadback` uses the published `GPUReadbackRing`.
- **`@luma.gl/core/test-utils`**: `NullDevice` is used. I would not adopt `getLeakedResources`: it reads a global
  `luma.stats` (`test-utils/utils/resource-tracker.js`).

## 4. Register corrections

1. **"Bind-group reuse" (section 2).**
   - The local code it names (`createChangedBindingResolver`, `createCachedBindingResolver`) has no `src` caller.
   - The identity reuse inside `createWGSLKernelNode` does not avoid bind-group creation: `Kernel.dispatch` →
     `ComputePass.setBindings` creates a bind group and an error scope on every dispatch at alpha.13.
   - A public workaround exists: `ComputePipeline.setBindings` plus `Kernel.dispatch` without bindings (SC-01).
   - The "what luma.gl needs" column still holds (a cache key on `Kernel.dispatch` or `ComputePass.setBindings`).
2. **"H3 face tables as data".** Add `cpu-cell-measures-operation.ts:74-96`, which parses two integer tables
   (base-cell homes, face neighbors) and two float tables. Widen the request to all four tables (SC-05).
3. **"Double-float WGSL".**
   - Replace "the public `WGSLShaderAssembler` ... sets `LUMA_FP64_INTEGER_ARITHMETIC` for Apple WebGPU" with: it does
     that for Apple and otherwise selects classic **without** `LUMA_FP64_CODE_ELIMINATION_WORKAROUND`. A caller must
     pass defines.
   - Classic reads a uniform at `@group(0) @binding(100)` that must be bound. `shader-fp64-module.spec.ts` covers only
     the Apple integer path and not `sin`/`cos`.
   - The per-vendor helpers are still `@internal` at alpha.13.
   - The same correction applies to README section 4, second bullet.
4. **"64-bit integer words".** "No multiply or divide" is accurate. Add that `dggs` does publish add, subtract,
   compare, shift, negate and to-float (`dggs_u64_add`, `dggs_u64_subtract`, `dggs_u64_compare`, `dggs_u64_shift_*`,
   `dggs_i64_negate`, `dggs_i64_compare`, `dggs_u64_to_f32`, `dggs_i64_to_f32`), with the same `(high, low)` order. A
   kernel that already includes `dggs.source` could call them instead of the matching `UINT64_WGSL` functions. Other
   kernels would pay 70 KB of compile text for it, so `UINT64_WGSL` stays.

## 5. Design note: FP64-01, `createWGSLKernelNode` with shader modules and defines

**Public calls.** Signatures are from the `.d.ts` files:
- `new WGSLShaderAssembler().assembleWGSLShader(props: AssembleShaderProps)` from `@luma.gl/shaders` (or
  `/wgsl`). It returns `{source, bindingAssignments: {moduleName, name, group, location}[], shaderLayout, modules,
  getUniforms, ...}`. The props to pass are `{platformInfo, source, modules, defines, shaderStage: 'compute',
  computeEntryPoint: 'main', scanVertexAttributes: false}`.
- `getPlatformInfo(device: Device): PlatformInfo` from `@luma.gl/graph`. It uses `device.info.gpu`. The spec instead
  derives the GPU from the adapter label; production must not.
- `ShaderModule` type from `@luma.gl/shaders`.
- For classic arithmetic only: `new UniformStore(device, {fp64arithmetic})`, then `.setUniforms({fp64arithmetic:
  fp64arithmetic.defaultUniforms})`, `.getManagedUniformBuffer('fp64arithmetic')` and `.destroy()` (`@luma.gl/core`).

**What the assembler does.**
- It resolves and de-duplicates modules (`getShaderModuleDependencies`).
- It runs `preprocess` over the application source and every module with platform, module and caller defines, in
  that order; caller defines win.
- It emits the modules before the application source.
- It moves module `@binding(auto)` declarations to group 0, location ≥ 100: `RESERVED_APPLICATION_GROUP_0_BINDING_LIMIT`
  (`assemble-wgsl-shaders.js:17`). Application explicit bindings must stay below 100.
- The local kernels' explicit `@binding(0..n)` storage and texture bindings are left alone. The probe confirmed this:
  `values@0:0`, `out@0:1`, `fp64arithmetic@0:100`.
- Override declarations and `const xOffset` lines pass through unchanged. The preprocessor touches only `#if`,
  `#ifdef`, `#ifndef`, `#else` and `#endif` lines.

**Defines per platform** (`wgsl-shader-assembler.js:69-81`):
- Apple WebGPU: `LUMA_FP64_INTEGER_ARITHMETIC`.
- Every other GPU: nothing, which means classic with no barrier, which means float32 accuracy on fast-math back ends
  (SC-03).
- `LUMA_FP32_TAN_PRECISION_WORKAROUND` is set for all but NVIDIA and AMD.

**`@binding(auto)` against explicit bindings.**
- Integer variant: the uniform `fp64arithmetic` is declared, but no function reachable from the entry point reads it.
  The probe found 1 reference, inside the uncalled `fp64_runtime_zero_bits`. With `layout: 'auto'` it is not part of
  the pipeline layout, so nothing is bound and the explicit `shaderLayout` stays as it is. The spec confirms this: its
  raw bind group has no entry 100 and it runs.
- Classic variant: `split()` reads `fp64arithmetic.SPLIT` (`fp64-arithmetic-wgsl.js:646-647`), so the uniform is
  statically used. The node must then:
  - add `{name: 'fp64arithmetic', type: 'uniform', group, location}` to `shaderLayout.bindings`, taken from
    `bindingAssignments` (no regex needed, unlike `geospatial-utils.js:381-397`);
  - add `_ = fp64arithmetic.ONE;` to the generated `main` (the node writes `main`, so no source surgery);
  - own a `UniformStore`, bind its buffer, and destroy it in `destroy`;
  - count one uniform buffer against `maxUniformBuffersPerShaderStage`.

**What `shader-fp64-module.spec.ts` already proves.**
- On the test GPU (Apple), `fp64` + `fp64arithmetic` assembled by the public assembler compile in a compute shader
  beside explicit application bindings 0 and 2.
- The integer path needs no uniform binding.
- Accuracy: sum, multiply, divide and square root at most 1e-13 relative; `log` and `exp` at most 1e-9; a long sum at
  1e-10.

**What it does not prove.**
- Classic, or any non-Apple GPU.
- Going through `Kernel` with an explicit `shaderLayout` and override constants.
- `sin_fp64`/`cos_fp64`, which FP64-02 needs for `cellIndexH3ExtSinCosDegrees`.
- Compile time with `dggs.source` beside the 46 KB integer module.
- `getPlatformInfo(device)` in place of the hand-made platform info.

**Smallest safe change.**
- Add optional `modules?: readonly ShaderModule[]` and `defines?: Record<string, boolean | number>` to
  `WGSLKernelProps`.
- In `createWGSLKernelNode`, when `modules` is non-empty:
  - assemble right after building `source` (`:387-408`), with `getPlatformInfo(graph.device)`;
  - use one module-level `WGSLShaderAssembler`, so auto-binding locations stay stable;
  - **always add `LUMA_FP64_INTEGER_ARITHMETIC: true`** when an fp64 module is present.
- That keeps the uniform out of the layout on every platform, gives the same WGSL that was measured, and leaves
  `shaderLayout`, encode and destroy unchanged.
- Store the assembled text in the node record (`:514`), so `describeGPUCommandGraph` shows what runs.
- Reject `modules` that declare a statically used uniform until the classic path exists. In practice: any defines that
  set `LUMA_FP64_INTEGER_ARITHMETIC: false`.
- Cost: integer arithmetic is slower (README: about 3 times the local code in the loop probe), and luma.gl's own
  comment says it compiles very slowly on D3D12. Choosing classic per vendor is FP64-05. It needs the `@internal`
  helpers exported upstream; copying them here would re-implement luma.gl.
- Spec: a headless twin of `shader-fp64-module.spec.ts` that builds one `createWGSLKernelNode` with
  `modules: [fp64]`, runs it through a compiled graph, and adds `sin`/`cos`. Also a node spec that the assembled
  source keeps bindings 0..n and records the assembled text.

**Other helpers as real modules.** The same `modules` prop lets the dependency-carrying helpers of SC-04 become
`{name, source, dependencies}` modules. That retires the 16 hand-kept ordering rules and the duplicate-`guardedAtan2`
trap. It is worth it for the `dggs`/H3/S2/cell-key family and `GUARDED_ATAN2` (about 30 kernels). It is not worth it
for the 140-odd single-function snippets: each assembly costs 5-17 ms of graph-build CPU, and they have no
dependencies to resolve.
