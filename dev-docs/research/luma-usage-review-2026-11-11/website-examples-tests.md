# Stream `website-examples-tests`: consumers of the library against published luma.gl alpha.13

Reviewer: read-only. Published side read from `node_modules/@luma.gl/*` (10.0.0-alpha.13).
Helper scripts: `review/website-examples-tests/deck-imports.js` (patched deck.gl imports against
`exports-installed.json`), `review/website-examples-tests/doc-imports.js` (docs snippet imports).

## 1. Coverage

Read in full:
- Website framework: `website/src/gpu/device.ts`, `gpu-canvas.tsx`, `runs/harness.ts`,
  `runs/README.md`, `runs/define-run.ts` (lines 440-760, 855-890, 1100-1400), `demos/kit.ts`
  (90-180, 600-889), `previews/runtime.ts` (240-320), `runs/text/text-layer.ts` (head),
  `runs/text/font.ts`, the three `kernel.ts` copies, readback helpers in
  `runs/entries/{categories-cells-tests,raster-time-series}/_lib/readback.ts`, and 4 sampled runs
  (`medoid-composite`, `conservative-remapping`, flagship `hot-spots`, `k-means`).
- Examples: `spatial-analysis-showcase/src/engine/deck-host.ts` (270-320, 855-866, 1100-1150),
  `engine/layers.ts` (737-900), `engine/summary-reader.ts`, `scenes/raster/spectral-texture-display.ts`
  (225-255), `docs/live-lens-adapter.ts`; `spatial-analysis-shared/{summary-reader,vector-timing,readback-scheduling}.ts`;
  `spatial-analysis-explorer/modes/buffer-mode.ts` (290-315), `visibility-mode.ts` (805-835).
- Deck.gl patch `.yarn/patches/@deck.gl-core-npm-9.4.0-707f3fb147.patch` (all luma.gl import hunks),
  `.yarnrc.yml`, root `package.json` resolutions, `yarn why` for the 9.x luma packages.
- Tests: `modules/analysis/src/testing/*` (3 files), `modules/analysis/test/utils/{require-test-device,gpu-contributor-test-utils,webgpu-test-device.spec}.ts`, `vitest.config.ts`, sampled kepler GPUVector readers.
- Docs: `docs/api/gpu-spatial-analysis/concepts.md` (205-300), `gpu-network/operators.md` (285-300),
  `website/src/components/meta/pages/anatomy/kernel-density.example.ts`; script scan of every
  `docs/**/*.md` and `website/src/components/meta/pages/**` for luma.gl imports (15 pages, 9 names).
- Published: core `Device`/`DeviceProps`/`PresentationContext`/`Buffer` d.ts, `webgpu-buffer.js`
  (`readAsync`, `mapAndReadAsync`, `_getMappableBuffer`), `webgpu-device.js` (error scopes,
  `uncapturederror`, `submit`), `webgpu-adapter.js` (`featureLevel`), `luma.js` (`best-available`),
  `test-utils` (`create-test-device.js`, `null-device-features.js`, `resource-tracker.js`),
  `device-resource.js` (stats names), graph `gpu-readback-ring.d.ts`, `gpu-reduction.d.ts`,
  `gpu-gather.d.ts`, `gpu-command-graph.d.ts/.js`, `cpu-gpu-vector-transfer.d.ts`, `cpu-vector.d.ts`,
  engine `text/index.d.ts`, `text-renderer.d.ts`, `gpu-text-data.d.ts`, `arrow-text-renderer.d.ts`,
  `shader-inputs.d.ts` (head), shaders `wgsl-binding-scan.d.ts`.

Not covered (a later pass should not assume these were checked):
- 532 of 540 run entry files were only grepped, not read; `previews/builders.ts`, `previews/shaders.ts`,
  `runs/layers.ts`, `runs/pick.ts`, `runs/record-demo.ts`, `website/src/components/meta/scenes/*`
  (5 explanatory scenes with their own `Model`s) were not read.
- Showcase scenes and explorer modes other than those named above; `spatial-analysis-shared/{resources,graph-swap,analysis-effect}.ts`.
- Whether published `ShaderInputs` can express the website kit's array uniforms and generate a WGSL
  struct (WE-05 is inferred on that point). The published text shaders were grepped for outline/halo,
  not read in full (WE-04).
- `modules/lenses/test` and `modules/kepler/test` helpers beyond the GPUVector readers.
- Nothing was run on a GPU; no node specs were run.

## 2. Findings

### WE-01 Website runs read back on every run with a fresh staging buffer and no pacing
- Class: `misuse` (the GPUReadbackRing half is `adopt-now`).
- Local: `website/src/gpu/runs/define-run.ts:1140-1164` (`refreshRanges`, called at `:1341` after
  every run, no in-flight guard), `:1236-1282` (`refreshExtra`, every run), flagship demos
  `website/src/gpu/demos/hot-spots.ts:133-138`, `demos/k-means.ts:104-109`. A pointer gesture that
  writes values calls `changeValues` (`define-run.ts:731-738`), so each frame runs the operation and
  issues these reads. Only `peek` is coalesced (`define-run.ts:650-699`).
- Published: `node_modules/@luma.gl/core/dist/webgpu/adapter/resources/webgpu-buffer.js:127-182,189-195`:
  every `readAsync` on a buffer without `MAP_READ` creates a temporary `MAP_READ` buffer, records
  and submits its own copy encoder, maps, and destroys it. `@luma.gl/graph/gpu-core`
  `GPUReadbackRing` (`dist/gpu-core/gpu-readback-ring.d.ts:1-90`) gives reusable slots and
  `tryAcquire()` drop-on-pressure.
- Evidence: read both.
- Change: route `refreshRanges`, the `afterRun` readbacks and the demo reads through one
  `GPUReadbackRing` per build, with the same one-in-flight/queued pattern `refreshPeeks` already uses.
  The examples already do exactly this: `examples/deck/spatial-analysis-shared/summary-reader.ts:27-135`
  (`tryAcquire`, `markEncoded`, one read in flight), used from 171 showcase files. Call sites: 3 in
  the framework (`define-run.ts:664`, `:1145`, the `afterRun` hand-off) plus 60 `readAsync` lines in
  48 run entry files and 3 demo files; 4 local copies of the bytes-to-typed-array helper
  (`entries/categories-cells-tests/_lib/readback.ts`, `raster-time-series/_lib/readback.ts`,
  `raster-bands-indices/support/labels.ts`, `space-time-models/_lib/flash-cube.ts`) collapse into
  one `RunBuffer` read in the framework.
- Optional further step (`adopt-with-cost`): compute auto ranges on the GPU with `GPUReduction`
  `operation: 'extent'` (`gpu-reduction.d.ts:15,24-34`) and read 8 bytes. Loss: `readRange`
  (`define-run.ts:453-467`) skips NaN, Infinity and |v| >= 1e20; `GPUReduction` has no such filter,
  so a mask kernel is needed first, and WGSL `min`/`max` on NaN are implementation-defined.
- Verify: `node website/scripts/verify-runs.mjs --group <group>` pointer sweep (five positions, graph
  must not recompile); count staging allocations with `luma.stats.get('GPU Resource Counts').get('Buffers Created')` before/after a sweep.
- Risk: none for values; a range can lag one more frame under pressure (already asynchronous).
- Side note (not luma): `readRange` keeps `0xffffffff` "unreached" sentinels of `uint32` buffers
  (4.29e9 < `INVALID_LIMIT` 1e20), contrary to `runs/README.md:240-241`, if a uint32 layer uses a
  non-categorical ramp with `range: 'auto'`. Lead, not checked per run.

### WE-02 85 runs round-trip GPU to CPU to GPU only to draw
- Class: `adopt-with-cost` (lead; per-run judgement).
- Local: 95 run files define `afterRun`; 85 of them `upload(...)` a CPU array derived from a readback,
  creating new buffers every run (`runs/README.md:94-103`). Examples:
  `entries/raster-composites-climate/medoid-composite.ts:88-99` (copies band 3 of a 4-band
  interleaved composite), `entries/raster-bands-indices/conservative-remapping.ts:101-119`
  (nearest-neighbour spread of a coarse grid onto the fine grid).
- Published: `GPUGather` (`@luma.gl/graph/gpu-core`, `dist/gpu-core/gpu-gather.d.ts:10-26`, "optionally
  strided" source rows) and strided graph data views keep such selections on the GPU. The website
  already has an in-graph helper kernel (`entries/regression-weights/shared/kernel.ts:24-106`, an
  engine `Computation` inside `graph.addComputePass`) used by 32 runs, in 3 near-identical copies
  (`geometry-construction-overlay/shared/kernel.ts`, `space-time-events/_lib/kernel.ts`).
- Evidence: read one (two runs and the published d.ts); the other 83 inferred from the grep.
- Change: for display-only reshapes (band pick, upsample, re-index) add a helper kernel or `GPUGather`
  node in `build` so the layer binds the result; keep `afterRun` for genuine CPU summaries
  (snapshot arenas, recorded graphs, readouts). Hoist the three `kernel.ts` copies into one shared file.
- Verify: posters from `node scripts/snapshot-runs.mjs <id>` must stay byte-identical (`pages-deploy`
  memory says posters regenerate byte-identically); `verify-runs.mjs` for the run.
- Risk: none for exact copies/gathers; a reshaping kernel that does arithmetic could round differently
  from the CPU version (float32 vs float64 on CPU).

### WE-03 One GPU device per card, capped at 12, recompiled on every restart
- Class: `adopt-with-cost`.
- Local: `website/src/gpu/device.ts:27-52` creates a device per `<GpuCanvas>` (`gpu-canvas.tsx:220-228`),
  with `MAXIMUM_LIVE_DEVICES = 12` and a wait queue (`device.ts:54-77`). Leaving the viewport or a
  theme change destroys the device and a revisit builds a new one (`gpu-canvas.tsx:276-280`), so every
  pipeline is compiled again. One path only (all 18 `<GpuCanvas>` users and the harness go through it).
- Published: `Device.createCanvasContext()` / `createPresentationContext()`
  (`core/dist/api/adapter/device.d.ts:418-422`); on WebGPU a presentation context renders directly
  into its canvas (`presentation-context.d.ts:3-9`, `webgpu-device.js:230-232`). Pipeline and shader
  factories are per device, so one device shares compiled pipelines across cards.
- Evidence: read one (published d.ts and constructor; did not run).
- Change: one page device (lazy, WebGPU), one canvas context per card; pass the card's framebuffer to
  `beginRenderPass` (7 call sites: `define-run.ts:1196`, `demos/kit.ts:840`, `previews/runtime.ts:495`,
  4 in `components/meta/scenes/*`) and replace `getDefaultCanvasContext()` (5 sites). The 12-slot
  queue stays only for the WebGL2 fallback.
- What is lost: isolation. A device loss or out-of-memory error ends every live card at once; the
  first-frame validation scope (`gpu-canvas.tsx:519-533`) must stay synchronous per card because error
  scopes are device-wide; on WebGL the presentation context borrows the default canvas and copies.
- Verify: `node scripts/check-all-pages.mjs` (every page reaches `data-ready=true`), posters unchanged.

### WE-04 Website text is a 2,236-line local SDF text stack beside `@luma.gl/engine/text`
- Class: `adopt-with-cost`.
- Local: `website/src/gpu/runs/text/` (`text-layer.ts` 371, `text-layout.ts` 561, `labels.ts` 471,
  `font-metrics.ts` 211, `generate-font-atlas.mjs` 483 lines, prebuilt base64 atlas). Roadmap
  `dev-docs/roadmap.md:480` (WEB-INT-16) lists its gaps: glyphs cannot be coloured, no `^`, `'` or
  superscript two.
- Published: `@luma.gl/engine/text` `TextRenderer`, `GPUTextResources`, `buildSdfFontAtlas`,
  `loadMsdfFontAtlas` (`engine/dist/text/index.d.ts`), and `ArrowTextRenderer`
  (`@luma.gl/engine/arrow`, `arrow-text-renderer.d.ts:18-55`: per-row colours, angles, sizes, pixel
  offsets, `textAnchor`, `alignmentBaseline`, SDF/MSDF modes, `modelProps.fs` override).
- Evidence: read one (published d.ts; grep of the published text shaders found no outline or halo).
- What it would gain: coloured labels and any glyph set. What it loses: the halo in the background
  colour (would need an `fs` override through `modelProps`), swatches and rules, the crowding/thinning
  layout of `labels.ts`, and determinism of posters if the atlas is rasterised at runtime from a system
  font (`buildSdfFontAtlas` uses canvas 2D); a checked-in MSDF atlas via `loadMsdfFontAtlas` keeps it.
  Inputs must become Arrow vectors or `GPUTextData`.
- What the consumers use today: website GPU SDF text (above); showcase deck.gl `TextLayer` (5 files);
  explorer draws side charts with 2D canvas `fillText` (`modes/point-pattern-chart.ts:74-138`), not
  pictures. No consumer imports `@luma.gl/engine/text` (only the parked kepler-compute does).
- Verify: `verify-runs.mjs --themes dark,light` on the labelled runs; poster diff.

### WE-05 Hand-packed uniform buffers instead of `ShaderInputs`
- Class: `adopt-with-cost` (inferred on WGSL struct generation and array uniforms).
- Local: website 5 uniform buffers with hard-coded float offsets, largest the 120-float `Look`
  (`demos/kit.ts:651-719`, offsets 0-119 written by index), plus `previews/runtime.ts:254-317`,
  `runs/text/text-layer.ts:252`, `runs/layers.ts:298`, `components/meta/scenes/scene-kit.ts:74`.
  Examples: 39 `Buffer.UNIFORM` lines, e.g. showcase `engine/layers.ts:836-841`, explorer
  `modes/*-layers.ts` (17 files). `ShaderInputs` is imported nowhere in the repository.
- Published: `@luma.gl/engine` `ShaderInputs` (`engine/dist/shader-inputs.d.ts:15-41`: modules with
  `uniformTypes`, managed uniform buffer per block, change tracking) and `Model.shaderInputs`.
- Change: declare each layer's uniforms as a shader module and set them with `setProps`; for deck
  layers this is deck's own convention. 44 sites.
- Loss: none in values; needs confirmation that the 6- and 16-entry `vec4` arrays of `Look` are
  expressible in `uniformTypes` and laid out as the WGSL struct expects. Low priority: the current code
  works and passes the verifier.

### WE-06 Raw `GPUTextureUsage` where `Texture` constants exist
- Class: `adopt-now`.
- Local: `examples/deck/spatial-analysis-showcase/src/scenes/raster/spectral-texture-display.ts:236,245`.
- Published: `Texture.STORAGE = 0x08`, `Texture.SAMPLE = 0x04` (`core/dist/api/adapter/resources/texture.js:25-27`),
  equal to `GPUTextureUsage.STORAGE_BINDING | TEXTURE_BINDING`. Evidence: read both.
- Change: `Texture.STORAGE | Texture.SAMPLE` on 2 lines. Verify: `check-sprint-scenes.mjs` for the
  raster scene. Risk: none (same bits). These are the only raw WebGPU constants in non-script source.

### WE-07 Showcase requests a raw adapter to read limits
- Class: `adopt-with-cost`.
- Local: `examples/deck/spatial-analysis-showcase/src/engine/deck-host.ts:1113-1140`
  (`navigator.gpu.requestAdapter()` then `requiredLimits` from `adapter.limits`).
- Published: `DeviceProps.featureLevel: 'max'` requests every adapter limit; explicit `requiredLimits`
  still override (`core/dist/webgpu/adapter/webgpu-adapter.js:184-186`, `device.d.ts:199-214`).
- Loss: `'max'` also requests every supported feature; published `GPUReduction` switches to subgroup
  collectives when the device has them (`gpu-reduction.d.ts:5-8`), which can change float-sum
  rounding; the cap `REQUESTED_STORAGE_BUFFERS_PER_STAGE` would no longer apply. Keep unless the
  showcase wants adapter maxima anyway. Explorer (`app.ts:359-360`) does not raise limits at all.

### WE-08 Examples never observe device loss
- Class: `unused-feature`.
- Local: 0 uses of `.lost` in showcase, explorer and shared code; the website does it
  (`gpu-canvas.tsx:258-261`). Published: `Device.lost` (`device.d.ts:383-386`).
- Change: one `device.lost.then(...)` in `deck-host.ts` after device creation (and in explorer
  `app.ts`) to show an error state instead of a frozen map. Verify by `device.destroy()` from the
  console. Risk: none.

### WE-09 Node specs fake a WebGPU device because `getNullTestDevice` cannot build graphs
- Class: `blocked` (not in the register).
- Local: `modules/analysis/src/testing/gpu-contributor-test-utils.ts:21-38` (`createNullWebGPUDevice`:
  `NullDevice` with `type` and `limits` overwritten by `Object.defineProperty`), used by 466 of 888
  `*.node.spec.ts` files (483 files overall). This is how node specs get a device for construction-time
  validation and binding-count checks (8 storage buffers, 16 KiB workgroup memory).
- Published: `getNullTestDevice()` returns `type = 'null'` (`test-utils/null-device/null-device.d.ts:18-25`)
  with every compute limit 0 (`null-device-features.js:10-45`), and `GPUCommandGraph.addNode` throws
  for compute nodes on non-WebGPU devices (`graph/dist/gpu-core/gpu-command-graph.js:624-628`).
- Needs from luma.gl: a null/test device that reports the WebGPU type and core limits (or honours
  `requiredLimits`), or a graph flag that allows declaration-only graphs on the null device. Until then
  the local shim is correct; it is fragile because it overwrites class fields.

### WE-10 No leak check, though specs destroy diligently; published helper is half stale
- Class: `unused-feature` (with a defect to report upstream).
- Local: 864 of 1,110 GPU spec files call `.destroy()` (4,370 calls); no `afterEach`/`onTestFinished`
  hook anywhere and no resource-count assertion.
- Published: `getResourceCounts`/`getLeakedResources` (`core/dist/test-utils/utils/resource-tracker.js:5-26`)
  read `luma.stats` 'GPU Resource Counts'. `'Buffers Active'` is still maintained
  (`api/adapter/resources/device-resource.js:12,217-220`), but `'Texture2Ds Active'` is not: luma 10
  names the bucket `'Textures'`, so the `Texture2D` count is always 0.
- Change: a browser-project setup file that records `getResourceCounts().Buffer` (and
  `luma.stats.get('GPU Resource Counts').get('Textures Active').count` directly) before each spec and
  reports `getLeakedResources` after it. Start as a report, not a failure. Report the stale texture
  stat name upstream.
- Risk: false positives from the cached shared test device, graph transient pools and deliberate
  per-file fixtures; needs a baseline like `contributor-contract-baseline.json`.

### WE-11 Kepler specs read GPUVector chunks by hand
- Class: `adopt-with-cost`.
- Local: 23 test files read `chunk.buffer.readAsync` / `data[0].buffer.readAsync`, e.g.
  `modules/kepler/test/aggregation.spec.ts:269-280`, `geojson-property-runtime.spec.ts:111-122`,
  `h3-runtime.spec.ts:62`. `makeCPUVectorFromGPUVector` is used nowhere.
- Published: `makeCPUVectorFromGPUVector` (`graph/dist/gpu-data/cpu-gpu-vector-transfer.d.ts`:
  preserves chunks, list offsets, validity), `CPUData` (`buffer`, `byteOffset`, `byteStride`,
  `valueOffsets`, `nullBitmap`).
- Loss: none in values, but `CPUData` has no typed accessor, so a small typed-view helper remains;
  the gain is correct handling of multi-chunk and strided vectors. Evidence: read one.

### WE-12 Docs snippet converts readback bytes element-wise
- Class: `misuse` (docs).
- Local: `docs/api/gpu-network/operators.md:297`:
  `decodeGPUNetworkStatistics(new Uint32Array(await statsBuffer.readAsync()))`.
- Published: `Buffer.readAsync` returns `Uint8Array` (`core/dist/api/adapter/resources/buffer.d.ts:58`);
  `new Uint32Array(uint8Array)` widens each byte into its own word.
- Change: `const bytes = await statsBuffer.readAsync(); new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)`.
  1 site (no other occurrence in docs, website, examples or modules).

### WE-13 Deck.gl patch: still needed, public surface, but deck still runs luma 9 shader modules
- Class: `adopt-with-cost` (lead; not run).
- What the patch does (`.yarn/patches/@deck.gl-core-npm-9.4.0-707f3fb147.patch`, 1,875 lines): rewrites
  `@luma.gl/webgl` to `@luma.gl/core/webgl`, `@luma.gl/webgl/constants` to `@luma.gl/core/webgl-constants`,
  `@luma.gl/gpgpu` to `@luma.gl/graph/gpu-simple-graph` (+ `/webgpu`); switches math helpers to
  `@math.gl/core`; WebGPU fixes (scissor and picking-readback y origin, depth on the first effect
  framebuffer, a WGSL `select` rewrite, clearing hover on canvas leave). `.yarnrc.yml` adds
  `@luma.gl/graph` as a dependency of `@deck.gl/core@9.4.0` so those rewritten imports resolve; both
  are still needed while deck 9.4.0 runs on luma 10.
- Public at alpha.13: yes. Every `@luma.gl/core|engine|graph` name the patched deck core imports
  resolves to a public export (script check; `gpu-simple-graph` and `gpu-simple-graph/webgpu` are in
  `@luma.gl/graph` exports; `GL` is an enum in `core/webgl-constants`).
- Gap: patched deck core and unpatched deck layers still import 7 names from `@luma.gl/shadertools@9.4.2`
  (`ShaderAssembler`, `picking`, `fp32`, `fp64arithmetic`, `gouraudMaterial`, `phongMaterial`,
  `initializeShaderModule`), so a luma 9 `ShaderAssembler` and luma 9 picking module feed luma 10
  `Model`s. All 7 exist in `@luma.gl/shaders@10.0.0-alpha.13`. `@luma.gl/webgl`, `gpgpu`, `constants`,
  `gltf` 9.4.2 are installed only as declared-but-unused deck dependencies. kepler.gl pulls a second,
  unpatched `@deck.gl/core@9.4.1`.
- Change: extend the patch hunk set to map `@luma.gl/shadertools` to `@luma.gl/shaders` in deck core
  (and patch `@deck.gl/layers`, 5 requires). Verify with the showcase scene checks and picking.
  Risk: luma 10 picking/project module uniforms may differ from what deck 9.4 expects; that is why this
  is a lead, not a recommendation to apply blindly.

### WE-14 Minor
- `adopt-now`: showcase `engine/layers.ts:737-741` finds declared bindings with a regex that also
  matches commented-out declarations; apply published `maskWGSLComments` (`@luma.gl/shaders/wgsl`,
  re-exported by `@luma.gl/core/wgsl-binding-scan`) first. 1 site. Risk: none.
- `adopt-now`: `website/src/gpu/runs/harness.ts:118-121` adds its own `uncapturederror` listener on
  `device.handle`; luma's `WebGPUDevice` already listens and routes to `DeviceProps.onError`
  (`webgpu-device.js:58-64,89`). Passing `onError` through `createGpuDevice` removes the raw handle use.
  Harness only.

## 3. Checked and fine

- Website device creation is one luma path: `luma.createDevice({type, adapters: [webgpuAdapter] | [webgl2Adapter], createCanvasContext: {autoResize, useDevicePixels}})` (`device.ts:27-52`). The `navigator.gpu.requestAdapter()` probe is justified: `type: 'best-available'` decides by `isSupported()` only (`core/dist/api/adapter/luma.js:113-117`) and would not fall back to WebGL when the adapter request returns null.
- `gpu-canvas.tsx:519-533` uses `device.handle.pushErrorScope` directly: correct, because luma's `WebGPUDevice.pushErrorScope/popErrorScope` are no-ops unless `props.debug` (`webgpu-device.js:356-375`).
- Device loss on the website: handled (`gpu-canvas.tsx:258-261`), keeps the static fallback.
- Website rendering: every run and demo draws the operation's storage buffers directly with engine `Model` (`demos/kit.ts:669-677`, bindings, instanced quads); no 2D-canvas copies. Of 515 `defineOperationRun` runs, 430 draw only from GPU buffers (plus the CPU auto-range of WE-01); 85 re-upload readbacks (WE-02). Previews use `Model` with WGSL or GLSL by backend.
- No `AnimationLoop`: right for the website. Cards render on demand through `DrawScheduler` (one rAF per change, `kit.ts:856-881`), gated by `IntersectionObserver` and reduced motion; `AnimationLoop` redraws continuously. The examples run inside deck's own `AnimationLoop`.
- Pointer input is coalesced to one rAF (`canvas-pointer.ts:78`); `peek` readbacks are coalesced (`define-run.ts:650-699`).
- Example readback: `GPUReadbackRing` everywhere it is per-frame (showcase `SummaryReader`, 171 files; explorer 35 files); ad hoc `readAsync` only in one-off verification paths (6 showcase, 4 explorer files).
- Example timing: `spatial-analysis-shared/vector-timing.ts` uses `createQuerySet` + `timeProfilingQuerySet` + published `encoding.readTimings()` with a wall-clock fallback; no hand timing of GPU work.
- The double-rAF deferred destroys (about 20 sites, e.g. `deck-host.ts:862`, explorer `buffer-mode.ts:303`) wait for deck to swap layers, not for GPU completion; a luma `Fence` would not express that (deck's `onAfterRender` would, but that is deck, not luma).
- Test devices: `getWebGPUTestDevice` (1,111 files) caches and refreshes lost devices (`create-test-device.js:49-53`). It defaults to `'max'` (3,180 calls vs 40 `'core'`), but storage-binding and workgroup-memory limits are enforced at declaration in node specs through the null device's core limits (WE-09), so the old "max hides limit bugs" finding (`dev-docs/archive/analysis-contributors-quality-roadmap.md:173`) is mitigated. `getWebGPUTestDevices(['core','max'])` is available if a spec wants both.
- `checkType`, `deepCopy`, `getTestDevices`, `NullAdapter`: nothing local duplicates them.
- Docs: 15 pages import 9 luma.gl names, all public entry-point exports; `compileAsync()`, `encode(commandEncoder, {parameters})`, `device.commandEncoder`, `device.submit()` match alpha.13 (`gpu-command-graph.d.ts:311,324,393`, `device.d.ts:321`). The website "anatomy" example (`kernel-density.example.ts`) is idiomatic. No page shows device creation (all use `declare const device`); adding the two-line `luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]})` once in `concepts.md` would complete it.
- `requireTestDevice` (`src/testing/require-test-device.ts`) has no published counterpart; keep.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

The register covers only `modules/*/src` (plus the parked kepler-compute row). Missing rows:
1. Deck.gl 9.4.0 patch and `.yarnrc.yml` `packageExtensions` (WE-13): what it rewrites, that it must
   move with every alpha bump (today's bump did), and the remaining `@luma.gl/shadertools@9.4.2` imports.
2. `createNullWebGPUDevice` test shim (WE-09): what luma.gl needs (WebGPU-typed null device with core limits).
3. Optionally, a consumers section for the website text stack beside `@luma.gl/engine/text` (WE-04)
   with its reasons (halo, poster determinism, layout).
No existing row was found wrong for this stream's scope.
