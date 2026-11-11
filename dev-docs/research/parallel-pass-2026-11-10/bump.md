# UF-11 bump report: `@luma.gl/*` 10.0.0-alpha.10 to 10.0.0-alpha.12

Stream `bump`. Worktree `.claude/worktrees/agent-ad42cfd8f556b49d5`, base commit `74dddcf5` (fast-forwarded from `edf9ca10` as the coordinator asked). Nothing committed. Date 2026-11-10.

## Summary

- `10.0.0-alpha.12` exists for every pinned package: `@luma.gl/core`, `shaders`, `graph` and `engine`. It is npm's `beta` tag. The repository does not pin `@luma.gl/webgpu`, `webgl` or `constants`. Those have no alpha.12, but they only come in as deck.gl's 9.4 dependencies, so that does not matter here.
- **No source change is needed.** Every gate passes with only the version pins, `.yarnrc.yml` and `yarn.lock` changed.
- **No workaround can be retired.** alpha.12 removes nothing public. What it adds is CPU and portable graph execution, WebGL render passes in `GPUSimpleGraph`, PBR and lighting in engine and shaders, and CPU and GeoArrow projection helpers. None of that is what a register row is waiting for.
- **NW-CENT is not unblocked.** `gpu-command-graph-types.d.ts` is byte-identical. The `GPUCommandGraph` compile and encode JS for WebGPU compute is unchanged apart from a base-class split.

## 1. What I did

1. Bumped all 29 `@luma.gl/{core,engine,graph,shaders}` pins together:
   - root `package.json` (`devDependencies` and `resolutions`)
   - the `~` peer ranges in `modules/{analysis,kepler,lenses}/package.json`
   - `website/package.json`
   - `examples/deck/spatial-analysis-{explorer,showcase}/package.json`
   - plus the `packageExtensions` entry for `@deck.gl/core@9.4.0` in `.yarnrc.yml` (`"@luma.gl/graph": "10.0.0-alpha.12"`), which also pinned alpha.10.
2. Did not bump `examples/deck/kepler-compute/package.json`. It is not a workspace (it is parked) and still says alpha.10.
3. Ran `yarn install` (Yarn 4.13.0, Node 22.23.4 from `.nvmrc`). The first attempt died with ENOSPC (full disk). After the coordinator freed space, `rm -rf node_modules && yarn install` finished cleanly: "Done with warnings", only the existing peer-dependency warnings (YN0060, YN0002, YN0086). `node_modules` is 1.8 GB.
4. Did not deduplicate `@math.gl/*`. I tried `yarn dedupe '@math.gl/*'` once. It moves the repository's own `@math.gl/geospatial` to 5.0.0-beta.6, and beta.6 **drops the `@math.gl/geospatial/geoarrow` subpath**. Kepler imports it in `modules/kepler/src/analysis/catalog-geoarrow.ts` and `catalog-gpu-pack.ts`. `yarn verify-package-exports` then failed with `Package subpath './geoarrow' is not defined by "exports"`. I reverted `yarn.lock` and reinstalled without the dedupe.
   - **The file `scratchpad/reports/bump-pins.patch` is superseded.** It was saved while the dedupe was in the lock. Use `bump.patch`.

## 2. Public API differences (index exports only)

Method:
- Export maps of all four packages compared.
- For every typed entry point, the exported names and declaration text were resolved through the TypeScript checker (`getExportsOfModule`), alpha.10 (main checkout `node_modules`, read-only) against alpha.12 (worktree).
- Then file-level `diff` of `.d.ts` and `.js` under `@luma.gl/graph/dist` and `@luma.gl/core/dist`.
- Neither package ships a CHANGELOG or release notes; the tarballs hold only `LICENSE`, `README.md`, `dist` and `src`.

**Removed public symbols: none, in any entry point.**

New entry points:
- `@luma.gl/graph/gpu-project/geoarrow`
- `@luma.gl/graph/gpu-project/graph`
- `@luma.gl/engine/lighting`

Changes that touch APIs this repository uses:

| Area | Change | Effect here |
| --- | --- | --- |
| `GPUCommandGraph` (`graph`, `gpu-core`, `gpu-graph`) | Split into `GPUCommandGraphBuilder` (base, `@internal`, exported from `gpu-graph`) and `GPUCommandGraph extends GPUCommandGraphBuilder`. Same constructor `(device, {id?, autotuner?})`. The WebGPU-only check moved into the subclass constructor. The base now throws `'WebGL graphs support render passes only'` for non-render nodes on non-WebGPU devices. Adds `getUniqueResourceId` (`@internal`). | None. The 1,395 `gpu-core` imports compile unchanged. Compute behaviour on WebGPU is unchanged (JS diff is only the class split, the WebGL guards and the id helper). |
| `gpu-command-graph-types.d.ts` | Byte-identical: node, condition, indirect, preflight and timing types. | No new gate or continuation. See NW-CENT below. |
| `composite`, `GPUCompositeOperation` | Now generic over the operation type (`GPUCompositeOperation<Operation>`). | None (no repository use). |
| `GPUGraph`, `CompiledGPUGraph`, `GPUGraphRunBindings`, `GPUGraphOutput`, `GPUGraphCompileOptions` | Accept `CPUVector` bindings and outputs, plus a new `outputLocations?: Record<string, 'cpu' \| 'gpu'>`. | None (no repository use of `GPUGraph` programs). |
| New `GPUCPUOperation`, `GPUShaderPassOperation`, `GPURenderOperation` (`gpu-core`) | Portable CPU operations and shader passes for `GPUGraph` and `GPUSimpleGraph`. | Same name as the branch's dropped `GPUCPUOperation`, but a different design (CPU nodes in `GPUGraph`, not CPU fallbacks for analysis kernels). Not a replacement. |
| `@luma.gl/graph/gpu-data` | New `CPUData`, `CPUVector`, `makeGPUVectorFromCPUVector`, `makeCPUVectorFromGPUVector`. | None. Possibly useful for tests and oracles later. |
| `@luma.gl/graph/arrow` | GeoArrow option and inference signature changes, plus `makeCPUVectorFromArrow`, `projectCPUVector`, `tessellateGeoArrowColumnToCPUVectors` and others. | Repository uses compile unchanged. |
| `GPUSimpleGraph` (`gpu-simple-graph`) | Generic `Parameters`, `addRenderOperation`, `invalidate(inputIds)`, WebGL render pass support. | The deck.gl patch (`.yarn/patches/@deck.gl-core-npm-9.4.0-*.patch`) imports `backendRegistry`, `cleanEvaluateSync`, `GPUDataEvaluator`, `interleave` and `gpu-simple-graph/webgpu` `interleave`. All are still exported. |
| `@luma.gl/core` | `Texture.computeMemoryLayout(options, result?)`. `RenderPassParameters.scissorRect` accepts `null`. `DataTypeInfo` fields are `readonly`. New `WebGLRenderPassParameters`. `Device`, `DeviceProps`, adapter and `PresentationContext` declarations are unchanged. | None (no repository use of the changed members). |
| `@luma.gl/shaders`, `@luma.gl/engine` | PBR, shadow, contact-shadow, clustered-light and lightmap modules. Engine adds scenegraph interaction and `engine/lighting`. | None. |
| `@math.gl/*` | luma.gl alpha.12 depends on `@math.gl/*` `~5.0.0-beta.6`. The repository stays on beta.4, so both are installed: beta.6 is nested under `@luma.gl/*`. | Works. See risks. |

Runtime JS is identical between the versions for all of these: `GPUReduction`, `GPUSort`, `GPUScan`, `GPUCompaction`, `GPUCrossfilter`, the graph algorithms, `gpu-raster`, `gpu-vector-search`, `gpu-dataframe` and `geospatial`. The only changed graph JS is in `arrow/geoarrow`, `gpu-core` (command graph, compiled graph, expression plan and operation), `gpu-simple-graph`, `gpu-data`, `gpu-project` and the new CPU files. So I expected no numerical drift, and none showed up.

## 3. Gates (worktree, base `74dddcf5` plus the patch)

| Gate | Exact command | Result |
| --- | --- | --- |
| Source typecheck | `NODE_OPTIONS=--max-old-space-size=8192 yarn typecheck` | Pass, 0 errors (70 s) |
| Build | `yarn build` | Pass (112 s). Wrote 1,147, 997 and 133 CommonJS modules for analysis, lenses and kepler. |
| Package exports | `yarn verify-package-exports` | Pass, 17 entry points verified. Prints the usual "luma.gl: This version of luma.gl has already been initialized" notice. |
| Upstream-first | `node scripts/check-upstream-first.mjs` | Pass, "no violations" |
| Contributor contract | `yarn check-contributor-contract` | Pass, "no new findings". Totals unchanged: bare-assert 89, missing-producer-declaration 1, single-invocation-data-loop 46, internal-export 13, luma-reexport 0, multiple-classes 21, root-file 21, unregistered-contributor 2; 0 new. |
| Lint | `yarn lint` | Pass, 0 errors, 64 warnings (existing, mostly `noExplicitAny` in `examples/deck/kepler-compute`) |
| Test typecheck (extra, procedure step 6) | `NODE_OPTIONS=--max-old-space-size=8192 yarn typecheck-tests` | Pass (536 s) |
| Node tests | `yarn test-node` | First run: 774 of 784 files and 8,487 of 8,873 tests passed, 376 skipped, **10 failed. All 10 were `Test timed out`** (30 s to 1,200 s limits) on the loaded machine. They are mostly CPU-executed WGSL models; one is `kernel-audit-statistics`. |
| Node tests, rerun of the 10 | `yarn vitest run --project node <the 10 files>` | Pass: 10 of 10 files, 582 of 582 tests. So the failures were load timeouts, not regressions. |
| Kernel-audit specs | Part of `yarn test-node` (all `kernel-audit*.node.spec.ts`, 50 files) | Pass. `kernel-audit-statistics` passed on the rerun. |
| Headless subset | `yarn vitest run --project headless --reporter=verbose <23 specs>` | Pass: 21 files passed, 1 skipped; 134 tests passed, 410 skipped, 0 failed (45 s). |
| Full headless | `yarn test-headless` | **Not run**, as instructed. |
| Benchmarks (procedure step 8) | `LUMA_CONTRIBUTORS_BENCHMARKS=true ...` | **Not run.** The machine is loaded, and the primitives' JS is unchanged. |

About the headless subset:
- The 410 skipped tests are all in `modules/lenses/test/computation-lenses/capture-records.spec.ts`. It is opt-in (`VITE_CAPTURE_LENS_RECORDS=1`) and writes `docs/lens-records`, so I left it off.
- `modules/kepler/test/point-position-source.spec.ts` was on my list but is excluded by `test/kepler-compute-example-specs.json`, so it never ran.
- These spec files passed, with no tolerance changes:
  - `gpu-primitive-large-input-plat01`: 13 tests
  - `sparse-resistance-solver` and `-connectivity` (EC03)
  - `gpu-network-centrality`: 12 tests
  - cost-distance, plus its continuation and parity-wp4 specs (raster gate node)
  - `gpu-local-frame-projection` and `raster-tile-reprojection` (`gpu-project`)
  - `segment-sum-parity` and `fixed-point-accumulator`
  - `computation-record-capture` and `gpu-transfer-mask` (compiled-graph data)
  - network analytics-columns, statistics and strong-components (graph algorithms)
  - `gpu-cell-compaction` (`GPUCompaction`)
  - `gpu-clustering-extras` (local `GPUKMeans`)
  - kepler `selection` and `time-selection`, and lenses `gpu-crossfilter-selection-backend` (`GPUCrossfilter`)

Numerical drift: none seen. No tolerance was touched.

## 4. Source changes in the patch

None. The patch (`scratchpad/reports/bump.patch`, 9 files, +147 / -84) contains:

- the seven `package.json` files: version strings only
- `.yarnrc.yml`: the deck.gl `packageExtensions` pin, alpha.10 to alpha.12
- `yarn.lock`: about 171 changed lines. It is moderate, not large:
  - the four `@luma.gl/*` entries are replaced, with new checksums
  - workspace peer ranges are updated
  - new `@math.gl/{arrow,core,geometry,geospatial,tasks,web-mercator}@5.0.0-beta.6` entries are added, needed by luma.gl alpha.12
  - `@math.gl/geometry@~5.0.0-beta.4` now resolves to beta.6 alongside the exact beta.4

## 5. Workaround register, row by row

"Still needed" means alpha.12 has no public replacement. The row's code stays.

### Section 1: dropped capabilities

| Row | Verdict | Evidence |
| --- | --- | --- |
| SCC on large graphs | Still needed (dropped) | No `StronglyConnected` symbol anywhere in alpha.12 |
| `GPUGraphAlgorithms*View`, command-graph views (UPS-01) | Still needed | `gpu-graph-algorithms` `.d.ts` and JS are unchanged; still `GPUGraphAlgorithmsTopology` over physical vectors only |
| Histogram or skip cube | No change (none needed) | |
| Residency arena and tile LOD | No change (none needed) | |
| Onesweep, look-back, double-single | Still an accepted regression | `GPUSort` and `GPUScan` JS unchanged |
| Arrow batch stream | No change (none needed) | |
| Partition descriptors | Still dropped | No public builder |
| `GPUCPUOperation` fallbacks and `GPUKernelNode` | Changed (name only) | A new public `GPUCPUOperation` exists, but it is a CPU node for `GPUGraph` and `GPUSimpleGraph` programs, not a CPU fallback for analysis kernels. Do not restore the old names onto it. |
| Dataframe query description | Still dropped | `gpu-dataframe` unchanged |
| Lens packs | No change | |
| `captureGPUCommandGraph` stepped capture | Still needed (`GPUSnapshotTap`) | No retain flag and no per-node capture |
| `makeGeometryGPUVectors` | Still needed | Absent. The new CPU GeoArrow helpers (`tessellateGeoArrowColumnToCPUVectors`, `makeCPUVectorFromGeoArrowColumn`) are CPU-side, not the GPU geometry-vector helper. `host-wkb-selection.spec.ts` stays excluded. |
| 3D datum (ECEF and Helmert) | Still dropped | No ECEF or Helmert in `gpu-project`. It adds only `CoordinateProjectionGraph` and `CPUProjectionOperation`. |

### Section 2: local workarounds

| Row | Verdict | Evidence |
| --- | --- | --- |
| Computation-record annotations | Still needed | No annotation, metadata or recording hook (`gpu-command-graph-types.d.ts` identical; the `annotation`, `withRecordScope` and `recording` matches are internal or unchanged) |
| `getSpatialOperationNodes` | Still needed | Present in one dist file, but not exported from any index |
| `GPUSegmentedReduction` | Still needed | Present in one dist file, not exported |
| Bind-group cache keys | Still needed | `bindGroupCacheKey` occurs only internally, same 6 files as alpha.10 |
| Indirect-dispatch gate (`setGPUComputeDispatchWorkgroups`) | Still needed | Not exported. `GPUCommandGraphGPUIndirectCondition` and `KernelDispatchIndirectOptions` were already public in alpha.10 and are unchanged. |
| `getBoundedDispatchOverrides` | Still needed | Absent. `getBoundedDispatchLayout` unchanged. |
| Raster validation helpers | Still needed | `validateRasterBand` is internal only |
| `verticalDatum` on a raster band | Still needed | Absent |
| Arrow analytics normalisation | Still needed | `prepareArrowTableForGPUAnalytics` absent |
| `waitForSubmittedWork` (fence) | No change (can stay) | |
| Out-of-memory detection | Still needed | No public error class |
| Crossfilter external masks, `invalidate`, `setExternalActive` | Still needed | `gpu-crossfilter` unchanged. The new `invalidate` is on `CompiledGPUSimpleGraph`, not `GPUCrossfilter`. |
| Primitive signature changes | No change | Typecheck passes |
| Kepler-compute example | Still an accepted regression | Unchanged. Its `package.json` still pins alpha.10 (not a workspace, not bumped). |
| Sparse resistance solve (EC03) | Still needed | `GPUJacobiPCG` exists in one dist file but is not exported. `createGPUConjugateGradientGraph` is still public. `GPUGraph` gains CPU and portable execution, but there is still no lowering of a `GPUGraph` program into a caller `GPUCommandGraph`. |

### Section 3: accepted regressions

| Row | Verdict | Evidence |
| --- | --- | --- |
| Bounded dispatch above about 4.2M rows | Still an accepted regression | Primitive JS unchanged; `plat01` spec passes, 13 tests |
| Compensated sums in `GPUReduction` | Still needed (keep `fixed-point-accumulator.ts`) | `GPUReduction` unchanged; no Kahan or compensated path |
| Block-compaction fixes | Still an accepted regression | `GPUCompaction` JS unchanged |
| Centrality compile cost (NW-CENT) | Still guarded | See below |
| Kernel throughput | Unmeasured | Benchmarks not run |

### Other items the brief named

| Item | Finding |
| --- | --- |
| Adapter-limit requests (PLAT-05) | Already public in **alpha.10** and unchanged in alpha.12: `DeviceProps.requiredLimits`, `featureLevel: 'max'`, `optionalFeatures`, `getRequiredWebGPULimits`. Lenses already use it (`device-profile.ts`). What the proposal still lacks upstream is the `'adapter-max'` policy shorthand. |
| Subgroups (PLAT-05) | `'subgroups'` is in `WebGPUDeviceFeature`, and `subgroupMinSize` and `subgroupMaxSize` are in device info, in both versions. No subgroup paths were added to `GPUScan` or `GPUSort`. |
| `buffer_view` (PLAT-05) | Absent in both |
| Published k-means (DUP-02) | `GPUKMeans`, `GPUKMeansLabels` and `GPUKMeansProps` are public from `@luma.gl/graph/gpu-vector-search` in both alpha.10 and alpha.12, unchanged. The collision with the local `GPUKMeans` stays as it is: no new break and no relief. |
| `device.createPresentationContext` | `abstract createPresentationContext(props?)` on `Device`, with the `PresentationContext` and `PresentationContextProps` exports, in both versions. `device.d.ts` is identical. Nothing new. |

### Recommendations (not applied)

- **Retire now: nothing.**
- After this bump, update the register's baseline line to "alpha.12, checked 2026-11-10".
- PLAT-05 docs: say that `requiredLimits` already exists upstream, so only the "adapter-max" policy is the proposal.

## 6. NW-CENT and PLAT-04

**Not unblocked.**
- alpha.12 adds no GPU-gated continuation, no batched indirect dispatch, and no public dispatch-workgroups metadata on compute nodes.
- `gpu-command-graph-types.d.ts` is identical to alpha.10.
- The only compiler changes are the `GPUCommandGraphBuilder` split and WebGL render-only guards.
- The 32,768-node constructor guard in `GPUNetworkCentrality` stays.

## 7. Risks for applying the patch to a changed tree

1. **The lock conflicts easily.** Apply the manifest hunks, then **regenerate `yarn.lock` with `yarn install`** instead of applying the lock hunk, if any other stream touched dependencies.
   - Do not run `yarn dedupe` on `@math.gl/*`. It moves the repository to `@math.gl/geospatial@5.0.0-beta.6`, which drops the `./geoarrow` subpath kepler uses. If the repository later moves its own `@math.gl/*` pins to beta.6, kepler must switch to `@math.gl/arrow/geoarrow` first; beta.6 `@math.gl/arrow` exports `./geoarrow`.
2. **Two `@math.gl/core` versions.** beta.4 for the repository and deck, beta.6 nested under `@luma.gl/*`. Typecheck and tests pass. But math.gl class instances passed into luma.gl APIs, or `instanceof` checks across the boundary, could misbehave. I did not see any in the gates.
3. **The `.yarnrc.yml` hunk is required.** The deck.gl `packageExtensions` pin must move with the others, or the lock keeps an alpha.10 `@luma.gl/graph` for deck.
4. **Other streams' new code.** New code written against alpha.10 in the main tree since `74dddcf5` is fine unless it uses `GPUCommandGraph` on a non-WebGPU device. Both versions throw there; the message differs only for the builder. Code that relies on `composite()` returning the non-generic type is very unlikely to break.
5. `examples/deck/kepler-compute/package.json` still pins alpha.10 (parked, not a workspace). Bump it when that example is unparked.
6. **Not run:** the full headless suite and the benchmarks. The full headless suite should run in wave 2 after the patch is applied.

## Files

- Patch: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/reports/bump.patch`
- Superseded early patch (contains the math.gl dedupe, do not use): `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/reports/bump-pins.patch`
- Surface dumps: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/bump/surface-a10.json` and `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/bump/surface-a12.json`
- Gate logs: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/bump/logs/` (files with the `-2` suffix are the final run)
