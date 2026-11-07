# Upstream workarounds register

Status date: 2026-11-07. Baseline: published `@luma.gl/*@10.0.0-alpha.10`.

This repository imports luma.gl only through published entry points (see [upstream-first-plan.md](./upstream-first-plan.md)). Where the published release is missing a capability or a fix that the pre-split branch had, we either dropped the capability, wrote a small local workaround, or accepted a regression. This file lists every one of them, so that each `@luma.gl` bump can retire the ones the release fixes.

Paths are relative to the repository root and were checked on the date above. The source tree is still changing, so re-grep before deleting anything.

## How to read the table

| Status | Meaning |
| --- | --- |
| dropped | The capability no longer exists here. Callers get an error or a smaller limit. |
| workaround | Local code stands in for a missing luma.gl API. Delete it when the API ships. |
| accepted regression | We use the published primitive as is and live with a performance or scale loss. No local code. |

The "what luma.gl needs" column says what would let us restore or delete. These are suggestions for luma.gl pull requests. This repository never waits on them (rule 3 of the plan).

## 1. Dropped capabilities

| Capability | Effect or limit | Local code to delete or restore | What luma.gl needs | Status |
| --- | --- | --- | --- | --- |
| Sparse resistance connectivity (circuit-theory effective resistance and current maps at scale, EC03) | Only the small CG recipe remains: at most 256 vertices, 65,536 edges, 4,096 iterations. `addSparseResistanceConnectivityRecipe`, `addSparseResistanceComponents`, `addSerialSparseResistanceComponents` and `GPURasterResistanceProps` are gone. The 65,538-vertex Circuitscape-validated path is lost. Not listed in [upstream-first-pending.md](./upstream-first-pending.md). Docs still describe it: `docs/api/gpu-spatial-analysis/landscape-ecology.md`, `urban-ecology-lenses.md`. | Restore on public APIs inside `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/`. Fix the stale comment at `resistance-connectivity.ts:33` ("use the general sparse solver for larger systems"). | Export `GPUJacobiPCG` (it exists in `@luma.gl/graph` at `dist/gpu-core/gpu-pcg.d.ts` but `gpu-core/index.d.ts` does not re-export it). `createGPUConjugateGradientGraph` is also not public. | dropped (not yet documented in user docs) |
| Strongly connected components on large graphs | `GPUNetworkStructure` keeps only the exact mutual-reachability SCC: at most 1,024 vertices and 65,536 CSR slots, O(V²) scratch (`gpu-network/network-structure/gpu-network-structure.ts:45-67`). The `'iterative'` method and the SCC lens scenario are gone. | Write an iterative SCC in `modules/analysis/src/gpu-network/` if a consumer needs it. | A public `GPUGraphAlgorithmsStronglyConnectedComponents`. | dropped |
| Command-graph-transient CSR and column views (`GPUGraphAlgorithms*View`) | Callers must hold physical `GPUVector<'uint32'>` CSR storage and physical outputs. | `gpu-network/network-statistics`, `network-structure` and `network-analysis/gpu-network-analytics-columns.ts` use physical vectors. Nothing to restore here. | Published algorithms that accept command-graph views. | dropped |
| Histogram-cube / skip-cube crossfilter | Replaced by published `GPUCrossfilter` (`@luma.gl/graph/gpu-crossfilter`). | None. | None needed. | dropped |
| Residency arena, `GPUResidentRowSelection`, tile LOD quadtree and selection | Gone. Kepler uses its own `resident-*` code. The explorer tile-LOD mode is removed from source (only stale `dist/` bundles mention it). | None. | None needed (not analysis). | dropped |
| Onesweep sort, decoupled look-back scan, double-single helpers, benchmark harness | Large-array sort and scan throughput follows the published `GPUSort` and `GPUScan` defaults. | None. | Faster sort and scan strategies in `@luma.gl/graph/gpu-core`. | accepted regression |
| Arrow batch stream (`appendGPUTableFromArrowBatches` and friends) | No streaming of Arrow batches into a table. Use `convertArrowRecordBatchesToGPURecordBatches` from `@luma.gl/graph/arrow`. | None. | None needed. | dropped |
| Partition descriptors (`getGPUPartitionDescriptorFromArrowTable`) | Callers build partition records locally or skip them. | `analysis-core/graph-support/` has no replacement. | A public partition-descriptor builder. | dropped |
| `GPUCPUOperation` CPU fallbacks, `GPUKernelNode`, planning ordered set | Gone. Nodes are written with `createWGSLKernelNode`. | None. | None needed. | dropped |
| Dataframe query description (`describeGPUDataFrameQueryIntent`, `...ShaderPlan`, `createGPUDataFrameExpressionFormulas`) and the 8 dataframe-query lens scenarios | Lenses cannot show query intent or formula views. | `modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/` no longer holds query-description scenarios. | Public query-plan metadata from `@luma.gl/graph/gpu-dataframe` (plan steps with source expressions). | dropped |
| Lens packs: `gpu-core` (25 scenarios), dataframe query (8), SCC (1), SQL (1) | Fewer lens scenarios than on the branch. | Scenario packs live in `modules/lenses/src/computation-lenses/scenarios/`. | None needed. | dropped |
| `captureGPUCommandGraph` stepped capture | Replaced by `GPUSnapshotTap` copy nodes. Taps end compute-pass coalescing and add copy cost. See [upstream-first-recording.md](./upstream-first-recording.md). | `modules/analysis/src/computation-record/gpu-snapshot-tap.ts` | Public per-node capture or a "retain" flag on `GPUCommandGraph`. | workaround |
| Geometry-vector helper `makeGeometryGPUVectors` (engine/arrow) | Three specs cannot run: `modules/kepler/test/host-wkb-selection.spec.ts`, `modules/analysis/test/arrow/arrow-column-info.spec.ts`, `arrow/get-arrow-data.spec.ts`. They are excluded in `vitest.config.ts:17-20`. | Excluded specs. | A public `makeGeometryGPUVectors` in `@luma.gl/engine` or `@luma.gl/graph/arrow`. | dropped |
| 3D datum conversion (ECEF, static Helmert, epoch resources) | Not available here or in the published `gpu-project`. Only supplied-grid horizontal and vertical shifts (`GPUDatumGridTransform`) exist. | `modules/analysis/src/geospatial-analysis/gpu-datum-grid-transform.ts` | Belongs in luma.gl `gpu-project`. | dropped (never in this repo) |

## 2. Local workarounds

| Capability | Effect or limit | Local code to delete or restore | What luma.gl needs | Status |
| --- | --- | --- | --- | --- |
| Computation-record annotations (`graph.recording`, `node.annotation`, `withRecordScope`, `compiled.describe()`) | `record: {...}` options on `createWGSLKernelNode` and the helper props are accepted and ignored. 32 call sites in 23 files under `modules/analysis/src` carry metadata that no code reads. Lens records get structure and timings only, no source text, bindings, formulas, loop provenance or `derivedFrom`. | `modules/analysis/src/computation-record/` (`computation-recorder.ts`, `describe-command-graph.ts`, `create-kernel-annotation.ts`, `gpu-snapshot-tap.ts`). Type `WGSLKernelRecordMetadata` and the `record` option in `analysis-core/kernels/wgsl-kernel-nodes.ts` (lines 73, 111, 296, 618, 709). | A public annotation or metadata hook on `GPUCommandGraph` nodes, readable from `CompiledGPUCommandGraph.preflight`. Then pass `record` through to it and delete the recorder. | workaround (recorder exists; no analysis contributor passes one) |
| Kernel node construction (`getSpatialOperationNodes`, packed read and write helpers) | Point-cloud kernels build nodes through a local helper. | `modules/analysis/src/pointcloud-analysis/pointcloud-kernel-nodes.ts` | Public `getSpatialOperationNodes` or an equivalent typed packed-binding helper. | workaround |
| Segmented sum (`GPUSegmentedReduction` fast path) | Always runs the local 256-lane tree kernel. Results are bit-identical; the upstream fast path is lost. `offsetsWithinInput` and `implementation` options are accepted and ignored. | `modules/analysis/src/analysis-core/kernels/sorted-segment-sums.ts` | Public `GPUSegmentedReduction`. | workaround |
| Bind-group reuse (`_bindGroupCacheKeys` on `Kernel.dispatch`) | `setBindings` is called without cache keys, so more bind groups are created per encode. | `createChangedBindingResolver`, `createCachedBindingResolver` in `analysis-core/kernels/wgsl-kernel-nodes.ts` | Public cache-key option on `Kernel.dispatch`. | workaround |
| Indirect-dispatch gate metadata (`setGPUComputeDispatchWorkgroups`) | Predicate-controlled indirect dispatch swap is lost. Behaviour is correct; only the optimisation is gone. Raster iteration uses its own gate node. | `raster-analysis/cost-distance/raster-relaxation.ts` (`createRasterIterationGateNode`); `createWGSLDispatchGate` in `wgsl-kernel-nodes.ts` | Public dispatch-workgroups metadata on compute nodes plus a GPU-gated continuation in the compiler. | workaround |
| Bounded dispatch overrides (`getBoundedDispatchOverrides`) | Analysis kernels use public `getBoundedDispatchLayout` through `createWGSLKernelNode`. Fine for analysis kernels. See section 3 for published primitives. | `analysis-core/kernels/wgsl-kernel-nodes.ts` | Public `getBoundedDispatchOverrides`. | workaround |
| Raster validation (`validateRasterBand`, `validateRasterMetadata`, `getRasterDispatchSize`, `getRasterLocalOrigin`, `validateRasterExtent`, `makeGPURasterBand`) | Replaced by domain helpers. | `modules/analysis/src/raster-analysis/raster-band-utils.ts` (for example `validateRasterGrid:38`, `validateRasterBand:266`, `validateRasterParameterExtent:252`) | Public raster validation helpers and band constructors. | workaround |
| Vertical-datum metadata beside a raster band | `makeGPURasterBand` is not public, so terrain keeps vertical-datum metadata beside the band. | `modules/analysis/src/raster-analysis/raster-grid/gpu-raster-grid.ts` (the only place that reads `verticalDatum`) | `verticalDatum` on public `GPURasterMetadata`. | workaround |
| Arrow analytics normalisation (`prepareArrowTableForGPUAnalytics`, time upload, wide numeric) | Rebuilt locally on public `@luma.gl/graph/arrow` and `gpu-data`. Used by kepler. | `modules/analysis/src/analysis-core/graph-support/` (`arrow-analytics-normalization.ts`, `arrow-analytics-time-upload.ts`, `arrow-analytics-wide-numeric.ts`). Note: `package.json` of `@luma-contributors/analysis` has no `./analysis-core/graph-support` export; kepler and examples need one or relative imports. | Public `prepareArrowTableForGPUAnalytics` in `@luma.gl/graph/arrow`. | workaround |
| `waitForSubmittedWork` | Replaced by `device.createFence()` and `await fence.signaled`. | `modules/kepler/src/session/compute-session.ts:469-474`, `modules/kepler/src/exploration/gpu-table-inspector.ts:539` | None; the replacement is public and adequate. | workaround (done; can stay) |
| Out-of-memory detection | Matches by `error.name` (`OutOfMemoryError`, `GPUOutOfMemoryError`) instead of an exported class. | `modules/kepler/src/runtime/dataset-registry.ts:852-854` | A public error class or type guard from `@luma.gl/core`. | workaround |
| Crossfilter extensions (external masks, `invalidate`, `setExternalActive`) | Kepler builds selection masks on the public crossfilter API or in its own kernel. | `modules/kepler/src/selection/` (`gpu-spatial-selection.ts`, `gpu-time-selection.ts`) | External-mask and invalidate options on `GPUCrossfilter`. | workaround |
| Primitive signature changes (`GPUGridIndex` `boundsBuffer`, `GPUCompaction` `strategy`, point-in-polygon `queryPositions`) | Call sites use the published signatures. | Call sites in `gpu-spatial-analysis` and `dataframe-analysis`. | None. | workaround (done) |
| Kepler-compute example | Parked: not in `workspaces`, depends on `@deck.gl-community/gpu-layers` and `@deck.gl/core`. 271 of 367 kepler spec files are excluded from every run because they need it. | `examples/deck/kepler-compute/`, `test/kepler-compute-example-specs.json`, `vitest.config.ts:14-21` | Published `@deck.gl-community/gpu-layers` compatible with luma.gl alpha.10. | accepted regression |

## 3. Accepted regressions on published primitives

These need no local code. They come from the published primitives lacking fixes the branch carried.

| Capability | Effect or limit | Local code | What luma.gl needs | Status |
| --- | --- | --- | --- | --- |
| Bounded dispatch above about 4.2M rows in published `GPUScan`, `GPUSort`, `GPUCompaction`, `GPUReduction` | Recipes that route through them may mis-handle or reject inputs above roughly 4.2M rows. Unverified here: no test in this repo pins the failure mode. | None. Add a test per primitive that records the actual behaviour, then add input-size assertions in the recipes that call them. | Bounded-dispatch layout in every published primitive. | accepted regression (risk: silent wrong results at scale) |
| Compensated (Kahan or fixed-point) sums in published reductions | Float32 sums over very large inputs drift more than on the branch. Analysis code that needs exactness uses the local fixed-point accumulator. | `modules/analysis/src/analysis-core/kernels/fixed-point-accumulator.ts` stays. No change needed for analysis kernels. | Compensated summation in `GPUReduction`. | accepted regression |
| Block-compaction fixes in published `GPUCompaction` | Large or adversarial compactions may fall back or fail. | None. | The block-compaction fixes. | accepted regression |
| Centrality compile cost | `GPUNetworkCentrality` unrolls reachability, sigma and delta rounds at compile time (default 64, at most 1,024; `gpu-network/network-centrality/gpu-network-centrality.ts:149`). At the maximum this is a very large command graph. The branch fix needs a GPU-gated continuation the published compiler does not provide. | `gpu-network-centrality.ts` | GPU-gated continuation or indirect-dispatch gate in `GPUCommandGraph`. | accepted regression |
| Exact GPU kernel throughput for point-in-polygon, hash join, and others | Not measured after the split. | None. | None. | unmeasured (see [roadmap.md](./roadmap.md) item UF-03) |

## 4. Upgrade procedure for each `@luma.gl` alpha bump

1. Read the luma.gl release notes and diff the public `.d.ts` files. Compare `node_modules/@luma.gl/graph/dist/**/index.d.ts` before and after, not individual files, since only index exports are public.
2. For each row above, check whether the release provides the "what luma.gl needs" item. Search for the symbol in the new `index.d.ts` files, for example `GPUJacobiPCG`, `GPUSegmentedReduction`, `getSpatialOperationNodes`, `makeGeometryGPUVectors`.
3. Delete the local workaround, switch callers to the public API, and move the row to a "Retired" list at the bottom of this file with the release version.
4. Run `node scripts/check-upstream-first.mjs`. It must report no violations.
5. Bump all `@luma.gl/*` versions together in the root `package.json`, `modules/*/package.json` peer ranges and `yarn.lock`.
6. Run `yarn typecheck`, `yarn build`, `yarn lint`, `node scripts/verify-package-exports.mjs`, then `yarn test-node` and `yarn test-headless`.
7. Investigate numerical drift in oracle and parity tests. Accept it with a note in the test, or fix it in the analysis code. Do not loosen tolerances silently.
8. Re-run the benchmarks that guard section 3 (`LUMA_CONTRIBUTORS_BENCHMARKS=true`) and record the numbers next to the row.
9. Add any new workaround or dropped capability to this file in the same change.

## Retired

None yet.
