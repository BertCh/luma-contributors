# Upstream-first foundation: what changed for area agents

Final state of the foundation layer. Scope: `modules/analysis/src/analysis-core/**`,
`src/testing/**` and `src/index.ts`. The former quarantine directory `src/_pending/` no longer exists.
`analysis-core` and `testing` typecheck against npm `@luma.gl/*@10.0.0-alpha.10` with zero guard
violations. Public luma.gl entry points are the keys of `node_modules/@luma.gl/*/package.json`
`exports` (no wildcards, and no `_internal` segment: `@luma.gl/graph` lists `./_internal/arrow` as an
explicit key and the guard rejects it all the same). Everything below is verified against the installed `.d.ts` files.

## 1. Helpers that kept their name and import path

Import from `modules/analysis/src/analysis-core/kernels/*` (relative) as before.

| Helper | File | Notes |
| --- | --- | --- |
| `createWGSLKernelNode`, `createWGSLActiveCountDispatch`, `createWGSLDispatchGate`, `createChangedBindingResolver`, `createCachedBindingResolver`, `getAutomaticWGSLKernelWorkgroupSize`, `getWGSLFloatLiteral`, ... | `kernels/wgsl-kernel-nodes.ts` | Same signatures. See section 2 for the dropped capabilities. |
| `GPUParameterBuffer`, `captureGraphCommandNodes`, `validateMaximumCount`, `validateCapacity`, `importGraphBuffer`, `validateSinglePackedView`, `getNextPowerOfTwo`, `getStorageBindingElementLimit`, ... | `kernels/gpu-contributor-utils.ts` | Unchanged. `captureGraphCommandNodes` works with published `addToGraph` contributors (for example `new GPUPairwisePointDistance(...).addToGraph(graph)`). |
| `GPUCompactOutput`, `GPUUint32Rows`, `GPUStatusScalar`, port types | `kernels/gpu-contributor-types.ts` | Unchanged. `GPUFloat32Positions` is no longer defined or re-exported here: import the type from `@luma.gl/graph/geospatial`. `scripts/check-upstream-first.mjs` rejects `export ... from '@luma.gl/...'` in `modules/*/src`. |
| `createSegmentSumNode`, `getSortedSegmentSumNodes`, `getSegmentTreeSumWGSL`, `getSegmentSumDispatchShape`, `SEGMENT_WORKGROUP_SIZE`, `getSortKeyBits` | `kernels/sorted-segment-sums.ts` | Same signatures; always uses the local kernel (see section 2). |
| exact predicates, Philox, fixed-point accumulator, `hash-uint32`, `scan-total`, `load-balanced-for-each`, `parallel-merge-tree`, `polygon-edge-slabs`, `stable-lexicographic-index-sort`, `study-window-wgsl`, `wgsl-common` | `kernels/*` | Unchanged. |
| contracts, `cell-key-wgsl`, `lon-lat-ring-wgsl`, `raster-grid`, `network-topology`, `polygon-triangulation/`, `ring-assembly/` | `analysis-core/*` | Unchanged. |

Test helpers: `@luma-contributors/analysis/testing` (`createNullWebGPUDevice`, `requireTestDevice`,
`getGPUCommandNodeKernelSources`, `importGraphBuffer`, ...) is unchanged, built only on
`@luma.gl/core/test-utils`, `@luma.gl/graph/gpu-core`, `@luma.gl/graph/gpu-data`, `vitest`, `wgsl_reflect`.

## 2. Capabilities removed from the kernel-node helpers

Published `GPUCommandGraph` and `Kernel` have no equivalent, so these are gone. Call sites keep
compiling; the options are accepted and ignored where noted.

| Removed | Effect on call sites | What to do |
| --- | --- | --- |
| Computation-record annotations (`graph.recording`, `node.annotation`, `GPUComputeCommandNode.annotation`) | The `record` prop and the `WGSLKernelRecordMetadata` type were deleted from `createWGSLKernelNode` / `createWGSLActiveCountDispatch` and the helper props. `node.annotation = ...` does not type check. | Use `GPUComputationRecorder` from `computation-record` (see `upstream-first-recording.md`); never set `node.annotation`. |
| `setGPUComputeDispatchWorkgroups` metadata (direct-dispatch geometry used by the forked compiler to swap in predicate-controlled indirect dispatch) | Nodes no longer carry it; the published compiler dispatches them directly. | Nothing; behaviour is correct, only the indirect-gate optimisation is lost. |
| `_bindGroupCacheKeys` on `Kernel.dispatch` | The bind-group reuse key is dropped; `setBindings` is called without cache keys. `createChangedBindingResolver` / `createCachedBindingResolver` still return `null` when nothing changed. | Nothing. Slightly more bind-group creation per encode. |
| Upstream `GPUSegmentedReduction` fast path in `createSegmentSumNode` | the local 256-lane tree kernel always runs (bit-identical results). | Nothing. |
| Test exports `addGPUQueryPredicatePass`, `makeGPUQueryExpressionShaderPlan`, `describeGPUDataFrameQuery*`, `createGPUDataFrameExpressionFormulas` | Removed from `testing/index.ts`; the dataframe query describe feature was dropped with its spec. Neither `makeGPUQueryExpressionShaderPlan` nor `addGPUQueryPredicatePass` is public in alpha.10: both are declared in `gpu-dataframe/gpu-expression-shader.d.ts`, and no entry point exports them (every `exports` key of `@luma.gl/graph` was loaded and checked on 2026-11-10; an earlier version of this row said the first one was public). | Nothing to import. Build queries with the public `GPUDataFrame` and `GPUExpression` API of `@luma.gl/graph/gpu-dataframe`. |

## 3. Deleted luma.gl extensions

These quarantined files were deleted and nothing imports them:
`gpu-kernel-node.ts`, `gpu-planning-ordered-set.ts`, `gpu-double-single.ts`, `gpu-scan-lookback.ts`
(look-back scan), `gpu-sort-onesweep.ts` (onesweep sort), `benchmarks.ts`, and the describe and record
files (`gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, `gpu-graph-describe.ts`),
replaced by `computation-record`.

Use published `GPUSort` / `GPUScan` defaults (`@luma.gl/graph/gpu-core`). Accepted limit: no onesweep
or look-back strategies, so large-array sort/scan throughput follows the published defaults.
See `upstream-first-pending.md` for every other dropped capability.

## 4. Replacements for commonly missing luma internals

"Public" means exported from a key of `exports`. Verified by resolving the module exports of every
public `@luma.gl/graph` entry.

| Missing symbol | Public status | Replacement / what to do |
| --- | --- | --- |
| `getBoundedDispatchLayout`, `getBoundedInvocationIndexSource`, `GPUBoundedDispatchLayout` | public, `@luma.gl/graph/gpu-core` | Use directly. `createWGSLKernelNode` already applies them. |
| `getBoundedDispatchOverrides` | not public | Use `createWGSLKernelNode` (constants `WGSL_KERNEL_DISPATCH_X/Y` are baked from `getBoundedDispatchLayout`), or read `x/y/z` off `getBoundedDispatchLayout(...)` and pass them as pipeline constants yourself. |
| `setGPUComputeDispatchWorkgroups` / `getGPUComputeDispatchWorkgroups` | not public | Drop (section 2). |
| `getSpatialOperationNodes` (only point-cloud uses it, 38 call lines) | not public | Build the node with `createWGSLKernelNode(graph, {bindings: [{name, view, type: 'f32'|'u32', access}], invocationCount, body})`. The packed read/write accessors (`readX(row)`, `writeX(row, v)`) are generated by a point-cloud-local helper named for the domain (`getPointCloudKernelNodes` in `pointcloud-analysis/pointcloud-kernel-nodes.ts`). Its `safeNormal` is identical to the published private function and is therefore a registered, pinned copy (see the `GPU_GRID_COORDINATE_WGSL` row). Multi-chunk inputs: iterate `view.data` (see `getGraphVectorData`) and add one node per chunk. |
| `GPUSegmentedReduction` | not public | Use `createSegmentSumNode` / `getSortedSegmentSumNodes` (`analysis-core/kernels/sorted-segment-sums.ts`). For min/max/count reductions use public `GPUReduction` (`gpu-core`, supports a mask and operations), `GPUHistogram`, `GPUSegmentOffsets`, `GPUSegmentedSort` or `GPUSegmentedLayout`, or a `createWGSLKernelNode` per-segment loop like `createSegmentSumKernelNode`. |
| `getGraphVectorData(v)` | not public | Inline: `v instanceof GraphVectorView ? v.data : [v]` (`GraphVectorView` is public from `@luma.gl/graph` and `/gpu-core`; `.data` and `.chunks` are public fields). |
| `alignGraphVectorViews` | not public | Require the operands to share chunk boundaries (compare `.data[i].length`), or split with public `graph.createDataView(buffer, {format, length, byteOffset, byteStride, rowByteLength})` per aligned span. If an area only needs one chunk, require `data.length === 1` with `validateRequirement`. |
| `GPU_GRID_COORDINATE_WGSL` | not public | Do not copy it: `AGENTS.md` forbids copied luma.gl source, and that rule wins over the earlier text of this row. A kernel that needs its own grid uses its own cell arithmetic (`cell-key-wgsl.ts` in `analysis-core`, or a domain helper such as `createPointCloudGridIndex`). Only a kernel that reads state a published grid class built (the cell tables of `GPUGridIndex`, the counts of `GPUGridBinning`) has to decide a cell exactly as that class does, and no public API gives it the function. There, a private copy of the official WGSL function is a registered workaround, not a practice: it needs a row in `upstream-workarounds.md` (section 2), an entry in `modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts`, and a pin spec that compares the copy with the published kernel text. That spec fails on any unlisted copy. The copy must be the form of the class it pairs with: `GPUGridIndex` and `GPUGridBinning` use different formulas. |
| `getRasterLocalOrigin`, `validateRasterExtent` | not public | Local origin: compute `[extent[0], extent[1]]` (or the metadata origin) inline in `raster-analysis`; extent check: use the domain helper `validateRasterParameterExtent` in `raster-analysis/raster-band-utils.ts` (no bare `assert`, per `AGENTS.md`). |
| `assertRasterStorageBindingFits` | declared in `gpu-raster/raster-utils.d.ts` but **not public** | Use `validateGPUStorageBindingByteLength` (`analysis-core/kernels/gpu-contributor-utils.ts`) or `getStorageBindingElementLimit`; both take the device limit. |
| `validateRasterBand`, `validateRasterMetadata`, `validateRasterScalarView`, `getRasterDispatchSize` | not public (exist in `gpu-raster/raster-utils.d.ts`) | Validate with `validateSinglePackedView` / `validateExtent` / `validateCapacity` from `gpu-contributor-utils.ts`, or the domain helpers `validateRasterGrid` and `validateRasterBand` in `raster-analysis/raster-band-utils.ts`. No bare `assert` (`AGENTS.md`). For 2D dispatch use `getBoundedDispatchLayout` or the public `planRasterDispatchStripes` (`@luma.gl/graph/gpu-raster`). |
| `GPURasterBand`, `GPURasterMetadata`, `GPURasterBufferBand` and friends | public types in `@luma.gl/graph/gpu-raster` | Import as types. `makeGPURasterBand` and `makeGPURasterMetadata` are **not public**: use the local ones in `raster-analysis/raster-grid` (they attach `verticalDatum` at runtime; see the workarounds register). |
| `prepareArrowTableForGPUAnalytics`, `getGPUPartitionDescriptorFromArrowTable`, `makeGPUTimeVectorFromPreparedArrowColumn` | not public | Only the pieces already moved to `analysis-core/graph-support` (arrow normalisation, time upload, wide numeric) remain. Build vectors with public `@luma.gl/graph/arrow` (`makeGPUVectorFromArrow`, `TEMPORAL_ORIGIN_METADATA_KEY`, `ArrowTemporalOriginPolicy`) and `@luma.gl/graph/gpu-data` (`GPUVector`, `GPUData`, `getGPUVectorFormatInfo`). Partition descriptors: build `GPUSceneTablePartition`-like records locally or drop. |
| `waitForSubmittedWork` | not public | `const fence = device.createFence(); await fence.signaled;` (public `@luma.gl/core`). |
| `GPUCommandGraph.withRecordScope`, `.recording`, `.describe` | not in published graph | Drop; see computation-record owner. |
| Bucket G primitives (`GPUGridIndex` `boundsBuffer`, `GPUCompaction` `strategy`, point-in-polygon `queryPositions`) | public classes with different options | Read the published `.d.ts` under `node_modules/@luma.gl/graph/dist/gpu-core/` (for example `gpu-grid-index.d.ts`, `gpu-compaction.d.ts`) and call the published signature. |

Useful public names to remember (all `@luma.gl/graph/gpu-core` unless noted): `GPUCommandGraph`,
`createTransientView`, `createGPUComputeCommandNode`, `createGPUCopyCommandNode`,
`createGPURenderCommandNode`, `getViewBinding`, `getViewBindingRange`, `getViewElementOffset`,
`GraphDataView`, `GraphVectorView`, `GPUCommandNodeProducer`, `GPUSort`, `GPUScan`,
`GPUCompaction`, `GPUMask`, `GPUGather`, `GPUReduction`, `GPUHistogram`, `GPUHashJoin`,
`GPUHashIndex`, `GPUBVH`, `GPUSegmentOffsets`, `GPUSegmentedSort`, `GPURunLengthEncode`;
`Kernel` from `@luma.gl/graph` (root). Not public: `GPUScatter`, `GPUSegmentedScan`,
`GPUSegmentedReduction`, `GPUPageRank` (look at `@luma.gl/graph/gpu-graph-algorithms` instead).

## 5. Kernel-audit and test infrastructure (`test/analysis-core`)

- `kernel-audit/analysis-contributors.ts` defines `ANALYSIS_BARRELS` (the seven analysis src
  barrels) and `selectAnalysisCases()`. Cases that name contributors owned by published luma
  (`geospatial`, `gpu-dataframe`, `gpu-raster` barrels) are skipped at run time; the audit no
  longer audits luma.gl code. Deleted: `cases-geospatial.ts`, `cases-raster-operations.ts`,
  `rejections-raster-operations.ts`, `kernel-audit-geospatial` and `kernel-audit-raster-operations` specs.
- Area agents: when you add, move or remove a contributor, update the matching `cases-*.ts`.
  Case files still import some classes from `@luma.gl/graph/gpu-raster` or `.../geospatial`; where an
  analysis class replaced them, change the import to `../../../src/<area>/index`.
- The 12 point-cloud and datum-grid contributors have audit cases in
  `kernel-audit/cases-pointcloud-and-datum.ts` (2026-11-07). `SKIPPED_CONTRIBUTORS` keeps only the four
  `GPUChunkedSpatialWeights*` contributors, with their documented reason. Rejection cases for the
  point-cloud maxima wait on roadmap AUD-02.
- The WGSL audit now ignores any published `x == x && abs(x) <= F32_MAX` finite test (generalised
  from the former `value` only idiom). Analysis source text is still checked for NaN self-compares.
- `analysis-layering`, `analysis-runtime-boundaries` and `analysis-vocabulary` specs now use the new
  directory names (`geospatial-analysis`, `dataframe-analysis`, `raster-analysis`,
  `pointcloud-analysis`, `gpu-terrain`, `gpu-spatial-analysis`, `gpu-network`). The vocabulary
  allowlist is empty (its entries were all luma-owned raster base classes). Point-cloud props are
  not covered by the vocabulary check (they have `iterations` / `selection` names).
- `doc-snippets` and `doc-parameters` specs run against the rewritten docs (`@luma.gl/*` and
  `@luma-contributors/analysis` imports). `doc-program.ts` / `doc-parameters.ts` entry lists are current.
- Removed tests: `wgsl-kernel-node-record.node.spec.ts` (recording) and the bind-group-cache test in
  `wgsl-kernel-nodes.node.spec.ts`.
