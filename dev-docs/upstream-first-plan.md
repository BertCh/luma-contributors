# Upstream-first plan

This repository contains geospatial and analysis code built on published luma.gl public entry points. The installed and pinned release is `10.0.0-alpha.13`. `10.0.0-alpha.14` changes colour conversion, expression operations and SQL; these do not change this plan.

## Rules

1. Depend on published `@luma.gl/core`, `@luma.gl/shaders` and `@luma.gl/graph`; use `@luma.gl/engine` only in examples and tests. A public entry point must be explicitly published by the package.
2. Never import an `_internal` segment, even if listed in `exports`, or a deep `dist` path. Never copy, fork, re-implement or re-export luma.gl APIs.
3. When an API is missing, use the public API even if verbose; adapt to luma.gl's design; drop the feature; only as a last resort write a small helper named for the analysis domain.
4. Send luma.gl improvements upstream as ordinary pull requests. Do not carry them here or wait for them. Remove a workaround when a published release supplies its replacement.
5. Source under `modules/analysis/src` belongs only in `analysis-core`, `computation-record`, `gpu-spatial-analysis`, `gpu-network`, `gpu-terrain`, `testing`, `dataframe-analysis`, `raster-analysis`, `pointcloud-analysis` and `geospatial-analysis`.
6. `scripts/check-upstream-first.mjs` enforces published entry points, forbidden `_internal` segments, analysis-owned directories and no luma.gl re-exports from `modules/*/src`.
7. Validate contributor props once at construction with named-limit helpers; follow [contributor-contract.md](./contributor-contract.md).

Exceptional grid WGSL copies are allowed only for kernels consuming published grid state. Each needs a row in [upstream-workarounds.md](./upstream-workarounds.md), an entry in `modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts` and a pin comparing the copy with upstream text. Match the paired class's arithmetic: `GPUGridIndex` and `GPUGridBinning` differ.

The dependency direction is `kepler → lenses → analysis → @luma.gl/{graph,core,shaders}`. Lenses remains private while dataset licensing is unresolved in `THIRD-PARTY-DATA.md`.

## Completed phases and maintenance

The fork, quarantine and compatibility layer are removed; analysis has its own entry points, consumers use published signatures, computation records use public graph data, and tests cover analysis rather than luma.gl internals. Keep analysis, oracle and parity tests.

Open work belongs in [roadmap.md](./roadmap.md). [upstream-workarounds.md](./upstream-workarounds.md) owns the capability register, local copies and deletion triggers; [upstream-proposals.md](./upstream-proposals.md) describes the requested upstream APIs. Bump `@luma.gl/*` with each alpha, all pins together; compare public entry-point exports, retire supplied workarounds and follow the register's verification procedure. Published `GPUSort` and `GPUScan` determine the default throughput; the old approximately 4.2M-row dispatch limit does not describe current primitives (`modules/analysis/test/gpu-core/gpu-primitive-large-input-plat01.spec.ts`, `ROW_COUNTS`, covers up to 33.5M rows; PLAT-01).

## Other split outcomes

Capabilities already listed in the workaround register are not repeated here. The fork also changed luma.gl-owned code (about 360 files in graph, core and engine); none of that is in published luma.gl, and [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md) records each change, where it is tracked and which ones this repository turned out not to need.

| Former extension | Current outcome |
| --- | --- |
| `getPhysicalBuffer`, `getAdjacencyVectors` from algorithms-utils | No analysis replacement; they have no consumer. Computation capture uses copy nodes rather than physical-buffer access. |
| `createGPUGraphAlgorithmsIterationGate` | No separate graph-algorithm gate. Raster iteration uses `createRasterIterationGateNode` in `modules/analysis/src/raster-analysis/cost-distance/raster-relaxation.ts`. |
| `addDirectLocalClusteringToGraph` | No local extension or consumer; the published class is `GPUGraphAlgorithmsLocalClusteringCoefficient`. |
| `kernel-dispatch` (`declareKernelConstants`, `dispatchHandlerKernel`, `KernelConstants`) | No consumer or replacement. |
| `rigid-body-dispatch` (`getRigidBodyDispatchCount`, `RIGID_BODY_LINEAR_WORKGROUP_SIZE`) | No consumer or replacement; physics is outside the analysis scope. |
| Sparse resistance adapters | `addRasterResistanceConnectivityRecipe` exists in `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts`; `addSparseResistanceComponents` and `addSerialSparseResistanceComponents` have no source implementation. The raster adapter is EC03-R; the shared solver is covered by the register (EC03). |
| Multi-chunk `GPUVariableStandardization` | `GPUVariableStandardization` in `modules/analysis/src/dataframe-analysis/variable-standardization/gpu-variable-standardization.ts` requires single-chunk `values`, `mask` and `output`; pack streamed batches before use. |

## Shared helpers on published luma.gl

Paths below are under `modules/analysis/src`. Import analysis contracts from `@luma-contributors/analysis/analysis-core`; `graph-support` has no separate package entry point. These are analysis helpers, not additional luma.gl APIs. Their upstream replacement conditions belong in the register.

| Path and symbols | Current role |
| --- | --- |
| `analysis-core/kernels/wgsl-kernel-nodes.ts`: `createWGSLKernelNode`, `createWGSLActiveCountDispatch`, `createWGSLDispatchGate`, `getAutomaticWGSLKernelWorkgroupSize`, `getWGSLFloatLiteral` | Builds published `Kernel` and command nodes with typed bindings, public bounded-dispatch layout and GPU indirect conditions. Binding-record identity caching saves JavaScript allocations; dispatch still supplies bindings each time. |
| `analysis-core/kernels/gpu-contributor-utils.ts`: `GPUParameterBuffer`, `importGraphBuffer`, `captureGraphCommandNodes`, `validateCapacity`, `validateMaximumCount`, `validateRequirement`, `validateSinglePackedView`, `validateGPUStorageBindingByteLength`, `getStorageBindingElementLimit`, `getNextPowerOfTwo` | Parameter upload, view import, attachment of published `addToGraph` contributors and construction-time limits. |
| `analysis-core/kernels/gpu-contributor-types.ts`: `GPUCompactOutput`, `GPUUint32Rows`, `GPUStatusScalar` | Analysis ports and bounded-output contracts. Import `GPUFloat32Positions` directly from `@luma.gl/graph/geospatial`. |
| `analysis-core/kernels/sorted-segment-sums.ts`: `createSegmentSumNode`, `getSortedSegmentSumNodes`, `getSegmentTreeSumWGSL`, `getSegmentSumDispatchShape`, `SEGMENT_WORKGROUP_SIZE`, `getSortKeyBits` | Published gather followed by a local 256-lane segment tree; no private `GPUSegmentedReduction` import. |
| `analysis-core/kernels/exact-orientation-wgsl.ts`: `EXACT_ORIENTATION_WGSL`; `philox-random.ts`: `PHILOX_WGSL`; `fixed-point-accumulator.ts`: `FIXED_POINT_WGSL` | Exact orientation, counter-based random draws and deterministic fixed-point sums for analysis kernels. |
| `analysis-core/kernels/hash-uint32.ts`: `HASH_UINT32_WGSL`; `scan-total.ts`: `createSaturatedScanTotalNodes`; `load-balanced-for-each.ts`: `createLoadBalancedForEachNode`; `parallel-merge-tree.ts`: `getParallelMergeTreeNodes` | Hashing, saturated scan totals and work distribution for analysis. |
| `analysis-core/kernels/polygon-edge-slabs.ts`: `getPolygonEdgeSlabNodes`; `stable-lexicographic-index-sort.ts`: `createStableLexicographicIndexSortNodes`; `study-window-wgsl.ts`: `getStudyWindowWGSL`; `wgsl-common.ts`: `GET_NAN_WGSL` | Polygon candidate slabs, stable key ordering, study-window sampling and shared WGSL constants. |
| `analysis-core/contracts/spatial-context.ts`: `GPUSpatialContext`, `validateGPUSpatialContext`; `analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts`: `getPolygonTriangulationIndexCount`; `analysis-core/ring-assembly/gpu-segment-ring-assembly.ts`: `GPU_SEGMENT_RING_ASSEMBLY_NONE` | Spatial contracts, polygon triangulation sizing and ring assembly. |
| `analysis-core/network-topology.ts`: `GPUNetworkTopology`, `validateGPUNetworkTopology`; `analysis-core/raster-grid.ts`: `getRasterGroundCellSize`; `analysis-core/lon-lat-ring-wgsl.ts`: `createLonLatPolygonCheckNode` | Network, raster ground-spacing and longitude/latitude ring contracts. |
| `analysis-core/graph-support/arrow-analytics-normalization.ts`: `prepareArrowTableForGPUAnalytics`; `arrow-analytics-time-upload.ts`: `makeGPUTimeVectorFromPreparedArrowColumn`; `arrow-analytics-wide-numeric.ts`: `isWideNumericType`, `convertWideNumericValue`, `convertWideNumericVector` | Arrow normalization and exact time upload over published Arrow and GPU-data APIs. |
| `testing/gpu-contributor-test-utils.ts`: `createNullWebGPUDevice`, `getGPUCommandNodeKernelSources`, `getGPUCommandNodePipelineKeys`, `getParsedKernelSources`; `testing/require-test-device.ts`: `requireTestDevice` | Workspace-only test helpers. The null-device shim supplies WebGPU type and limits for declaration and kernel inspection, not GPU execution. `testing/index.ts` is a TypeScript workspace alias, not a published package export. |

[upstream-first-recording.md](./upstream-first-recording.md) describes `recordWGSLKernelNodes` and `GPUComputationRecorder`. Public luma.gl helpers such as `getBoundedDispatchLayout`, `getBoundedInvocationIndexSource`, `getViewBinding` and `device.createFence()` should be used directly. For packed chunks use public `GraphVectorView.data` or `graph.createDataView`; do not restore private alignment utilities.

Kernel audits use `ANALYSIS_BARRELS` and `selectAnalysisCases` in `modules/analysis/test/analysis-core/kernel-audit/analysis-contributors.ts`; update the matching `cases-*.ts` when contributors change. `cases-pointcloud-and-datum.ts` covers point-cloud and datum-grid contributors. `SKIPPED_CONTRIBUTORS` in `modules/analysis/test/analysis-core/kernel-audit.node.spec.ts` records the four chunked spatial-weights exceptions. `analysis-layering.node.spec.ts`, `analysis-runtime-boundaries.node.spec.ts` and `analysis-vocabulary.node.spec.ts` enforce package boundaries; `doc-snippets.node.spec.ts` and `doc-parameters.node.spec.ts` check documented APIs.
