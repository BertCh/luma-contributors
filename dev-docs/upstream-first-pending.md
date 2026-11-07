# Quarantined luma.gl extensions: replacements

Result of resolving `modules/analysis/src/_pending/` (non-recording, non-foundation files).
Consumers are not yet edited; area agents apply this table.

| Quarantined symbol(s) | Replacement |
| --- | --- |
| `prepareArrowTableForGPUAnalytics`, `PrepareArrowTableForGPUAnalyticsOptions`, `PreparedArrowAnalyticsTable`, `PreparedArrowAnalyticsTimeColumn`, `ArrowAnalytics*` types | Rebuilt unchanged on public APIs: `modules/analysis/src/analysis-core/graph-support/arrow-analytics-normalization.ts` (barrel `graph-support/index.ts`). Uses only `apache-arrow` and the published type `GPUAnalyticsDictionary` from `@luma.gl/graph/arrow`. Consumers: kepler `session/*`, kepler-compute example, kepler tests (they imported it from `@luma.gl/graph/arrow`). |
| `makeGPUTimeVectorFromPreparedArrowColumn`, `GPUTimeVectorFromPreparedArrowColumn` | `graph-support/arrow-analytics-time-upload.ts` (public `GPUData`/`GPUVector` from `@luma.gl/graph/gpu-data`). |
| `isWideNumericType`, `convertWideNumericValue`, `convertWideNumericVector` | `graph-support/arrow-analytics-wide-numeric.ts`. |
| `appendGPUTableFromArrowBatches`, `getRecordBatchesFromArrowTableBatch`, `Structural*Arrow*`, `ArrowBatchStreamOptions` | Dropped. No consumer. Published `@luma.gl/graph/arrow` has `convertArrowRecordBatchesToGPURecordBatches` and `makeGPURecordBatchFromArrowRecordBatch`. |
| `GPUGraphAlgorithmsTopologyView`, `GPUGraphAlgorithmsAdjacencyView`, `GPUGraphAlgorithmsTopologyLike`, `GPUGraphAlgorithmsColumn`, `importGPUGraphAlgorithmsColumn/Overflow`, `validate*GPUGraphAlgorithms*`, `usesGPUGraphAlgorithmsViews` | Dropped. Published algorithms (`GPUGraphAlgorithmsDegree`, `ConnectedComponents`, `PageRank`, `CoreNumber`, `LabelPropagation`) accept only `GPUGraphAlgorithmsTopology` (from `@luma.gl/graph/gpu-graph-algorithms`) over caller-owned physical `GPUVector<'uint32'>` outputs, not command-graph views. Consumers (`gpu-network-analytics-columns`, `gpu-network-statistics`, `gpu-network-structure`, parity and recording tests) must hold physical `GPUVector` CSR storage (build with `GPUGraphAlgorithmsTopology` rebuild or `GPUGraphAlgorithmsData`) and physical outputs; graph-transient CSR/columns are no longer supported. Otherwise drop that feature. |
| `getPhysicalBuffer`, `getAdjacencyVectors` (algorithms-utils) | Dropped. No analysis consumer (`getPhysicalBuffer` in capture-command-graph/showcase is a different, compiled-graph or local function). |
| `GPUGraphAlgorithmsStronglyConnectedComponents`, `addGPUGraphAlgorithmsStronglyConnectedComponentsToGraph` | Dropped, no published equivalent. `GPUNetworkStructure`: remove the `'iterative'` method and `strongComponentsConverged`/`strongComponentsIterations`; keep the existing exact `'reachability'` kernel (limit 1024 vertices, 65536 CSR slots) for all strong components. Accepted limit: no large-graph SCC. Test `gpu-graph-strongly-connected-components.spec.ts` deleted. The SCC lens scenario (`modules/lenses/.../graph-strongly-connected-components.*`, `pack-communities-layouts.ts` SCC entries) should be dropped by the lenses owner. |
| `createGPUGraphAlgorithmsIterationGate`, `GPUGraphAlgorithmsIterationGate` | Dropped. Raster/spatial code already uses its own `createRasterIterationGateNode` in `raster-analysis/cost-distance/raster-relaxation.ts`; no consumer of the graph one. |
| `addDirectLocalClusteringToGraph` | Dropped. Published `GPUGraphAlgorithmsLocalClusteringCoefficient` is the replacement; no consumer. |
| `kernel-dispatch` (`declareKernelConstants`, `dispatchHandlerKernel`, `KernelConstants`) | Dropped. No consumer. |
| `rigid-body-dispatch` (`getRigidBodyDispatchCount`, `RIGID_BODY_LINEAR_WORKGROUP_SIZE`) and the `gpu-physics` README | Dropped. No consumer; physics is not analysis. |
| `histogram-cube` (`createHistogramCubeBuildNodes`, `createHistogramCubeReadoutNode`, `getHistogramCubeCellCount`, `HistogramCube*`) | Dropped. Only consumer was the skip-cube crossfilter tests (deleted with their oracle). Use published `GPUCrossfilter` from `@luma.gl/graph/gpu-crossfilter`. |
| `GPUResidencyArena`, `GPUResidentRowSelection`, `ResidencyArenaAllocator`, `ResidencyArenaFullError`, `GPU_RESIDENCY_ARENA_DEAD_SLOT`, `GPUResidencyArena*` types | Dropped. No src consumer (kepler-compute `resident-*` files use their own code). All residency-arena tests and oracles deleted. |
| `GPUTileLODSelection`, `makeGPUTileLODQuadtree`, `getGPUTileLODQuadtreeTile`, `getGPUTileLODViewParameterValues`, `GPU_TILE_LOD_*`, `TILE_LOD_*`, `createTileLOD*Node` | Dropped. Only consumer is the explorer example `examples/deck/spatial-analysis-explorer/modes/tile-lod-mode.ts` (plus `tile-lod-layers`, `modes/index.ts`, `mode-categories.ts` id `tile-lod`): remove the mode. Tile-LOD tests deleted. |

## Deleted tests

- `modules/analysis/test/gpu-tables/**` (residency arena, tile LOD).
- `modules/analysis/test/gpu-graph/gpu-graph-strongly-connected-components.spec.ts`.
- `modules/analysis/test/gpu-crossfilter/gpu-crossfilter-skip-cube{,.node}.spec.ts`, `histogram-cube-oracle.ts`.

## Entry point note

`graph-support` has no package export yet. kepler and examples need one (for example
`@luma-contributors/analysis/analysis-core/graph-support`) added to `modules/analysis/package.json`, or relative imports.
