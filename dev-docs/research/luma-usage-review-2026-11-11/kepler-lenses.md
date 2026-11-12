# Stream kepler-lenses: luma.gl usage review (2026-11-11, alpha.13)

## 1. Coverage

Read in full or in the relevant part (local side):

- kepler `exploration/`: `gpu-resident-explorer.ts` (constructor :155-430, encode/readback :500-900, helpers :1060-1100), `gpu-table-inspector.ts` (:1-80, :280-605), `gpu-column-statistics.ts` (all), `read-all-buffers.ts`.
- kepler `selection/`: `gpu-spatial-selection.ts` (all), `gpu-scalar-selection.ts` (:1-60, :395-431, :470-760), `gpu-selection-scopes.ts` (:600-760), `gpu-time-selection.ts` (imports, graph construction).
- kepler `styles/resident-style.ts` (:1-80, :360-380, :600-790); `aggregation/kepler-group-by.ts` (:1-140, :255-470), `kepler-group-by-passes.ts` (all), `gpu-kepler-aggregation.ts` (imports, graph); `analysis/analysis-scaling.ts` (`addKeplerLinearKernel`), `catalog-gpu-session.ts` (all), `point-weights-streamed.ts` (:60-200); `runtime/gpu-readback.ts`, `runtime/dataset-registry.ts` (:80-120, :820-855); `session/metered-command-graph.ts`, `compute-session.ts` (grep plus :449-490), `resident-dataset.ts` (:285-330), `arrow-stream-snapshot.ts` and `arrow-stream-publisher.ts` (imports and the upload call).
- `cluster/` (CPU only: supercluster, d3-scale, a local KD tree; no luma.gl import) and `execution/` (DuckDB router; no luma.gl import): imports only.
- `modules/analysis/src/analysis-core/graph-support/arrow-analytics-*.ts` (all three files).
- lenses: `computation-lenses/lens-view-snapshot.ts` (all), `computation-views/coordinator/gpu-crossfilter-selection-backend.ts` (all), `scenarios/gpu-dataframe/dataframe-lens-support.ts` (:79-118, :300-340), the four files that hold WGSL (`expansion-examples.ts`, `urban-ecology-examples.ts`, `interpolation.run.ts`, `s1-regression-model.ts`), and the imports and `new GPU*` lines of a sample of 13 scenario files across dataframe, network, raster, spatial-analysis and terrain, plus every scenario that constructs a published `gpu-core` class (`GPUHistogram` 5, `GPUGather` 4, `GPUScan` 2, `GPUReduction` 1).

Published side read: `gpu-crossfilter/gpu-crossfilter.d.ts`, `gpu-selection.d.ts`, `gpu-crossfilter.js:235-247`; `gpu-core/gpu-mask.d.ts` and `.js`; `gpu-readback-ring.d.ts` and `.js`; `gpu-incremental-execution.d.ts`; `gpu-visibility-workflow.d.ts`; `gpu-elementwise.d.ts`; `gpu-command-graph-inspector.d.ts`; `gpu-command-graph.d.ts:140-180` and `.js:1270-1293` (`GPUCommandGraphEncoding.readResource`); `gpu-resource-preview.js`; `gpu-dataframe/gpu-group-by-query.d.ts`, `gpu-group-aggregation-compiler.d.ts` and `.js` (validity handling), `gpu-query-compiler.d.ts` and `.js:88-170`, `gpu-data-frame.d.ts`; `gpu-tables/table/gpu-table.d.ts`, `gpu-record-batch.d.ts`, `gpu-schema.d.ts`, `utils/gpu-table-buffer-planner.d.ts` (head); `arrow` export list, `gpu/arrow-gpu-analytics-adapters.d.ts`, `vectors/arrow-temporal-gpu-vector.d.ts`; `gpu-data/cpu-gpu-vector-transfer.d.ts` and `.js`; `execution/compute/kernel.js` (pipeline factory); `@luma.gl/core` `webgpu-buffer.js:125-215` (`readAsync`), `webgpu-device.js:58-140, :356-375` (uncaptured errors, error scopes), `api/adapter/device.d.ts` (error API).

Not done (a later pass should pick these up):

- The 59 `modules/kepler/src/analysis/catalog-*.ts`, `point-*.ts`, `recipe-bindings.ts` and `polygon-weights.ts` files were only grepped: they hold 10 of the direct `new Kernel(` calls and most `addKeplerLinearKernel` callers (42 call sites). I did not compare their kernel bodies with published classes.
- `recipes/analysis-recipe.ts`, `recipes/sandbox-state.ts`, `session/selection-export.ts`, `session/adopted-arrow-dataset.ts` beyond the schema mutation, `styles/style-cache.ts`, `styles/style-plan.ts`, `cluster/*` bodies.
- Lenses: 791 of 804 scenario files were not opened; `computation-views/` (views, charts, scales, encoding, hosts, packs, registry, twin) was grepped only (no `createBuffer`, `readAsync` or WGSL outside the crossfilter backend). The per-domain `gpu-*/computation-pack` folders were not read.
- Nothing was run (no node specs either).

## 2. Findings

### KL-01 Interactive readbacks allocate a staging buffer and a queue submission per buffer; `GPUReadbackRing` is unused

- Class: misuse (adopt-now for the explorer, inspector and legend paths).
- Local: `modules/kepler/src/exploration/gpu-resident-explorer.ts:564-605` (`readSummary`: one `readAsync` per buffer for the count, every histogram and every domain, `:575`), `:735-830` (`gatherPage`: 1 + 2 per column output buffers created per page `:759`, `:784`, each read separately `:814`), `:723-725` (`inspectionReservationByteLength = maximumConcurrentReads * maximumReadbackBytes * 2`, i.e. a ring described in words), `:897` (`beginRead` drops a read when `activeReads >= maximumConcurrentReads`); `gpu-table-inspector.ts:348`, `:474-518`; `gpu-column-statistics.ts:184-188` (three reads); `styles/resident-style.ts:373-377` (two reads per legend, with its own `maximumConcurrentLegendReads` at `:56`); `session/resident-dataset.ts:296-311` (one read per column and per validity per page); `selection/gpu-spatial-selection.ts:351-366` (three separate 4-byte status buffers, three reads); lenses `computation-views/coordinator/gpu-crossfilter-selection-backend.ts:161`.
- Published: `node_modules/@luma.gl/core/dist/webgpu/adapter/resources/webgpu-buffer.js:127-181` (`readAsync` on a buffer without `MAP_READ` creates a temporary `MAP_READ` buffer, `:191-195`, logs `log.warn("... is not readable, creating a temporary Buffer")` at `:193`, and submits its own copy on a separate encoder, `:196-206`); `node_modules/@luma.gl/graph/dist/gpu-core/gpu-readback-ring.d.ts` (`GPUReadbackRing({byteLength, slotCount})`, `tryAcquire()` returns `null` under pressure, `acquire()` waits, a ticket's `buffer` is a reusable staging buffer, `markEncoded({byteLength})` after several caller copies, `read()` after the caller submits).
- Evidence: read both.
- Change: give `GPUResidentExplorer` one ring with `byteLength = maximumReadbackBytes` and `slotCount = maximumConcurrentReads`; `tryAcquire()` replaces the `activeReads` counter in `beginRead`. In `readSummary` and `gatherPage`, copy every output into the ticket buffer at increasing offsets on the encoder that already runs the graph (`encoder.copyBufferToBuffer`, then `markEncoded`), submit once, `read()` once. Do the same for the table inspector's derived page, the style legend (one small ring of `maximumConcurrentLegendReads` slots) and the resident page loader; pack the three spatial-selection status words into one 12-byte buffer. Call sites: 11 `readAsync` calls in 6 kepler files plus 10 calls of the `runtime/gpu-readback.ts` helpers in 6 files; the interactive ones are the 7 sites listed above.
- Gain: per summary today, 1 + H + D temporary buffers, 1 + H + D extra queue submissions and as many console warnings (H histograms, D domains); with the ring, zero allocations after warm-up, one submission, one map. The explorer pages also stop allocating 1 + 2C storage buffers per page (they can stay as reusable gather outputs sized `maximumPageSize`).
- Verify: `modules/kepler/test/exploration.spec.ts`, `inspection-read-lifetime.node.spec.ts`, `table-inspector.spec.ts`, `selection.spec.ts`, `resident-style-legend.node.spec.ts` (the last is in `excludedFromRuns` of `test/kepler-compute-example-specs.json`; check before relying on it). Add a node spec that counts `device.createBuffer` calls across ten `readSummary` calls.
- Risk: none on values (same bytes). Ordering: today each `readAsync` copy is submitted after the caller's own submission because it is queued later; with the ring the copy must be encoded after the graph on the same encoder, which the code already controls. A ticket that is encoded but never read must still be `read()` then cancelled (the ring's documented rule).

### KL-02 Dataframe filters used only for their mask run an unused compaction on every encode

- Class: misuse (adopt-now for the encode gate; `blocked` for a mask-only compile).
- Local: `modules/kepler/src/exploration/gpu-resident-explorer.ts:206-229` (each dimension is `frame.filter(predicate).compile(...)` and only `query.selectionMask` is consumed), `:510-524` (`encode` re-encodes every predicate and domain query on every call, "Encode before Deck draws"); `modules/kepler/src/selection/gpu-scalar-selection.ts:527-533` (same pattern) and `:413-427` (every encode re-runs every query and then copies its mask into a second buffer).
- Published: `node_modules/@luma.gl/graph/dist/gpu-dataframe/gpu-query-compiler.js:98-150`: every compiled filter allocates `rowIndices` (N words) and `selectedCounts`, and adds a `GPUVisibilityWorkflow` per batch (identity IDs, scan, stable scatter, count) after the predicate pass. `selectionMask` is a persistent output vector (`:98`, `createGPUQueryOutputVector`).
- Evidence: read both.
- Change: keep a per-predicate revision and encode a predicate query only when its parameters (or the dataset revision) changed since its last submitted encode; the mask persists between encodes. The same gate applies to the mask copy in `gpu-scalar-selection.ts:417-427`. Sites: 2 classes, 3 encode loops (`explorer` predicates, `explorer` domain predicates, scalar selection).
- Gain: per frame and per dimension, today one predicate pass plus a full scan-and-scatter over N rows and one N-word copy; after the gate, nothing while the user is not dragging.
- Blocked part (not in the register): the compaction and the `rowIndices` allocation (4N bytes per dimension, counted in `accountedResidentByteLength`) cannot be avoided because luma.gl has no mask-only compile of a dataframe filter. Needs an option on `GPUDataFrameQuery.compile` (or a public predicate-mask node) that skips `GPUVisibilityWorkflow`.
- Verify: `modules/kepler/test/exploration.spec.ts`, `scalar-selection.spec.ts`; add a check that a second `encode` with unchanged controls records no predicate node (`GPUCommandGraphEncoding.stats.nodeCount`).
- Risk: a caller that rewrites a borrowed source column in place without calling `invalidate()` would get a stale mask; the explorer already documents `invalidate()` for that case (`:503`), so the gate must key on `this.revision` too.

### KL-03 Mask composition chains binary `GPUMask` passes through transient masks; `GPUMask` is N-ary and has `difference`

- Class: adopt-now.
- Local: `modules/kepler/src/selection/gpu-selection-scopes.ts:620-645` (`compose`: for k inputs, k - 1 `GPUMask` passes and k - 2 transient N-word masks); inverted filters are a `not` pass into a transient followed by an `and` (`:653-654`, `:729-733`); `allRows` is `not(zeroMask)` into a transient (`:647-648`) and is then ANDed in as `live` when there is no live mask (`:649`).
- Published: `node_modules/@luma.gl/graph/dist/gpu-core/gpu-mask.js:31-53` (any number of inputs; only `not` requires exactly one), `:95-125` (one pass reads every input), `:160-178` (`and`, `or`, `xor`, `difference` = first AND NOT the rest, `not`).
- Evidence: read both.
- Change: `compose` adds one `GPUMask` with all inputs, split into groups of at most `device.limits.maxStorageBuffersPerShaderStage - 1` inputs (the published class does not check this limit itself). Use `operation: 'difference'` for "A and not B" instead of a `not` pass plus an `and` pass, and drop `allRows` from AND lists. 16 `compose(` calls in the file go through the one helper.
- Verify: `modules/kepler/test/selection-scopes.spec.ts`, `selection-scope-plan.node.spec.ts`.
- Risk: none on values (canonical 0/1 either way). Exceeding the storage-buffer limit would fail at pipeline creation, so the grouping is required.

### KL-04 Out-of-memory detection by `error.name` matches nothing luma.gl produces

- Class: stale-claim (register row "Out-of-memory detection") plus blocked.
- Local: `modules/kepler/src/runtime/dataset-registry.ts:851-855` (`isOutOfMemoryError`: `'OutOfMemoryError'` or `'GPUOutOfMemoryError'`), used by `isKeplerAllocationError` at `:105-112`. The reason `'device-out-of-memory'` (`:86`, `:98`) is never raised anywhere in `modules/kepler/src`.
- Published: `@luma.gl/core` alpha.13 exports no error class or type guard (`api/adapter/device.d.ts:216` `onError?: (error: Error, ...)`, `:412` `reportError`). WebGPU out-of-memory never throws from `createBuffer`; it arrives as an uncaptured error, which `webgpu-device.js:58-63` re-wraps as `new Error(event.error.message)`, so the name is `'Error'` and the GPU error class is lost. `WebGPUDevice.pushErrorScope` and `popErrorScope` (`webgpu-device.js:356-375`) do nothing unless the device was created with `debug: true`. No file in `@luma.gl/core/dist` contains `OutOfMemoryError`.
- Evidence: read both.
- Change: correct the register row: the name check is dead for every luma.gl path, not a stand-in for a missing class. To detect real allocation failure today, wrap the large allocations (dataset upload, explorer and style outputs) in a raw scope on the WebGPU handle: `(device as WebGPUDevice).handle.pushErrorScope('out-of-memory')` ... `await handle.popErrorScope()` (`WebGPUDevice` is a public export of `@luma.gl/core/webgpu`), and raise `KeplerAllocationError('device-out-of-memory')` when it returns an error. What luma.gl needs: keep the GPU error type (or a `cause`) in `reportError` for uncaptured errors, and an always-on `pushErrorScope('out-of-memory')` on `Device`.
- Verify: `modules/kepler/test/binding5.node.spec.ts:43-64` covers only the predicate; a headless spec that allocates past `maxBufferSize` would prove the scope route.
- Risk: an error scope adds one async round trip per wrapped allocation group; scope only admission-time allocations, never per frame.

### KL-05 Lens crossfilter backend tells users to set `skipUnchangedDimensions`, which the published crossfilter does not have

- Class: stale-claim.
- Local: `modules/lenses/src/computation-views/coordinator/gpu-crossfilter-selection-backend.ts:45` (doc comment) and `:212` (thrown error message: "set outputMask or skipUnchangedDimensions").
- Published: `node_modules/@luma.gl/graph/dist/gpu-crossfilter/gpu-crossfilter.d.ts` `GPUCrossfilterProps` has only `id`, `dimensions`, `views`, `outputMask`; no file under `node_modules/@luma.gl/graph/dist` contains `skipUnchangedDimensions`.
- Evidence: read both.
- Change: drop the option from both texts; the only persistent route is `outputMask` (and caller-owned `output` on `mask` views). 2 lines.
- Verify: `modules/lenses/test/gpu-spatial-analysis/computation-pack/gpu-crossfilter-selection-backend.node.spec.ts` (asserts on messages only if it matches this text; grep before editing).
- Risk: none.

### KL-06 Table inspector reads the sorted permutation to the CPU and uploads it again for the gather

- Class: misuse (adopt-now).
- Local: `modules/kepler/src/exploration/gpu-table-inspector.ts:336-355` reads `length` words of `query.globalRowIndices` (`:348`), then calls `explorer.readRows(Array.from(sourceRows))`, which writes them back into a fresh indices buffer (`gpu-resident-explorer.ts:759-771`, `indices.write(new Uint32Array(sourceRows))`) before `GPUGather`.
- Published: `GPUGather` takes any packed `uint32` `GraphDataView` as `indices` (`gpu-core/gpu-gather.d.ts`); `graph.importGPUData` / `createDataView` accept a byte offset, so the slice `globalRowIndices.data[0]` at `request.offset * 4` can be the gather's indices directly.
- Evidence: read one (local fully; published from the declarations of `GPUGather` and the explorer's own use of it at `:792-803`).
- Change: give the explorer's gather path an "indices view" variant and pass the permutation slice; read the gathered row IDs back together with the columns (they are needed for `sourceRowIndices`). One call site.
- Gain: one fewer readback and one fewer upload per sorted page, and one fewer GPU round trip of latency.
- Verify: `modules/kepler/test/table-inspector.spec.ts`, `table-inspector.node.spec.ts`.
- Risk: the permutation is only valid for the revision it was encoded at; the code already checks `program.revision` before reading (`:338-344`) and must keep that check (`:340`) before encoding the gather.

### KL-07 Kepler group-by hand-assembles a dense grouped aggregation that `GPUDataFrameGroupedAggregationQuery` publishes

- Class: adopt-with-cost.
- Local: `modules/kepler/src/aggregation/kepler-group-by.ts:255-470`: key normalization kernel, a `GPUGroupAggregation` count, a per-column finite-and-valid mask kernel (`:337-355`), a `GPUGroupAggregation` per measure (`:395-404`), and a per-measure validity kernel that NaNs empty groups (`:413-431`); passes planned in `kepler-group-by-passes.ts`.
- Published: `node_modules/@luma.gl/graph/dist/gpu-dataframe/gpu-group-by-query.d.ts` (`frame.groupBy(key, {groupCount}).aggregate({name: 'count' | {sum|min|max|mean: column}})`); `gpu-group-aggregation-compiler.js:76` (rows with an invalid key are dropped), `:184-204` (one shared null/NaN/infinity-filtered mask per value column and a per-group validity vector), `:339-362` (validity normalised to 0/1).
- Evidence: read one (local fully; published compiler read for validity, not for the kernels).
- What delegation keeps: the per-column finite mask, the per-measure grouped values and their validity (about 2 local kernels and 1 + M `GPUGroupAggregation` nodes per pass).
- What it loses: the null-key group (the published query drops null keys, the local code keeps slot `dictionaryCount`), the nonfinite input and output counts in `status` (published filters them silently), first-source-row ordering and compaction to occupied groups (local `GPUSort`, keep it), and key windows beyond `groupCount` (the windows would need a derived key column). Sums are float atomics in both, so values do not change.
- Verify: `modules/kepler/test/group-by.spec.ts`, `group-by.node.spec.ts`, `group-by-passes.node.spec.ts`, `host-group-by.spec.ts` (some are in `excludedFromRuns`); a parity spec on a table with null keys and NaN values is needed first.
- Risk: different handling of null keys and nonfinite rows; decide that before delegating. Lower priority than KL-01 to KL-03.

### KL-08 Dead dataframe-query helpers in the lens support file; one duplicates `makeCPUVectorFromGPUVector`

- Class: adopt-now (delete).
- Local: `modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support.ts:87-118` (`readGPUVectorValues`: strided per-chunk `readAsync` and decode), `:310-329` (`EncodableDataFrameQuery`, `submitDataFrameQuery`), `:331` (`applyValidity`), `:339` (`gatherValues`), `:374` (`createDataFrameQueryLensRecord`). `git grep` finds no caller of any of them in `modules`, `website` or `examples` (left over from the dropped dataframe-query lens scenarios, register section 1).
- Published: `node_modules/@luma.gl/graph/dist/gpu-data/cpu-gpu-vector-transfer.d.ts` `makeCPUVectorFromGPUVector` (handles chunks, `byteOffset`, `byteStride`, `cpu-gpu-vector-transfer.js:34`, `:71`).
- Evidence: read both.
- Change: delete the five exports; if a reader comes back, build it on `makeCPUVectorFromGPUVector`. 0 call sites.
- Verify: `yarn typecheck`; lens record specs.
- Risk: none.

### KL-09 Lens output copies are compiled WGSL kernels instead of copy nodes

- Class: adopt-with-cost (low value).
- Local: `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/expansion-examples.ts:84-95` and `urban-ecology-examples.ts:61-72`: `createWGSLKernelNode` with `operation: 'lens-output-copy'`, body `destination[i] = source[i]`.
- Published: `GPUCommandGraph.addCopyPass` (`gpu-core/gpu-command-graph.d.ts:303`; kepler uses it at `styles/resident-style.ts:907`), or `GPUElementwise({operation: 'copy'})` for scalar formats (`gpu-core/gpu-elementwise.d.ts`; kepler uses it at `analysis/point-hotspots.ts:163`).
- Evidence: read both (local) and declarations (published).
- Change: a byte-range copy node per captured view. 2 call sites.
- What it loses: the kernel facts that `recordWGSLKernelNodes` puts into the lens record for these nodes (the record would show a copy node instead), so the how-it-works pages change. Keep as is if those pages show the copy kernel on purpose.
- Verify: lens record specs for the two scenario families (record shape changes, values do not).

### KL-10 Lens selection counts reduce an all-ones buffer under a mask

- Class: misuse (minor).
- Local: `modules/lenses/src/computation-views/coordinator/gpu-crossfilter-selection-backend.ts:270-288` (`GPUReduction` `sum` of `getAllRows(keySpace)`, a `keyCount`-word buffer of ones uploaded at `:216-227`, with `mask`).
- Published/idiom: the crossfilter masks are canonical 0/1 (`gpu-selection.d.ts`), so `GPUReduction({operation: 'sum', input: mask})` gives the count with no extra buffer; kepler does exactly this at `gpu-resident-explorer.ts:390-401`.
- Evidence: read both.
- Change: sum the mask directly for spaces whose mask comes from the crossfilter; keep the all-ones input only for relation targets whose `GPUTransferMask` output may hold counts above one (not checked here). Saves `4 * keyCount` bytes of upload and storage per key space (key spaces are above 65,536 by construction, `:72`). 1 call site.
- Verify: `gpu-crossfilter-selection-backend.node.spec.ts`, `gpu-crossfilter-selection-backend.spec.ts`.
- Risk: wrong counts if a non-canonical mask is summed; gate on the mask's origin.

### KL-11 `nullable` set by mutating a built table's schema instead of passing `fields`

- Class: misuse (minor, adopt-now).
- Local: `gpu-resident-explorer.ts:197-203`, `gpu-table-inspector.ts:408-410`, `gpu-scalar-selection.ts:520-522`, `session/resident-dataset.ts:466`, `session/adopted-arrow-dataset.ts:176` (7 assignment lines in 5 files).
- Published: `gpu-tables/table/gpu-record-batch.d.ts:18-32` (`GPURecordBatchFromDataProps.fields?: GPUField[]`, with `nullable` in `gpu-schema.d.ts`), then `new GPUTable({batches})`. `GPUDataFrame` reads `field.nullable` to decide whether a validity mask is required (`gpu-dataframe/gpu-expression-shader.js:333-336`).
- Evidence: read both.
- Change: build batches with `fields` carrying `nullable`, or keep one local helper that does so; this states intent at construction instead of editing table and batch schemas in two loops.
- Risk: none; same schema values.

### KL-12 `MeteredCommandGraph` exists because compiled dataframe queries hide their graph statistics

- Class: blocked (not in the register).
- Local: `modules/kepler/src/session/metered-command-graph.ts` (overrides `compile` to keep `stats.physicalTransientResourceBytes`), used by every kepler dataframe compile (explorer, inspector, scalar selection).
- Published: `gpu-dataframe/gpu-query-compiler.d.ts:56-77`: `CompiledGPUDataFrameQuery` keeps `graph` private and exposes no `stats`.
- Needs: a public `stats` (or `transientByteLength`) on compiled dataframe queries. Then delete the subclass.
- Evidence: read both.

## 3. Checked and fine

- `GPUCrossfilter` in the explorer (`gpu-resident-explorer.ts:206-302`): dimensions, `visibility` view with caller-owned count, `mask` views with `includeOwnSelection`, `getViewMask`, `outputMask`, `destroy` in the error path. External masks and the live mask are range dimensions over a 0/1 mask with `setRange(SELECTED_MASK_RANGE)` and `clear`, the documented workaround. Histograms use `mask` views plus `GPUMask` with column validity plus `GPUHistogram` instead of `histogram` views because validity is per column; a `histogram` view would apply no validity. Correct.
- Register claim "GPUCrossfilter lacks external masks and `invalidate`": holds at alpha.13. `GPUCrossfilterDimension` is only `range` or `bounds`; `GPUCrossfilter` has `setRange`, `setBounds`, `clear`, `clearAll`, `getDimensionMask`, `getViewMask`, `addToGraph`, `destroy` and nothing else (`gpu-crossfilter.d.ts`, `gpu-selection.d.ts`).
- `GPUKeplerSpatialSelection` (`gpu-spatial-selection.ts:252-269`): analysis `GPUPointInPolygonJoin` writes feature IDs, then a published `GPUCrossfilterSelection` with range `[0, 0]` turns them into a source-aligned mask. `GPUMask` has no equality test, so this is the shortest published route. Buffers and the compiled graph are released after pending reads.
- `gpu-scalar-selection.ts` category filters: an OR of `parameter('enabled_i') && value == parameter('category_i')` capped at 64 (`:830-834`); a lookup or join would cost more at that size.
- Global style quantiles, colour and scalar scales (`resident-style.ts`): analysis `GPUColumnQuantiles`, `GPUColorScale`, `GPUScalarScale`. Published luma.gl alpha.13 has no column colour scale, size scale or column quantile (export search for Scale, Color, Quantile, Palette); `@luma.gl/graph/arrow` `convertColors` is a CPU colour format conversion. Nothing to delegate.
- `cluster/`: CPU supercluster, d3-scale and a KD tree; no GPU equivalent applies.
- `gpu-kepler-aggregation.ts`: delegates to analysis `GPUPointDensity` (register section 4 row `GPUGridBinning`), with a published `GPUMask` for selection and validity.
- Table sort (`gpu-table-inspector.ts:380-430`): `GPUDataFrame.filter(column('selected').equal(literal(1))).sortByGlobal(..., {nulls: 'last', nans: 'last'})`, the published global sort with null and NaN placement. Correct use.
- Arrow upload (`session/arrow-stream-snapshot.ts:35-36`): uses published `makeGPUAnalyticsTableFromArrowTable` after local normalization, so upload, validity vectors and dictionaries are already published code.
- Arrow analytics workaround (`analysis-core/graph-support/arrow-analytics-*.ts`): none of the 85 `@luma.gl/graph/arrow` value exports covers it. `makeGPUAnalyticsTableFromArrowTable` accepts only Float32, Int32, Uint32 and `Dictionary<Utf8, Int32|Uint32>` (`arrow-gpu-analytics-adapters.d.ts`): no plain Utf8, Bool, Int8/16, Float64 policy or Int64/Decimal. `convertArrowTemporalToGPUVector(s)` converts to relative Float32 from an origin (`arrow-temporal-gpu-vector.d.ts`), which loses the exact Int64 words that `makeGPUTimeVectorFromPreparedArrowColumn` keeps for `GPUTimeWindowFilter`. `getGPUDataBufferProps` is not exported from any entry point, so the hand-written buffer usages in `arrow-analytics-time-upload.ts:99-103` cannot use it.
- `GPUTableBufferPlanner`: plans vertex and storage binding layouts for rendering; `dataset-registry.ts` is a memory budget and lease registry. No overlap.
- `compute-session.ts` and `execution/`: no frame budget, time slicing or incremental work exists to delegate. Fences use `device.createFence()` and `Promise.race([fence.signaled, device.lost])` (`compute-session.ts:469-486`, `gpu-table-inspector.ts:540-547`), the public route. `GPUIncrementalExecution` and `GPUCommandGraphExecutionBudgetController` have no local counterpart in this stream.
- Direct `Kernel` construction: `analysis-scaling.ts:102` (`addKeplerLinearKernel`, 42 call sites), `resident-style.ts:668` and 8 more in `analysis/`: each uses public `getBoundedDispatchLayout`, `getBoundedInvocationIndexSource`, `getViewBinding`, `getViewElementOffset`, and returns `destroy: () => kernel.destroy()`. Pipelines go through the default `PipelineFactory` cache (`execution/compute/kernel.js:58-69`). Released and bounded. The same kernels could use analysis `createWGSLKernelNode` (not a luma.gl question).
- `catalog-gpu-session.ts`, `point-weights-streamed.ts`, `analysis5-terrain-resident-gpu.ts`: one-off batch runs with explicit `destroy`; one-shot `readAsync` is acceptable there.
- Lens runtime (`lens-view-snapshot.ts`): timings use published `CompiledGPUCommandGraph.encode` with a timestamp query set and `GPUCommandGraphEncoding.readTimings()` (through analysis `submitGPUCommandGraphWithTimings`); encoding stats come from `encoding.stats`. `GPUCommandGraphEncoding.readResource` is no substitute for the snapshot tap: it returns at most 256 bytes and refuses a transient whose allocation was reused (`gpu-command-graph.js:1287-1291`, `gpu-resource-preview.js:12`). `GPUCommandGraphInspector` keeps rolling sample histories for repeated encodings, which one-shot lens runs do not need. `@luma.gl/graph/gpu-trace` is a trace-visualisation toolkit (spans, lanes, critical path), not graph capture.
- Lens scenarios: the sampled files construct analysis contributors and recipes (`GPUTemporalReduction`, `GPUNetworkReachability`, `GPUIsolines`, `GPURasterSpectralUnmixing`, `GPUClassBreaks`, `GPUGeometryPredicates`, `GPULineNoding`, `GPURegionMask`, `GPUSpatialErrorGM`, `GPUTerrainFlow`, ...) with published `GPUHistogram`, `GPUGather` and `GPUScan` as glue. No lens holds a private copy of an analysis or published kernel; the only WGSL in lenses is the two output copies (KL-09), a strided column split in `s1-regression-model.ts:200-218` and a strided capture in `interpolation.run.ts:206-214` (presentation only; `GPUGather` has no stride). The 66 `@luma.gl/graph/gpu-raster` imports in lenses are all `type GPURasterMetadata`; no lens constructs a published raster class.
- `gpu-crossfilter` in lenses (2 imports): type-only, used by the selection backend over a caller-built crossfilter (apart from KL-05 and KL-10, correct: persistent masks enforced, one packed count readback). `gpu-dataframe` in lenses (1 import): type-only, in the dead helpers of KL-08.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

- Section 2, "Out-of-memory detection": the row says the code matches by name "instead of an exported class". At alpha.13 luma.gl never produces either name: uncaptured WebGPU errors are re-wrapped as plain `Error` (`webgpu-device.js:58-63`) and `pushErrorScope` is a no-op without `debug`. The detector is dead code; what luma.gl needs is to keep the error type in `reportError` and an always-on out-of-memory scope (KL-04).
- Section 2, "Crossfilter extensions": the local code column points at `selection/gpu-spatial-selection.ts` and `gpu-time-selection.ts`, which hold no crossfilter workaround (their `invalidate()` is their own revision counter). The workaround lives in `exploration/gpu-resident-explorer.ts:207-215` (external mask as a range dimension), `:476` (`setExternalActive`), `:504` (`invalidate`) and in lenses `computation-views/coordinator/gpu-crossfilter-selection-backend.ts:101` (explicit key sets rejected for the same reason). The claim itself holds at alpha.13.
- Section 2, "Arrow analytics normalisation": state that upload, validity and dictionaries already go through published `makeGPUAnalyticsTableFromArrowTable`; the local part is normalization only (plain Utf8 to codes, Bool, small integers, Float64 and wide numeric policy, exact Int64 time words). Published `convertArrowTemporalToGPUVector` is not a replacement because it produces relative Float32.
- New row (section 2): `MeteredCommandGraph` (KL-12), needs public stats on `CompiledGPUDataFrameQuery`.
- New row (section 3 or 2): mask-only dataframe filter (KL-02), needs a compile option that skips `GPUVisibilityWorkflow` and `rowIndices`.
