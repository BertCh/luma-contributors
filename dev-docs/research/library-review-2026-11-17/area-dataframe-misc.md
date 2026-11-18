# Dataframe, point cloud, geospatial, computation record, and testing

The highest-priority fixes are at public data boundaries. I reproduced a Decimal value changing under an `exact` policy with a reported error of zero, and a single NaN point invalidating the local coordinates of otherwise valid points. I also found a calendar period that the CPU helper accepts and the generated GPU predicate excludes, plus a computation-record spec that explicitly expects graph-owned buffers to be reported as caller-owned.

## What I checked

I reviewed the working tree read-only. The installed `@luma.gl/graph` is `10.0.0-alpha.14`, which matches the package peer declaration. I used that version for public-API comparisons. The supplied repository instructions still name alpha.13. See `node_modules/@luma.gl/graph/package.json:6` and `modules/analysis/package.json:89`.

| Source area | TypeScript files | Lines | Deep sample |
|---|---:|---:|---|
| `dataframe-analysis` | 141 | 41,430 | Calendar buckets, period index and its CPU helper, time windows, wide numeric upload and aggregation, key joins. Sampled the largest file, `natural-breaks-nodes.ts`, and the CPU quantile operation. |
| `pointcloud-analysis` | 22 | 6,279 | Local-origin helpers, grid construction, neighbourhood output and reuse, normal-scale composition. Sampled ICP, change, and surface callers. |
| `geospatial-analysis` | 4 | 959 | Public barrel, viewport bounds, point-buffer parameters and chunking, datum-grid validation and interpolation. |
| `computation-record` | 18 | 3,216 | Creation and identity, validation, snapshots, graph description, metrics, trace export, public docs and consumers. |
| `testing` | 4 | 437 | Public/private boundary, device acquisition, resource counters, pipeline capture and counting. |
| `analysis-core/graph-support` | 4 | 563 | Arrow normalization, wide conversion, exact timestamp upload and public barrel. |

Full-file reads included `arrow-analytics-normalization.ts`, `arrow-analytics-wide-numeric.ts`, `period-index-calendar.ts`, `pointcloud-local-origin.ts`, `geographic-viewport-bounds.ts`, `gpu-snapshot-tap.ts`, `gpu-snapshot-arena.ts`, `validate-computation-record.ts`, `gpu-resource-counts.ts`, `require-test-device.ts`, and the scoped entry barrels. I sampled matching specs in `analysis-core`, `gpu-dataframe`, `dataframe-analysis`, `gpu-pointcloud`, `gpu-geometry`, `geospatial-analysis`, and `computation-record`. The recent chunking and column-profile changes came from the scope-filtered `git log` and `git show` of `69305eb4`.

Representative census commands, from the repository root:

```sh
rg --files modules/analysis/src/{dataframe-analysis,pointcloud-analysis,geospatial-analysis,computation-record,testing,analysis-core/graph-support} -g '*.ts' | wc -l
rg -n 'apache-arrow' modules/analysis/src
rg -n 'spatialContext|createPointCloudGridIndex\(' modules/analysis/src/pointcloud-analysis/gpu-*.ts
rg -l -F '@luma-contributors/analysis/computation-record' modules/lenses/src -g '*.ts' | wc -l
rg -l 'getGPUCommandNodePipelineKeys\(' modules/analysis/test | wc -l
rg -l 'countDistinctPipelines\(|recordComputePipelines\(' modules/analysis/test | wc -l
```

I also walked static ESM imports and literal CommonJS `require()` edges in the existing `dist` files, including resolved dependencies, and ran small extracted CPU-function probes. I did not build, test, run GPU code, or produce a package tarball. I did not audit every classification, pair-statistics, forest, or ICP kernel. GPU execution outcomes below are marked `UNVERIFIED` where applicable.

Paths beginning `src/` or `test/` below are relative to `modules/analysis/`. Other paths are repository-relative.

## Findings

### DPC-01 Arrow Decimal normalization can report an exact conversion after losing data [P0] [correctness]

- **Evidence.** `src/analysis-core/graph-support/arrow-analytics-wide-numeric.ts:37` converts the Decimal coefficient with `Number(integer)`. `src/analysis-core/graph-support/arrow-analytics-normalization.ts:226` returns immediately when `Object.is(value, rounded)`. I ran the extracted normalization functions on a real one-row Arrow `Decimal(0, 38, 128)` containing `9007199254740993`. With `{float64: {mode: 'exact'}}`, the output was **`9007199254740992`**, with both reported maximum errors **`0`**.
- **Why it matters.** A caller explicitly requesting lossless input can receive changed data and a conversion report certifying it. This happens entirely on the CPU, before upload.
- **Proposed change.** Carry the integer coefficient and decimal scale through the comparison against the Float32 candidate. Compute exactness and the error budget against that original rational value. Reuse the integer-rounding logic at `src/dataframe-analysis/column-profile/numeric-narrowing-diagnostics.ts:111` for scale zero. Add the reproducer and a nonzero-scale case to the public preparation specs. **M**, behavioral correction. Previously accepted lossy `exact` conversions will throw.
- **Confidence.** High. This affects an already-supported type, distinct from the additional type coverage in roadmap `ARROW-01`.

### DPC-02 Calendar period indexing excludes valid periods within its advertised range [P0] [correctness]

- **Evidence.** `src/dataframe-analysis/period-index/gpu-period-index.ts:78` permits `count` up to `0xfffffffd`. The generated calendar predicate at line 365 uses `candidate < ${Math.min(periods.count, 0x7fffffff)}`. For an hourly axis beginning `1970-01-01`, `count = 2^31`, and input `+246953-10-09 07:00`, the exported CPU helper returns **`2147483647`**. The emitted comparison is `2147483647 < 2147483647`. No arithmetic overflow is needed for this counterexample. At line 352, larger dates also form `daysFromCivil(...) * 24` in i32 before subtraction.
- **Why it matters.** One input row suffices to produce the disagreement. `count` describes the axis and does not require allocating billions of rows. The input date is within the calendar decoder’s documented range at `src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts:217`.
- **Proposed change.** Form the calendar step difference at sufficient width, compare its sign, divide, then compare the resulting unsigned period with the full `count`. The fixed-width path already separates ordering, division overflow and the unsigned range check at `src/dataframe-analysis/period-index/gpu-period-index.ts:382`. Add CPU/GPU parity cases at `2^31−1`, `2^31`, and with a negative origin. **M**, nonbreaking correction.
- **Confidence.** High for the CPU result and emitted comparison. Actual GPU readback `UNVERIFIED`.

### DPC-03 One nonfinite point poisons the recommended local origin for the whole cloud [P0] [correctness]

- **Evidence.** `src/pointcloud-analysis/pointcloud-local-origin.ts:25` applies `Math.min` and `Math.max` to every coordinate. Line 38 subtracts the resulting origin from every row. For packed points `[1,2,3, NaN,4,5, 7,8,9]`, the extracted helpers returned origin `[NaN,5,6]` and NaN for **all three X coordinates**. This is the documented preparation sequence for ICP and change detection at `docs/api/pointcloud-analysis.md:573`. The grid itself has an invalid-point sentinel and a finite-position check at `src/pointcloud-analysis/pointcloud-grid-index.ts:175`.
- **Why it matters.** An isolated missing observation can invalidate both clouds when they share the recommended origin. The caller loses valid rows before the contributor can apply its row-level invalid-data policy.
- **Proposed change.** Compute the bounding box over complete finite XYZ triples. Define the all-invalid result explicitly, and reject a trailing partial triple. Test a mixed cloud, an all-invalid cloud, and two clouds sharing the derived origin. **S**, preserves finite, correctly packed inputs.
- **Confidence.** High. Reproduced on the CPU.

### DPC-04 Decimal128 upload accepts other physical widths and changes row identity [P1] [correctness]

- **Evidence.** `src/dataframe-analysis/column-profile/wide-numeric-gpu-decimal-aggregate.ts:160` accepts a generic `Vector`. Line 169 copies `values.subarray(0, chunk.length * 4)`. The higher-level entry accepts `Vector<Decimal>` at line 220, which also admits Decimal256. Arrow defines Decimal stride as `bitWidth / 32` at `node_modules/apache-arrow/src/type.ts:751`. For a two-row Decimal256 vector containing `[1, 2]`, I verified that Arrow reports stride 8. The slice used by the uploader contains the first row’s low and high halves, interpreted as two four-word rows **`[1, 0]`**.
- **Why it matters.** A typed caller can pass a valid Arrow Decimal column and get a plausible answer computed from the wrong rows. The existing upload fixture fixes the type at 128 bits at `test/dataframe-analysis/column-profile/wide-numeric-upload.node.spec.ts:11`.
- **Proposed change.** Narrow the factory argument to `Vector<Decimal>` and validate `DataType.isDecimal(column.type) && column.type.bitWidth === 128` before the first device allocation. Include the actual type and width in the error. Add a two-row Decimal256 rejection case. **S**, rejects inputs the implementation cannot represent.
- **Confidence.** High for the exact upload bytes. Downstream GPU aggregate `UNVERIFIED`.

### DPC-05 A snapshot tap can capture bytes outside the supplied logical view [P1] [correctness]

- **Evidence.** `src/computation-record/gpu-snapshot-tap.ts:61` reduces a `GraphDataView` to `{buffer, byteOffset}`. `src/computation-record/gpu-snapshot-arena.ts:118` bounds the copy against `source.buffer.byteLength`, and line 132 copies `slot.byteLength`. A one-word view into a larger buffer therefore passes the checks for an eight-byte slot if the backing allocation has room. A strided view likewise becomes a contiguous copy. The lenses consumer already rejects strided views and derives slot bytes from `view.length * view.rowByteLength` at `modules/lenses/src/computation-lenses/lens-view-snapshot.ts:109`.
- **Why it matters.** A capture labeled as one intermediate result can contain its neighbour’s data. The safer behavior currently depends on reaching the tap through the lenses wrapper.
- **Proposed change.** Validate slot names, packed stride and copy length against each logical view in `GPUSnapshotTap` before adding a copy node. Keep the arena’s backing-buffer check for direct callers and whole-buffer handles. Test an offset subview with an oversized slot, plus a strided subview. **S**, rejects ambiguous captures.
- **Confidence.** High for the copy descriptor and bounds check. GPU copy `UNVERIFIED`.

### DPC-06 Computation records assert caller ownership for graph-owned resources [P1] [correctness]

- **Evidence.** `src/computation-record/describe-command-graph.ts:230` says “ownership is unknown” and immediately emits `ownership: 'imported'`. The public type defines imported as caller-owned at `src/computation-record/computation-record-types.ts:281`. The existing fixture constructs transient buffers at `test/computation-record/computation-recorder.node.spec.ts:31`, then explicitly expects every resource to be imported at line 112.
- **Why it matters.** A consumer inspecting an ordinary graph without annotations gets affirmative, incorrect lifecycle information. The existing test protects that behavior. A related size problem is visible at `src/computation-record/describe-command-graph.ts:150`, which uses the largest individual view length as the resource size. Two disjoint four-byte views can describe a backing resource of at least eight bytes.
- **Proposed change.** Represent unobserved ownership as `unknown` or an omitted field. Distinguish known logical size from an observed lower bound, using `max(byteOffset + byteLength)` for the latter. Change the fixture to expect unknown without handles and transient with handles. **M**, schema-visible. Existing `/1` records need a compatibility policy.
- **Confidence.** High. The fixture and serializer establish the contradiction directly.

### DPC-07 Arrow is optional at the package root but required by two general-purpose entry points [P1] [packaging]

- **Evidence.** `modules/analysis/package.json:94` marks Arrow optional. `src/index.ts:5` exports only computation-record. Both static module walks found **no Arrow edge** from the root, computation-record, pointcloud-analysis, or geospatial-analysis. Both reached Arrow from `analysis-core` through `src/analysis-core/index.ts:6` → `src/analysis-core/graph-support/index.ts:13` → `arrow-analytics-normalization.ts:24`. Both also reached Arrow from `dataframe-analysis` through `src/dataframe-analysis/index.ts:7` → `column-profile/index.ts:28` → `wide-numeric-gpu-decimal-aggregate.ts:7`. That last import is a runtime dependency because line 186 constructs `new Vector([chunk])`.
- **Why it matters.** A consumer using only a named limit helper or a GPU calendar contributor still needs to resolve Arrow during ordinary package import. The existing export verifier loads entries in the installed workspace at `scripts/verify-package-exports.mjs:19`, where Arrow is available.
- **Proposed change.** Add an explicit `@luma-contributors/analysis/arrow` entry for normalization and Arrow-vector factories. Keep resident-vector Decimal aggregation in dataframe-analysis. Publish the import matrix now, then remove eager re-exports at an announced compatibility boundary. Add an installed-package fixture with the optional peer absent. **M**, additive initially. Making the existing barrels Arrow-free requires an export migration.
- **Confidence.** High for static ESM and CommonJS resolution in the existing distribution. A clean tarball install was outside this review.

### DPC-08 The default run ID cannot distinguish repeated executions when joining sidecars [P1] [architecture]

- **Evidence.** The docs describe one real execution and say the sidecar join prevents mixing timings at `docs/api/computation-record.md:5`. `src/computation-record/create-computation-record.ts:71` defaults `runId` to a hash of graph/node IDs, backend, variant, dataset version and seed, with the inputs listed at line 97. Two executions with identical inputs receive the same ID by construction. `src/computation-record/validate-computation-record.ts:370` accepts matching run, graph and node IDs. The dataframe lens recorder uses the default at `modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support.ts:253`.
- **Why it matters.** The stable hash serves comparisons and permalinks. It also permits metrics or captures from a previous execution to validate against a new execution, even when per-frame inputs or adapter hardware differ.
- **Proposed change.** Retain the deterministic hash as structural identity. Add an execution identity to records and sidecars and validate it when present. Allow fixtures to supply it explicitly. Preserve existing permalink resolution during migration. **M**, additive with a `/1` compatibility path.
- **Confidence.** High. The ambiguity requires no hash collision.

### DPC-09 The test utilities undercount the pipeline configurations luma.gl actually caches [P2] [testing]

- **Evidence.** The installed cache hashes shader source plus `JSON.stringify([props.shaderLayout, props.entryPoint ?? 'main', props.constants])` at `node_modules/@luma.gl/core/src/api/factories/pipeline-factory.ts:369`. `src/testing/gpu-contributor-test-utils.ts:128` sorts constants, includes layout only by option, and omits entry point. The second recorder at line 314 uses source and serialized constants. The census found **155** analysis test files calling `getGPUCommandNodePipelineKeys` and **14** calling `recordComputePipelines` or `countDistinctPipelines`; the groups can overlap.
- **Why it matters.** A test asserting stable pipeline count can pass when two descriptors differ only in entry point or layout. Conversely, callers measuring distinct WGSL modules have a legitimate reason to ignore descriptor differences. Those measurements currently share names and machinery.
- **Proposed change.** Record the descriptor once. Expose separate helpers for unique shader modules and distinct pipeline configurations. For the latter, prefer observing public factory cache misses on the recording device. Add one source with two entry points and one source with two layouts as fixtures. **M**, workspace-test-only.
- **Confidence.** High for the counting mismatch. This is evidence about test coverage, not a measured production regression.

### DPC-10 Pre-epoch timestamps can fall on different calendar days in presentation and analysis [P2] [consistency]

- **Evidence.** `src/dataframe-analysis/temporal-reduction/timestamp-format.ts:10` truncates microsecond and nanosecond ticks toward zero. Thus `getTimestampPresentationMilliseconds(-1n, 3)` returns `0`, the first millisecond of 1970. The exact formatter in the same file adjusts a negative remainder at line 21 and renders the instant in 1969. `GPUCalendarBuckets` documents flooring negative sub-millisecond times at `src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts:138`. A consumer selects between the two presentation helpers at `examples/deck/kepler-compute/src/original-column-values.ts:45`.
- **Why it matters.** A date label or a host-side daily grouping can disagree with the analysis for the same instant. Passing the correct Arrow unit does not resolve the disagreement.
- **Proposed change.** Provide a floor-to-containing-millisecond helper for calendar display, and use it for dates. Preserve the existing truncation helper for callers that need truncating numeric conversion. Test `−1 ns`, `−1 µs`, zero and a positive fraction together across formatting and bucketing. **S**, additive.
- **Confidence.** High for both host results and the documented calendar rule.

### DPC-11 The public Decimal uploader makes callers own an unused validity buffer [P2] [ergonomics]

- **Evidence.** `src/dataframe-analysis/column-profile/wide-numeric-gpu-decimal-aggregate.ts:183` creates a flag array for every chunk. Line 189 allocates a device buffer even when the whole column has no nulls. Line 208 returns it as `unusedValidity`. `test/dataframe-analysis/column-profile/wide-numeric-upload.node.spec.ts:75` explicitly expects `validity` to be absent, `unusedValidity` to exist, and the caller to destroy it.
- **Why it matters.** For a non-null Decimal128 column this allocates another four bytes per row, about **25%** of the 16-byte-per-row word payload, plus buffer overhead. A caller that correctly destroys the words and optional validity can still leave an allocation alive.
- **Proposed change.** Inspect chunk `nullCount` metadata before allocating. If any chunk has nulls, preserve validity topology across every chunk. Otherwise omit validity allocations. Return one `destroy()` method following the existing time-upload pattern at `src/analysis-core/graph-support/arrow-analytics-time-upload.ts:80`. Retain a deprecated optional `unusedValidity` field during migration. **S**, compatible with callers using optional cleanup.
- **Confidence.** High. The allocation path and its ownership expectations are explicit.

### DPC-12 The dataframe guide omits the execution cost of its public one-shot operations [P2] [docs]

- **Evidence.** `docs/api/gpu-dataframe-analysis.md:17` describes contributors as adding command nodes and says “It never compiles, submits, encodes, or reads back.” `src/dataframe-analysis/column-profile/index.ts:38` publicly exports `runWideNumericKernel`. `wide-numeric-gpu-runner.ts:46` compiles, then encodes, submits and awaits readback at lines 48–50. Public `aggregateInteger64OnGpu` delegates to it at `wide-numeric-gpu-aggregate.ts:31`. `aggregateDecimal128Column` additionally uploads and frees an Arrow column at `wide-numeric-gpu-decimal-aggregate.ts:223`.
- **Why it matters.** Someone building a per-frame workflow needs to know whether an operation composes into their existing graph, submits another graph, uploads CPU data, or waits for a result.
- **Proposed change.** Add a small execution/ownership table to the guide for `GPUWideNumericScan`, the resident-vector one-shot functions, and the Arrow convenience functions. Show one graph-resident path and one explicit readback path. **S**, nonbreaking.
- **Confidence.** High.

### DPC-13 The key-join backlog understates the shipped aggregate set [P3] [docs]

- **Evidence.** `dev-docs/roadmap.md:125` lists five aggregates. The public union includes a sixth, `'count-distinct'`, at `src/dataframe-analysis/key-join/gpu-key-join.ts:75`. It has a dedicated execution branch at line 778 and an independent Set-based expectation at `test/gpu-dataframe/key-join/key-join.node.spec.ts:163`.
- **Why it matters.** A planning pass can schedule work that has already landed. “Other key-join operations” also gives a future implementer no concrete acceptance criterion.
- **Proposed change.** Mark distinct count delivered. Name the remaining requested operation and its semantics, or close the row until there is a consumer request. **S**, nonbreaking.
- **Confidence.** High. `KNOWN(KJ-01)` — the row is stale.

## Cross-cutting patterns

- **Test the representation that crosses the boundary.** DPC-01, DPC-02 and DPC-04 each pass through a narrower representation before a result is accepted. Three inexpensive fixtures cover the failures: a Decimal coefficient just above `2^53`, the last valid signed-positive hourly period, and two Decimal256 rows with visibly different low words. Each catches a class of error that ordinary small values conceal.
- **Time has a shared representation, with uneven metadata propagation.** Of five central operations checked, period index and time-words-relative require `temporalContext`, calendar buckets accepts it optionally, and time-window filter and temporal reduction select their mode from the view format. A `uint32x2` view establishes word layout but carries no unit or epoch. The Arrow uploader already returns timestamps, validity and context together. Use that result shape in examples and offer context-aware window and reduction parameter packers, while retaining raw views for existing callers. See `src/analysis-core/graph-support/arrow-analytics-time-upload.ts:75`, `src/dataframe-analysis/period-index/gpu-period-index.ts:165`, `time-words-relative/gpu-time-words-relative.ts:66`, `calendar-buckets/gpu-calendar-buckets.ts:143`, `time-window-filter/gpu-time-window-filter.ts:113`, and `temporal-reduction/gpu-temporal-reduction.ts:99` (last four under `src/dataframe-analysis/`).
- **Point-cloud reuse exists at two different levels.** Consumers can reuse neighbourhood results: features accepts `indices` and `counts`, and normal-scale computes the largest neighbourhood once before iterating radii. Spatial metadata and grid reuse are less uniform. Only **1 of 11** exported classes across the ten `gpu-pointcloud-*.ts` files resolves `spatialContext`. Five files call `createPointCloudGridIndex` internally. Some classes consume summaries and need no coordinate context, so a blanket rule for all eleven would be inappropriate. Apply the contract to coordinate readers first. See `dev-docs/contributor-contract.md:36`, `src/pointcloud-analysis/gpu-pointcloud-features.ts:60`, `gpu-pointcloud-normal-scale.ts:138` and `:309`, and `gpu-pointcloud-neighborhoods.ts:113` in the same directory.

## Keep

- **Keep the dataframe/upstream division of responsibility.** Published `@luma.gl/graph/gpu-dataframe` already exports query, grouped aggregation, histogram, sort, join and lookup operations. Its join explicitly assumes unique right keys. The analysis join handles repeated right keys, 64-bit keys, masks, deterministic first-row selection and 1:n aggregation. `GPUGroupStatistics` adds moments, order statistics and caller-owned compact output. These are concrete reasons to retain the contributors. The guide already explains this distinction well at `docs/api/gpu-dataframe-analysis.md:23`. See `node_modules/@luma.gl/graph/src/gpu-dataframe/index.ts:23`, `gpu-join-query.ts:23` in that directory, `src/dataframe-analysis/key-join/gpu-key-join.ts:169`, and `src/dataframe-analysis/group-statistics/gpu-group-statistics.ts:103`.
- **Keep the explicit Arrow round trip.** On input, local preparation normalizes selected columns, retains batch identity, separates exact timestamp words and validity, and reports narrowing. Use published `makeGPUAnalyticsTableFromArrowTable` for scalar upload. On output, use published `makeArrowTableFromGPUAnalyticsTable` with the result table, appropriate validity, dictionaries and row selection. Time words have their own upload and ownership. An example should carry those sidecars all the way through. See `src/analysis-core/graph-support/arrow-analytics-normalization.ts:64` and `:192`, `src/analysis-core/graph-support/arrow-analytics-time-upload.ts:75`, and `node_modules/@luma.gl/graph/src/arrow/index.ts:18`.
- **Keep computation-record in analysis.** Its public API includes the `/1` record and sidecar schemas, recorder, graph description, validators, snapshots, timing and exporters. The docs explicitly disclaim external schema compatibility. I counted 93 lenses source files mentioning the explicit subpath, and Kepler also creates workflow records using its own execution ID. Graph descriptions depend directly on analysis kernel facts. Moving the recorder wholesale into lenses would put that dependency on the wrong side of the package boundary. See `src/computation-record/index.ts:64` and `:83`, `src/computation-record/describe-command-graph.ts:13`, `docs/api/computation-record.md:10`, and `modules/kepler/src/workflows/workflow-computation-record.ts:73`.
- **Keep `geospatial-analysis` as a real entry point.** It exports a datum-grid transform, a point-buffer generator and a viewport helper. The first two have direct consumers and distinct contracts: ordered correction grids with per-row failure status, and centre-relative ring offsets that preserve small geographic radii. Four source files is a reasonable size for this boundary. See `src/geospatial-analysis/index.ts:5`, `gpu-datum-grid-transform.ts:56` and `gpu-point-buffer.ts:82` in that directory, and `modules/kepler/src/analysis/catalog-gpu-point-buffer.ts:8` and `catalog-network-categorical-datum.ts:5`.
- **Keep testing private.** The barrel explains the workspace-only contract, the package manifest excludes both source and distribution testing directories, and the package export map has no `./testing` entry. I found zero explicit `@luma-contributors/analysis/testing` references in analysis or lenses production source, versus 2 analysis test files and 43 lenses test files. The workspace wildcard alias can still resolve the path in production code, so a source-import guard would preserve the split. See `src/testing/index.ts:5`, `modules/analysis/package.json:75`, and `tsconfig.json:41`.

## Future

1. **Give optional Arrow integration its own entry and compatibility fixture.** Publish one example from a nullable, chunked Arrow table through analysis and back to Arrow. Use it as the acceptance fixture for any upstream normalization proposal. The published upload and readback adapters already provide the transport. The local precision, dictionary and exact-time policies are the work to propose upstream. The trade-off is one import migration in exchange for genuinely Arrow-free core and dataframe entries. See `dev-docs/upstream-workarounds.md:49`.
2. **Introduce a versioned computation-record reader before expanding the schema.** Separate structural identity from execution identity, support unknown ownership, and distinguish known allocation size from observed byte ranges. Keep one decoder for historical `/1` artifacts. This adds reader maintenance but lets recording fidelity improve without silently changing the meaning of saved records. See `src/computation-record/computation-record-types.ts:17` and `:274`.
3. **Expose graph-scoped point-cloud index reuse after measuring a composed workflow.** Start with neighbourhoods and ICP over the same target. Let a graph-owned descriptor carry positions, grid geometry, ordering and invalidation requirements. Preserve the current private builder behind it. The existing rationale for a local index is specific about cell arithmetic, half-open bounds, stable order and raster mode. The new opportunity is avoiding duplicate builds across consumers. The cost is explicit invalidation when positions or grid parameters change. See `src/pointcloud-analysis/pointcloud-grid-index.ts:108`, `gpu-pointcloud-neighborhoods.ts:308`, and `gpu-pointcloud-icp.ts:465`.
4. **Define one boundary-case suite for temporal contributors.** Run the same signed ticks, validity, unit and epoch through Arrow preparation, calendar decode, period index, filtering, reduction and presentation. Include `−1 ns`, a DST transition, a non-Unix relative origin, an empty chunk and a period above `2^31−1`. Keep operation-specific expectations for inclusive windows and half-open buckets. The fixtures add maintenance but make intentional differences visible. See `src/dataframe-analysis/time-window-filter/time-words.ts:53` and `src/dataframe-analysis/temporal-reduction/temporal-reduction-parameters.ts:17`.

## Top 5

1. Fix Decimal exactness against the original coefficient and scale. The CPU reproducer already demonstrates silent loss.
2. Repair calendar-period arithmetic and add the signed-boundary CPU/GPU parity case.
3. Make local-origin preparation preserve valid points in a mixed-validity cloud.
4. Reject incompatible Arrow widths before Decimal128 upload, and remove the unused validity allocation in the same factory.
5. Validate snapshot copies against logical views and correct computation-record ownership reporting. The existing lens wrapper and transient-resource fixture provide concrete starting points for both.

## Questions for the lead

- Is optional Arrow support intended for `analysis-core` and `dataframe-analysis` consumers individually? If so, DPC-07 needs an export migration. If the promise is limited to selected entry points, the immediate deliverable is a tested import matrix.
- Which persisted consumers require the exact `/1` record shape or deterministic `runId` permalinks? That determines whether execution identity and unknown ownership can land additively or should ship behind a versioned reader.
- Which real workflow builds two point-cloud indexes over the same resident positions? That is the benchmark I would use before committing to a public index-handle lifecycle.