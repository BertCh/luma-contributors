# Dataframe, point cloud, and computation records as a user

## Summary

The most valuable change before alpha is a **role-aware Arrow preparation path for keys and measures**. I reproduced two plausible, incorrect answers with real Arrow tables: both labels in a string join picked up the other label’s value, and two nullable measures acquired different, incorrect group means. Both operations completed normally.  
The most consequential point-cloud issue is row identity. A statistical-filter mask can have the right length for ground classification while addressing a different point on every row.  
The most consequential recording issue is persistence. A captured float NaN can become JSON `null`, pass sidecar validation, and become zero when restored to a typed array. The private lenses package already has a binary codec.  
Some conventions work well. The group and join guides specify ordering and tie behavior, the time-filter quick start explains the active prefix, and the ICP guide distinguishes local and absolute transforms. Small host probes confirmed the corresponding reference behavior.  
I recommend renaming `GPUKeyJoin` before publication. Its stable, left-aligned lookup and aggregation behavior is useful, but SQL `INNER JOIN` predicts the wrong output cardinality. I counted 51 actual constructor expressions that would change.

## Census

Paths below use `D` for `modules/analysis/src/dataframe-analysis`, `P` for `modules/analysis/src/pointcloud-analysis`, `R` for `modules/analysis/src/computation-record`, and `A` for `modules/analysis/src/analysis-core/graph-support`. `KI` denotes the newer copy in `luma-contributors-ki`. Both family guides are staging-owned. I checked the KI guides as well as the checkout guides. Source references are to this checkout unless marked KI.

I counted public runtime exports from the four source barrels, bundling the probe on stdin and leaving dependencies external. The producer predicate was `Object.hasOwn(value.prototype ?? {}, 'getCommandNodes')`. Operation counts used the exported names `^CPU.*Operation$` and `^GPU.*Operation$`. File counts used `rg --files` on the four roots.

| Public convention | Dataframe | Point cloud | Record | Arrow graph support |
|---|---:|---:|---:|---:|
| TypeScript files at inspection | 149 | 27 | 20 | 5 |
| Direct public graph producers | 34 | 11 | 2 | 0 |
| Named GPU operation forms | 7 | 0 | 0 | 0 |
| Named CPU operation forms | 7 | 0 | 0 | 0 |
| Direct producers taking an Arrow `Table` | 0 | 0 | 0 | 0 |

The seven paired dataframe forms are ClassBreaks, ColumnQuantiles, CoregionalizationFit, Inequality, RipleyEnvelope, SpaceTimeDiggleTest, and VariogramModelFit. GroupStatistics and KeyJoin have **test oracles**, which I executed separately from the public CPU twins.

| Convention in the requested workflows | Count | Names and consequences |
|---|---:|---|
| Group population and finite-value population have separate counts | 2 producers | `GPUGroupStatistics` has group and column counts. `GPUKeyJoin` has `matchCounts`, aggregate `count` without a column, and finite `count` with a column. The latter aggregate outputs are float32 even though `matchCounts` is uint32. |
| Compact lists identify different entities | 3 producers | `GPUTimeWindowFilter.output.rows` identifies source rows. `GPUKeyJoin.output.rows.ids` identifies matched **left** rows. `GPUTemporalReduction.output.occupiedSlots.ids` identifies **dense slots**. Capacity comes from the respective destination length. |
| A sorted time path is a caller declaration | 2 producers | `GPUTimeWindowFilter` can measure order violations. `GPUTemporalReduction` documents unchecked order. Neither ordering fact establishes Arrow validity. |
| One scale word controls sums for multiple groups or bins | 3 producers | `GPUGroupStatistics`, `GPUKeyJoin`, `GPUValueCube`. The join’s scale can include right keys that no left row matches. |
| A mask changes row domains in the point-cloud workflow | 3 producers in the chain | `GPUPointCloudNeighborhoods` can reorder queries, `GPUPointCloudStatisticalFilter` preserves that order, and `GPUPointCloudSMRF.mask` addresses input positions. |
| A capture category can contain typed values | 3 categories | `snapshots`, `stamps`, `replays`. All use `GPUComputationCapture`. The record barrel exports no standalone capture wire codec. |

**Docs-first controls.** These are the lines I would write from the guides, with caller-owned views already allocated. They distinguish a few real successes from the findings.

```ts
// Several statistics on one measure.
const column = {
  values, statistics: ['count', 'mean', 'median'] as const,
  output: {counts, means, medians}
};

// Stable attribute lookup, including duplicate right keys.
const join = new GPUKeyJoin({
  leftKeys, rightKeys, kind: 'inner',
  gather: [{column: rightValue, output: leftValue}],
  output: {rightRows, matchCounts, rows: {ids, status: {count, overflow}}}
});

// Select first, then aggregate using the source-aligned mask.
const filterOutput = {rows, outputMask, status: {count, overflow, requiredCount}};
const reduction = new GPUTemporalReduction({
  cellIds, timestamps, values, mask: outputMask, parameters, cellCount, bucketCount,
  output: {counts, minimums, maximums, first, last, occupiedSlots}
});

// Convert the source-to-target ICP readback before applying it to absolute XYZ.
const absoluteTransform = icp.getAbsoluteTransform(localTransform);
```

- **Group and join.** On keys `[2,1,2,1]` and values `[NaN,6,3,Infinity]`, the group oracle returned `(key, rows, finite, mean) = (1,2,1,6), (2,2,1,3)`, agreeing with NumPy. On duplicate keys, the join oracle retained the smallest right-row index and compacted matched left rows in ascending source order. UDF-14 gives the full example.
- **Quantiles and classification.** The public CPU twin on `[NaN,0,10,20,Infinity,300]`, with the last row masked, gave linear quartile `5`, median `10`, valid count `3`, and mask `[0,1,1,1,0,0]`. NumPy 2.0.2 on the same finite, selected sample gave `[5,10]`. UDF-08 shows where invalid host requests become apparently valid results.
- **Window and cube.** The time-filter quick start takes 26 lines from first import through closing brace, assumes five supplied views plus device and graph, and returns a parameter buffer. Its readback table correctly says to consume only `rows.slice(0, count[0])`. The selection window is inclusive. Reduction buckets are half-open. A dense value cube is cell-major. UDF-07 and UDF-15 identify the two places where following those workflows loses information.
- **Outliers, ground, and ICP.** I constructed the neighborhood → statistical filter → SMRF chain on a null device. Equal view lengths pass construction, including the permutation in UDF-03. For the separate progressive-ground route, the guide calls for an unclassified cell-minimum surface and coverage before classification. For ICP, a local 90° rotation about Z, zero local translation, and origin `[100,200,0]` produced absolute translation `[300,100,0]` through the public transform helper, as expected.
- **Recording and replay.** The recording example at `docs/api/computation-record.md:112` compiles and describes a graph. Submission and readback are separate. `createGPUComputationCaptures({replays})` packages observations supplied by the caller. UDF-04 and UDF-13 concern what a second process can actually recover.

## Proposed convention

At each public boundary, document and preserve **what one row identifies, how missingness is represented, and which metadata is needed to interpret the result after the call**. Check each documented composition with a small fixture whose rows are permuted, whose nullable measures have different validity, or whose capture must survive serialization. Compare decoded labels, source rows and values, not just lengths and successful construction.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UDF-01 | W | ORDINARY | ADDITIVE | M | Arrow preparation → KeyJoin | Independently encoded string keys join different labels. |
| UDF-02 | W | ORDINARY | ADDITIVE | M | Arrow preparation → GroupStatistics | One row mask cannot express independent nulls in several measures. |
| UDF-03 | W | ORDINARY | ADDITIVE | S | Neighborhoods → StatisticalFilter → SMRF | A valid-looking inlier mask can select the wrong source points. |
| UDF-04 | W | ORDINARY | ADDITIVE | M | Capture persistence | Ordinary JSON loses float missingness, and one lossy representation validates. |
| UDF-05 | W | ORDINARY | ADDITIVE | S | Record construction | Reusing graph metadata can rewrite an already-created execution record. |
| UDF-06 | W | UNUSUAL | ADDITIVE | M | Group, join, value cube | An unrelated key can reduce a singleton sum of exactly `1` to `0`. |
| UDF-07 | W | ORDINARY | ADDITIVE | S | Arrow time → sorted filter | A null and epoch zero can pass order validation and enter the same range. |
| UDF-08 | W | ORDINARY | ADDITIVE | S | Classification packers | Descending custom edges and out-of-domain probabilities produce ordinary numerical answers. |
| UDF-09 | W | UNUSUAL | ADDITIVE | S | Sidecar validation | A capture can validate although no graph contains its resource and observation node together. |
| UDF-10 | W | UNUSUAL | ADDITIVE | M | StatisticalFilter | With multiplier zero, float32 reduction can classify a constant cloud as entirely outlying. |
| UDF-11 | W | UNUSUAL | ADDITIVE | S | Calendar period origins | February 29 in a non-leap year silently becomes March 1. |
| UDF-12 | W | UNUSUAL | ADDITIVE | S | Metrics → Chrome trace | One NaN timestamp corrupts every timed node, including valid measurements. |
| UDF-13 | E | ORDINARY | ADDITIVE | S | Record identity and replay | The record hashes the seed and dataset version, then discards both. |
| UDF-14 | C | ORDINARY | BREAKING | M | KeyJoin | `kind: 'inner'` returns matched left rows, with no duplicate-pair expansion. |
| UDF-15 | D | ORDINARY | ADDITIVE | S | TemporalReduction guide | The documented slot decoder yields fractional JavaScript array indices. |
| UDF-16 | S | UNUSUAL | ADDITIVE | S | Arrow ordered dictionaries | Preparation can return a dictionary it rejects on the next call. |

### UDF-01: Independently prepared string keys can join the wrong labels

**Where.** `A/arrow-analytics-normalization.ts:265,301`, `D/key-join/gpu-key-join.ts:149`, KI `docs/api/gpu-dataframe-analysis.md:888`. The same encoding matters when comparing `GPUGroupStatistics` keys across tables.

**EXECUTED.** I prepared two real Arrow UTF-8 columns separately. Left `['A','B']` produced codes `[0,1]`. Right `['B','A']` also produced `[0,1]`. With right values `[20,10]`, the production preparation plus the join test oracle returned `[20,10]` on the left. Label equality requires `[10,20]`. Preparing the right side with `{dictionaries: left.dictionaries}` produced right codes `[1,0]` and the correct result. Both wrong matches had ordinary matching keys.

**Now.** `prepareArrowTableForGPUAnalytics(left)` and `prepareArrowTableForGPUAnalytics(right)` look like symmetric, independent operations. Also, a wide numeric precision budget is a poor guide for identifiers. With Uint64 IDs `9007199254740992n` and `9007199254740993n`, exact narrowing rejected, while an absolute budget of `1` produced identical float values. The error named the physical Uint64 column “Float64”.

**Wanted.** Put the shared-dictionary example directly under `GPUKeyJoin`, including different column names on the two sides. Add a paired key-preparation helper that returns compatible code or word chunks, validity, and the dictionary needed to decode results. For integer keys, preserve low/high words and handle the reserved all-ones key explicitly. Report the actual Arrow type in errors.

**Callers, value, cost.** No existing signature changes. GIS users joining CSV attributes to boundaries gain label and identifier equality across independently loaded tables. M, with a documentation fix first.

### UDF-02: Multi-measure groups need per-column validity

**Where.** `A/arrow-analytics-normalization.ts:152`, `D/group-statistics/gpu-group-statistics.ts:189,208,237`, KI dataframe guide `:837–859`. `GPUKeyJoinAggregate.column` has the analogous limitation for several nullable right measures at `D/key-join/gpu-key-join.ts:93`. The published Arrow adapter already supplies per-field validity at `node_modules/@luma.gl/graph/src/arrow/gpu/arrow-gpu-analytics-adapters.ts:69`.

**EXECUTED.** For one key and real Arrow columns `x=[null,0,2]`, `y=[12,null,4]`, the prepared vectors retained the correct logical nulls. Their physical float arrays were `[0,0,2]` and `[12,0,4]`. Passing those arrays to the group oracle gave `(count, mean)` of `(3,0.6666667)` and `(3,5.3333333)`. Converting each column’s nulls to NaN first gave `(2,1)` and `(2,8)`, with group row count `3`. Intersecting validity would wrongly leave just the last row.

**Now.** `columns: [{values: x, ...}, {values: y, ...}], mask` has a place for group eligibility and none for `x` and `y` validity independently. Arrow preparation itself did preserve the information.

**Wanted.** Accept `columns[i].validity?: Uint32Rows` on grouping and `aggregates[i].validity` on key lookup, or supply an analysis adapter that materializes per-column missing floats once. Show `COUNT(*)`, `COUNT(x)`, `AVG(x)` and `AVG(y)` together with the fixture above. Keep key eligibility independent.

**Callers, value, cost.** None required. Arrow users can aggregate heterogeneous nullable columns without rebuilding a different table for each measure. M.

### UDF-03: Neighborhood query order can silently reverse a ground-selection decision

**Where.** KI `P/operations/gpu-pointcloud-neighborhoods.ts:259,436`; `P/operations/gpu-pointcloud-statistical-filter.ts:54,142,257`; `P/operations/gpu-pointcloud-smrf.ts:119`; KI point-cloud guide `:539,849`.

**EXECUTED host, SOURCE shader.** Positions at X `[0,1,10]`, `queryRows=[2,0,1]`, radius `20`, capacity `1`, and multiplier `0` give NumPy nearest-neighbor distances `[9,1,1]` in query order. Query mask `[0,1,1]` means source mask `[1,1,0]`. I constructed all three production contributors with the same three-row mask wired directly into SMRF. Every constructor accepted it. The filter output’s TSDoc calls it “Original-row”.

**Now.** `new GPUPointCloudSMRF({positions, mask: filterValidity, ...})` can exclude source point 0 and keep the actual outlier, source point 2. The neighborhood guide correctly explains `queryRows`, but the filter does not carry that mapping onward.

**Wanted.** Describe both filter outputs as **neighborhood-query aligned**. Show scattering the inlier flags through `queryRows` before classifying the original cloud. There is already a suitable `GPUTransferMask` index relation with `mode: 'any'`. A three-point GPU probe should read the query mask and scattered mask separately, then check which original SMRF row gets the excluded bit.

**Callers, value, cost.** None required. Tiled and reordered PDAL-style pipelines retain the correct point identity. S.

### UDF-04: A saved capture can turn a missing value into zero

**Where.** `R/computation-record-types.ts:653–657`, `R/gpu-snapshot-arena.ts:163–177`, `R/computation-metrics.ts:104`, `R/validate-computation-record.ts:474`. All three capture categories use this representation. A working binary encoder exists at `modules/lenses/src/computation-lenses/registry-implementation.ts:245`.

**EXECUTED.** With the repository’s valid record fixture, serializing `Float32Array.of(NaN,-0,2)` produced `{"0":null,"1":0,"2":2}`. The reloaded sidecar failed with “Values must contain whole components”. Following the type comment and converting values to a number array first produced `[null,0,2]`. That reloaded sidecar **validated with no issues**. Restoring it as float32 turns `null` into `0`.

**Now.** `JSON.stringify(createGPUComputationCaptures(...))` has no documented, lossless inverse in the analysis entry. Missing quantiles, empty-bin means and genuine zero measurements can become indistinguishable.

**Wanted.** Expose `serializeGPUComputationCaptures` and `deserializeGPUComputationCaptures` from the record entry. Reuse the lenses binary representation, make element type and byte order explicit, preserve all float bits, and validate payload length. Reject `null` lanes in legacy numeric arrays. Show saving and loading a capture with one finite value and one missing result.

**Callers, value, cost.** No required migration for existing readers. Lenses can delegate to the shared implementation. Users can reopen actual analytical results without changing their meaning. M.

### UDF-05: A record continues to borrow mutable graph facts after construction

**Where.** `R/create-computation-record.ts:73–80,110–120`. Affected facts include node attributes, graph edges and resources, axes, formulas, decisions, and workflow descriptors.

**EXECUTED.** I constructed a valid record from a cloned repository fixture, then set `input.graphs[1].nodes[0].attrs.radius = 9`. The already-created record immediately reported radius `9`. It still validated with no issues. Node IDs were copied; the attributes object was shared.

**Now.** `const first = createGPUComputationRecord(props); updateMetadataForNextRun(props)` can change a record already associated with an execution ID. This extends the earlier identity fixes to the *meaning* of that execution.

**Wanted.** Snapshot all plain record metadata once at construction, after source-reference merging. Test mutation of one node attribute, resource, edge and workflow invocation on the caller side. Keep large capture buffers on their explicit sidecar ownership path.

**Callers, value, cost.** None required. Application authors can reuse graph descriptions without retroactively changing saved evidence. S.

### UDF-06: An unmatched right key can erase a matched singleton sum

**Where.** `D/key-join/gpu-key-join.ts:893,951–980,1011`; `D/group-statistics/group-statistics-moments.ts:247–258`; KI `D/value-cube/gpu-value-cube.ts:370–385,428–431`; `analysis-core/kernels/fixed-point-accumulator.ts:21–29,81–83`. KI dataframe guide `:860,911,1827`.

**SOURCE, unverified GPU.** Use left keys `[1]`, right keys `[1,2]`, and right values `[1,2**100]`. The join scale pass reads valid right groups without consulting left matches. For two rows, headroom is `2`, so the shared unit is `2**(100+1+2-95) = 256`. The matched singleton `1` rounds to zero units. Its mathematically exact sum and mean are both `1`. Both inputs are normal, exactly representable float32 values.

**Now.** Adding a right row that cannot match the query can change a presented sum. The cube guide does describe a cross-bin error bound. The join and group summaries also make stronger exact-sum claims. This differs from KNOWN(DPC-109), where the total overflows before a representable mean is formed.

**Wanted.** Put the cross-key bound and this example beside all three aggregate APIs. For key lookup, grouping and cubes, offer a per-group/bin scale when independent results are required. Smallest GPU probe: run the one-left-row join before and after adding key `2`, reading `matchCounts`, sum and mean. Use two occupied bins as the cube control.

**Callers, value, cost.** None for documentation and an optional scale policy. Analysts can tell whether filtering an unrelated category may change their answer. M for the alternate path, S for the specification.

### UDF-07: Sorted Arrow time can pass order checks while treating null as the epoch

**Where.** `A/arrow-analytics-normalization.ts:363–378`; `D/time-window-filter/gpu-time-window-filter.ts:143,491–495`; `D/time-window-filter/time-window-sorted-nodes.ts:149–175`; KI dataframe guide `:226–249`.

**EXECUTED host, SOURCE shader.** A real nullable Int64 column `[null,0n,10n]` prepared as milliseconds produced words `[0,0, 0,0, 10,0]` and validity `[0,1,1]`. The physical word sequence is sorted. The sorted contributor constructed successfully without validity. Adding validity as a selection predicate produced `time-window-filter sorted does not support additionalPredicates or trackIds`.

**Now.** For window `[0n,0n]`, binary search sees two equal epoch rows. Checking adjacent order cannot distinguish them. This extends KNOWN(DPC-123): the sorted path specifically rejects the existing means of passing the validity predicate.

**Wanted.** Show the unsorted path with `additionalPredicates: [{kind:'selection', mask: timeValidity}]` for nullable Arrow time. Enable the sorted shortcut only on an all-valid view or a separately established contiguous valid slice with its source-row offset. GPU probe: expect source row `[1]` and count `1`. Also test a real epoch followed by a null at the other end of an Arrow batch.

**Callers, value, cost.** None for the documented safe route. Temporal analysts keep real epoch observations while excluding nulls. S.

### UDF-08: Two classification packers turn invalid requests into valid-looking results

**Where.** `D/column-classification/class-breaks-parameters.ts:69,122–146`; `D/column-classification/cpu-class-breaks-operation.ts:294–306`; `D/column-classification/column-quantiles-parameters.ts:35,84–86`; KI `D/column-classification/cpu-column-quantiles-operation.ts:89–95`.

**EXECUTED.** The custom-edge packer accepted `[0,10,5,20]`. The CPU twin classified `[1,6,9,15]` with counts `[1,0,3]`. Sorted edges `[0,5,10,20]` gave `[1,2,1]`. NumPy rejected the nonmonotone bins. Independently, host probabilities `[-1e-50,1+Number.EPSILON]` packed as `[-0,1]`. On finite values `[0,10,20]` the CPU twin returned `[0,20]`, despite the settings type promising NaN for probabilities outside `[0,1]`.

**Now.** A GIS user gets a complete-looking legend from a transposed threshold. A dataframe user gets both extrema for an invalid percentile request.

**Wanted.** Before publishing the parameter buffer, reject the first decreasing packed custom edge and name both indices. Preserve supported repeated edges. Test probability range in binary64 before conversion and encode NaN for out-of-domain inputs, retaining the documented policy. These are additional host-side cases beyond KNOWN(DPC-108).

**Callers, value, cost.** No valid caller changes. Both fixes happen once while packing settings. S.

### UDF-09: A capture can validate against two different graphs at once

**Where.** `R/validate-computation-record.ts:443–460`; `R/computation-record-types.ts:638–647`. Applies to `snapshots`, `stamps`, and `replays`.

**EXECUTED.** In the valid repository fixture, resource `first` exists only in `compiled`, and node `prepare` exists only in `intent`. A capture with `{resource:'first', afterNode:'prepare'}` and omitted `graph` returned `{valid:true, issues:[]}`. No recorded graph contains that observation. A replay using the same ID as a snapshot also validated, although the type calls IDs sidecar-unique.

**Now.** Independent searches for the two names accept a reference that cannot identify one capture location. Duplicate capture IDs also leave consumers without a unique lookup key.

**Wanted.** Intersect the candidate graphs for all graph-local references on one capture. Require one candidate when inference is needed, and request an explicit graph for ambiguity. Check ID uniqueness across all three arrays.

**Callers, value, cost.** No well-formed unambiguous capture changes. Viewers and replay tools can identify one actual observation. S.

### UDF-10: Equal neighbor distances can yield an all-outlier answer

**Where.** `P/operations/gpu-pointcloud-statistical-filter.ts:182–185,232,241,250,257`; published reduction tree at `node_modules/@luma.gl/graph/src/gpu-core/gpu-reduction.ts:438,532`.

**SOURCE, unverified GPU; EXECUTED arithmetic control.** Use five rows, capacity `1`, count `1` each, squared distance `961` each, multiplier `0`. NumPy gives five distances of `31`, mean `31`, standard deviation `0`, and five inliers. Emulating the published binary reduction tree in float32 gave each scaled term `6.199999809265137` and mean `30.999998092651367`. Comparing `31 <= mean` rejects every row on that arithmetic path.

**Now.** The caller explicitly choosing the mean threshold can lose the entire constant population. This is a stronger failure than one borderline point changing classification. It is independent of PDAL’s sample-versus-population deviation convention.

**Wanted.** Preserve a constant mean exactly, for example by reducing distances relative to a supported reference before restoring it. GPU probe: read both `meanDistances` and `validity` on the five-row case, expecting `[31,31,31,31,31]` and `[1,1,1,1,1]`. Include one larger-distance control.

**Callers, value, cost.** No signature changes. Survey users retain uniform scans at a documented threshold. M.

### UDF-11: An impossible civil date can define period zero

**Where.** `D/period-index/period-index-calendar.ts:30–38,105–115`; KI `D/period-index/gpu-period-index.ts:480–493`.

**EXECUTED.** The KI `GPUPeriodIndex` constructor accepted a day axis with origin `{year:2023, month:2, day:29}`. `getGPUPeriodIndexCalendarPeriod` assigned March 1, 2023 to period `0` for that origin. Python rejected the supplied date with `day is out of range for month` and accepted February 29, 2024.

**Now.** An invalid project date silently shifts the start of every bucket. Independent checks for month `1..12` and day `1..31` miss it.

**Wanted.** Share a proleptic-Gregorian days-in-month check between the calendar-origin constructor and the public host calendar path. Validate day/month combinations for day, week and hour axes. Name `periods.origin.day` and the supplied calendar date in the error.

**Callers, value, cost.** None for valid origins. A typo becomes actionable before graph construction. S.

### UDF-12: One bad timing endpoint destroys unrelated valid timings

**Where.** `R/computation-metrics.ts:51–60,74–75`; `R/validate-computation-record.ts:419–430`; `R/export-chrome-trace.ts:66–75`.

**EXECUTED.** Built metrics for `init` with millisecond endpoints `[NaN,2]` and `step:0` with `[3,4]`. Both nodes acquired NaN begin and end. The sidecar validator returned `{valid:true, issues:[]}`. Serialized Chrome trace events were `[{"ts":null,"dur":null},{"ts":null,"dur":null}]`.

**Now.** `Math.min(origin, NaN)` poisons the report’s time origin. The validator tests endpoint order, which does not detect NaN.

**Wanted.** Select the origin from finite begins only. Omit invalid endpoint pairs or return a field-specific error for the offending measurement. Validate endpoint finiteness when loading a sidecar. The good node in this example should retain a 1 ms duration.

**Callers, value, cost.** No valid timing input changes. Profilers keep independent usable measurements. S.

### UDF-13: The record cannot identify its own replay dataset and seed

**Where.** `R/create-computation-record.ts:58–59,100–120,147–148`; `R/computation-record-types.ts:55`; `docs/api/computation-record.md:119–129`.

**EXECUTED.** Creating a valid record from the repository fixture with `datasetVersion:'table-v4'` and `seed:42` produced `run-f6bf1d91`. Both fields on the result were `undefined`. Passing the record’s own identity fields to `getGPUComputationRunId` produced `run-5072d00e`.

**Now.** The guide asks authors to supply both fields, but a reader cannot tell which table to reopen or which seed to use. A comparison hash cannot recover those values.

**Wanted.** Persist optional `datasetVersion` and `seed` verbatim. Add a round-trip identity test for generated run IDs. Follow the guide’s recording snippet with the application-level replay sequence: resolve the saved dataset and implementation revision, rebuild with recorded parameters and seed, submit, then attach the new observations with replay provenance. This makes the provenance proposal in the direction report concrete without adding a generic execution engine.

**Callers, value, cost.** None. Legacy records can omit the fields. Reproducibility improves without additional GPU work. S.

### UDF-14: Rename the left-aligned key lookup before publication

**Where.** `D/key-join/gpu-key-join.ts:70,81,130–140,601`; KI dataframe guide `:893–922`.

**EXECUTED.** Left keys `[2,1,2]`, right keys `[2,2,1]`, right values `[10,20,30]`, `kind:'inner'`: the repository oracle returned `innerRows=[0,1,2]`, `rightRows=[0,2,0]`, `matchCounts=[2,1,2]`, gathered values `[10,30,10]`. The stable smallest-right-row rule works. NumPy’s equality pairs were `[[0,0],[0,1],[1,2],[2,0],[2,1]]`. A SQL or pandas inner join has five rows here.

**Now.** `new GPUKeyJoin({kind:'inner', ...})` sounds like a request for the five pairs. The guide eventually explains that compaction still contains each matching left row once.

**Wanted.** Rename the producer **`GPUKeyLookup`** and replace `kind` with **`compactMatchedLeftRows?: boolean`**, or infer compaction from a supplied compact destination. Lead the example with its duplicate-key output. Reserve pair-join terminology for an operation that can publish each `(leftRow,rightRow)` pair. No pair-expanding implementation is needed for this rename.

**In-repository callers that change.** `rg 'new GPUKeyJoin\('` found 55 lines in 22 files across source, tests, examples, website and docs. Parsing the TypeScript matches as `NewExpression` nodes gave **51 actual constructors in 19 files**: 44 tests, two analysis production calls, one lens, three examples, and one website call. Four further textual examples need editing. Update exported type names, imports and catalog display names too.

**Value and cost.** SQL and GIS users can predict cardinality from the call. BREAKING, M. I recommend doing it before alpha.

### UDF-15: The documented occupied-slot decoder gives a fractional cell index

**Where.** `docs/api/gpu-dataframe-analysis.md:1522` in this checkout and `:1535` in KI. The producer’s slot definition is at `D/temporal-reduction/gpu-temporal-reduction.ts:139`. The same integer decoding rule applies to cell-major `GPUValueCube` readback.

**EXECUTED.** Using the page’s `cell = id / bucketCount` with slot `5` and three buckets produced `1.6666666666666667`. `['north','south'][cell]` returned `undefined`. The intended cell is `1`, bucket `2`, label `'south'`.

**Wanted.** Print executable TypeScript: `const cell = Math.floor(slot / bucketCount); const bucket = slot % bucketCount;`. Put slot `5` in the example. Use a nonzero remainder in the documentation test so ordinary division cannot pass by accident.

**Callers, value, cost.** Documentation only. The first readback maps back to the correct GIS entity. S.

### UDF-16: The ordered-dictionary preparer rejects its own output on reuse

**Where.** `A/arrow-analytics-normalization.ts:264–286,301–308`.

**EXECUTED.** I constructed an Arrow ordered UTF-8 dictionary with labels `['x','x']` and row codes `[0,1]`. The first preparation returned dictionary `['x','x']`, recoded both rows to `[1,1]`, and reported success. Preparing the same table with `{dictionaries: first.dictionaries}` then threw `Category dictionary contains duplicate labels`.

**Now.** The first-batch path checks uniqueness before importing the ordered source labels. The reuse path checks after those labels are present. A preparation result offered for reuse can therefore fail on the identical next input.

**Wanted.** Apply the same duplicate-label policy while installing the first ordered dictionary and when reusing it. Reject ambiguous ordered labels with the field name and duplicate positions before returning a reusable dictionary. Test one preparation followed by reuse of its returned metadata.

**Callers, value, cost.** No change for unique dictionaries. Batch-by-batch callers discover ambiguous encodings at the same boundary every time. S.

## Departures by name

| Boundary | Departures in the audited workflows |
|---|---|
| Arrow key identity and per-column validity | `prepareArrowTableForGPUAnalytics`, `GPUKeyJoin`, `GPUGroupStatistics` |
| Query rows versus original point rows | `GPUPointCloudNeighborhoods`, `GPUPointCloudStatisticalFilter`, `GPUPointCloudSMRF` |
| Selection, time validity and decoded slot identity | `GPUTimeWindowFilter`, `GPUPeriodIndex`, `getGPUPeriodIndexCalendarPeriod`, `GPUTemporalReduction`; the dense decoder also applies to `GPUValueCube` |
| Numerics that change meaning across a boundary | `GPUGroupStatistics`, `GPUKeyJoin`, `GPUValueCube`, `GPUPointCloudStatisticalFilter`, `getGPUColumnQuantilesParameterValues`, `getGPUClassBreaksParameterValues` |
| Execution facts lost, shared, or accepted inconsistently | `createGPUComputationRecord`, `getGPUComputationRunId`, `createGPUComputationCaptures`, `validateGPUComputationSidecars`, `createGPUComputationMetrics`, `toChromeTraceEvents` |

## Quick wins

The ten changes I would take first, in implementation order:

- [ ] Fix the TypeScript slot decoder and give it a non-divisible example.
- [ ] Add the reversed-dictionary join fixture and shared-codebook example.
- [ ] Add a two-measure group example with independent nulls, showing `COUNT(*)` and finite counts.
- [ ] Show an Arrow nullable-time filter with validity wired in and the sorted shortcut disabled.
- [ ] Label the outlier outputs as query-aligned and show the `queryRows` scatter into source order.
- [ ] Validate monotonic custom edges and out-of-domain probabilities before packing.
- [ ] Validate actual Gregorian origin dates and ordered-dictionary uniqueness on first use.
- [ ] Persist dataset version and seed, and snapshot nested record metadata at construction.
- [ ] Validate capture IDs, jointly resolve graph-local references, and reject lossy JSON `null` lanes.
- [ ] Isolate bad timing endpoints and add the unmatched-right-key precision counterexample to the aggregate guides.

## Larger proposals

1. **Build the Arrow key and measure preparation path.** Reuse published Arrow validity vectors and existing word conversion. Make shared dictionary identity, numeric key width and per-measure validity explicit. This addresses the two ordinary wrong answers with the largest reach. M.
2. **Promote the capture codec into analysis.** Lenses already implements binary encoding and restoration. Moving the shared representation down the package dependency chain gives standalone analysis consumers a lossless archive format. M.
3. **Rename `GPUKeyJoin` to `GPUKeyLookup` before alpha.** A deprecated alias would ease migration but leave new users inferring SQL cardinality. With 51 actual constructors and no published consumers, I recommend the direct rename. M.
4. **Qualify the two numerical changes independently.** Use the two-right-key singleton probe before designing per-key scales. Use the five-equal-distance probe before changing the statistical filter’s reduction. Neither needs a large benchmark to establish correctness. M each.

## Not read

I read the Arrow normalization and timestamp-upload implementation, the direct CPU quantile and class-break paths relevant to these fixtures, the statistical-filter stages, the calendar-step arithmetic, and the record construction, metrics and sidecar-validation paths. I sampled the group and join sort/aggregation stages, both point-cloud ground routes, the ICP frame conversion, the temporal range searches, and the cube scale and accumulation stages. I chose those paths because they change row identity, missingness, ordering, numerical units or persistence.

I checked staged copies for the family guides, CPU quantiles, period indexing, value cube and neighborhoods. I checked candidates against the synthesis, the earlier dataframe/point-cloud/record report, API/docs and direction reports, class-name matches in the other area reports, and `NEEDS-DECISION.md`. I did not audit the full pair-statistics family, every classification method, ICP’s eigensolver, all point-cloud shader bodies, or every lens and Kepler adapter. All executed results above are host results, null-device construction, test-oracle results, or NumPy controls. Shader predictions are marked `SOURCE` and have a proposed device probe. No files were written.