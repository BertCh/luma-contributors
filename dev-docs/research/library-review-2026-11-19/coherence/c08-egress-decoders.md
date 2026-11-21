# Getting answers out of GPU analysis

## Summary

Ten of the twenty producers traced have a public decoder that accepts the `Uint8Array` returned by `Buffer.readAsync()` and can silently produce a wrong answer. A real host-buffer read decoded **300 nodes as 44**. A contiguity profile made the same mistake and still returned `exact: true`.

Two other failures affect correctly typed inputs. The network statistics decoder can read one degree histogram as another. The cell-period codec rejects a valid level-30 S2 cell when there is only one period to encode.

The strongest existing design is `GPUKShortestPaths`. Its allocation layout and decoder share dimensions, its readback mirrors the GPU ports, and its result retains interruption flags while removing padding. Arrow egress also already exists in the pinned upstream package. I round-tripped `[5, null, 7]` without a GPU.

**The highest-value change is to make the byte-to-value boundary explicit.** Give each packed output a public decoder with an unambiguous input type, logical dimensions, and a fixture that starts with the bytes a caller actually receives.

## Census

`S/` means `modules/analysis/src/`, `T/` means `modules/analysis/test/`, and `D/` means `docs/api/`. Paths and line numbers are for this checkout unless marked **KI** for the newer `luma-contributors-ki` copy.

**COUNT.** I bundled the nine public family barrels from source over stdin and counted exported values with `typeof value === 'function' && value.prototype?.getCommandNodes`. The result was spatial **272**, raster **86**, terrain **46**, network **41**, dataframe **34**, point cloud **11**, core **4**, geospatial **2**, and computation record **2**. These sum to **498**. I searched the barrels for `decode`, `read`, `unpack`, `Readback`, `ModelResult`, and result-specific `get` functions. A prefix search alone misses the raster fixed-point decoder, both Vecchia readers, and the signed-word helper. It also counts input parsers such as `readXGBoostForestModel`.

I selected twenty producers across eight families to cover mixed scalar formats, flags, signed and unsigned words, split floats, compact prefixes, offsets, and a capture sidecar. The following counts describe the **selected output** of each producer, not an estimate of coverage across all 498.

| Route to a printable value | Classes | Producers and selected output |
|---|---:|---|
| Public `decode…` function | 9 | `GPUNetworkStatistics` summary → counts and histograms; `GPUNetworkCoarsening` summary → eight counts; `GPUNetworkSubgraphFilter.counts` → four counts; `GPUKShortestPaths` → retained routes and batch status; `GPURegionStatistics.summary` → statistics, histogram and truncation flags; `GPUAnalysisSpacing.summary` → spacing profile; `GPUAnalysisContiguityStatistics` → exact or incomplete profile; `GPUSphericalConvexIntersection.counts/status` and `GPUSphericalArcOverlay.counts/status` → named fields |
| Public `get…` function interprets readback | 3 | `GPUVecchiaLogLikelihood.logLikelihood/status` → binary64 total and site counts; `GPUVecchiaFisherScoring.totals/status` → total, score and information; `GPURasterStreamingAccumulator.state.sums/moments` → binary64 values using different exponents |
| Public `unpack…` function | 2 | `GPUTerrainHorizon.output.horizon` → degrees; `GPUCellPeriodKeys.output.keys` → cell and period, after the caller joins the two words |
| Domain exporter or reader object | 2 | `GPUHealpixCellAggregation` → `createHealpixDGGSJSON` after prefix selection and key conversion; `GPUSnapshotTap` → `arena.read()` → named typed arrays |
| Caller assembles the domain answer | 4 | `GPUGroupStatistics` signed scaled sum; `GPUIsolines` stitched level lines; `GPUDatumGridTransform` split positions; `GPUPointCloudChange` displacement, interval, support and qualified status |
| **Total** | **20** | |

**Boilerplate.** I counted physical lines excluding blanks and `//` comments. Decoder blocks include their braces and returned object. These are measured examples, not theoretical minimums.

| Example | Lines between read request and usable result |
|---|---:|
| `D/README.md:58,69`, shared float and unsigned readers | 5 each, 10 shared |
| `D/README.md:133,169,208,249`, result callbacks for terrain derivatives, distance field, time-window filter and point-cloud filter | 1, 4, 7, 1 respectively, after the shared readers |
| `D/operation-forms-and-cpu-twins.md:316`, density and overflow | 6 |
| `D/gpu-network/operators.md:407`, statistics buffer → statistics object | 4 |
| `examples/deck/spatial-analysis-showcase/src/scenes/interpolation/dot-density.compute.ts:407`, three buffers → typed columns | 10 |
| `examples/deck/spatial-analysis-showcase/src/scenes/cells/cell-neighborhoods.compute.ts:96`, one staged payload → named fields | 41 |
| `examples/deck/spatial-analysis-showcase/src/scenes/geometry/route-windowing.compute.ts:98`, two scalar readers and one clip decoder | 24, before assembling the full window result |

For a wider check, I scanned the **116** API Markdown pages for fenced `ts`, `typescript`, `tsx`, `js` and `javascript` examples. **7 of 213** fences contained a literal `.readAsync(`, `.read(`, or `await readResult(`. Under `examples/`, excluding `dist` and `node_modules`, **58 of 1,960** JS/TS source files contained a direct `.readAsync(` call, on **149** non-comment lines. The distribution was 24 files with one such line, 14 with two, 12 with three, 3 with four, 2 with five, and one each with 10, 11 and 18. This measure intentionally excludes ring-ticket wrappers and CPU-output operation graphs. The first-run callbacks above are present in this checkout, an update to `KNOWN(ERG-06)` and `KNOWN(DOC-111)`.

## Proposed convention

For every public output whose interpretation depends on packing, counts, offsets, scale, residuals or status, export `decode<Producer><Output>(readback, layout?)` beside the producer. Its inputs should follow the producer’s port names and nesting, distinguish raw bytes from scalar elements, and return the evidence needed to interpret the result.

Check each decoder through the public barrel with an exact logical view, a nonzero byte offset, dirty spare capacity, a short view, and an incomplete or missing result where applicable. Use the same layout for allocation and decoding. Leave submission and readback timing with the caller.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| EGR-01 | W | ORDINARY | ADDITIVE | M | Ten sampled producers | A valid byte readback satisfies the decoder signature and changes the answer. |
| EGR-02 | W | ORDINARY | ADDITIVE | S | `GPUNetworkStatistics` | A 64-bin summary decoded with the default mixes three histograms. |
| EGR-03 | W | ORDINARY | ADDITIVE | S | `GPUHealpixCellAggregation` → DGGS-JSON | A dropped cell is serialized exactly like an absent cell. |
| EGR-04 | W | UNUSUAL | BREAKING | S | `GPUCellPeriodKeys` | A one-period cube consumes a cell bit and can lose valid cells. |
| EGR-05 | W | UNUSUAL | BREAKING | S | `unpackGPUCellPeriodKey` | A missing key decodes to a cell and an impossible period. |
| EGR-06 | W | UNUSUAL | ADDITIVE | S | Spherical, network, Vecchia, raster and terrain readers | Truncated metadata becomes `true`, a finite likelihood, or scientific no-data. |
| EGR-07 | W | UNUSUAL | ADDITIVE | S | `GPUSnapshotArena.read` | An empty metadata map can attach functions to a capture. |
| EGR-08 | S | UNUSUAL | ADDITIVE | S | Arrow table egress | A valid column called `constructor` breaks the natural upload/readback round trip. |
| EGR-09 | S | UNUSUAL | ADDITIVE | S | `GPUSnapshotArena.read` | One rejected mapping permanently prevents retrying the captured frames. |
| EGR-10 | E | ORDINARY | ADDITIVE | S | Core, testing, snapshot arena | Prefix readers transfer whole allocations; the test reader can expose a neighboring word. |
| EGR-11 | C | ORDINARY | ADDITIVE | S | `GPUAnalysisContiguityStatistics` | The caller must move certificate fields before invoking the decoder. |
| EGR-12 | E | ORDINARY | ADDITIVE | S | `GPUGroupStatistics` | The signed sum primitive exists, but its signature and output omit the group layout and scale. |
| EGR-13 | E | ORDINARY | ADDITIVE | M | `GPUIsolines` | The caller must reconstruct line geometry from six readback fields and two overflow flags. |
| EGR-14 | E | ORDINARY | ADDITIVE | S | `GPUDatumGridTransform` | A natural typed-array expression discards the precision the split output preserved. |
| EGR-15 | E | ORDINARY | ADDITIVE | S | `GPUPointCloudChange` | A single status word combines an outcome code and an independent completeness bit. |

### EGR-01: Raw bytes pass as words, floats and counts

**Where.** The ten sampled producers are covered by the following decoder implementations.

| Implementations under `S/` | Observed error |
|---|---|
| `gpu-network/network-statistics/gpu-network-statistics.ts:148`, `network-coarsening/gpu-network-coarsening.ts:112`, `network-subgraph-filter/network-subgraph-filter-parameters.ts:107` under the same family | First count `300` becomes `44`. Filter counts `[300,1,2,3]` become `[44,1,0,0]`. |
| `gpu-spatial-analysis/global-coordinates/spherical-status.ts:51,56` | Convex-intersection and arc-overlay counts read individual bytes as separate counts. |
| `gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.ts:96` | A 300-slot profile with median 1 becomes 44 slots with median 0 and **`exact:true`**. |
| `gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood.ts:228` and `gpu-vecchia-fisher-scoring.ts:372` | Float words `[-4,0.25]` and status `[0,0,1]` yield likelihood `0` from bytes, expected `-3.75`. The likelihood reader also reports zero used sites. |
| `gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts:152` | Packed `0xffff0001` decodes to `[-90,NaN]` from bytes, expected `[-90,90]` degrees. |
| `raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:98` | One 96-bit sum equal to 1 produces four values `[1,0,0,0]` from its 12 bytes. |

**Evidence — EXECUTED.** This is the code a caller can write from “decode the summary read back from the view.” `Uint8Array` satisfies `ArrayLike<number>`. I ran the actual `Buffer.readAsync` call on a null-device buffer containing one prefix word followed by eight summary words.

```ts
const bytes = await buffer.readAsync(4, 32);
decodeGPUNetworkCoarseningSummary(bytes).liveNodeCount; // 44
decodeGPUNetworkCoarseningSummary(
  new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
).liveNodeCount; // 300
```

An additional downstream example is `D/gpu-raster/raster-eof-analysis.md:23`. It starts the CPU step with `words`. Substituting the return from `readAsync()` into `getGPURasterSnapshotCovarianceMatrix` changed the valid two-slice Gram matrix `[1,-1,-1,1]` to `[1,0,0,0]`. This occurs before the eigensolver and is independent of `KNOWN(RT-120)`.

The reverse surprise occurs at `D/gpu-spatial-analysis/analysis-experience.md:37` and `S/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.ts:83,229`. The guide says to read 2,176 bytes and call the decoder. The signature requires `Uint32Array`. With version 1 and zero points, 544 words returned `undefined` as specified. Passing their bytes threw `summary.length must be an integer in [544, 544]`. By contrast, `decodeGPURegionStatistics` correctly decoded a byte view at the byte boundary. Its **KI** implementation at `region-statistics-readback.ts:21` retains that behavior.

**Wanted.** Add explicitly typed byte entry points for single packed buffers and typed port-object decoders for multiple buffers. Show the actual conversion at the call site wherever a low-level word reader remains documented. A byte entry point must respect both `byteOffset` and `byteLength`. Keep existing numeric-array overloads available to callers who intentionally supply decoded elements.

**In-repository callers that change:** None required for the additive entry points. **Value and cost:** A caller can use the result of `readAsync()` without publishing a plausible wrong number. M.

### EGR-02: Histogram width must travel with the network summary

**Where.** `S/gpu-network/network-statistics/gpu-network-statistics.ts:148,311,345`; `D/gpu-network/operators.md:407`, **KI** page at `:421`.

**Evidence — EXECUTED.** I packed 64 linear bins for a directed star with 33 leaves. There are 34 nodes, one node with in-degree zero, and 33 with in-degree one. `decodeGPUNetworkStatistics(words)` returned **32 bins, `[0,1]` for the first two in-degree bins, and only 33 nodes summed over out-degree**. Passing `{degreeBinCount:64}` returned **64 bins, `[1,33]`, and 34 nodes**. The decoder accepts any length at least as large as its default. It starts the in-degree histogram halfway through the real out-degree histogram.

**Now.** `decodeGPUNetworkStatistics(words)` is the documented example. Changing the contributor’s `degreeBinCount` is enough to make this example wrong.

**Wanted.** Bind decoding to the contributor’s `degreeBinCount`, or add a checked decoder that takes the exact logical summary and derives `(wordCount - 16) / 3`, requiring an integer in `[1,4096]`. For a capacity-sized backing buffer, take the logical view first. Update the example to use a nondefault width.

**In-repository callers that change:** None required if the checked form is added. **Value and cost:** Network analysts can change histogram resolution without changing what “in-degree” means. S.

### EGR-03: DGGS-JSON loses the distinction between absent and dropped cells

**Where.** `D/gpu-spatial-analysis/healpix-cells-dggs-json.md:62–68`; `S/gpu-spatial-analysis/healpix-cells/gpu-healpix-cell-aggregation.ts:64` and `healpix-dggs-json.ts:42,85,127`. Kepler already checks overflow on this route in `modules/kepler/src/analysis/catalog-network-categorical-cells.ts:157`.

**Evidence — EXECUTED.** I followed the page’s instruction to keep the first `status.count` keys. For order 1, ancestor 0, one retained cell `0n`, value 4, both `{count:1,requiredCount:1,overflow:0}` and `{count:1,requiredCount:2,overflow:1}` serialized to **`[4,null,null,null]`**. A downstream reader cannot know that one null represents a dropped observation. `KNOWN(DGGS-108)` concerned an invalid UNIQ input key. This loss occurs after a legitimate aggregation.

The output-to-exporter handoff is also incomplete. The CPU point-to-cell producer returned `Uint32Array [4,0]` for `[0°,0°]` at order zero. `cells: out.cells` raised `Cannot mix BigInt and other types`. A `BigUint64Array` also fails because the builder uses typed-array `.map()` with a numeric return value. Joining the words, passing `[4n]` and setting `ancestor:4n` produced the expected count data `[1]`. The declared `readonly bigint[]` explains both direct-call failures.

**Wanted.** Add `createHealpixDGGSJSONFromReadback({output, order, scheme, ancestor, ancestorOrder, …})`. Accept `output.cells` as packed `Uint32Array`, inspect `output.status.overflow`, slice through `count`, convert keys losslessly, and name `requiredCount` in an incomplete-export error. Keep the plain sparse-data builder for callers who explicitly supply sparse data.

**In-repository callers that change:** None required. **Value and cost:** A GIS export preserves the difference between missing data and insufficient output storage. S.

### EGR-04: One period should occupy zero bits

**Where.** `S/dataframe-analysis/value-cube/cell-period-key-packing.ts:23,60,82`; `gpu-cell-period-keys.ts:167,248` in this checkout and **KI** `:167,249,300`; `D/gpu-dataframe-analysis.md:1851`, **KI** `:1866`.

**Evidence — EXECUTED.** The documented formula gives `ceil(log2(1)) = 0`. The public helper returned `getGPUCellPeriodKeyPeriodBits(1) === 1`. `getGPUCellKeyFreeBits('s2',30)` returned 0. For the level-30 S2 leaf `1n`, `packGPUCellPeriodKey(1n,0,1,'zeros')` returned `undefined`. Decoding the documented identity representation, `unpackGPUCellPeriodKey(1n,1,'zeros')`, returned `{cellKey:0n,period:1}`. Period 1 is outside this cube.

**SOURCE GPU probe.** Both producer copies use the host bit count to form `periodMask`. Submit one row with cell words `[1,0]`, period `[0]`, `periodCount:1`, `fill:'zeros'`, and an `invalidCount` output. The documented result is key `[1,0]`, invalid count 0.

**Wanted.** Special-case `periodCount === 1` to zero bits in the common layout function. Pack and unpack period zero as the identity. This is **breaking** for previously stored one-period keys with `'ones'` fill: for example the old packer encoded `3n` as `2n`. Mark that distinction in the migration note.

**In-repository callers that change:** `rg 'getGPUCellPeriodKeyPeriodBits\('` found **11 non-declaration TS/JS call sites**. One assertion explicitly requires the old one-bit result, `T/gpu-dataframe/value-cube/gpu-cell-period-keys.spec.ts:71`. The producer, packer, unpacker and two website uses derive their masks from the helper. **Value and cost:** A single-period export retains the full resolution of its source cell identifiers. S.

### EGR-05: Unpacking a missing key invents a group

**Where.** `S/dataframe-analysis/value-cube/cell-period-key-packing.ts:47,77`; producer output contract in `gpu-cell-period-keys.ts:64`, also checked in **KI**. This extends `KNOWN(DPC-101)`, which fixed the forward CPU packer.

**Evidence — EXECUTED.** `unpackGPUCellPeriodKey(0xffffffffffffffffn,3,'zeros')` returned `{cellKey:18446744073709551612n,period:3}`. The producer reserves that input for “not packed.” Even `unpackGPUCellPeriodKey(3n,3,'zeros')` returns period 3 with a declared range of 0–2.

**Wanted.** Return `undefined` for the documented sentinel, matching the forward helper’s missing result. Reject a non-sentinel payload whose low bits encode a period outside `periodCount`. Document the distinction between an excluded row and malformed packed data.

**In-repository callers that change:** Four executable calls found by `rg 'unpackGPUCellPeriodKey\('`. Widening the return type requires guards at the two destructuring sites, `website/src/gpu/runs/entries/flows-cubes/cell-period-keys.ts:87` and `T/gpu-dataframe/value-cube/gpu-cell-period-keys.spec.ts:194`. Two equality assertions already pass valid keys. **Value and cost:** Row-by-row decoding cannot insert a phantom cell into a group table. S.

### EGR-06: Short buffers turn into scientific results

**Where.** `S/gpu-spatial-analysis/global-coordinates/spherical-status.ts:51,56`; `S/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters.ts:107`; the two Vecchia readers, terrain unpacker and raster fixed-point reader in EGR-01; `S/dataframe-analysis/column-profile/wide-numeric-gpu-reference.ts:33`.

**Evidence — EXECUTED.** These probes supplied numeric elements, so this is independent of the byte/word problem.

| Input | Returned | Expected |
|---|---|---|
| `decodeGPUSphericalArcOverlayCounts([3,1,4])` | `referenceInside:true` | Missing fourth word |
| `decodeGPUSphericalConvexIntersectionCounts([3])` | `requiredVertexCount:undefined` | Missing second word |
| `decodeGPUNetworkSubgraphFilterCounts([3])` | Three `undefined` fields declared as numbers | Four required count words |
| `getVecchiaLogLikelihoodResult([10,.5],[])` | `logLikelihood:10.5` | Missing failure certificate |
| `getVecchiaFisherEvaluation([10,.5],[])` | Likelihood 10.5, three NaN scores, six NaN information entries | Missing totals and certificate |
| `unpackGPUTerrainHorizonUnorm16([0xffff0001],3)` | `[-90,90,NaN]` | Missing second packed word |
| `getGPURasterFixedPointValues([1,0,0,2],0)` | `[1]` | Incomplete second triple |
| `joinWordsToBigInt([],true)` | `0n` | Missing pair |

For overlay, `undefined !== 0` answers a topological question in the affirmative. For terrain, a missing word becomes the genuine no-data code. Both are indistinguishable from successful decoding.

**Wanted.** Validate required component counts before interpreting any field. Check divisible-by-three for an entire 96-bit column. Use minimum lengths for explicitly bounded prefixes. Include the port, expected length and received length in the error.

**In-repository callers that change:** None supplying complete layouts. **Value and cost:** A bad read range fails where it was introduced, before it becomes a likelihood, cell zero, or no-data measurement. S.

### EGR-07: Snapshot metadata can contain inherited functions

**Where.** `S/computation-record/gpu-snapshot-arena.ts:119,168`; compare the tap’s own-key check in `gpu-snapshot-tap.ts:71`. Related input-side precedent is `KNOWN(DPC-104)`.

**Evidence — EXECUTED.** I recorded one four-byte slot named `toString` using a host buffer double, then called `arena.read({resources:{},keySpaces:{}})`. The capture had an own `resource` property. Both `typeof capture.resource` and `typeof capture.keySpace` were **`"function"`**. Both fields should be absent.

**Wanted.** Use `Object.hasOwn` for optional slot metadata and for direct arena source lookup. Retain arbitrary user slot names. Test `toString`, `constructor` and an ordinary missing name.

**In-repository callers that change:** None. **Value and cost:** Capture provenance retains its declared string type for user-named outputs. S.

### EGR-08: Arrow egress exists, with a reproducible field-name failure

**Where.** `node_modules/@luma.gl/graph/src/arrow/index.ts:27,196`; `arrow/gpu/arrow-gpu-analytics-readback.ts:60,111,204,223` in that package; `examples/deck/kepler-compute/src/resident-sql-gpu-runtime.ts:975`. This adds an executable route to `KNOWN(DIR-102)` and identifies a different boundary from `KNOWN(DPC-104)` and `KNOWN(DPC-115)`.

**Evidence — EXECUTED.** On a null device, an `Int32` Arrow column `[5,null,7]` survived this full round trip:

```ts
const resident = makeGPUAnalyticsTableFromArrowTable(device, source);
const result = await makeArrowTableFromGPUAnalyticsTable(resident);
Array.from(result.getChild('score')!); // [5, null, 7]
```

With one non-null `Int32` field named `constructor` holding `42`, upload succeeded. The same readback threw **`Cannot read properties of undefined (reading 'length')`**. Reading `{table:resident.table}` returned `42`. Passing the actual metadata entries in null-prototype maps also returned `42`. The ordinary empty maps expose inherited values to upstream’s dictionary and validity lookups.

**Wanted.** Add the working route to `D/README.md`, including how validity and dictionaries travel with a result table. Preserve supplied metadata in own-key dictionaries at the application boundary. Fix the four lookups upstream. For individual numeric vectors, the public routes also include `makeCPUVectorFromGPUVector` and `makeArrowVectorFromCPUVector`. A graph buffer handle alone does not carry the table schema.

**In-repository callers that change:** One direct application materializer found, the Kepler example above. The upstream fix requires no call-site change. **Value and cost:** Analysts can retrieve a nullable Arrow table using the installed API, including datasets with legal user-defined field names. S for the guide, fixture and call-site workaround.

### EGR-09: A rejected snapshot read cannot be retried

**Where.** `S/computation-record/gpu-snapshot-arena.ts:111,154`.

**Evidence — EXECUTED.** I supplied a buffer whose first `readAsync` rejected with `temporary mapping failure` and whose next call would succeed. Two `await arena.read()` calls both rejected. The underlying buffer was called **once**. The arena caches the rejected promise and has already closed the frame set.

**Wanted.** Keep the frame set sealed after reading starts. Share an in-flight read, cache successful bytes, and clear only the failed promise so retained frames can be read again. Test two simultaneous callers and a later retry.

**In-repository callers that change:** None. **Value and cost:** A recoverable host read failure no longer requires replaying the observed computation. S.

### EGR-10: Read the logical view and the live prefix

**Where.** `S/testing/gpu-contributor-test-utils.ts:294,302,310`; `S/computation-record/gpu-snapshot-arena.ts:154`; `modules/analysis/package.json:78`; `D/README.md:58,69`.

**Evidence — EXECUTED.** Given a returned byte view covering only the middle word of `[999,7,888]`, `readUint32(buffer,2)` returned **`[7,888]`**. It honors the byte offset but lets the requested length extend past the returned view into neighboring backing storage. `readCompactIds(ids,count)` with count zero still called `ids.readAsync()` without a range. An arena with capacity 1,024 and one recorded four-byte slot requested **4,096 bytes** to return four. The current first-run documentation already demonstrates the better bounded request and zero-length fast path.

**Wanted.** Fix the test readers first. For raw packed application outputs, provide a small public reader taking the caller’s buffer, logical byte offset, scalar format and element count. Check the logical range, call `readAsync(offset,count * elementSize)`, skip zero-length reads, and return an owned typed array. Use existing upstream vector transfer for strided or chunked vectors. Limit arena mapping to `count * frameByteLength`. Publishing the existing testing module would also expose test-device machinery and retain the bugs above.

**In-repository callers that change:** None required. **Value and cost:** A compact result can be read in proportion to its live size, without reading a neighboring allocation. S for the bounded readers and arena change.

### EGR-11: Contiguity decoding makes the caller move the certificate

**Where.** `S/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.ts:51,96`; `modules/kepler/src/profile/dataset-profile-gpu.ts:377`; `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/analysis-profile-run.ts:159`.

**Evidence — EXECUTED.** After replacing views with read-back arrays, `decodeGPUAnalysisContiguityStatistics(output)` threw `Cannot read properties of undefined (reading 'length')`. The arrays are at `output.weights.status.overflow` and `.candidateOverflow`. The decoder requests both at the top level. `decodeGPUAnalysisContiguityStatistics({...output,...output.weights.status})` returned the expected `{exact:true,slots:2,islands:1,minimum:0,median:1,maximum:1,…}`.

**Wanted.** Accept the producer-shaped object with `counts`, `median`, `validity` and `weights.status`. Keep the flat overload. Preserve the existing refusal to report exact statistics after overflow.

**In-repository callers that change:** None required. Kepler and lenses can remove their field rearrangement. **Value and cost:** A generic port reader can feed the decoder directly. S.

### EGR-12: Fixed-point group sums require an undocumented cross-family recipe

**Where.** `S/dataframe-analysis/group-statistics/gpu-group-statistics.ts:133`; `key-join/gpu-key-join.ts:114` in the same family; `S/gpu-spatial-analysis/cell-aggregation/cell-table.ts:53`; `T/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts:502`; `D/gpu-dataframe-analysis.md:843`, **KI** `:859`.

**Evidence — EXECUTED.** The public dataframe helper `joinWordsToBigInt(pair,signed)` exists. For `[0xffff0000,0xffffffff]`, it returned **`-65536n`** with `true` and **`18446744073709486080n`** with `false`. At the default scale the answer is `-1`. Its declared input is `readonly number[]`, so the natural `Uint32Array.subarray()` does not match its parameter type. The group test reconstructs a bigint with indexing and a shift.

**Wanted.** Reuse the existing word join. Add `decodeGPUGroupSum(words: Uint32Array, row: number, sumScale: number)` returning `{scaledInteger:bigint,value:number}`. Check the pair boundary and retain the exact integer for sums beyond binary64 integer precision. Cross-link the same signed representation from key joins and cell tables. Keep `sumValues` documented as a distinct statistic.

**In-repository callers that change:** None required. **Value and cost:** A negative group total cannot accidentally become a large positive measurement through unsigned interpretation. S.

### EGR-13: Stitched isolines have no host geometry decoder

**Where.** `S/raster-analysis/isolines/gpu-isolines.ts:77,101`; `cpu-isolines-operation.ts:37,73`; `T/gpu-raster/isolines/gpu-isolines.spec.ts:188`; `D/gpu-raster/isolines-tiles.md:93`.

**Evidence — EXECUTED.** On a 3×3 raster with each row `[0,1,2]`, levels `[0.5,1.5]`, segment capacity 8 and vertex capacity 12, the CPU twin returned two lines, six live vertices, **24 coordinate scalars** and offsets **`[0,3,6,0,0,0,0,0,0]`**. The first line is `[[1,2.5],[1,1.5],[1,0.5]]`. To obtain it the caller needs `polylineCount`, the next offset, the factor of two for XY, and the level index. Before exporting a complete dataset the caller also needs both overflow flags. The docs explain resident overlay composition but give no host conversion.

**Wanted.** Add `decodeGPUIsolinesPolylines({polylines,status}, {levels,localOrigin})`. Return retained lines with level index, level value, closure and coordinates, plus segment and polyline completeness. Support the CPU twin through a documented adapter. Show GeoJSON `LineString` export with an explicit coordinate transform when output is projected. The decoder should not label local projected XY as longitude/latitude.

**In-repository callers that change:** None required. **Value and cost:** A GIS developer gets two correctly located features instead of interpreting spare vertices as observations. M.

### EGR-14: The obvious split-coordinate expression rounds away the correction

**Where.** `S/geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts:60,132`; `D/geospatial.md:34`; `D/gpu-spatial-analysis/global-coordinates.md:42`, also checked in **KI**; `T/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts:697`.

**Evidence — EXECUTED.** With float32 high longitude `120` and float32 residual `0.000001`, `high[0] + low[0]` printed **`120.000001`**. The natural column expression `high.map((value,i) => value + low[i])[0]` printed **`120`**, because the result of `Float32Array.map` is another `Float32Array`. The test adds individual components in JavaScript numbers.

**Wanted.** Add `decodeGPUDatumGridTransformPositions({positions,positionsLow,status})` returning `Float64Array` coordinates and preserving row statuses. Accept the documented aligned chunks or decode one identified chunk at a time. Publish the one-line equivalent `Float64Array.from(high,(value,i) => value + low[i])`.

**In-repository callers that change:** None. **Value and cost:** Exported coordinates retain the displacement the user specifically requested split precision to preserve. S.

### EGR-15: Point-cloud change needs a decoder for its mixed status

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-change.ts:39,102`; `D/pointcloud-analysis.md:435`, **KI** `:441`; `T/gpu-pointcloud/gpu-pointcloud-change-grid.spec.ts:127,268`.

**Evidence — EXECUTED host interpretation.** Using the exported flag, words `[0,1,2,3,256,258,259]` decompose to base codes `[0,1,2,3,0,2,3]` and incomplete flags `[false,false,false,false,true,true,true]`. `word === VALID` loses the distinction between a published displacement with incomplete spatial support and an invalid core. The test implements the mask itself. The two support counts and the order of each three-float result are further facts the caller must reconstruct.

**Wanted.** Export `decodeGPUPointCloudChangeStatus(word)` returning `{code,gridIncomplete}` with named outcomes. Add `decodeGPUPointCloudChangeRow(output,row)` returning signed displacement, interval bounds, detection limit, support for each epoch and that status. Keep the original word available for logging.

**In-repository callers that change:** None. **Value and cost:** Survey analysts can decide whether incomplete support is acceptable without discarding a published change or treating it as complete. S.

## Departures by name

Within the twenty-producer trace, the packed outputs still requiring application-level assembly are:

| Departure | Outputs |
|---|---|
| No complete domain decoder for the selected result | `GPUGroupStatistics.output.columns[].sums` with its scale; `GPUIsolines.output.polylines` with level and completeness; `GPUDatumGridTransform.output.positions/positionsLow/status`; `GPUPointCloudChange.output.changes/roughness/counts/status` |
| A decoder exists for metadata, but geometry or topology assembly remains with the caller | `GPUSphericalArcOverlay.output.positions/ringOffsets/counts/status` for open unit-XYZ rings; `GPUSphericalConvexIntersection.output.positions/counts/status` for retained vertices; `GPUNetworkCoarsening.output.edges/edgeTargetGroups/edgeCounts/edgeWeights` for retained superedges; `GPUNetworkSubgraphFilter.output.liveNodes/liveEdgeSlots/inducedTopology` for compact IDs and induced CSR. Their summary/count decoders do not decode these companion outputs. |
| Exporter or unpacker requires a representation change first | `GPUHealpixCellAggregation.output.cells` and `GPUCellPeriodKeys.output.keys` are low/high word columns. Their domain consumers take JavaScript bigints. `joinWordsToBigInt` is available through dataframe analysis. |
| Decoder accepts the wrong native readback unit | `GPUNetworkStatistics`, `GPUNetworkCoarsening`, `GPUNetworkSubgraphFilter`, `GPUSphericalArcOverlay`, `GPUSphericalConvexIntersection`, `GPUAnalysisContiguityStatistics`, `GPUVecchiaLogLikelihood`, `GPUVecchiaFisherScoring`, `GPUTerrainHorizon`, `GPURasterStreamingAccumulator` |
| Decoder needs words where the first-use text says bytes | `GPUAnalysisSpacing` |
| Readback nesting differs from output nesting | `GPUAnalysisContiguityStatistics.weights.status`; the direct CPU twin of `GPUKShortestPaths` also flattens status, but its guide gives the precise adaptation at `D/gpu-network/k-shortest-paths.md:106`. |

Two examples are worth preserving. Given a padded route with vertices `[8,9,777]`, the public K-shortest decoder returned `{length:2,cost:3,vertices:[8,9],edges:[4]}`. `GPURegionStatistics` offers both a pure byte decoder and an optional readback ring. Its histogram length follows the **supplied byte view**: an exact view produced `[3]`, while the whole backing allocation produced `[3,999]`. This is why the common reader needs the logical port range even when the domain decoder itself is correct.

## Quick wins

- [ ] Make short count records, incomplete word pairs/triples and missing Vecchia status fail with the offending port name.
- [ ] Replace `words` placeholders in the spacing and raster EOF guides with actual `readAsync` conversions. Use a nonzero buffer offset in at least one example.
- [ ] Bind the degree histogram decoder to the producer’s bin count and demonstrate a 64-bin result.
- [ ] Add the producer-shaped contiguity overload, signed group-sum helper, float64 datum-position decoder and point-cloud change status decoder.
- [ ] Check HEALPix completeness before DGGS-JSON export. Show the exact conversion from retained GPU words to cell identifiers.
- [ ] Fix snapshot metadata lookup, retry after failed mapping and the captured-prefix byte range.
- [ ] Document the existing upstream Arrow materializer using a nullable column and a compact selection. Add the `constructor` regression at the integration boundary.

## Larger proposals

1. **Adopt output layouts and pure domain decoders incrementally.** Start with the ten byte-ambiguous producers and isolines. Use K-shortest’s dimension sharing, explicit status and copied live prefix as the worked example. Keep scalar, already typed outputs simple.
2. **Make the two cell-period changes before alpha.** Zero bits for one period changes persisted keys, and preserving the sentinel changes a return type. Both migrations are small today. Keeping the current representation would require documenting why an S2 leaf cannot be grouped into a single period and why decoding a missing row returns a cell.
3. **Keep materialization at the boundary.** Use upstream Arrow and vector readers when the caller has a `GPUTable` or `GPUVector`. A bounded raw-view reader plus family-specific geometry and statistics decoders covers the demonstrated gaps without introducing the generic execution runner discussed in `KNOWN(DIR-102)` and the direction report.

## Not read

I read the K-shortest output layout and decoder, spherical count/status helpers, region readback, cell-period codec, HEALPix JSON builder and snapshot arena in full. For the other producers I followed output declarations, the relevant packing or decoding sections, named tests, and selected Kepler, lens or example consumers. I checked the supplied synthesis, area reports, API/catalog and direction reports, decision list, and staging ownership. In particular, the newer KI cell-period producer still uses the one-bit mask, the KI spherical overlay still publishes four count words, and the KI region reader retains byte-view decoding.

The 498-producer count and documentation/example scans are broad censuses. The named missing-decoder inventory is bounded to the twenty traces. I did not inspect every shader, every GeoJSON consumer, every output port of the other 478 producers, or generated distribution files. All numerical results labeled executed came from host code, CPU twins, a null-device buffer, or an identified host double. No GPU, browser, network, build or test suite was used, and no files were written.