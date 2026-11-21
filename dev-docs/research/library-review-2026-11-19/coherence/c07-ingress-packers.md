# Getting data into analysis without changing its meaning

## Summary

The package has 498 public command-node producers and 214 parameter-value packers, but only seven public functions named `prepare*`. Most first-use work falls between those surfaces. The caller has to turn source rows into typed arrays, choose a graph format, and reconstruct the counts the contributor expects.

The worst ordinary defect is in Vecchia prediction. A documented query outside the sample extent returned the same site three times and omitted two nearer sites. A single masked NaN also erased the neighborhoods of otherwise valid sites.

The most misleading cross-family instruction is that network topology can be passed wherever spatial weights are read. The same two-edge input passed routing and failed the library’s own spatial-weights validator.

The largest opportunity is a small set of **domain input adapters that return the data, its graph format, its logical row counts and its provenance together**. Start with geometry. The installed public luma.gl Arrow entry already preserves GeoArrow batches. Analysis needs to translate those batches into its geometry ports and enforce their semantics.

All numerical results below are host executions. No GPU result is inferred from them.

## Census

I enumerated the 11 entries in `modules/analysis/package.json#exports`, bundled their **source** barrels, and deduplicated by public name. I also followed the barrels with the TypeScript syntax parser to identify definitions, parameter types and constructor props. The checkout was live during the review.

```sh
node -e '
const entries = Object.values(require("./modules/analysis/package.json").exports);
const names = entries.map((entry, i) => {
  const name = `entry${i}`;
  console.log(`import * as ${name} from "./modules/analysis/src/${entry.types.slice(7, -5)}.ts"`);
  return name;
});
console.log(`const exports = Object.entries(Object.assign({}, ${names.join(",")}))`);
console.log(`const functions = exports.filter(([,v]) => typeof v === "function")`);
console.log(`console.log(exports.length, functions.length,
  functions.filter(([,v]) => /^class\\b/.test(String(v))).length,
  functions.filter(([,v]) => v.prototype && "getCommandNodes" in v.prototype).length)`);
console.log(`console.log(functions.filter(([n]) => /^(pack|encode|prepare|create\\w*Input)/.test(n)).map(([n]) => n).sort())`);
' | node_modules/.bin/esbuild --bundle --platform=node --format=esm --loader=ts --log-level=warning \
  | node --input-type=module -
```

**COUNT:** 2,997 runtime export names, 1,694 function-valued exports, 761 classes, 933 other functions, 498 producers. Counting inherited methods matters. Four raster spectral indices inherit `getCommandNodes`.

| Source family | Public producers | Public functions ending in `ParameterValues` | `prepare*` | `pack*` | `encode*` |
|---|---:|---:|---:|---:|---:|
| Spatial analysis | 270 | 88 | 2 | 2 | 0 |
| Raster | 86 | 49 | 0 | 0 | 0 |
| Terrain | 46 | 27 | 0 | 0 | 1 |
| Network | 41 | 12 | 3 | 0 | 0 |
| Dataframe | 34 | 23 | 1 | 2 | 0 |
| Point cloud | 11 | 11 | 0 | 0 | 0 |
| Analysis core | 6 | 3 | 1 | 1 | 0 |
| Geospatial | 2 | 1 | 0 | 0 | 0 |
| Computation record | 2 | 0 | 0 | 0 | 0 |
| **Total** | **498** | **214** | **7** | **5** | **1** |

`create*Input*` matched **zero** public functions. `createInputBuffer` is a test helper. `packGPUSpatialParameterValues` is counted in both its applicable columns.

| Caller starts with | Public ingress and preparation found |
|---|---|
| Arrow table or scalar column | `prepareArrowTableForGPUAnalytics`, `convertWideNumericValue`, `convertWideNumericVector`, `makeGPUTimeVectorFromPreparedArrowColumn`, `getGPUTemporalContextFromPreparedArrowColumn`, `makeDecimal128WordVectors`. `aggregateDecimal128Column` also accepts a vector and owns its upload. `rg -l "from 'apache-arrow'" modules/analysis/src` found four source files. No public producer constructor directly takes an Apache Arrow `Table`. |
| XY points, lines, polygons and multipart features | `createGPULocalFrame().projectArray()` prepares XY coordinates. `preparePolygonContiguityTopology` prepares equality tokens for one algorithm. There is no analysis host packer from ordinary point/line/polygon columns to the five corresponding single- or multipart geometry port shapes. `createGPUMultipartRowOwnersNode` operates after offsets are resident. |
| XYZ points | `getPointCloudLocalOrigin`, `getPointCloudLocalOriginSummary`, `makePointCloudLocalPositions`. `flattenPointCloudAttribute` reinterprets an existing tightly packed graph view and preserves its byte offset. |
| Raster band, stack or embedding | `makeGPURasterMetadata` builds the grid. `makeGPURasterBand` borrows a grid-sized prefix of an **existing view**. Neither packs host band planes. Fourteen raster producers expose both `values` and `sliceCount` in the constructor-prop census. `GPUCubeLayoutTranspose` is an available resident scalar-cube transpose. |
| Network, spatial weights or neighborhoods | `prepareGPUNetworkFlowDemands`, `prepareGPUNetworkRestrictionStates`, `prepareGPUNetworkPointRoutingTurns`, `buildVecchiaConditioningSets`, `buildVecchiaPredictionNeighborhoods`, `getVecchiaMaxMinOrdering`, and the contiguity topology preparer. The two Vecchia builders return `{offsets, neighbors}`. `planGPUSpatialWeightsChunks` plans storage. `computeSpatialWeightsValidationOnCPU` checks an already packed CSR. Neither constructs general spatial weights from a caller’s edge table. |
| Time, trajectories or encoded numeric lanes | `prepareGPUTrajectoryObservations`, `getGPUSpaceTimeRelativeTimes`, `getGPUSpaceTimeGuardedRelativeTimes`, `splitTimestamps`, `splitTimeWords`, `getInt64TimeWords`, `getGPUTimeZoneTransitionTable`, `getGPURasterCentredFractionalYears`, `splitBigIntToWords`, `splitDecimal128ToWords`, `splitDoubleToWords`. These cover relative floats, high/low floats, signed integer words and calendar tables. Those encodings have different meanings. |
| Scalar and auxiliary controls | `prepareGPUScalarScaleControls`, `packGPUColor`, `packGPUCellPeriodKey`, `getGPUFlowPairKey`, `packGPUSpatialRelate`, `packGPUSpatialRelatePattern`, `encodeGPUTerrainHorizonUnorm16`, `getGPUColumnQuantilesRationalProbabilityValues`, `getGPUSolarIrradianceSunTable`, and the 214 `*ParameterValues` functions. Generated weights and sample lattices also appear under `createGPUPointDensityGaussianKernel`, `createGPUPointDensityGaussianKernel1D`, `getGPUTextureShadingKernel`, `getGPUPointHorizonDistanceLattice` and `getGPUTerrainHorizonStepDistances`. |

The contracts already distinguish several cases correctly. **EXECUTED:** nonfinite terrain horizon → code `0`; `100°` → saturated code `65535`. An out-of-range restriction endpoint threw. A complete turn graph with costs `[1,2]` and arc weight `3` produced turn `[0,1]` and penalty `[1]`; setting `overflow:1` threw. Contiguity preparation rejected both `2²⁴+1` and `2³²+1` rows against its declared 4,096-row limit before allocating. The Arrow scalar route preserved `2²⁴+1` rows exactly. A universal 24-bit count cap would break working inputs.

## Proposed convention

Name a host data adapter `prepare<Domain>Input`. Return `{arrays, formats, counts, metadata}` with live counts distinguished from allocated capacity. Where binding adds value, a separate graph adapter returns `{views, counts, metadata}` and uses the existing public luma.gl upload APIs.

Validate source shape and row identity before allocation, validate the representation actually written, and apply validity masks before computing shared bounds or dictionaries. A checker can pair each adapter with a real consuming constructor and exercise empty, singleton, sliced, multi-batch, masked, truncated, narrowing and overflow fixtures. Format and lane count should come from the adapter, with callers choosing any extra replay capacity.

## Findings

`S/` means `modules/analysis/src/`. `V` is `S/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets.ts`. `A` is `S/analysis-core/graph-support/arrow-analytics-normalization.ts`. `P` is `S/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology.ts`.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| ING-01 | W | ORDINARY | ADDITIVE | S | `V:56,115,124` | Outside-extent prediction duplicates sites in nearest-neighbor CSR. |
| ING-02 | W | ORDINARY | ADDITIVE | S | `V:25,179,263` | One masked NaN removes the neighborhoods of valid sites. |
| ING-03 | W | UNUSUAL | ADDITIVE | S | `S/analysis-core/contracts/local-frame.ts:129` | Overlapping source and destination overwrite unread coordinates. |
| ING-04 | W | UNUSUAL | ADDITIVE | S | `A:265,276,302` | Ordered dictionary preparation changes valid ordinal codes. |
| ING-05 | W | UNUSUAL | ADDITIVE | S | `A:89,112,221` | A misspelled precision mode permits unbudgeted loss. A misspelled time-column name can route ticks through Float32. |
| ING-06 | W | UNUSUAL | ADDITIVE | S | `prepare-flow-demands.ts:63` | An incomplete edge becomes an edge to node zero. Sparse input can misalign capacities. |
| ING-07 | W | UNUSUAL | ADDITIVE | S | `prepare-trajectory-observations.ts:126` | A later caller edit changes the reported origin of already projected positions. |
| ING-08 | W | EXTREME | ADDITIVE | S | `space-time-parameters.ts:243` | The guarded time packer certifies collapsed or infinite timestamps. |
| ING-09 | W | EXTREME | ADDITIVE | S | `gpu-solar-irradiance.ts:133,197` | Solar ingress admits an infinite irradiance and rounds an integer active count. |
| ING-10 | W | EXTREME | ADDITIVE | S | `arrow-analytics-wide-numeric.ts:40` | Standalone Decimal conversion maps a representable number to zero and an exact zero to NaN. |
| ING-11 | W | EXTREME | ADDITIVE | S | `A:183,200` | A prepared table can advertise 4,294,967,297 rows while its scalar vector exposes one. |
| ING-12 | W | UNUSUAL | ADDITIVE | S | `spatial-relate-types.ts:77` | A NaN slot count turns a valid spatial predicate into an empty upload. |
| ING-13 | S | ORDINARY | ADDITIVE | S | `V:36,43,417` | A one-point or all-coincident dataset throws during grid allocation. |
| ING-14 | S | UNUSUAL | ADDITIVE | S | `V:177,183,235` | A duplicate ordering can hang preparation. Omitting a required time scale silently changes the metric. |
| ING-15 | D | ORDINARY | ADDITIVE | S | `docs/api/gpu-network/concepts.md:61` | A valid road CSR fails the spatial-weights invariants despite the documented handoff. |
| ING-16 | E | ORDINARY | ADDITIVE | M | `P:25,220`, `catalog-geoarrow.ts:39` | A native Polygon is classified as malformed. Three consumer readers lose later batches if reused. |
| ING-17 | E | ORDINARY | ADDITIVE | M | `gpu-raster-medoid-composite.ts:42` | Acquisition rasters need a manual transpose that shape validation cannot distinguish. |
| ING-18 | E | ORDINARY | ADDITIVE | S | `scalar-scale-parameters.ts:92`, `V:247` | Preparers omit the minimum allocation or constructor field names needed immediately afterward. |
| ING-19 | S | ORDINARY | ADDITIVE | S | `A:126,152,175` | Float16 and LargeUtf8 are refused despite lossless routes to supported representations. |
| ING-20 | C | ORDINARY | BREAKING | S | `gpu-network-time-routing.ts:398` | One of three network preparers uses positional arguments and a second edge spelling. |

### ING-01: A query outside the extent returns duplicate neighbors

**Where and evidence.** `V:56,60,84,115,124,251`. **EXECUTED** from the documented “arbitrary query points” interface in `docs/api/gpu-spatial-analysis/vecchia-likelihood.md:27`. The KI page has the same wording.

```ts
const positions = [0,0, 2,0, 0,2, 2,2, 1,1, 1,0, 1,2, 2,1];
const {neighbors} = buildVecchiaPredictionNeighborhoods({
  positions, queries: [-10, 1], neighborCount: 4
});
// Actual [0, 0, 0, 2]. Brute-force squared distance, then row ID: [0, 2, 4, 5].
```

**Now.** A negative bucket column can enter a flattened address belonging to another row. The result has the expected length and looks usable. **Wanted.** Clamp the starting cell on both axes, check both bounds on every visited cell, and compare all four outside edges and corners against brute force. **Callers:** none. **Value and cost:** correct interpolation inputs outside the survey footprint, S.

### ING-02: Masked coordinates poison the search for every other row

**Where and evidence.** `V:25,30,179,190,263,265`. **EXECUTED** with `positions:[0,0, NaN,0, 1,1]`, `validity:[1,0,1]`, `neighborCount:1`. Conditioning produced offsets `[0,0,0,0]` and no neighbors. Expected `[0,0,0,1]` and `[0]`. Prediction at `[0,0]` also returned no neighbor instead of site `0`.

**Now.** The mask is consulted after the full-column min/max pass. **Wanted.** Build the extent from eligible finite XY rows and use the same eligibility rule for insertion. Keep original row IDs. **Callers:** none. **Value and cost:** one missing observation no longer invalidates an entire survey, S.

### ING-03: Reusing a typed-array backing buffer changes projected geometry

**Where and evidence.** `S/analysis-core/contracts/local-frame.ts:129,137,144,148`. **EXECUTED** with origin `[0,0]`, source `Float32Array.of(0,0,1,1,99,99)`, input `source.subarray(0,4)` and destination `source.subarray(2)`. Fresh output was `[0,0,111183.7890625,111200.7265625]`; overlapping output was `[0,0,0,0]`. The inverse reproduced the same loss. This adds an aliasing case to KNOWN(CORE-101), whose short-target check now passes.

**Now.** Both views meet the advertised length requirement. A separate shape discrepancy is visible with `[0,0,3]`: projection returns two elements, inverse projection returns three, including a fabricated trailing zero. **Wanted.** Support overlapping byte ranges using a source snapshot, or reject hazardous overlap before writing. Reject partial source rows in both directions. **Callers:** none for complete, nonoverlapping inputs. **Value and cost:** safe reuse of staging buffers, S.

### ING-04: An ordered Arrow dictionary loses ordinal identity

**Where and evidence.** `A:263,276,302,308`. **EXECUTED** with an ordered UTF-8 dictionary `['a','a','b']` and source indices `[0,1,2]`. The prepared dictionary still had all three entries, but its indices were `[1,1,2]`. Reusing the returned dictionary also produced `Category dictionary contains duplicate labels`.

**Now.** The first ordered dictionary is copied into the output, then each row is decoded to a label and looked up in a map. The last duplicate wins. **Wanted.** With a matching ordered dictionary, preserve each valid source index. If the analytics contract requires unique ordinal labels, reject the dictionary before producing any output and name the field and duplicate indices. **Callers:** none for unique labels. **Value and cost:** stable category ranks across ingestion and replay, S.

### ING-05: An invalid Arrow precision instruction fails open

**Where and evidence.** `A:89,101,112,221,242`. **EXECUTED** on Float64 `16777217`. `{mode:'exact'}` threw; JavaScript configuration `{mode:'excat'}` returned `16777216`. Separately, an Int64 column named `time`, `timeColumns:{tyme:'second'}` and a valid absolute rounding budget of `1` produced scalar `16777216` and **no time column**.

**Now.** The first case is caught by a TypeScript literal, but configuration read from JSON reaches runtime. The second typo passes the declared `Record<string, …>` type. **Wanted.** Validate the mode discriminant, the unit enum and time-column option keys against the source schema. Allow a real source field to be omitted from a particular projected selection. Name the unknown key in the error. **Callers:** none for valid policies. **Value and cost:** protects identifiers and timestamp ticks before destructive narrowing, S.

### ING-06: Flow-demand preparation manufactures endpoints or shifts capacities

**Where and evidence.** `S/gpu-network/network-optimization/prepare-flow-demands.ts:48,63,69,163`. **EXECUTED** with two nodes and zero demands. `edges:[[1]], capacities:[8]` returned packed endpoints `[1,0]`. A sparse two-slot array `edges:[, [0,1]], capacities:[3,8]` returned endpoints `[0,1]`, capacities `[3,8]` and `originalEdgeCount:2`.

**Now.** `every` and `flatMap` skip holes. `Uint32Array.from` converts a missing endpoint to zero. A sparse outer array can occur even when its TypeScript element type is a tuple. **Wanted.** Visit every integer edge index and require exactly two present, integral, in-range endpoints before committing its capacity or cost. Emit `edges[<row>].target` for the short tuple. **Callers:** none for well-formed edges. **Value and cost:** prevents a believable connection to the first vertex, S.

### ING-07: Trajectory metadata can describe a different projection from the data

**Where and evidence.** `S/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:51,126,133`. **EXECUTED** using a mutable origin `[0,0]`. After preparation, `origin[0] = 90` changed `prepared.projection.origin` to `[90,0]`; the already produced coordinates retained the original frame.

**Now.** The helper owns its arrays and borrows the one value needed to interpret them. `createGPULocalFrame` already snapshots its origin at `S/analysis-core/contracts/local-frame.ts:115`. **Wanted.** Store a frozen copy of `[longitude, latitude]` in the trajectory result. **Callers:** none. **Value and cost:** stable frame identity in cached observations, S.

### ING-08: The guarded time helper certifies a resolution the output does not have

**Where and evidence.** `S/gpu-spatial-analysis/space-time-tests/space-time-parameters.ts:243,250` and `space-time-bands.ts:126,143`. **EXECUTED**: `getGPUSpaceTimeGuardedRelativeTimes([0,1e40],1e35)` returned `[0,Infinity]` and resolution `6.490371073168535e32`. With `[0,1e-46]` and width `1e-51`, it returned `[0,0]` and resolution `1.044048714879764e-53`. The smallest positive Float32 spacing is `2^-149`, approximately `1.4013e-45`.

**Now.** The guard calculates spacing from the unencoded double. **Wanted.** Reject nonfinite stored lanes, account for subnormal spacing, and evaluate the guard against the encoded magnitude. Keep `[2²⁴,2²⁴+1]` as a positive test: default origin subtraction correctly returned `[0,1]`. **Callers:** none for resolutions the output actually meets. **Value and cost:** the time-resolution guarantee becomes enforceable, S.

### ING-09: Solar table controls lose both finiteness and cardinality

**Where and evidence.** `S/gpu-terrain/terrain-illumination/gpu-solar-irradiance.ts:108,133,151,197,209`. **EXECUTED** for noon UTC at `[0,0]`, 2026-06-01, one five-minute interval, `directNormalIrradiance:1e40`. The table’s fourth lane was `Infinity`. The public parameter packer encoded `sampleCount:16777217` as `16777216` and `16777219` as `16777220`.

**Now.** The API validates a finite JavaScript irradiance and a safe JavaScript count, then writes both as Float32. The latter can round **up** into spare capacity. **Wanted.** Require finite encoded irradiance, angular radius and diffuse controls. For the existing layout require `Math.fround(sampleCount) === sampleCount` and a representable consumer index. A future mixed-word layout could keep the count as Uint32. **Callers:** none for representable inputs. **Value and cost:** exact active table length and usable irradiance words, S.

### ING-10: Standalone Decimal input conversion loses values before Float32 is involved

**Where and evidence.** `S/analysis-core/graph-support/arrow-analytics-wide-numeric.ts:24,40,45`. **EXECUTED** with Decimal128 coefficient `2^96` and scale `309`. Both `convertWideNumericValue` and `convertWideNumericVector` returned `0`; the representable JavaScript number is `7.922816251426434e-281`. Coefficient `0` at scale `-324` returned `NaN`.

**Now.** `10 ** scale` becomes infinity or zero before division. This is distinct from KNOWN(CORE-02), which addressed the table preparer’s Float32 precision policy. **Wanted.** Convert the original coefficient and scale with bounded rational arithmetic or a correctly rounded decimal-to-binary64 conversion. Fast-path zero. **Callers:** none. **Value and cost:** correct standalone values and callbacks for wide scientific columns, S.

### ING-11: A prepared Arrow table has two incompatible row counts above Uint32

**Where and evidence.** `A:183,194,200`; installed `apache-arrow/src/util/chunk.ts:65` and `src/vector.ts:97`. **EXECUTED** with 4,096 references to the same 1,048,576-row Float32 record batch and a final one-row batch. This needs only one large backing array. `prepared.sourceRowCount` and `prepared.table.numRows` were `4294967297`; `prepared.table.getChild('x').length` was **1**. The last batch still contained the distinct value `9`. At `16777217` rows, all counts agreed.

**Now.** The source Arrow table already has the wrapped vector-index limitation. Preparation republishes it as an analytics-ready scalar table. **Wanted.** Before returning a combined scalar table, enforce the supported aggregate row limit. Offer independently addressable prepared batches with safe source offsets for larger inputs. **Callers:** none for correctly addressable combined tables. **Value and cost:** prevents a whole-column consumer from processing one row of a multibillion-row table, S for the guard.

### ING-12: A NaN capacity makes a predicate disappear

**Where and evidence.** `S/gpu-spatial-analysis/spatial-join/spatial-relate-types.ts:77,85,94`. **EXECUTED**: `packGPUSpatialRelatePattern('T********', NaN)` returned `Uint32Array(0)`. Capacity `1.5` returned three words `[4294967294,15,0]` for a two-word-per-slot layout.

**Now.** The content grammar is validated, but `patterns.length > NaN` is false and the typed-array constructor coerces the length. **Wanted.** Validate a positive integer `slotCount` and its two-word product before checking capacity or allocating. **Callers:** none for valid capacities. **Value and cost:** a malformed size cannot turn a valid predicate into an empty upload, S.

### ING-13: The smallest useful Vecchia inputs cannot be prepared

**Where and evidence.** `V:36,39,43,174,251,392,417`. **EXECUTED**: one site `[2,3]` and two coincident sites `[2,3,2,3]` each produced `Invalid typed array length: Infinity` in both builders. `getVecchiaMaxMinOrdering([2,3])` also threw. Both builders threw on empty coordinates.

**Now.** Both extent spans can be `Number.MIN_VALUE`; both cell-size terms underflow to zero. **Wanted.** Handle zero logical sites directly and represent a point extent as one finite bucket. One-site conditioning should yield `{offsets:[0,0],neighbors:[]}`, one-site prediction should find site `0`, and one-site ordering should return `[0]`. **Callers:** none. **Value and cost:** supports singleton features, coincident observations and empty batches, S.

### ING-14: The Vecchia builder must validate the ordering before linking rows

**Where and evidence.** `V:154,177,183,188,235`. **EXECUTED** with sites `[0,0, .1,0, 1,1]`. Order `[0,1,2]` completed and returned neighbors `[0,1]`; `[0,0,1]` entered the function and hit a 500 ms subprocess timeout. The source inserts the repeated row into its own linked list. An out-of-range order `[1,4]` on two sites instead returned an empty CSR without a diagnostic.

A second input illustrates the same missing preflight. With positions `[0,0, 2,0, .1,0]`, times `[0,1000,1000]` and one neighbor, omitting `timeScale` chose site `0` for the last observation. Supplying `timeScale:1` chose site `1`. The type comment says times require a scale.

**Wanted.** Before allocation or insertion, check complete XY rows, aligned masks and times, an in-range permutation without duplicates, documented neighbor capacity, safe count products, and an explicit finite scale when times are supplied. This is separate from KNOWN(PTS-120), which concerns consumers of already prepared over-width CSR. **Callers:** none for inputs satisfying the documented contract. **Value and cost:** prevents a host-side hang and accidental spatial-only inference, S.

### ING-15: A road network and a spatial-weights matrix share storage, but have different invariants

**Where and evidence.** `docs/api/gpu-network/concepts.md:61,84`, `S/analysis-core/contracts/network-topology.ts:15`, `docs/api/gpu-spatial-analysis/operators-weights.md:17,49`. **EXECUTED** with the first-use composition suggested by the network page:

```ts
const offsets = Uint32Array.of(0, 2, 2);
const neighbors = Uint32Array.of(1, 1);       // two roads from 0 to 1
const weights = Float32Array.of(2, 3);
const topology = {offsets, neighbors, weights};

computeNetworkReachabilityOnCPU({topology, startNodes: Uint32Array.of(0)}).costs;
// [0, 2]
computeSpatialWeightsValidationOnCPU(offsets, neighbors, weights);
// {flags: [4, 0], invalidCount: [1]}
```

**Now.** The network guide says this object can be passed “wherever spatial weights are read.” The weights guide requires strictly ascending unique neighbors. Also, travel cost and statistical influence need an explicit relationship. The earlier network report identified the differing invariants; the new evidence is the contradictory public handoff and a two-edge fixture that distinguishes them.

**Wanted.** Change both network-guide claims to say that reuse requires checking spatial-weights invariants and choosing what a weight means. Show the validator and an explicit parallel-edge aggregation policy. The KI weights guide at lines 20 and 54 and KI CPU validator at line 60 retain the same requirement. **Callers:** none. **Value and cost:** prevents a type-compatible but invalid analysis, S.

### ING-16: Geometry ingestion needs a port adapter with a schema check

**Where and evidence.** `P:25,70,152,220,240`; `docs/api/reused-analysis-preparation.md:28`; `modules/kepler/src/analysis/catalog-geoarrow.ts:39,58,74`. **EXECUTED** on an interleaved Float64 Arrow Polygon containing the valid closed ring `[[10,20],[30,20],[10,35],[10,20]]`. Supplying that vector as `batches[0].polygons` returned `geometryMask:[0]`, `invalidRows:[{sourceRowIndex:0,reason:'invalid-topology'}]`. Adding one list level for MultiPolygon produced mask `[1]` and positions `[0,0,1,0,2,0]`. Those are equality tokens. The native Polygon’s geometry was valid.

The installed public `iterateGeoArrowBatchesFromArrowVector` from `@luma.gl/graph/arrow` recognized the original Polygon and returned `(chunkIndex,rowOffset,rowCount)` of `(0,0,1)` and `(1,1,1)` for two concatenated batches. In contrast, the Kepler multipoint, multilinestring and multipolygon readers each returned feature offsets `[0,1]` when passed two one-feature chunks. Their current catalog producer builds one chunk, so this is a **reuse hazard**, not a claim that today’s catalog loses a feature.

**Wanted.** Build on KNOWN(DIR-102) with `prepareGeometryInput(column, options)`. Use the existing upstream iterator, normalize Polygon versus MultiPolygon nesting, retain batch offsets, distinguish nulls from empty geometries, and carry the frame and source IDs. Validate unsupported column encodings once at the schema boundary. Keep exact-equality token output explicitly separate from metric XY. The staged geometry ports and join guide retain the relevant shapes. **Callers:** none for an additive adapter. **Value and cost:** real Arrow geometry can reach joins or weights without being discarded or reinterpreted, M.

### ING-17: A stack of valid acquisitions can produce a plausible result for the wrong pixels

**Where and evidence.** `docs/api/gpu-raster/raster-time-series.md:29,202`, `S/raster-analysis/raster-time-series/gpu-raster-medoid-composite.ts:42`, `cpu-raster-medoid-composite-operation.ts:55`. The KI time-series guide also specifies cell-major layout at line 29. **EXECUTED** with three one-band acquisitions over two cells:

```ts
const acquisitions = [[0, 10], [100, 20], [200, 30]];
const dimensions = {cellCount: 2, sliceCount: 3, bandCount: 1};
computeRasterMedoidCompositeOnCPU(Float32Array.from(acquisitions.flat()), dimensions);
// composite [10, 30], status [0, 0]

computeRasterMedoidCompositeOnCPU(Float32Array.of(0,100,200, 10,20,30), dimensions);
// composite [100, 20], status [0, 0]
```

**Now.** The indexing formula is documented. The missing piece is a checked host packing operation from the layout in which acquisitions arrive. Both buffers have the same length, and both results report success. `makeGPURasterBand` handles a single resident pixel view.

**Wanted.** Add `prepareRasterStackInput({slices, grid, bandCount, sourceLayout})` with an explicit mask layout. Return cell-major values plus the constructor counts. Document the resident single-band alternative using `GPUCubeLayoutTranspose`. **Callers:** none. **Value and cost:** a GIS developer can ingest acquisition planes without independently implementing the index arithmetic, M.

### ING-18: Prepared counts and consuming field names should travel with the arrays

**Where and evidence.** `S/dataframe-analysis/column-classification/scalar-scale-parameters.ts:85,92,129`, `gpu-scalar-scale.ts:59,127`; `V:139,247,313`; `gpu-vecchia-log-likelihood.ts:68`, `gpu-vecchia-fisher-scoring.ts:85`, `gpu-vecchia-space-time-log-likelihood.ts:72` in the same directory as `V`.

**EXECUTED**: `prepareGPUScalarScaleControls({scale:'point',domain:[42]})` returned a one-element domain. Empty domain returned zero elements. `GPUScalarScale` requires allocated capacity at least two. The source explains padding, but the result does not say how much. Vecchia users have to write the equivalent of:

```ts
const csr = buildVecchiaConditioningSets({positions, neighborCount});
const inputs = {
  neighborhoodOffsets: uploadUint32(csr.offsets),
  neighborhoods: uploadUint32(csr.neighbors),
  neighborCount
};
```

`uploadUint32` here denotes the caller’s upload. The count cannot be recovered reliably from a particular result when every observed neighborhood is smaller than the requested maximum.

**Wanted.** Add named `counts` and `formats` to the results. Return `minimumDomainCapacity:Math.max(2,domain.length)` for scalar controls and `{rowCount,neighborCount,entryCount}` for Vecchia. Provide the consumer-field mapping once. Flow preparation already returns the expanded `nodeCount` that `GPUNetworkFlow` needs; preserve that example. Solar similarly needs both active `sampleCount` and allocated `sampleCapacity` when a padded target is reused. **Callers:** none if existing result fields remain. **Value and cost:** fewer duplicate counts and fewer graph views measured in words instead of rows, S.

### ING-19: Two common Arrow types have lossless preparation routes

**Where and evidence.** `A:126,136,152,175`. **EXECUTED** on an Arrow Float16 vector `[1.5,null,-3.25]` and a LargeUtf8 vector `['a','b']`. Both threw `Unsupported analytics type … for "x"`.

**Now.** Narrow integers and Boolean already widen exactly; ordinary Utf8 already reconciles to dictionary codes. Float16 needs no error budget to widen to Float32. LargeUtf8 changes the source string-offset width, while the prepared categorical representation stores label codes. **Wanted.** Widen Float16 while retaining validity. Route LargeUtf8 through category preparation with an explicit capacity check for the resulting dictionary. Document both in the type matrix. **Callers:** none. **Value and cost:** fewer application-specific Arrow casts for imagery and large string tables, S.

### ING-20: Use one public network-preparation signature

**Where and evidence.** `S/gpu-network/network-optimization/prepare-flow-demands.ts:28`, `prepare-point-routing-turns.ts:15`, `gpu-network-time-routing.ts:398`. **COUNT:** two of three network preparers take one options object; restriction preparation takes three positional arguments. Flow accepts `[source,target]` tuples; restrictions require `{source,target}` objects. **EXECUTED**: `prepareGPUNetworkRestrictionStates(2,[{source:0,target:1}],[])` returned one transition encoded in two Uint32 lanes, plus `stateVertices`, `originalEdges` and `initialStates`.

**Now.** The Chicago showcase’s displayed snippet at `chicago-route-workbench.scene.ts:290` passes the returned CPU arrays directly as graph views. `GPUNetworkTimeRouting` requires a `uint32` state view and `uint32x2` transition view. The snippet omits precisely the distinction a newcomer needs.

**Wanted.** Adopt `prepareGPUNetworkRestrictionStates({nodeCount, edges, restrictions})`, share an endpoint input type with flow preparation, and include formats and counts in the result. Fix the displayed snippet to show the two uploads. **In-repository callers that change:** `rg -n -U '\bprepareGPUNetworkRestrictionStates\s*\('` found **three executable calls**: `chicago-route-workbench.compute.ts:238`, `network-optimization.node.spec.ts:170`, and `network-movement-advanced.spec.ts:105`. There is also **one displayed code call** in `chicago-route-workbench.scene.ts:290`. **Value and cost:** one learnable edge-input shape before publication, S.

## Departures by name

- **Host preparation with no common result descriptor:** `prepareArrowTableForGPUAnalytics`, `preparePolygonContiguityTopology`, `prepareGPUTrajectoryObservations`, `prepareGPUNetworkFlowDemands`, `prepareGPUNetworkPointRoutingTurns`, `prepareGPUNetworkRestrictionStates`, `prepareGPUScalarScaleControls`, `buildVecchiaConditioningSets`, `buildVecchiaPredictionNeighborhoods`, `makePointCloudLocalPositions` and `getGPUSolarIrradianceSunTable`. Preserve the useful domain-specific information each already returns.
- **Raster `values` plus `sliceCount` constructors in the prop census:** `GPURasterBestAvailablePixel`, `GPURasterBFASTMonitor`, `GPURasterCOLDMonitor`, `GPURasterEOFProjection`, `GPURasterGeometricMedianComposite`, `GPURasterHANTS`, `GPURasterHarmonicRegression`, `GPURasterLaggedCorrelation`, `GPURasterMedoidComposite`, `GPURasterSeasonalBreak`, `GPURasterSnapshotCovariance`, `GPURasterTemporal`, `GPURasterTemporalResample`, `GPURasterWhittakerSmoother`. A shared host packer would address the common handoff without changing these constructors.
- **Network topology vocabulary:** 24 of 41 network producers exposed a top-level `topology` option in the constructor-prop census, including optional and alternative routes. The other 17 were `GPUAdjacencyMatrixOrder`, `GPUEdgeBundling`, `GPUFlowAggregation`, `GPUKernelFlowSmoothing`, `GPUMapMatchedEdgeVolumes`, `GPUNetworkAccessibility`, `GPUNetworkCostWeights`, `GPUNetworkFacilityOptimization`, `GPUNetworkInhomogeneousK`, `GPUNetworkKernelDensity`, `GPUNetworkNoding`, `GPUNetworkPathExtraction`, `GPUNetworkPointRouting`, `GPUNetworkRationalAccess`, `GPUNetworkTimeRouting`, `GPUNetworkTravelTimePercentiles`, `GPURadiationFlowModel`. Their algorithms justify different inputs. The guide’s “every contributor” wording hides those distinctions.
- **Packed capacities that silently coerce or fail late:** `packGPUSpatialRelatePattern` and `getGPUSolarIrradianceParameterValues` above. Two additional **EXECUTED** examples are worth small fixes. `makeGPURasterMetadata({width:1.5,height:2})` returned the fractional-width grid, and accepted width `4294967297` at height one (`S/raster-analysis/raster-grid/gpu-raster-grid.ts:44`). `makePointCloudLocalPositions([0,0,1e40],[0,0,0])` returned `[0,0,Infinity]`; origin `[NaN,0,0]` turned a finite input row into `[NaN,2,3]` (`S/pointcloud-analysis/operations/pointcloud-local-origin.ts:69`). Validate the dimension product at grid creation and the finite origin and finite-row conversion at point preparation. Preserve intentional missing observations.
- **Reusable targets with observable failed writes:** `getGPUColumnQuantilesRationalProbabilityValues`, `S/dataframe-analysis/column-classification/column-quantiles-parameters.ts:141`. **EXECUTED**: packing `[[1,2],[1,0]]` into `[9,9,9,9]` threw on the second denominator and left `[1,2,9,9]`. Validate all supplied ratios before writing the caller’s buffer.
- **Deliberately different scalar policies that need to remain legible:** `encodeGPUTerrainHorizonUnorm16` reserves zero for nonfinite input; `packGPUSpatialRelate` rejects malformed text; `packGPUCellPeriodKey` returns `undefined` for excluded rows; `packGPUColor(256,-1,NaN)` executed as `0xff00ff00`. The last helper should document byte masking or offer a checked channel packer. Do not generalize any one of these policies to measurement columns.

## Quick wins

- [ ] Fix Vecchia bounds, mask handling, zero extents and permutation validation together. Pin distinct rows against brute-force distance for an outside query.
- [ ] Correct the two CSR handoff claims and add the parallel-road fixture to both guides.
- [ ] Reject unknown Arrow precision modes and unknown time-column names. Preserve ordered dictionary indices.
- [ ] Check the actual float lanes and integer round trips before publishing time, irradiance and active-count data.
- [ ] Guard overlapping projection targets, failed rational-probability writes and incomplete flow endpoints. Snapshot trajectory origins.
- [ ] Refuse oversized aggregate Arrow scalar tables before preparing their batches.
- [ ] Publish field formats and minimum capacities for Vecchia, scalar controls and restriction transitions. Correct the Chicago snippet.

## Larger proposals

1. **Ship geometry ingress first.** An analysis adapter over the public GeoArrow iterator is smaller than a general reader. Qualify it with Polygon and MultiPolygon, interleaved and separate coordinates, a nonzero slice offset, null and empty features, two batches, holes, and Float64 positions that coincide after Float32 conversion. Keep coordinate projection and equality-token preparation explicit. Cost M.
2. **Add raster stack and spatial-weights adapters independently.** Raster callers should declare plane ordering once. Weights callers should declare the meaning of edge values and the duplicate/self-edge policy once. Keep source-edge identity available for road routing. Neither operation belongs in a generic upload wrapper. Cost M each.
3. **Make the restriction-state signature change before alpha.** An options-object overload would avoid three executable edits but leave both conventions public. I recommend migrating the three calls and the displayed example now. Existing resident-view constructors can continue to accept caller-owned views.

## Not read

I read the full bodies used for the Arrow normalization and time-column paths, polygon topology preparation, network demand and turn preparation, trajectory preparation, point-cloud local-origin conversion, raster grid/band construction, and Vecchia grid search and ordering. I sampled the wide-numeric implementation, raster consumers, public parameter packers, class constructors, docs and consumer examples by following the prepared fields to their first use. The public-export census covered all entries. The 214 parameter-value functions were counted, not individually fuzzed.

For staging-owned evidence I checked the KI copies of the Vecchia guide, geometry ports and joins guide, spatial-weights validator and weights guide, network reachability twin, raster stack and embedding guides, and terrain operator guide. The primary host defects identified above were observed in the working checkout. I did not open every shader, recipe, lens or example, or run a compiler, test suite, browser or GPU. No files were written.