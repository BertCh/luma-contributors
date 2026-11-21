# Validation and error messages

## Summary

The library usually rejects a bad count at construction. Its weakest boundary is the element type of a supplied GPU view. Of 409 public producers tested with a replacement packed view, **52 accepted a format outside the tested port’s declared contract**. Some then generated a shader that reads the caller’s bytes as a different numeric type.

Three composite producers accepted invalid scalar props, reserved graph resources, and then threw. Correcting the prop and retrying on the same graph failed in all three cases. Eight public parameter packers returned an undersized destination as a successful result. Three modified a previously valid destination before throwing.

The common diagnostics identify the expected range much more reliably than the received value. **None of 1,117 wrong-format refusals named the supplied format.** Applications get a plain `Error`, a native `TypeError`, or a later graph error for comparable mistakes.

The most valuable change is to validate each required port against its **declared format and row relationship** in the public constructor, using the same descriptor to produce a structured error and a negative test. This would catch the largest group of defects without changing valid calls.

## Census

`S/` means `modules/analysis/src/`, `T/` means `modules/analysis/test/`. Counts are from the live R19 checkout. `KI` identifies a separate check of the newer staging copy.

I enumerated runtime exports from the nine `ANALYSIS_BARRELS` in `T/analysis-core/kernel-audit/analysis-contributors.ts:16`, filtered on `typeof value.prototype?.getCommandNodes === 'function'`, and matched the resulting **498 names to 498 TypeScript class declarations**. I parsed constructor bodies with `ts.createSourceFile`, including constructors that delegate. The following counts are lexical, so a helper called by a base class or by another validator does not count as a direct call.

| Direct constructor-body pattern | Public classes |
|---|---:|
| Named core helper, no direct `throw` | 214 |
| Named core helper and direct `throw` | 252 |
| Direct `throw`, no named core helper | 17 |
| Neither direct pattern | 15 |
| Bare `assert(...)` | 0 |

The helper predicate was `validate` followed by one of `Capacity`, `MaximumCount`, `FiniteNumber`, `NonZeroFiniteNumber`, `Requirement`, `ViewLength`, `Extent`, `PowerOfTwo`, `CommandNodeBudget`, `CompactOutput`, `SinglePackedView`, `GraphOutputsDisjointFromInputs`, `GraphViewTopology`, `GraphViewsBelongToGraph`, or `GPUStorageBindingByteLength`. This broader set explains why its 466 direct users exceed the earlier named-limit count. The 15 classes in the fourth row should **not** be described as having no validation.

For runtime evidence, I bundled the existing kernel-audit fixture builders with esbuild in memory. I intercepted `new GPU…(props)` **in the test builders only**, captured the original props of the first successful fixture for each class, and called the real, unmodified constructor once per mutation. The graph supplied transient view descriptors on a null device. Constructor trials did not add nodes or allocate GPU memory. The core operations of the probe were:

```ts
new Producer({...props, id: 'probe-id', [port]: view('unorm8x4', original.length)});
new Producer({...props, id: 'probe-id', [port]: view(original.format, original.length - 1)});
for (const received of [0, 1.5, NaN, -1]) {
  new Producer({...props, id: 'probe-id', [countProp]: received});
}
```

| Family | Successful baseline constructors | Classes in the four-mistake comparison |
|---|---:|---:|
| Spatial | 246 | 106 |
| Raster | 74 | 51 |
| Terrain | 44 | 12 |
| Network | 38 | 19 |
| Dataframe | 34 | 25 |
| Point cloud | 11 | 4 |
| Analysis core | 4 | 4 |
| Geospatial | 2 | 1 |
| Computation record | 1 | 0 |
| **Total** | **454** | **222** |

The 222-class comparison used the first top-level view longer than one row, excluding names beginning with `output` and the legacy output names `pairCount` and `features`. It used the first positive numeric fixture prop ending in `Capacity` or `Count`, or named `capacity`, `k`, `maximumIterations`, or `maximumSteps`. I also tested **all** top-level `GraphDataView` and numeric props in the 454 fixtures. This wider pass detects cases hidden by choosing the first port, such as `GPUPointCloudFeatures` checking `positions` but accepting the wrong format on `indices`.

| Mutation in the 222-class comparison | Rejected in constructor | Accepted | Rejection contained custom ID and mutated prop spelling |
|---|---:|---:|---:|
| Packed `unorm8x4`, same logical length | 185 | 37 | 165 |
| Original format, one fewer row | 200 | 22 | 166 |
| Count or capacity `0` | 206 | 16 | 201 |
| Count or capacity `1.5` | 216 | 6 | 215 |
| Count or capacity `NaN` | 217 | 5 | 215 |
| Count or capacity `-1` | 216 | 6 | 215 |

Acceptance alone is not a defect. Some zero counts are legal, some shortened views still cover the required prefix, and some props apply only to a selected route. For example, KI `GPUTrajectoryLocations` explicitly permits zero rounds and correctly rejected `-1`, `0.5` and `1.5` in my rerun. I excluded that R19 departure from the findings.

Across all baseline fixtures, the wider pass made **1,284 format substitutions** in 409 classes, **1,210 shortening trials**, and four numeric substitutions on **793 props**. The constructors accepted 167 format substitutions across 52 classes. They rejected 1,062 shortening trials and 759 `NaN` trials. Testing `sint32` as a control also showed why the accepted-format list matters: `GPUClassBreaks` and `GPUColorScale` legitimately accept signed scalar columns. Both rejected `unorm8x4`.

**A first use from the docs.** I used the four-row fixture in `docs/api/pointcloud-analysis.md:54`, with separate transient views standing in for the caller’s uploaded arrays. The same quick start appears in the staging-owned KI page.

```ts
const props = {
  id: 'lidar',
  squaredDistances: view('float32', 4), // [1, 1, 100, 0]
  counts: view('uint32', 4),            // [1, 1, 1, 0]
  capacity: 1,
  standardDeviations: 2,
  output: {meanDistances: view('float32', 4), validity: view('uint32', 4)}
};
const filter = new GPUPointCloudStatisticalFilter(props);
```

The valid form declared **12 nodes**. At construction, three distances produced `lidar squaredDistances must have 4 rows`. Capacities `0` and `1.5` produced the identical `lidar capacity must be an integer in [1, 64]`. A multiplier of `-1` produced `lidar standardDeviations must be a finite number in [0, Infinity]`. Replacing the distances with a four-row **`uint32`** view still declared **12 nodes**. Omitting `counts` produced `TypeError: Cannot read properties of undefined (reading 'length')`. The quick start gives the correct types. Its promise that bad props receive a contributor-prefixed construction error does not hold for these two mistakes. A precisely typed TypeScript literal catches the format substitution statically. JavaScript, deserialized configurations and dynamically assembled port maps reach this runtime boundary.

## Proposed convention

A public producer validates every host-knowable precondition against its declared port and prop contract before construction succeeds. A packer validates its complete input and destination before its first write, and a graph-dependent check runs before the first graph mutation.

Each refusal exposes `code`, `contributor`, `id`, `path`, `received`, and `expected` on an `AnalysisValidationError extends Error`. Its message starts with the instance ID and gives the expected value or relationship and the received value. A fixture checker can mutate one prop at a time and assert the phase, fields, and absence of side effects.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| VAL-01 | W/S | ORDINARY | ADDITIVE | M | Fixed-format ports across 52 producers | A packed view can pass because the validator compares it with its own format. |
| VAL-02 | W | ORDINARY | ADDITIVE | S | Eight parameter packers | A short destination returns successfully after dropping parameter words. |
| VAL-03 | W | ORDINARY | ADDITIVE | S | Four raster time-series packers | An invalid active slice count becomes a shorter or empty observation period. |
| VAL-04 | W | ORDINARY | ADDITIVE | S | Geospatial and record mode selectors | A misspelled selector can choose a different coordinate transform or aggregation. |
| VAL-05 | W | UNUSUAL | ADDITIVE | S | Standardized-index calibration | A negative magnitude bound passes packing and excludes positive calibration samples. |
| VAL-06 | W | UNUSUAL | ADDITIVE | S | Three streaming raster packers | An invalid stream bound changes the caller’s destination before throwing. |
| VAL-07 | W | EXTREME | ADDITIVE | S | Point buffer and point-cloud change | Values required to remain positive arrive as float32 zero. |
| VAL-08 | S | ORDINARY | ADDITIVE | M | Local Moran, spatial Markov, polygon subdivision | An invalid constructor prop reserves resources and prevents a corrected retry. |
| VAL-09 | S | ORDINARY | ADDITIVE | S | `validateCompactOutput` | The shared validator skips two mandatory fields as though they were optional. |
| VAL-10 | E | ORDINARY | ADDITIVE | M | Required input ports | Omitting a required view frequently gives a JavaScript property-access error. |
| VAL-11 | C | ORDINARY | ADDITIVE | S | Geometry refinement and point-process helpers | A caller-supplied ID and the actual prop disappear at helper boundaries. |
| VAL-12 | E | ORDINARY | ADDITIVE | M | Core validators and refusal fixtures | Callers have neither the received format nor a stable way to classify a refusal. |
| VAL-13 | T | ORDINARY | ADDITIVE | S | Kernel-audit rejection runner | An expected error after graph mutation can certify a zero-node refusal. |
| VAL-14 | C | UNUSUAL | ADDITIVE | S | Shared finite-number validation | A float prop silently coerces booleans and strings that a count prop rejects. |

### VAL-01: Check the declared element type

**Where.** `S/pointcloud-analysis/operations/pointcloud-kernel-nodes.ts:222`, `S/gpu-terrain/terrain-processing/terrain-processing-utils.ts:40`, `S/raster-analysis/remote-sensing/raster-kernel-utils.ts:45`, both `S/raster-analysis/raster-time-series/raster-time-series-chunks.ts:90` and `raster-time-series-chunks-b.ts:90`, and `S/gpu-spatial-analysis/spatial-interpolation/interpolation-view-validation.ts:28`. The named constructor census is below.

**Evidence and current behavior.** `EXECUTED` — 167 of 1,284 substitutions accepted. `validateSinglePackedView(view, [view.format], id)` checks layout but cannot detect a wrong format for that port. For `GPULocalRBF`, substituting a `sint32` `values` view declared one node whose recorded source contained `var<storage, read> values: array<f32>` and `right[row] = values[valuesOffset + neighbors[row]]`. Integer word `1065353216` reinterpreted on the host is float `1`. For the documented point-cloud filter, the recorded accessor instead returned `u32`, which the generated body passed to `sqrt`. Thus the same newcomer mistake can lead either to numeric reinterpretation or to a shader type error.

**GPU confirmation, `SOURCE`.** Run the RBF fixture with noncollinear samples, all values supplied as the integer word `1065353216`, and query a sample location. Compare with the float32 constant-one fixture. For the filter, compile the documented four-row example with `uint32` distances and inspect the diagnostic at `sqrt(readSquaredDistances(...))`. I verified the generated sources, not either device outcome.

**Wanted.** Give fixed-format validators a named port map with explicit allowed formats. For this example, reject with `lidar squaredDistances must be packed float32; received uint32`. Keep the existing format unions for genuinely polymorphic columns. An automatic check should flag a port validated exclusively against `[view.format]`, while allowing this expression in a generic layout-only helper whose caller has already checked the port.

**In-repository callers that change.** No valid API calls. **Value and cost.** Dynamic GIS pipelines get a local wiring error before interpreting bytes or compiling a shader. M.

### VAL-02: Reject truncated parameter destinations

**Where.** `getGPUPointBufferParameterValues` in `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:58`. Under `S/raster-analysis/raster-time-series/`, the affected `getGPU…ParameterValues` functions are `RasterStreamingAccumulator:62`, `RasterClimateRunAccumulator:44`, `RasterStandardizedIndexCalibration:47`, `RasterStandardizedIndex:41`, `RasterLaggedCorrelation:41`, `RasterGeometricMedianComposite:50`, and `RasterSnapshotCovariance:60`, in the corresponding `gpu-raster-*.ts` files.

**Evidence and current behavior.** `EXECUTED` — passing a target one word shorter than each function’s fixed layout succeeded in all eight cases. With `reset:true`, three slices and magnitude `10`, the streaming packer normally returned `[1,3,-66,-61,10]`. Given `new Float32Array(4)`, it returned `[1,3,-66,-61]`. The correlation and geometric-median packers returned `[]` instead of `[3]` and `[0.000009999999747378752]`. These calls type-check with the public target type. `GPUParameterBuffer` allows prefix writes, so a subsequent write can retain the old trailing parameter. This extends `KNOWN(CORE-101)` from coordinate projection to per-frame parameter updates.

**Wanted.** Check `target.length >= requiredLength` before any assignment. Raster already has `validateParameterTarget` at `S/raster-analysis/raster-band-utils.ts:132`. Keep padded targets and existing explicit legacy layouts, including the shorter documented zonal-statistics layout.

**In-repository callers that change.** No valid calls. **Value and cost.** A reused scratch array cannot silently combine this frame’s settings with the previous frame’s bound. S.

### VAL-03: Validate active slice counts before packing them

**Where.** In `S/raster-analysis/raster-time-series/`, `gpu-raster-streaming-accumulator.ts:75`, `gpu-raster-climate-run-accumulator.ts:61`, `gpu-raster-standardized-index-calibration.ts:60`, and `gpu-raster-standardized-index.ts:46`. `GPURasterRollingMean` also consumes the streaming packer at `gpu-raster-rolling-mean.ts:46`.

**Evidence and current behavior.** `EXECUTED` — the three accumulator/calibration packers emitted `-1` in the active-count word for `{activeSliceCount:-1}` and a three-slice chunk. The index packer directly assigns the same setting. `SOURCE` — the readers clamp the word to `[0, chunkSliceCount]` before converting to `u32`, so a negative count reads as zero and a positive fractional count drops the fractional part. With `reset:true`, that can clear an accumulator and process none of the intended data. The smallest GPU probe is one cell, one finite observation, `reset:true`, and `activeSliceCount:-1`, followed by readback of the observation count.

**Wanted.** Validate `chunkSliceCount` against the contributor’s declared range, then `activeSliceCount` as an integer in `[0, chunkSliceCount]`. Preserve explicit zero for a deliberately empty chunk. Include both counts in a refusal.

**In-repository callers that change.** No valid calls. **Value and cost.** Streaming users can distinguish an intentionally empty update from a bad observation count. S.

### VAL-04: Reject unknown selectors before choosing an algorithm

**Where.** `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:119,259`, `S/geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts:246,281`, and `S/computation-record/gpu-transfer-mask.ts:49,95,176`.

**Evidence and current behavior.** `EXECUTED` — `coordinateSystem:'latlon'` declared a point-buffer node with variant `latlon`. Datum operation `'horizonal'` declared `horizonal-inverse`, and direction `'reverse'` declared `combined-reverse`. Mask mode `'counts'` declared two `index-counts` nodes. `SOURCE` — point buffer chooses planar arithmetic for every value other than `geographic`. Datum transform treats the unknown operation as both horizontal and vertical and the unknown direction as forward. Mask transfer selects `atomicMax` for every value other than `count`, even while recording the supplied mode. For a device probe, scatter two selected sources to the same target: a count is `2`, while the fallback produces `1`.

**Wanted.** Validate the existing literal unions at the public boundary. List the accepted strings and the supplied string. Use defaults only when an optional selector is omitted. The TypeScript unions already specify the intended behavior.

**In-repository callers that change.** No valid calls. **Value and cost.** A typo in an imported configuration cannot change the units, direction, or meaning of a result. S.

### VAL-05: Give calibration the same magnitude guard as its streaming sibling

**Where.** `S/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration.ts:47,246`; compare `gpu-raster-streaming-accumulator.ts:69`.

**Evidence and current behavior.** `EXECUTED` — calibration with `{reset:true, maximumValueMagnitude:-1}`, two slices, returned `[1,2,0,-69,-63,-67,-1]`. The streaming accumulator refused the same bound. `SOURCE` — calibration skips `positive > maximumMagnitude`, before incrementing the slot count. A one-cell gamma calibration with values `[1,2]`, location zero and bound `-1` tests the predicted empty fit.

**Wanted.** Reuse a positive-finite-float32 magnitude check before constructing the exponent or writing the target. Report `maximumValueMagnitude`, its received value and the float32 requirement.

**In-repository callers that change.** No valid calls. **Value and cost.** Climate analysts get a configuration error instead of a missing calibration period. S.

### VAL-06: A failed pack must preserve the previous settings

**Where.** `S/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:73`, `gpu-raster-climate-run-accumulator.ts:60`, `gpu-raster-standardized-index-calibration.ts:58`, and `S/analysis-core/kernels/fixed-point-accumulator.ts:49`.

**Evidence and current behavior.** `EXECUTED` — I filled correctly sized targets with `777`, requested `maximumStreamSliceCount:-1`, and caught `fixed-point maximumTermCount must be an integer in [0, 2^32 - 1]`. The surviving targets were `[1,3,777,777,777]`, `[1,3,5,777]`, and `[1,3,0,777,777,777,777]` respectively. A caller that catches the error has already lost the prior reset flag. The exception also names an internal argument instead of the supplied prop.

**Wanted.** Validate the stream bound and compute all exponents into locals before writing the first word. On error, preserve the entire destination and name `maximumStreamSliceCount`.

**In-repository callers that change.** None. **Value and cost.** Applications can safely reject an edit and reuse the last good settings. S.

### VAL-07: Check positivity after float32 rounding

**Where.** `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:63` and `S/pointcloud-analysis/operations/gpu-pointcloud-change.ts:141,162,319,419`.

**Evidence and current behavior.** `EXECUTED` — `getGPUPointBufferParameterValues(1e-100)` returned `[0,0]`. `getGPUPointCloudChangeParameterValues({confidenceMultiplier:1e-100})` emitted float32 zero in word 2. `new GPUPointCloudChange({...valid, halfDepth:1e-100})` also succeeded. Its constructor checks positivity on the double and finiteness on the float, then the kernel builder rounds the depth to zero. The neighboring radius check already tests the rounded quantity. These are additional paths beyond `KNOWN(CORE-102)`.

**Wanted.** Require `Number.isFinite(rounded) && rounded > 0` for these three positive float32 fields, using the rounded value that will actually be packed or embedded. Include the original and rounded values on underflow.

**In-repository callers that change.** No representable positive inputs. **Value and cost.** The public packers cannot emit the zero value their own contracts exclude. S.

### VAL-08: Reject invalid child props before reserving parent scratch

**Where.** `S/gpu-spatial-analysis/space-time-statistics/gpu-space-time-local-moran.ts:160`, KI `:164`; `distribution-dynamics/gpu-spatial-markov.ts:320,400`, KI `:341,421`; `geometry-refinement/gpu-subdivide-polygons.ts:150,297`.

**Evidence and current behavior.** `EXECUTED` on fresh null graphs. All three public constructors succeeded. I measured the new graph buffer descriptors at the later exception, then retried the corrected props on the same graph with the same ID.

| Producer and bad prop | Extra buffers at refusal, R19 / KI | Corrected retry failed on |
|---|---:|---|
| `GPUSpaceTimeLocalMoran`, `maximumNeighbors:NaN` | 3 / 3 | `…-current` already in use |
| `GPUSpatialMarkov`, `periodLag:-1` | 42 / 41 | `…-padded-values` already in use |
| `GPUSubdividePolygons`, `maximumWork:NaN` | 8 / 8 | `…-indices` already in use |

These are public producer paths beyond the sorted-reduction failure in `KNOWN(CORE-109)`. The newer KI copy already moves `GPULISAMarkov.minimumTransitions` validation earlier; that class is excluded here.

**Wanted.** Extract the child’s device-independent prop checks and call them from the parent constructor. Keep the device-size checks at graph declaration, before reserving scratch. Pin corrected same-ID retries, including resource IDs as well as added nodes.

**In-repository callers that change.** No valid calls. **Value and cost.** A user fixing one bad setting can keep the rest of an assembled graph. M.

### VAL-09: Validate the two required compact status words

**Where.** `S/analysis-core/kernels/gpu-contributor-types.ts:22,35,46`, `S/analysis-core/kernels/gpu-contributor-utils.ts:444`, `S/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter.ts:251`. `rg 'validateCompactOutput\(' modules/analysis/src` found **33 call sites in 30 files**, excluding the definition.

**Evidence and current behavior.** `EXECUTED` — deleting either `output.status.count` or `output.status.overflow` from the valid `GPUBoundsFilter` fixture passed **both** `validateCompactOutput` and the public constructor. `getCommandNodes` then threw `TypeError: Cannot read properties of undefined (reading 'byteOffset')`. The common loop skips every absent scalar. The type requires these two and permits omission of `requiredCount` and `candidateOverflow`.

**Wanted.** Check presence of `status`, `status.count`, and `status.overflow` before validating their layouts. Retain the optional-field behavior. Give the helper an optional caller path so a nested compact result can report, for example, `output.matches.status.count`.

**In-repository callers that change.** None required by the existing type. **Value and cost.** One helper improves diagnostics for compact selections, joins and trajectory outputs. S.

### VAL-10: Identify a missing required view before dereferencing it

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter.ts:106,140`, `S/analysis-core/kernels/gpu-contributor-utils.ts:276`, and the published packed validator in `node_modules/@luma.gl/graph/src/gpu-core/graph-data-view-utils.ts:39`.

**Evidence and current behavior.** `EXECUTED` — replacing each of 1,284 supplied top-level views with `undefined` yielded **871 native `TypeError`s in 375 classes**, 114 explicit `Error`s, and 299 accepted calls. The last group includes legitimate optional views. Examples include `GPUPointCloudStatisticalFilter.counts` reading `length`, `GPUSpaceTimeLocalMoran.cube` reading `format`, and `GPURasterTemporalResample.values` reading `length`. The latter two reproduced in KI. No contributor or prop appears in those native errors.

**Wanted.** Mark required ports in the same descriptor used for format checks. Assert presence before any format inspection, length product, snapshot of a required nested map, or call into luma.gl. For the quick start, the first error should identify `lidar counts`, expected `GraphDataView<'uint32'>`, received `undefined`.

**In-repository callers that change.** No valid calls. **Value and cost.** A developer wiring a missing dataset column can identify it from the first exception. M.

### VAL-11: Preserve the instance ID and the failing prop through helpers

**Where.** `S/gpu-spatial-analysis/point-process/point-process-utils.ts:12`, `S/gpu-network/gpu-network-kernel-density.ts:141`, `S/gpu-network/gpu-network-inhomogeneous-k.ts:159`, and the geometry-refinement sites listed below.

**Evidence and current behavior.** `EXECUTED` — with `id:'my-layer'`, wrong distance and intensity formats in the two network classes both produced `point-process must be packed, uint32-aligned float32 GPU data`. `GPURatioConcaveHull` with `ratio:-1` gave `Invalid hull ratio`. Shortening one port in `GPUNormalizeGeometry` gave `Canonical tree output size mismatch`. The nine named geometry examples below reproduced against both trees, including the three staging-owned declarations. Across the wider probe, **116 of 1,117 format refusals omitted the mutated prop’s literal spelling**. This adds loss of instance identity and port information to the specific misleading ranges in `KNOWN(ERG-102)`.

**Wanted.** Pass `(id, path, view, formats, expectedRows)` through the point-process helper. Replace pooled geometry checks with individual port checks. Report the relation and both counts, such as `my-layer output.nodes must have 8 rows to match nodes; received 7`. Retain grouped checks only when the relation itself needs multiple props.

**In-repository callers that change.** Internal validator call sites and diagnostic assertions. No public producer calls. **Value and cost.** A multi-layer graph can direct the user to the failing instance and field. S for the named sites.

### VAL-12: Expose a stable error that carries the received value

**Where.** `S/analysis-core/kernels/gpu-contributor-utils.ts:296,318,585,605,641`, `S/analysis-core/index.ts:32`, `T/analysis-core/kernel-audit/kernel-audit-runner.ts:32`.

**Evidence and current behavior.** `EXECUTED` — capacity `1.5`, received length `3` for expected `4`, and radius `NaN` each threw `Error` with `code === undefined` and `Object.keys(error) === []`. All 1,117 rejected `unorm8x4` trials omitted `unorm8x4`. `rg 'extends .*Error\b' modules/analysis/src` found a private `PolygonContiguityCapacityError`, not an exported common constructor error. GPU result status codes describe different events and cannot classify these exceptions.

**Wanted.** Export `AnalysisValidationError` from `analysis-core`. Start with codes `MISSING_PROP`, `VIEW_FORMAT`, `VIEW_LENGTH`, `INTEGER_RANGE`, `NUMBER_RANGE`, and `INVALID_OPTION`. Keep `Error` inheritance and the existing instance-ID prefix. For a short view, carry `received:3`, `expected:{rows:4}`, `path:'squaredDistances'`, and the producer identity even when the instance is named `lidar`. A size relation should also identify its source, such as `counts.length * capacity`.

**In-repository callers that change.** Zero constructor call sites. There are **208 registered rejection matchers, all 208 end-anchored**. Appending received values indiscriminately would require migrating the affected matchers. Introduce the structured fields first, migrate the tests, then enrich the text. **Value and cost.** UI and GIS integrations can recover programmatically and display a useful message without parsing English. M.

### VAL-13: The rejection audit can report zero after a node was added

**Where.** `T/analysis-core/kernel-audit/kernel-audit-runner.ts:169,175` and `kernel-audit-rejections.ts:37`.

**Evidence and current behavior.** `EXECUTED` — I used the registered `GPUTerrainSpikeRepair` 4,194,304-cell refusal identity in a synthetic builder. It added a compute pass, observed `graph.nodes.length === 1`, then threw the expected limit message. `auditKernelCase` returned `{findings:[], nodeCount:0, moduleCount:0, rejected:true}`. Its catch branch reports literal zero. This explains how an error with precisely the right text can conceal the failure phase seen in VAL-08.

**Wanted.** Distinguish constructor refusals from device-dependent declaration refusals. Measure node and resource counts immediately before the operation under test and after the exception, excluding fixture input allocation. Report actual deltas. Include a synthetic mutation-then-throw fixture and a corrected same-ID retry.

**In-repository callers that change.** Test infrastructure and refusal records only. **Value and cost.** The gate can enforce the stated construction contract and detect partial declaration. S.

### VAL-14: A finite-number validator should reject non-numbers

**Where.** `S/analysis-core/kernels/gpu-contributor-utils.ts:605` and `S/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter.ts:38`.

**Evidence and current behavior.** `EXECUTED` — `getGPUPointCloudStatisticalFilterParameterValues({standardDeviations:true})` returned `[1]`; the string `"2"` returned `[2]`. `Math.fround` coerces before the validator checks finiteness. On the same API, `validateCapacity` uses `Number.isInteger` and rejects equivalent non-number count inputs.

**Wanted.** Check `typeof value === 'number'` before rounding, then validate the actual rounded range. Preserve `true` and `"2"` as the received values in diagnostics. Applications that accept numeric strings can convert them explicitly.

**In-repository callers that change.** None conforming to the declared `number` type. **Value and cost.** An incorrect configuration type cannot silently become a different statistical threshold. S.

## Departures by name

**Wrong-format construction successes.** These are all 52 observed in the 454-fixture pass. Each name below includes its indicated prefix. Four declarations on the staging-owned list were checked again in KI.

| Prefix | Classes accepting at least one substituted `unorm8x4` view |
|---|---|
| `GPURaster` | `BFASTMonitor`, `BestAvailablePixel`, `COLDMonitor`, `ClimateAnomaly`, `ClimateRunAccumulator`, `ConservativeRemapping`, `EOFProjection`, `GeometricMedianComposite`, `HANTS`, `HarmonicRegression`, `LaggedCorrelation`, `MedoidComposite`, `PhenologyMetrics`, `RegionGraph`, `Registration`, `RollingMean`, `SeasonalBreak`, `SnapshotCovariance`, `SpectralUnmixing`, `StandardizedIndex`, `StandardizedIndexCalibration`, `StreamingAccumulator`, `Temporal`, `TemporalResample`, `Texture`, `WhittakerSmoother` |
| `GPUPointCloud` | `Change`, `Features`, `Ground`, `ICP`, `NormalScale`, `SMRF`, `StatisticalFilter` |
| `GPUTerrain` | `BranchingFlow`, `Breach`, `CatchmentSummary`, `DEMCoregistration`, `DepressionHierarchy`, `FlowFractions` |
| `GPU` | `ConditionalGaussianSimulation`, `IsobandRings`, `LinearReferencing`, `LocalKriging`, `LocalRBF`, `NaturalNeighbor`, `NetworkIsochrones`, `RegionStatistics`, `SpaceTimeLocalMoran`, `SubdividePolygons`, `VecchiaFisherScoring`, `VecchiaLogLikelihood`, `VecchiaSpaceTimeLogLikelihood` |

KI `GPURasterTemporalResample:116` still accepted wrong `values` and `parameters`; `GPUNetworkIsochrones:369` accepted wrong `startCosts`, `startCount`, and `costLimit`; `GPUSpaceTimeLocalMoran:98` accepted wrong `parameters`. KI `GPURegionStatistics:146` rejected wrong `positions` in its rectangle fixture but still accepted wrong `sourceIds`. The R19 counts describe construction success, including ports that a selected route might not consume.

**Named geometry diagnostics from VAL-11.** Relative to `S/gpu-spatial-analysis/geometry-refinement/`:

| Class | Reproduced message site |
|---|---|
| `GPUCanonicalPolygonEquality` | `gpu-canonical-polygon-equality.ts:61` |
| `GPUConvexStraightSkeleton` | `gpu-convex-straight-skeleton.ts:104` |
| `GPUMaximumInscribedCircle` | `gpu-maximum-inscribed-circle.ts:98` |
| `GPUNormalizeGeometry` | `gpu-normalize-geometry.ts:132` |
| `GPUNormalizePolygons` | `gpu-normalize-polygons.ts:61` |
| `GPUPolygonPrecisionReduction` | `gpu-precision-reduction.ts:76`, KI `:174` |
| `GPULinePrecisionReduction` | `gpu-precision-reduction.ts:303`, KI `:512` |
| `GPURatioConcaveHull` | `gpu-ratio-concave-hull.ts:72,92` |
| `GPUVectorStraightSkeleton` | `gpu-vector-straight-skeleton.ts:93` in both copies |

**Direct `throw`, no direct named helper in the static census.** `GPUGammaStatistic`, `GPUIsobands`, `GPUIsolines`, `GPULocalDominance`, `GPUMultiScaleRelief`, `GPURasterArithmetic`, `GPURasterConditional`, `GPURasterNDWI`, `GPURasterReclassify`, `GPURasterSAVI`, `GPUReliefBlend`, `GPUReliefShading`, `GPUSimpleLocalRelief`, `GPUSolarIrradiance`, `GPUSolarShadowMask`, `GPUTextureShading`, `GPUTransferMask`.

**Neither direct pattern.** `GPUCellAggregation`, `GPUCellRollup`, `GPUChunkedSpatialWeightsValidation`, `GPUDotDensity`, `GPUExpectationBasedSpaceTimeScan`, `GPUGeneralizedGeographicallyWeightedRegression`, `GPUGeometryCleanup`, `GPULocalFrameProjection`, `GPUNetworkCentrality`, `GPURandomPointsInPolygon`, `GPURasterEVI`, `GPURasterNBR`, `GPUSpatialLag`, `GPUSpatialPearson`, `GPUSpatialScanStatistic`. Delegation and inheritance explain why these names are a navigation aid rather than a defect list.

## Quick wins

- [ ] Add target-length guards to the eight fixed-layout packers in VAL-02. Test exact, one-short, empty, and padded destinations.
- [ ] Share active-slice and positive-magnitude validation across the raster streaming packers. Compute all fixed-point exponents before writing any words.
- [ ] Require `status.count` and `status.overflow` in `validateCompactOutput`. Add a regression through `GPUBoundsFilter`.
- [ ] Validate the three mode selectors in VAL-04 and the three float32-positive fields in VAL-07.
- [ ] Pass the caller’s ID and port name through `validatePointProcessView`. Replace the nine reproduced anonymous geometry errors.
- [ ] Make the audit report actual node and resource deltas on refusal.

## Larger proposals

1. **Use named port descriptors at the constructor boundary.** Start with the point-cloud helper, the two raster time-series helpers, terrain processing and interpolation. These cover many of the 52 observed format departures. Describe allowed formats, presence, topology and row relationships once. Keep GPU-resident row values out of these checks. Cost M.
2. **Ship the structured error before the alpha.** Adding an `Error` subclass and fields preserves existing `catch` behavior. Changing the English text has a real migration cost because the audit currently pins all 208 refusal patterns to the end of the message. Migrate those assertions to fields and retain a few tests of the user-facing message. Cost M.
3. **Make the mutation probes part of contributor acceptance.** Start from a passing fixture, replace one port or prop, assert the public phase and error fields, and then correct the call on the same graph. This checks what the user experiences even when a class delegates through several helpers. Cost M.

## Not read

I read the shared compact-output types and point-process validator in full, traced the constructor-to-binding or packer-to-reader paths cited above, and sampled the public guides, audit runner, related area reports, synthesis and decision list. The static census covered all 498 public producers. Runtime coverage was the first successfully captured audit fixture for 454, with a comparable numeric-and-view matrix for 222. I did not manually review all 498 implementations, every route of those fixtures, every nested output, or the 44 producers without a captured baseline.

I used the newer KI copy for the staging-owned findings identified above and checked the point-cloud quick start in both copies. The checkout changed during the review, so line numbers and census totals describe the observed snapshots. No files were written. No browser, GPU, network, Vitest, compiler or build was used. Shader consequences are identified as `SOURCE`; constructor outcomes, generated source, packed words, graph resource counts and retry failures were observed on the host.