# Boundary inputs across the analysis library

## Summary

I exercised **105 distinct public producers** through their CPU twins, constructors, packers or graph declaration. In a controlled zero-row comparison, **19 of 59 CPU twins returned a result and 40 rejected the input**. A caller cannot infer the behavior from the family or the `compute…OnCPU` name.

The most consequential result was statistical. An all-missing Pettitt series contributes a seemingly valid `p = 1` to multiple-testing correction. Adding one geography with no observations changed a neighboring geography’s adjusted p-value from **0.03141 to 0.06282**.

Two movement operators changed a geographic centroid by **half a degree** when one observation’s longitude changed from `180` to `-180`. A third returned `pairStatus = 0` for a Fréchet decision it could not compute.

The worst performance case required just **three collinear vertices**. Rotating a horizontal line onto the diagonal took the CPU label-point and inscribed-circle twins from immediate answers to timeouts.

The highest-value pre-alpha change is an executable boundary contract shared by a contributor and its operation forms. Start with zero logical rows, insufficient observations, and exact versus one-short capacity. The existing time-window filter already demonstrates how to publish scalar status for an empty batch.

## Census

`S/` below means `modules/analysis/src/`. `KI` means the staging checkout named in the brief. Counts are from the working trees read during this review.

I enumerated the nine public producer barrels through `ANALYSIS_BARRELS` in `modules/analysis/test/analysis-core/kernel-audit/analysis-contributors.ts`. The counting predicates were `typeof value?.prototype?.getCommandNodes === 'function'` and `/^compute.*OnCPU$/`, deduplicating exported values. This returned **498 producer classes and 131 public pure functions**. The source search `rg -l 'export function compute\w*OnCPU\(' modules/analysis/src` was used to locate implementations, not to infer public visibility.

| Implementation family | Distinct producers exercised |
|---|---:|
| `gpu-spatial-analysis` | 59 |
| `gpu-network` | 13 |
| `raster-analysis` | 12 |
| `gpu-terrain` | 10 |
| `dataframe-analysis` | 5 |
| `analysis-core` | 2 |
| `geospatial-analysis` | 2 |
| `pointcloud-analysis` | 1 |
| `computation-record` | 1 |
| **Total** | **105** |

I executed 99 corresponding pure functions and six additional contributor classes. I also exercised named operation forms for several of those same producers. Constructor and graph-declaration results below establish host behavior only.

For the zero-row comparison I used typed arrays of length zero, `[0]` for CSR and nested terminal offsets, positive model dimensions where independent of the rows, and valid parameter layouts. I repeated the 57-function main batch with the KI implementation substituted for each of its **31 staging-owned files**. The accept/reject counts agreed. Two further movement functions brought the cohort to 59.

| Boundary and observed variant | Number | Examples |
|---|---:|---|
| Zero logical rows accepted by pure function | 19 | `TrajectoryKalman`, `LineLoopRemoval`, `PointToCell`, `OrdinaryLeastSquares` |
| Zero logical rows rejected by pure function | 40 | `TrajectorySplit`, `LabelPoint`, `NetworkReachability`, `SpatialWeightsTraces` |
| Additional null-device producers emitting no nodes for zero rows | 4 | `GPUPointBuffer`, `GPUDatumGridTransform`, `GPUPointCloudStatisticalFilter`, `GPUFittingMinibatchOrder` |
| Additional null-device producers emitting nodes for zero rows | 2 | `GPUTimeWindowFilter` emitted clear-count and publish nodes. `GPUTransferMask` emitted clear and scatter with a one-row destination. |
| Cell-index families assigning the same key to equivalent `-150°` and `210°` | 2 of 5 | H3, S2 |
| Cell-index families assigning different keys, both marked valid | 3 of 5 | Quadbin, quadkey, geohash |

The CPU probes were bundled from source into stdout and executed with `node --input-type=module -`. Representative controls help interpret the findings:

| Input | Executed result |
|---|---|
| One triangle, Delaunay capacities `1` and `0` | `(count, requiredCount, overflow)` was `(1,1,0)` and `(0,1,1)`. |
| Three identical points, then three distinct collinear points | Delaunay returned zero triangles and converged in both cases. `duplicateCount` distinguished them, `2` versus `0`. |
| One point, OPTICS `minimumSamples: 3` | Ordering `[0]`, core distance `Infinity`, no cluster. `maximumDistance: 0` gave `maximumDistance must be positive`. |
| One node, no edges, source equals destination | Reachability cost `[0]`. Alternative routes returned one route of cost `0` even with `routeCount: 3`. |
| Reciprocal parallel edges plus an isolated node | Network centrality returned edge betweenness `[0.5,0.5,0.5,0.5]` and reached counts `[2,2,1]`. A positive-cost self-loop contributed zero betweenness. |
| `1 × 1` raster `[2]` | Terrain flow returned accumulation `[1]`. Raster medoid returned value `[2]`, slice `[0]`, status `[0]`. Isolines rejected the width with an explicit minimum of `2`. |
| `1 × 3` raster `[NaN,NaN,NaN]` | Raster reconstruction returned validity `[0,0,0]`. Medoid returned three NaNs and three no-slice sentinels. Fractional zonal statistics preserved covered area `1` and reported valid area `0`. |
| One zone and one finite value `2` | Inequality returned Gini `0`, mean `2`, count `1`. Class breaks requested three quantile classes and returned one occupied class. |
| `179.9° → -179.9°` on the equator | Geodesic distance `22,240.373 m` on the packed float32 inputs. Both movement-center functions returned `-180°` for the pair. |

## Proposed convention

**Distinguish the number of rows supplied, the number of usable observations, and the number of output slots.** Accept a well-formed zero-row batch with terminal offsets `[0]`, publish zero counts and the documented empty or insufficient-support result, and reject a dimension only when the operation actually requires it to be positive. A conformance fixture should run the pure function, both named forms and the contributor on the same logical input, including a second submission over previously populated outputs.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| DGN-01 | W | ORDINARY | BREAKING | S | Pettitt → multiple testing | An untested cell can reverse another cell’s significance decision. |
| DGN-02 | W | ORDINARY | BREAKING | M | Staypoints, locations | Equivalent spellings of the antimeridian get separate centroid votes. |
| DGN-03 | W | ORDINARY | ADDITIVE | S | Track similarity | Invalid coordinates return an ordinary negative decision with clear status. |
| DGN-04 | W | ORDINARY | ADDITIVE | S | Point-cloud statistical filter | Float32 arithmetic can reject every row of a constant-distance population. |
| DGN-05 | W | UNUSUAL | ADDITIVE | S | Radiation flow | An unknown distance is treated as having zero intervening population. |
| DGN-06 | W | EXTREME | ADDITIVE | S | Radiation CPU twin | The CPU still loses all outside population after the GPU fix for NET-111. |
| DGN-07 | W | EXTREME | ADDITIVE | S | Geodesic pairs | An exactly representable longitude gives zero distance to a point 19,570 km away. |
| DGN-08 | S | UNUSUAL | ADDITIVE | S | Label point, maximum inscribed circle | Rotating three collinear vertices defeats the degeneracy shortcut. |
| DGN-09 | C | ORDINARY | ADDITIVE | M | Nested geometry and movement | `[0]` is accepted by some consumers of an empty geometry and rejected by their siblings. |
| DGN-10 | C | ORDINARY | ADDITIVE | S | Network queries | An unchanged one-node topology accepts zero path-extraction targets and rejects zero route queries. |
| DGN-11 | S | ORDINARY | ADDITIVE | S | Delaunay, network path extraction | Named operations block zero-capacity count and overflow results. |
| DGN-12 | W/S | UNUSUAL | ADDITIVE | S | Point-buffer packer | A positive radius packs as zero, and a short target is returned as successful. |
| DGN-13 | C/D | UNUSUAL | ADDITIVE | S | Point-to-cell | Changing the index family changes the treatment of wrapped input. |
| DGN-14 | D | ORDINARY | ADDITIVE | S | Spatial weights traces | The documented infinity norm of an edgeless graph is returned as `1`. |

For migration counts, I used `rg --pcre2` over `modules`, `examples` and `website`, matching direct calls to `compute<Name>OnCPU` and `new (CPU|GPU)<Name>(Operation)?`, excluding function declarations. These are direct textual call sites. Required syntax edits and calls whose *results* can change are distinguished below.

### DGN-01: An all-missing cell enters the multiple-testing family

**Where.** `S/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation.ts:55` and `:67`, `gpu-pettitt-test.ts:171`, and `multiple-testing/cpu-multiple-testing-adjustment-operation.ts:40`. Both Pettitt implementations in KI retain the behavior. An actual composition wires the per-cell p-values directly into correction at `website/src/gpu/runs/entries/categories-cells-tests/raster-false-discovery-rate-mask.ts:61` and `:74`.

**Evidence and current use — EXECUTED.** From the API pages, the natural CPU composition is two calls:

```ts
const test = computePettittTestOnCPU(values, {
  cellCount: 2, sliceCount: 12, layout: 'cell-major'
});
const corrected = computeMultipleTestingAdjustmentOnCPU(test.pValues, 'bonferroni');
```

Cell A had six zeros followed by six ones. Cell B had twelve NaNs. Pettitt returned statistics `[36,0]`, change slices `[6,4294967295]`, and p-values `[0.03140780,1]`. Correction returned `[0.06281560,1]`, `testedCount = 2`. Replacing only the untested p-value with NaN returned `[0.03140780,NaN]`, `testedCount = 1`. The decision at 0.05 changes. This is a composition defect even though the change-slice sentinel is correct.

**Wanted.** Return a NaN p-value below two finite observations. Keep `p = 1` for a tested constant series. Document both cases together. The smallest GPU probe is the same two-cell cube passed through Pettitt and the raster correction, reading the significance mask and family count where available.

**In-repository callers that change.** Zero signature edits. **20 direct calls, four application calls** are exposed to the changed meaning. The existing CPU assertion for a single finite observation at `modules/analysis/test/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation.node.spec.ts:76` must change. Review the website correction and Kepler’s `catalog-spacetime-series.ts`.

**Value and cost.** Raster analysts can pass a p-value band between operators without reconstructing a support mask. **S.**

### DGN-02: A repeated geographic point gets a second vote at the seam

**Where.** `S/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts:149`, `cpu-trajectory-locations-operation.ts:123`, `gpu-trajectory-staypoints.ts:654`, and `gpu-trajectory-locations.ts:427`. The staged locations twin and shader retain raw-coordinate equality at KI lines `173` and `513`. The staged movement guide describes centers as means of distinct coordinates.

**Evidence and current use — EXECUTED.** I used one track on the equator, timestamps `[0,1,2]`, great-circle context, a 500 km threshold, and included the final staypoint. Both CPU functions returned:

| Longitudes | Center longitude |
|---|---:|
| `[180,180,177]` | `178.5` |
| `[180,-180,177]` | `179` |
| `[180,540,177]` | `179` |

All three inputs contain the same two physical points. With `[179.9,-179.9]`, both functions correctly produced `-180`. The error is the equality test before averaging. The KI locations twin reproduced `179`.

**Wanted.** In geographic mode, compare canonical longitudes for distinct voting, identify `-180` and `180`, and give exact poles a longitude-independent identity. Use that same representative when accumulating offsets. Preserve planar equality. **SOURCE GPU probe:** run the first two rows of the table through both contributors and require the same center, `178.5`.

**In-repository callers that change.** Zero signature edits. The distinct-coordinate rule changes for **32 staypoint and 21 location call sites**, including seven application call sites across Kepler and the website. Only inputs containing equivalent geographic representations should change.

**Value and cost.** A GIS developer can concatenate tracks exported on different longitude branches without moving a stop by tens of kilometres. **M.**

### DGN-03: Invalid track coordinates report a successful negative decision

**Where.** `S/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation.ts:373`, `:386` and `:390`. KI has the same unflagged `continue` after its coordinate check at `:409`. The existing `invalidPair` bit is `4` in `gpu-track-similarity.ts:156`.

**Evidence and current use — EXECUTED.** I compared a one-vertex track with itself, pair indices `[0],[0]`, `frechetEpsilon = [1]`, requesting `frechetWithin` and `hausdorff`. At `(5,90)` the result was `(within=1, distance=0, pairStatus=0)`. At `(NaN,NaN)` and `(5,91)` it was `(0,NaN,0)`. The KI twin also returned `(within=0, pairStatus=0)` for NaN. A caller requesting the decision alone sees an ordinary “outside tolerance.”

**Wanted.** Set a named invalid-coordinate bit, or extend `invalidPair` to cover coordinates, before leaving the CPU branch. State the decision value for flagged pairs. On the GPU, validate while traversing resident vertices and publish status for the same two-row probe. I did not establish the current device result.

**In-repository callers that change.** No required edits. Existing callers inspecting `pairStatus` gain a way to distinguish an invalid comparison.

**Value and cost.** Track matching can distinguish a negative answer from an unavailable one. **S for the CPU fix and contract.**

### DGN-04: Three equal neighbor distances can all be classified as outliers

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter.ts:184`, `:232`, `:250` and `:258`. The first-run example explicitly uses a multiplier of zero at `docs/api/README.md:238`.

**Evidence — SOURCE, with EXECUTED float32 arithmetic.** From the example, use squared distances `[11,11,11]`, counts `[1,1,1]`, capacity `1`, and `standardDeviations: 0`. The expected validity is `[1,1,1]`: every row equals the population mean. The kernel divides each term by the supported count before summing.

```ts
const f = Math.fround;
const distance = f(Math.sqrt(11));                  // 3.316624879837036
const term = f(distance / 3);                       // 1.1055415868759155
const threshold = f(f(term + term) + term);          // 3.316624641418457
const retained = distance <= threshold;             // false
```

**Wanted.** Preserve the exact constant-population case when forming the threshold, for example by centering the reduction on a supported value. Avoid a blanket tolerance that also changes genuinely unequal classifications. The smallest GPU confirmation is the three rows above, reading both means and validity. `[0,0,0]` is a prediction from source.

**In-repository callers that change.** No signature changes. The documented population-threshold definition stays the same.

**Value and cost.** Uniform point spacing can pass the documented mean-distance filter at multiplier zero. **S.**

### DGN-05: Missing coordinates become plausible radiation flows

**Where.** `S/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts:104` and `:138`. Invalid distances are placed in a separate list, then passed to `writeFlow(destination, 0)`.

**Evidence and current use — EXECUTED.** I used three unit populations, outflows `[10,0,0]`, finite-size normalization, and x coordinates `[0,1,x]` with y zero. For `x = 0.5`, the first row was `[0,2.5,7.5]`. For `x = 2`, it was `[0,7.5,2.5]`. For `x = NaN`, it was **`[0,7.5,7.5]`**. Neither the destination order nor the intervening population can be determined. The CPU publishes two finite estimates and provides no validity signal. The documented strict tie rule applies to known equal distances.

**Wanted.** Reject a nonfinite position in the pure function with a zone-indexed diagnostic. For resident input, publish an invalid-input result for affected origin rows. An unknown third zone can affect the flow between two otherwise valid zones. **SOURCE GPU probe:** use these same three completions and inspect the entire first row.

**In-repository callers that change.** None for valid inputs.

**Value and cost.** A missing centroid stops looking like an observed high-flow connection. **S for CPU rejection.**

### DGN-06: Finite-size radiation still loses the outside population on CPU

**Where.** `S/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts:68` and `:117`. The GPU correction is in `gpu-radiation-flow-model.ts:204`. **KNOWN(NET-111)** used `[1e9,1]`, for which the earlier report explicitly found the CPU result correct.

**Evidence and current use — EXECUTED.** With two distinct points, populations `[2**60,1]`, outflows `[100,100]` and `normalization:'finite-size'`, the CPU returned `[0,0,100,0]`. At `[2**20,1]` it returned `[0,100,100,0]`. Each origin has exactly one possible destination. In binary64, `1 - sourcePopulation / total` still rounds to zero for the larger input. Both input populations are exactly representable in float32.

**Wanted.** Accumulate outside-origin opportunities independently and evaluate the normalized expression using that quantity, as the public guide already promises. Add this larger CPU fixture beside the existing GPU regression.

**In-repository callers that change.** No edits required. This restores the documented finite-size formula.

**Value and cost.** The pure reference remains a valid oracle when opportunity weights have a large dynamic range. **S.**

### DGN-07: Longitude reduction can erase nearly half the globe

**Where.** `S/analysis-core/kernels/cpu-geodesic.ts:19` and `:31`, called by `S/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation.ts:179`. The KI twin uses the same reduction at `:320`. **KNOWN(DGGS-106)** concerns device-side cell indexing. This probe executes the CPU geodesic inverse.

**Evidence and current use — EXECUTED in both copies.** On the equator, `2**62` degrees to `0` gave `distance=0, converged=1, approximation=0`. Integer remainder gives `2**62 mod 360 = 184`, equivalent to `-176°`. Supplying `-176` gave **19,570,334 m**, with the same flags. The huge input is itself exactly representable in float32. The precision guide limits its measured envelope to canonical longitudes, but the function accepts this input as a resolved coincidence.

**Wanted.** Reduce each endpoint modulo 360 before subtracting outside the canonical branch. Preserve the precise short-seam path. If a backend cannot support a longitude magnitude, expose an invalid result at that magnitude. Compare these two equivalent endpoints in the GPU probe and check both distance and status.

**In-repository callers that change.** None required for the CPU arithmetic correction.

**Value and cost.** A finite wrapped coordinate cannot silently become a certified zero-length pair. **S.**

### DGN-08: Three collinear vertices trigger an unbounded pole search

**Where.** `S/gpu-spatial-analysis/label-point/cpu-label-point-operation.ts:142`, `:204`, `:206` and `:232`. Both `computeLabelPointOnCPU` and `computeMaximumInscribedCircleOnCPU` use the search. KI’s circle twin still calls it on its default unbudgeted route at `cpu-maximum-inscribed-circle-operation.ts:181`.

**Evidence and current use — EXECUTED.** One ring, offsets `[0,3]`, one polygon, one feature. `[(0,0),(1,0),(2,0)]` immediately produced label `(0,0)`, radius `0`, `degenerate=1`. The circle also returned radius zero. For `[(0,0),(1,1),(2,2)]`, both functions exceeded an **800 ms enforced execution deadline**, including the KI circle twin, with circle tolerance `0.000002`. The first unbounded label probe also passed ten seconds. This establishes pathological runtime on a three-vertex fixture. I did not establish mathematical nontermination.

The shortcut recognizes a collapsed x or y extent. A diagonal has positive extent on both axes and zero interior. The implementation scans all live cells to select a winner and again to prune after subdivision.

**Wanted.** Use an exact collinearity test on the finite float32 vertices before subdivision. Return the existing boundary fallback, zero radius and zero search error for a collinear feature. Keep this test distinct from a signed-area test, since signed areas can cancel across rings.

**In-repository callers that change.** None.

**Value and cost.** A collapsed polygon cannot occupy a CPU worker simply because a transform rotated it. **S.**

### DGN-09: Empty nested geometries require a fabricated feature or track

**Where.** `S/gpu-spatial-analysis/shared/cpu-polygon-rows.ts:45`, `geometry-refinement/cpu-canonical-geometry.ts:56`, `geometry-refinement/cpu-segment-snap-operation.ts:106`, `line-simplification/cpu-line-simplification-operation.ts:194`, `spatial-weights/cpu-contiguity-weights-operation.ts:39`, and `S/analysis-core/operations/cpu-trajectory-rows.ts:35`. The affected names are listed below.

**Evidence and current use — EXECUTED.** With `positions=[]` and `lineOffsets=[0]`, loop removal returned `lineOffsets=[0]` and zero status. Line simplification and segment snap threw `lineOffsets must contain at least two entries`. For polygons with all three offset arrays `[0]`, triangulation returned `triangleCount=[]` and `triangleOffsets=[0]`. Label point, maximum inscribed circle and shape descriptors threw `ringOffsets must contain at least two rows`.

For movement, `[0]` means zero tracks and `[0,0]` means one track with no fixes. Brownian bridge rejected the former and returned density `[0]` on a one-cell grid for the latter. Modal split rejected the former and returned `modalSplit=[0]` for the latter. Fabricating a track changes aggregate output cardinality. Trajectory Kalman accepted `[0]`.

**Wanted.** Accept the single terminal zero at each nesting level. Preserve `[0,0]` as a distinct input with one empty parent. Return `[0]` for an empty offsets output and preserve explicitly requested group or raster shapes. Check both cases in the same fixture.

**In-repository callers that change.** No edits required for existing valid calls. Reversed, out-of-range and missing terminal offsets still need their own checks.

**Value and cost.** Filtered Arrow geometry and trajectory batches can continue through a pipeline without inventing rows. **M** across the shared helpers and named forms.

### DGN-10: Zero work and an invalid network are conflated

**Where.** `S/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.ts:84`, `S/gpu-network/alternative-routes/cpu-alternative-routes-operation.ts:67`, and `S/analysis-core/operations/cpu-network-routing.ts:22`. The staged shared routing validator retains the two-offset minimum. Six regression consumers also share `S/gpu-spatial-analysis/spatial-regression/cpu-regression-weights.ts:37`, KI `:80`.

**Evidence and current use — EXECUTED.** I held a valid one-node network fixed with `offsets=[0,0]`, no edges, and passed zero queries. Path extraction returned `count=0`, `lineOffsets=[0]`, `pathFound=[]`. Service areas with zero facilities returned assignment `0xffffffff` and cost `Infinity`. K-shortest paths threw `sources.length must be an integer in [1, 4294967295]`. Alternative routes threw `sources must contain at least one query`.

Separately, for zero observations, CSR `[0]` really does start at zero and contain `rowCount + 1` entries. Spatial regression diagnostics, two-stage least squares, weights traces, error GM, error ML and lag ML all rejected it with a message saying it must satisfy exactly those two conditions.

**Wanted.** First support zero **queries** over a valid topology in both route producers. Then distinguish a zero-node topology from malformed CSR. For a fit whose result needs observations, return its existing insufficient-support state or name the minimum observation count in the error. Keep malformed offsets a separate error.

**In-repository callers that change.** None. Route counts, costs and certificates have zero query rows in the new case.

**Value and cost.** A routing client can submit the result of a filter directly. **S** for the query case, **M** for the shared zero-topology change.

### DGN-11: Zero-capacity status works until the contributor is given a graph name

**Where.** `S/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation.ts:50` and `cpu-delaunay-tessellation-operation.ts:231`, `S/gpu-network/network-analysis/gpu-network-path-extraction-operation.ts:154` and `cpu-network-path-extraction-operation.ts:243`. The port opt-in already exists at `S/analysis-core/operations/gpu-contributor-operation.ts:140`. The KI Delaunay forms reproduced the rejection.

**Evidence and current use — EXECUTED.** For a real triangle, the pure Delaunay function with capacity zero returned `requiredCount=1, count=0, overflow=1`. The raw contributor declared a graph at capacities zero and one. Both named Delaunay forms rejected zero with `outputs.triangles.length must be an integer in [1, 4294967295]`. Both named path-extraction forms did the same for `outputs.ids.length`. All four constructors accepted capacity one.

**Wanted.** List the compact payload ports in `emptyOutputPorts`, allow zero in the CPU wrapper’s own capacity check, and retain the mandatory scalar status ports. For path extraction, include optional row-aligned outputs when their declared lengths are zero. Confirm the count-only route through the executor with a single triangle and a one-node path.

**In-repository callers that change.** None.

**Value and cost.** Callers can size an allocation or inspect overflow through the same named graph they use for the real run. **S.**

### DGN-12: The point-buffer packer accepts a zero ring through underflow

**Where.** `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:58–67`. **KNOWN(CORE-101)** fixed short output targets in local-frame helpers. **KNOWN(ERG-102)** concerns a different point-buffer constructor diagnostic.

**Evidence and current use — EXECUTED.** `getGPUPointBufferParameterValues(0)` threw `GPUPointBuffer radius must be positive`. `getGPUPointBufferParameterValues(1e-50)` returned `[0,0]`. With radius `10`, target lengths `0`, `1`, `2` returned `[]`, `[10]`, `[10,0]` respectively. The one-short target was modified before being returned. The packer promises two words.

**Wanted.** Validate `Math.fround(radius) > 0` and target length `>= GPU_POINT_BUFFER_PARAMETER_LENGTH` before the first write. Name `radius after float32 conversion` or `target.length` in the corresponding message. The packer probe establishes the defect without a device. To confirm the downstream planar result on GPU, use one center, three ring vertices and the returned zero parameter.

**In-repository callers that change.** None for valid inputs and correctly sized targets.

**Value and cost.** The packer catches a collapsed buffer or truncated parameter block at its point of origin. **S.**

### DGN-13: A valid cell key can mean the clipped longitude

**Where.** `S/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts:82`, `:92`, `:178` and `:272`. The contributor source describes per-family clipping at `gpu-point-to-cell.ts:119` and `:126`. The API family table starts at `docs/api/gpu-spatial-analysis/operators-cells.md:242`. I checked its staged copy. **KNOWN(DGGS-106)** concerns huge-angle float32 reduction. The example here uses exactly representable integer degrees.

**Evidence and current use — EXECUTED.** At latitude `10`, resolution `5`, I indexed `-150°` and `210°`. H3 and S2 assigned the same key within each family. Quadkey returned low words **`174` and `170`**, Quadbin returned different high words, and geohash returned low words **`8711246` and `30801247`**. Every family returned `validity=[1,1]`. A GIS developer changing the `family` setting can silently place the second point in a different part of the world.

**Wanted.** Put a longitude-policy column in the family table and show normalization before the call for continuous-track input. Provide an opt-in `longitudePolicy: 'family-default' | 'wrap'` if the library should own this preparation for resident columns. Preserve the documented native clipping behavior by default.

**In-repository callers that change.** None for the docs or opt-in policy. A default change would expose 97 direct calls and is unnecessary.

**Value and cost.** Users can switch index families without discovering coordinate policy through key comparisons. **S** for the table and fixture.

### DGN-14: `infinityNorm` describes two quantities on edgeless graphs

**Where.** `S/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation.ts:150` and `:160`. `docs/api/gpu-spatial-analysis/operators-regression.md:648`, KI `:795`, describes the returned scalar as the largest absolute row sum. The KI CPU implementation retains the fallback at `:160`.

**Evidence and current use — EXECUTED in both copies.** For one node, `offsets=[0,0]`, no neighbors or weights, exact mode and `maximumPower=2`, the result was `powerTraces=[0,0]`, `powerSums=[0,0]`, **`infinityNorm=[1]`**. The largest absolute row sum is exactly zero. The CPU JSDoc does explain the use of one as a normalization scale. The public equation identifies that scale with the mathematical norm.

**Wanted.** Change the page to say “`infinityNorm` is the scale used to form powers. It equals the maximum absolute row sum when positive and is 1 for an all-zero matrix.” Apply the same wording to the bundled trace outputs of spatial lag ML. If consumers need the actual norm, expose it separately.

**In-repository callers that change.** None. Keeping the scale preserves the traces-to-impacts pipeline.

**Value and cost.** A developer inspecting an edgeless graph can interpret the scalar without reading the twin. **S.**

## Departures by name

The following is the complete named inventory for the **59-function zero-row cohort**. Names abbreviate `compute<Name>OnCPU`. This is a measured sample of the library, not an extrapolation to all 498 producers.

| Outcome | Names |
|---|---|
| **Returned a result, 19** | `CellMeasures`, `ConstrainedDelaunayTriangulation`, `ConvexStraightSkeleton`, `DelaunayTessellation`, `GeodesicPairs`, `HealpixPointToCell`, `LineLoopRemoval`, `MinimumBounds`, `NearestFeatureJoin`, `NormalizeGeometry`, `OrdinaryLeastSquares`, `PointToCell`, `PolygonTriangulation`, `ProbitRegression`, `RadiationFlowModel`, `RatioConcaveHull`, `SpatialPredicateJoin`, `TrajectoryKalman`, `VectorStraightSkeleton` |
| **Rejected empty nested geometry or tracks, 15** | `BrownianBridgeUtilization`, `ContiguityWeights`, `LabelPoint`, `LineSimplification`, `MaximumInscribedCircle`, `ModalSplit`, `NormalizePolygons`, `SegmentSnap`, `ShapeDescriptors`, `TrackSimilarity`, `TrajectoryRecursions`, `TrajectorySegments`, `TrajectorySplit`, `TrajectoryStaypoints`, `TrajectoryThinning` |
| **Rejected zero points, queries or centroid capacity, 5** | `ConditionalGaussianSimulation`, `OPTICS`, `PopulationSupportAlignment`, `SpatialCrossValidation`, `TrajectoryLocations` |
| **Rejected empty network, 10** | `AlternativeRoutes`, `KShortestPaths`, `NetworkCentrality`, `NetworkCostMatrix`, `NetworkCostTable`, `NetworkLouvain`, `NetworkPathExtraction`, `NetworkReachability`, `NetworkServiceAreas`, `NetworkStructure` |
| **Rejected empty adjacency, 4** | `MapColoring`, `NeighborhoodSummary`, `SpatialWeightsAlgebra`, `SpatialWeightsValidation` |
| **Rejected correct zero-row regression CSR, 6** | `SpatialErrorGM`, `SpatialErrorMaximumLikelihood`, `SpatialLagMaximumLikelihood`, `SpatialRegressionDiagnostics`, `SpatialTwoStageLeastSquares`, `SpatialWeightsTraces` |

A returned result need not have zero storage everywhere. `MinimumBounds` preserved the requested empty group with hull size zero and NaN bounds. OLS and probit returned model-sized diagnostics with insufficient-support status. These are useful precedents for fixed-shape outputs.

## Quick wins

- [ ] Add the two-cell Pettitt → correction fixture. Keep a tested constant cell as a separate control.
- [ ] Set invalid track-coordinate status at the branch that already detects the invalid vertex.
- [ ] Add the seam spelling substitution to both movement-center specs.
- [ ] Check the point-buffer radius after conversion and reject a short target before mutation.
- [ ] Short-circuit exact collinearity in the shared CPU polygon-pole search.
- [ ] Add zero-capacity Delaunay and path-extraction fixtures through both named forms, including status readback.
- [ ] Document the three cell-index longitude policies and the fallback scale for zero weights.

## Larger proposals

1. **Adopt logical empty batches before alpha.** Start with query-only and compact-output operations. Then update shared geometry, trajectory and CSR validators. Keep fixed model dimensions explicit and use the model’s insufficient-support result when observations disappear. The change predominantly adds accepted inputs. Cost **M** for the shared paths, **L** to qualify every producer on hardware.
2. **Change the missing Pettitt result to NaN.** The alternative is to require every correction consumer to combine `changeSlice` with `pValues`. The repository already contains a direct composition that omits this step. I recommend the breaking numerical change and the two-cell regression. Cost **S**.
3. **Define geographic distinctness once for movement.** Use one geographic equality and accumulation rule in both CPU and shader paths. Apply it to distinct voting while preserving original row IDs and source coordinates. Cost **M**.
4. **Check equivalent inputs, not just literal outputs.** Rotate a collapsed ring, substitute `-180` for `180`, append an all-missing cell, permute identical observations, and run at exact and one-short capacity. These small fixtures expose failures that an implementation and its twin can share.

## Not read

I read the shared CPU polygon-pole search, radiation-flow twin, map-coloring twin, transfer-mask contributor and the website Pettitt-to-correction wiring in full or through their complete computation paths. I sampled the remaining families by public entry, choosing pure twins for independently observable values and null-device declarations for families without twins. I checked the synthesis, area reports, API/catalog and direction reports, KI digest and decision list against the candidate findings. I excluded the asymmetric map-coloring diagnostic after finding its exact limitation documented in staging.

I did not compile shaders, run a GPU or browser, run the test suite, inspect every recipe, or exhaust every combination of boundary and producer. GPU predictions above include the fixture needed to confirm them.

**Operational limit.** The initial unbounded diagonal label-point probe was still present as Node PID `28095` at the last check. The sandbox denied termination of that process. Subsequent reproductions used an enforced 800 ms JavaScript deadline. No project files were written.