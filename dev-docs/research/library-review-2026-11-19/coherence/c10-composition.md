# Composition review — plugging one analysis result into the next

## Summary

All six journeys can run on resident buffers when the caller starts with the necessary row identities, time coordinates and grid metadata. I declared all six on the CPU-only graph device. No GPU result is claimed from that exercise.

The most consequential failure was a capacity-limited group table feeding a key join and regression. Dropping one group changed the fitted slope from **10.2 to 2**, with solver status `0` in both cases.

Two ordinary GIS representations also changed answers. A point and a one-part multipoint at the same coordinate produced different polygon counts. An absent cell gathered from a sparse count table became **4,294,967,296 observations** at the hot-spot input.

The hardest journey was stops to emerging hot spots. Its obstacles are recovering event time, excluding unused stop slots, retaining the period during aggregation, and placing each slice on the same cell dictionary. Transposition alone solves only the last layout change.

The most valuable pre-alpha change is to expose a **consumable result port** where a bare view loses meaning. `GPUContiguityWeights.outputWeights` is an existing example. A consumer should receive the readable rows, their identity and the producer’s completeness evidence together.

## Census

`R` denotes this checkout. `K` denotes the corresponding file in `../luma-contributors-ki` when listed in `.typecheck/r19/staging-owned.txt`. Source paths below start at `modules/analysis/src`. I abbreviate `gpu-spatial-analysis/`, `gpu-network/`, `raster-analysis/`, `gpu-terrain/` and `dataframe-analysis/` as `S/`, `N/`, `Ras/`, `Ter/` and `D/`. The checkout was being edited during review.

I used `rg --files modules/analysis/src`, substituted the staging path by exact filename, then parsed the files with `typescript.createSourceFile` in Node. I counted exported `GPU*` class declarations with their own `getCommandNodes` member and classified their declared public members with `/^output([A-Z]|$)/`. This excludes inherited graph methods and measures discoverability on the class, not the existence of caller-owned output buffers.

| Declared public member | Classes | Interpretation |
|---|---:|---|
| `outputWeights` | 15 | A named port the next contributor can consume |
| `output` | 3 | `GPUAnalysisSpacing`, `GPURasterExtremaPyramid`, `GPUSpatialWeightsMinimumSpanningTree` |
| `outputSpatialContext` or `outputTemporalContext` | 3 | `GPUCellGeometry`, `GPUCellSetOutline`, `GPUTimeWordsRelative` |
| No member matching the expression | 475 | Includes useful outputs under `props.output` and other names |
| **Total** | **496** | Of 25 principal journey classes inspected, only `GPUContiguityWeights` declared a matching member |

The 15 weights producers were `GPUContiguityWeights`, `GPUFuzzyContiguity`, `GPUIslandAttachment`, `GPULatticeWeights`, `GPUNearestFeatureWeights`, `GPUNeighborSearch`, `GPUNetworkCostWeights`, `GPUSpatialPredicateJoin`, `GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsTransform`, `GPUSpatialWeightsTranspose`, `GPUSpatialWeightsValidation`, `GPUChunkedSpatialWeightsAlgebra`, `GPUChunkedSpatialWeightsTransform` and `GPUChunkedSpatialWeightsTranspose`.

For the six journeys, the distinction that predicted trouble was whether **identical physical types could denote different data**.

| Journey | Direct handoff | Glue the types do not supply | Final evidence |
|---|---|---|---|
| 1. Points to corrected local Moran | Feature rows to zonal aggregation, polygon CSR and feature means to inference, raw pseudo-p to correction | Select rows versus IDs, preserve assignment loss independently of weights loss | Correction alone cannot distinguish missing input from untested hypotheses |
| 2. Roads to accessibility and allocation | Noded CSR to search, demand-by-facility costs to allocation | Live node prefix, noding loss, matrix axes, forward versus reverse travel | Both scorers can carry a matrix certificate. Search convergence alone says nothing about the road build |
| 3. Raster to polygon cost statistics | Elevation band to derivatives, wrapped slope to friction, polygon rows to zones | Choose a friction model and combine two independent producer flags | Whole-cell zonal has one suppression input for the combined flag |
| 4. Stops to emerging hot spots | Stop centroids to point indexing, cell-major counts to Gi* | Stop mask and time gather, stable cell dictionary, zero fill, period layout | Stop-search loss, stop capacity and cell-table loss must accompany the categories |
| 5. Points to classified density colours | Exact grid prefix to breaks, breaks and class count to colour scale | Bound the readable grid and distinguish class count from edge count | Colours carry neither KDE exclusions nor approximation diagnostics |
| 6. Groups to regression | Group keys to key join, one gathered float predictor to OLS | Group-capacity guard, left-row alignment, pack multiple predictors row-major | Solver success is local to the surviving regression rows |

## Proposed convention

A public producer whose allocation admits padding, external IDs, multiple axes, missing sentinels or bounded loss should expose a borrowed result port carrying the logical shape, row identity, missing-value rule and applicable status views. A consumer or named adapter should accept that port and preserve supplied loss through its final result.

Check each advertised composition with a graph-declaration fixture and a small adversarial data fixture. Use permuted IDs, asymmetric axes, nonzero padding, an absent key and a frame with producer loss followed by a clean frame. Equal lengths and successful graph construction are insufficient checks.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| CMP-01 | W | ORDINARY | ADDITIVE | M | Group statistics → key join → OLS | Group overflow becomes ordinary missing data and changes a successful fit |
| CMP-02 | C/W | ORDINARY | ADDITIVE | M | Point-in-polygon → zonal counts | A one-part multipoint changes the definition of `featureCounts` |
| CMP-03 | W | ORDINARY | ADDITIVE | S | Key join → emerging hot spots | An absent unsigned count becomes a finite four-billion-event bin |
| CMP-04 | W | ORDINARY | ADDITIVE | S | Staypoints → cell aggregation | Unused stop slots are valid geographic coordinates |
| CMP-05 | W | ORDINARY | ADDITIVE | S | KDE → class breaks | Spare raster capacity changes every class boundary |
| CMP-06 | W | ORDINARY | ADDITIVE | M | Network matrix → two scorers | An accepted square matrix selects a different facility when its axes are reversed |
| CMP-07 | W | UNUSUAL | ADDITIVE | M | Zonal statistics → local Moran | Complete weights can certify a statistic computed from incomplete feature values |
| CMP-08 | W | UNUSUAL | ADDITIVE | M | Network noding → matrix | An unused node slot has zero travel cost to itself |
| CMP-09 | W | UNUSUAL | ADDITIVE | S | Cost distance → raster zonal | A failed solve and a completed unreachable zone have identical statistics |
| CMP-10 | D/W | ORDINARY | ADDITIVE | S | Polygon rasterization → raster zonal docs | Copying `ignoredZone: 0` discards the first polygon |
| CMP-11 | E | ORDINARY | ADDITIVE | S | Joined columns → multivariable OLS | Concatenating two columns passes the shape check and fits the wrong model |
| CMP-12 | E | ORDINARY | ADDITIVE | M | Cell aggregation → temporal cube | An all-period cell table has already discarded the time axis |
| CMP-13 | C/E | ORDINARY | BREAKING | S | Class breaks → colour scale | `domainCount` counts edges in settings and classes on the resident port |
| CMP-14 | C | UNUSUAL | ADDITIVE | S | Permutations → multiple testing | `testedCount` can exceed the number of rows with a test result |

### Six end-to-end journeys

These are wiring excerpts. Identifiers such as `means`, `parameters` and `status` stand for caller-owned graph views of the indicated producer or consumer type. Each journey uses its own graph. The probes used `GPUCommandGraph`, `createTransientView` and the repository’s `createNullWebGPUDevice`. The graph accepted all six routes, including the guarded two-predictor fit. I checked the excerpts against source props and exercised construction without running `tsc`, compiling kernels or submitting commands.

**1. Points → polygons → zonal means and polygon weights → local Moran → permutations → correction**

```ts
// Omit polygon sourceIds on this route so the assignment is a feature ROW.
const polygons = {kind: 'polygons' as const, positions, featureOffsets, polygonOffsets, ringOffsets};
graph.add(new GPUPointInPolygonJoin({
  points: {kind: 'points', positions: pointPositions}, polygons, candidateCapacity,
  output: {pointFeatureIds: featureRows, status: joinStatus}
}));
graph.add(new GPUZonalStatistics({
  features: {kind: 'feature-rows', pointFeatureRows: featureRows, featureCount, status: joinStatus},
  values: pointValues, output: {means, status: zonalStatus}
}));
const adjacency = new GPUContiguityWeights({criterion: 'queen', polygons, output: csrOutput});
graph.add(adjacency);
const weights = adjacency.outputWeights;

graph.add(new GPULocalMoran({
  values: means, weights, mask: analysisMask, parameters: analyticParameters,
  output: {zScores, localI}
}));
graph.add(new GPULocalPermutationTest({
  statistic: 'localMoran', values: means, weights, mask: analysisMask,
  parameters: permutationParameters, maximumPermutations: 99,
  output: {exceedances, pseudoPValues, status: permutationStatus}
}));
graph.add(new GPUMultipleTestingAdjustment({
  pValues: pseudoPValues, method: 'benjamini-yekutieli',
  output: {adjustedPValues, testedCount}
}));
```

There are two branches. Contiguity reads polygon geometry. Both Moran contributors read the **same original feature values**. Passing `localI` as `GPULocalPermutationTest.values` would permute an already-computed statistic. If only the permutation estimate and adjusted probabilities are wanted, request `observed` and `adjustedPValues` on the permutation contributor and set its `multipleTesting` method. The analytic pass and standalone correction then disappear.

All statistical columns are one element per polygon. The joins and CSR need their own capacities. In the guarded declaration I ORed the planar zonal loss words, negated the scalar with public `GPUMask`, and broadcast it to `analysisMask` with `GPUGather` from one scalar using a zero-filled index column. Keep the original failure word alongside the adjusted probabilities. A family with no tests does not say why it had no tests. `KNOWN(GEO-114, GEO-116)` covers assignment IDs and the earlier missing status input to zonal. CMP-07 concerns the *following* arrow.

**2. Network build → shortest-path cost matrix → location allocation and catchment accessibility**

```ts
graph.add(new GPUNetworkNoding({
  lines, tolerance, intersectionCapacity,
  output: {pieces, nodes, edges, topology, status: buildStatus}
}));
graph.add(new GPUNetworkCostMatrix({
  topology, startNodes: demandNodes, targetNodes: facilityNodes, costLimit,
  output: {costs: demandByFacility, status: {converged}}
}));
graph.add(new GPUMask({
  id: 'route-complete', operation: 'difference',
  inputs: [converged, buildOverflow, buildUncertainCount], output: routeComplete
}));
graph.add(new GPULocationAllocation({
  costs: demandByFacility, facilityCount, demandWeights, maximumFacilityCount: 1,
  operation: 'p-median', costsComplete: routeComplete, costLimit, output: allocation
}));
graph.add(new GPUTranspose({
  id: 'facility-by-demand', rows: demandCount, columns: facilityCount,
  input: demandByFacility, output: facilityByDemand
}));
graph.add(new GPUNetworkAccessibility({
  orientation: 'opportunity-rows', costs: facilityByDemand, opportunityWeights: supply,
  parameters: scoringParameters, costsComplete: routeComplete, costLimit,
  catchment: {method: '2sfca', demand: demandWeights, output: accessibility},
  status: {complete: accessibilityComplete}
}));
```

`buildOverflow` and `buildUncertainCount` are the destinations in `buildStatus`. Starts and targets must already be **live node rows**. With named facilities, column `i` is the row of `facilityNodes[i]` in the target list. `allocation.assignments` indexes that list. For shared resident costs the extra work is one transpose and one `D × F` float buffer. Running from facilities on the reverse CSR produces the catchment orientation directly. On a directed network, transposing a matrix rearranges the same trips. Searching the forward CSR from the other endpoint asks a different travel question.

Use an unbounded matrix for ordinary p-median. A finite horizon can be adequate for threshold catchment while insufficient for p-median. I also executed the staged CPU matrix with `costLimit=[NaN]`. It completed the same unrestricted search as `+Infinity`, but forwarding that NaN to allocation cleared its `complete` word. The accessibility source expressly documents that conservative convention. `+Infinity` avoids the disagreement. `KNOWN(NET-101, NET-124)` covers the earlier matrix-to-score completeness defects.

**3. Raster → terrain slope → cost distance → whole-cell statistics by polygon**

```ts
const grid = raster.metadata;
graph.add(new GPUTerrainDerivatives({
  grid, elevation: raster.getBand('dem'), slopeUnits: 'degrees',
  parameters: terrainParameters, output: {slope, outputValidity: slopeValid}
}));
// Example model: one base cost unit per ground unit, plus one per degree of slope.
const friction = makeGPURasterBand(slope, grid, {id: 'friction', validity: slopeValid, offset: 1});
graph.add(new GPUCostDistance({
  grid, friction, sources: sourceCellRows, parameters: costParameters,
  output: {costs, status: {iterationLimitReached}}
}));
graph.add(new GPUPolygonRasterization({
  grid, parameters: polygonParameters, positions, featureOffsets, polygonOffsets, ringOffsets,
  crossingCapacity, output: {zones, status: {overflow: polygonOverflow}}
}));
graph.add(new GPUMask({
  id: 'cost-zone-loss', operation: 'or', inputs: [iterationLimitReached, polygonOverflow],
  output: incompleteInput
}));
graph.add(new GPURasterZonalStatistics({
  grid, zones, zoneCapacity: featureCount, producerOverflow: incompleteInput,
  values: makeGPURasterBand(costs, grid),
  output: {cellCounts, valueCounts, means, status: {incomplete}}
}));
```

All three raster values share the same grid and row-major cell order. `makeGPURasterBand` borrows the active prefix and preserves validity. No resampling or copy is needed. The friction formula is an explicit modelling choice. Directly assigning slope degrees as cost makes level ground free. The polygon path produces dense feature rows with `0xffffffff` outside. `valueCounts` measures finite reachable cells, while `cellCounts` measures polygon support. The example requests costs alone. With back-links enabled, the cost contributor’s overall iteration flag can also include an unfinished tie phase.

**4. Trajectories → stops → cell aggregation → emerging hot spots**

This version uses Unix Int64 millisecond timestamps, assigns a stop to its **start** period, and counts stops. `dictionary` contains one fixed 64-bit key per analysis cell, in precisely the row order of `weights`. Each `periodLabels[t]` is a one-row `uint32` view containing `t`. `fixOnes` contains one `1` per input fix.

```ts
graph.add(new GPUTrajectoryStaypoints({
  positions, timestamps, trackOffsets, temporalContext, parameters: stayParameters,
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
  output: {ids: trackIds, startRows, centroids, status: stopStatus}
}));
graph.add(new GPUGather({id: 'stop-time', source: timestamps, indices: startRows, output: stopTimes}));
graph.add(new GPUGather({id: 'live-stop', source: fixOnes, indices: startRows, output: liveStops}));
graph.add(new GPUPeriodIndex({
  temporalContext, timestamps: stopTimes, mask: liveStops,
  periods: {kind: 'fixed', width: 86400000, origin, count: T}, output: {periods}
}));
graph.add(new GPUPointToCell({
  family: 'quadbin', resolution, positions: centroids, mask: liveStops, output: {cells: stopKeys}
}));

for (let t = 0; t < T; t++) {
  const {mask, table, rightRows, plane} = perPeriod[t];
  graph.add(new GPUKeyJoin({
    id: `period-${t}`, leftKeys: periods, rightKeys: periodLabels[t],
    leftMask: liveStops, output: {matched: mask}
  }));
  graph.add(new GPUCellAggregation({
    id: `cells-${t}`, family: 'quadbin', resolution, cells: stopKeys, mask, output: table
  }));
  graph.add(new GPUKeyJoin({
    id: `lookup-${t}`, leftKeys: dictionary, rightKeys: table.cells, output: {rightRows}
  }));
  graph.add(new GPUGather({
    id: `counts-${t}`, source: table.counts, indices: rightRows, output: plane
  }));
}
graph.add(new GPUCubeLayoutTranspose({
  values: periodMajor, from: 'period-major', cellCount: C, periodCount: T, output: cellMajor
}));
graph.add(new GPUEmergingHotSpots({
  values: cellMajor, weights, sliceCount: T, parameters: hotSpotParameters, output: hotSpots
}));
```

`plane` is a `C`-row subview of `periodMajor` starting at byte offset `periodMajor.byteOffset + 4*t*C`. The second join supplies dictionary-aligned **row indices**. Public upstream `GPUGather` writes zero for an out-of-range index, the desired count for an observed period with no stops in a cell. `GPUKeyJoin.gather` has a different unsigned fill. The `GPUEmergingHotSpots.mask` is per cell. It cannot describe an individually unobserved cell-period. For unknown bins use a float cube with NaN there.

This literal route costs two initial gathers, four declarations per slice, and a transpose, in addition to time and point indexing. On a fixed dictionary the shorter route is `GPUPointToCell → GPUKeyJoin.rightRows → GPUValueCube.counts → GPUEmergingHotSpots`. `GPUValueCube` accepts the per-event dense cell row and period, and writes the required `cell*T + period` layout directly. Supply a finite value column, such as ones, for a pure count. The existing space-time recipe also bins events directly. Neither route reconstructs periods from an already collapsed cell table.

Keep stop final overflow, stop candidate overflow and per-slice table overflow beside the result. The count cube and categories alone contain none of those words. A relative-time trajectory also needs an explicit epoch/unit conversion or the original epoch-time column before calendar slicing.

**5. Points → kernel density → classification → colour scale**

```ts
graph.add(new GPUKernelDensity({
  positions, grid, indexGridSize, indexBounds, parameters: densityParameters,
  output: {values: densityAllocation, status: densityStatus}
}));
const density = makeGPURasterBand(densityAllocation, grid).storage.values;
graph.add(new GPUClassBreaks({
  values: density, methods: ['equal-interval'], maximumClassCount: K,
  parameters: breakParameters, output: {breaks, classCount}
}));
graph.add(new GPUColorScale({
  values: density, domain: breaks, domainCount: classCount, palette,
  maximumDomainCount: K + 1, maximumPaletteCount: K,
  parameters: colorParameters, output: {colors, classIndices}
}));
```

The classification input is `width*height` scalar floats in grid order. It needs no conversion or readback. The colour scale reads the same pixels, the `k+1` edge values and the resident class count `k`. Use a threshold scale for these edges. The packed settings still require a numeric `domainCount` even on the resident-count route. Preserve `densityStatus` separately, including `invalidCount` and, on a binned route where requested, `insufficientPadding` and `bandwidthBelowThreeCells`. A coloured pixel is not evidence that the density estimate met its approximation conditions.

**6. Dataframe group statistics → key join → regression**

```ts
graph.add(new GPUGroupStatistics({
  keys: sourceKeys,
  columns: [
    {values: sourceX, statistics: ['mean'], output: {means: groupX}},
    {values: sourceY, statistics: ['mean'], output: {means: groupY}}
  ],
  output: {keys: groupKeys, counts: groupCounts, status: groupStatus}
}));
graph.add(new GPUKeyJoin({
  leftKeys: observationKeys, rightKeys: groupKeys, output: {rightRows, matched}
}));
for (const [j, column] of [groupX, groupY].entries()) {
  graph.add(new GPUGather({
    id: `predictor-${j}`, source: column, indices: rightRows, output: columnPlanes[j]
  }));
}
graph.add(new GPUTranspose({
  id: 'ols-design', rows: 2, columns: N, input: columnMajor, output: predictors
}));
graph.add(new GPUOrdinaryLeastSquares({
  predictors, response, predictorCount: 2, mask: trainingMask, output: fit
}));
```

`response` and `matched` have `N` rows in **left observation order**. `columnPlanes[j]` is the `N`-float subview of `columnMajor` beginning at `j*N`. The transpose produces `predictors[row*2 + j]`. For one predictor, `GPUKeyJoin.gather: [{column: groupX, output: predictors}]` removes the separate gather and transpose. The graded two-column case used `trainingMask = matched AND broadcast(!groupStatus.overflow)`. Retain the group flag so a suppressed model can be distinguished from a genuinely undersized training set.

### CMP-01 — Group loss becomes a successful fit

**Where.** R `D/group-statistics/gpu-group-statistics.ts:199`, R `D/key-join/gpu-key-join.ts:142,822`, K `S/spatial-regression/gpu-ordinary-least-squares.ts:108,277`.

**Evidence — EXECUTED.** I chained the repository group and key-join CPU oracles into the public OLS twin. Keys and predictor values were `[0,1,2,3,4]`, response was `[1,3,5,7,50]`. Group capacity 5 gave coefficients `[-7.1999998,10.1999998]`. Capacity 4 still reported `requiredCount=5`, but the joined predictor became `[0,1,2,3,NaN]` and the fit returned `[1,2]`. Both solver codes were `0`. Passing `mask: matched` reproduced the loss exactly.

**Now → wanted.** `rightKeys: groupKeys` carries the retained keys. Add a grouped-table input such as `rightTable: {keys, status}` to the join and a composition helper that carries its completeness into the fit. Preserve the distinction between a genuinely unmatched observation and a missing right key caused by truncation. **Callers changing:** none required. **Value/cost:** regression users can detect a changed training population, M.

### CMP-02 — Point representation changes what a polygon count counts

**Where.** K `S/spatial-join/gpu-point-in-polygon-join.ts:176,1409,1441`, K `S/spatial-join/cpu-point-in-polygon-join-operation.ts:128`, R `S/zonal-statistics/gpu-zonal-statistics.ts:123`.

**Evidence — EXECUTED.** Two identical square polygons, one interior point `(1,1)`, candidate capacity 2. With ordinary points, `pointFeatureIds=[0]`, `featurePolygonMembership=[1,1]`, `featureCounts=[1,0]`. Adding only `points.featureOffsets=[0,1]` yielded the same assignment and membership, but `featureCounts=[1,1]`. Both overflow flags were zero. Grouping the returned assignment gives `[1,0]` in both cases.

**Now → wanted.** `counts: joinFeatureCounts` cannot safely substitute for counts from zonal first-owner rows. Publish `assignedFeatureCounts` and `intersectingFeatureCounts` with stable meanings on both representations. Use the former in the zonal journey. This is independent of the ID/row ambiguity in `KNOWN(GEO-114)`. **Callers changing:** none for additive ports. **Value/cost:** GIS users can choose exclusive assignment or overlapping membership explicitly, M.

### CMP-03 — A sparse unsigned count has the wrong empty-bin fill

**Where.** R `D/key-join/gpu-key-join.ts:84,822`, K `S/emerging-hot-spots/gpu-emerging-hot-spots.ts:137,663`. Public alternative in `@luma.gl/graph` `gpu-core/gpu-gather.ts:50`.

**Evidence — EXECUTED and SOURCE.** Key-join oracle with left cell keys `[10,20]`, right keys `[20]`, right count `[3]` returned gathered counts `[4294967295,3]`. Conversion to float32 gave `[4294967296,3]`. The hot-spot kernel reads every unsigned bin with `f32(values[…])`. A device probe needs just this pair of cells over two slices. Compare global moments with explicit `[0,3]` for the sparse slice.

**Now → wanted.** `gather: [{column: table.counts, output: plane}]` is the natural spelling and has the wrong fill here. Add `unmatchedValue?: number` to a gather column, validated against its format. Document `rightRows → GPUGather` as the existing zero-fill route. **Callers changing:** none. **Value/cost:** absent cells no longer become the largest observations in the study, S.

### CMP-04 — The tail of a stop table maps to a real place

**Where.** R `S/trajectory-staypoints/gpu-trajectory-staypoints.ts:151,640,664`, R `S/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts:127`, R `S/cell-aggregation/gpu-cell-aggregation.ts:94,499`.

**Evidence — EXECUTED.** Three stationary fixes at `(12,30)`, times `[0,2,4]`, distance and time thresholds `1`, `includeLast=true`, capacity 3. The staypoint twin returned `count=1`, `startRows=[0,0xffffffff,0xffffffff]`, centroids `[(12,30),(0,0),(0,0)]`. The resolution-6 Quadbin oracle produced counts **1 at the stop and 2 at the geographic origin**. Using `[1,0,0]` as the mask removed the second cell.

**Now → wanted.** `positions: centroids` is both type-correct and insufficient. Offer `output.validity` per compact stop or a documented adapter from `startRows`/`status.count` to a live-row mask. The stop-specific mask in `KNOWN(RCP-03)` fixed a fleet recipe. The direct cell consumer still needs this connection. **Callers changing:** none. **Value/cost:** movement users avoid inventing stops, S.

### CMP-05 — A padded density grid changes the map legend

**Where.** R `S/point-density/gpu-kernel-density.ts:210,607,1164`, R `D/column-classification/gpu-class-breaks.ts:319`, R `Ras/raster-grid/gpu-raster-grid.ts:87`.

**Evidence — EXECUTED and SOURCE.** A declared 2×2 KDE with a five-float destination fed a class-breaks object accepting **five** rows. The class-breaks twin on `[2,4,6,8]` produced edges `[2,5,8]`, counts `[2,2]`. Adding a spare `1000` produced `[2,501,1000]`, counts `[4,1]`. The prefix route declared four rows. To confirm the shader handoff, prefill a five-row allocation’s tail with `1000`, run the 2×2 KDE, and compare breaks on the full view and its four-row prefix.

**Now → wanted.** `values: kde.props.output.values` exposes allocation length. Return a grid-sized `outputBand` or document `makeGPURasterBand(allocation, grid).storage.values` at the classification example. The latter already preserves byte offset with no copy. **Callers changing:** none. **Value/cost:** raster users can reuse larger buffers without changing the legend, S.

### CMP-06 — Matrix orientation can reverse the chosen facility

**Where.** K `N/network-accessibility/gpu-network-cost-matrix.ts:76,168`, R `N/network-accessibility/gpu-network-accessibility.ts:268,293`, R `S/location-allocation/gpu-location-allocation.ts:172`.

**Evidence — EXECUTED.** A directed network with demands `0,1`, facilities `2,3` and edge costs `0→2:3`, `0→3:10`, `1→2:1`, `1→3:1` produced demand-major `[3,10,1,1]`. P-median selected facility 0, objective 4. Passing the facility-major representation `[3,1,10,1]` unchanged selected facility 1, objective 2. Both certificates said complete. The correct answer to the original problem is facility 0.

**Now → wanted.** `costs: matrix` loses both axis identities. Expose a matrix result containing `values`, `rowCount`, `columnCount`, start/target row maps, travel direction, convergence and horizon. Provide named allocation and catchment adapters. They can reuse public `GPUTranspose` when needed. **Callers changing:** none for an overload. **Value/cost:** planners avoid selecting a different site from the same network, M.

### CMP-07 — Membership loss survives zonal aggregation but disappears at Moran

**Where.** R `S/zonal-statistics/gpu-zonal-statistics.ts:136,198`, K `S/spatial-autocorrelation/gpu-local-moran.ts:140`, K `S/permutation-inference/gpu-local-permutation-test.ts:314`, K `S/multiple-testing/gpu-multiple-testing-adjustment.ts:84`.

**Evidence — EXECUTED.** Four adjacent polygons, five points, one point per polygon plus a second in the last polygon. Point values `[1,3,2,4,100]`. The public join twin at capacities 4 and 5 returned assignment tails `NO_FEATURE` and `3`, respectively, with candidate loss `1` and `0`. The arithmetic means were `[1,3,2,4]` and `[1,3,2,52]`. On the same complete chain CSR, the local-permutation oracle changed observed Moran at row 0 from **−0.45 to +0.248135**. It returned finite pseudo-p values for both datasets. In this small fixture all BY-adjusted values were 1.

**Now → wanted.** `values: means, weights: adjacency.outputWeights` guards the adjacency branch only. Accept a scalar `valuesComplete` in the local statistic and permutation contributors, or supply a public composition helper that builds their common mask and publishes the original membership failure alongside correction. Keep per-point truncation indexed by points. It cannot directly serve as a per-polygon mask. **Callers changing:** none. **Value/cost:** inference users can distinguish incomplete observations from weak spatial association, M.

### CMP-08 — Search uses node capacity as its universe

**Where.** K `N/network-noding/gpu-network-noding.ts:405,415,889`, `analysis-core/contracts/network-topology.ts:24,77`, K `N/network-accessibility/gpu-network-cost-matrix.ts:196`.

**Evidence — EXECUTED.** For two live nodes joined in both directions and capacity 3, the padded CSR was `offsets=[0,1,2,2]`, `neighbors=[1,0]`, costs `[1,1]`. A matrix from start `[2]` to target `[2]` returned **`[0]`**. That is correct for an explicitly supplied three-node graph with an isolated node. It is a false reachable pair when the third row is spare space in the noding result. A device probe should build one segment with capacity 3, retain `nodes.status.count`, then query the first spare row.

**Now → wanted.** `topology: noding.props.output.topology` drops the live node count and build diagnostics. Add a noded-topology port carrying `nodes.status.count` and the build status. Matrix construction from that port should exclude start and target rows beyond the live prefix, and distinguish search convergence from source-network completeness. `KNOWN(NET-116)` concerns unsafe edge offsets in different producers. **Callers changing:** none. **Value/cost:** routing users avoid treating capacity slots as isolated facilities, M.

### CMP-09 — Unfinished and unreachable zones have identical summaries

**Where.** K `Ras/cost-distance/gpu-cost-distance.ts:131,177`, R `Ras/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:232,891`, R `Ras/raster-zonal-statistics/raster-producer-completeness.ts:72`.

**Evidence — EXECUTED.** A 32×1 unit-friction row, all cells in zone 0. A one-round CPU cost solve from cell 0 produced NaN costs and `iterationLimitReached=1`. A completed solve with no seeds produced infinite costs and flag `0`. The zonal oracle returned **`cellCounts=[32], valueCounts=[0], sums=[0], means=[NaN], overflow=0` for both**. The unrestricted seeded control returned `valueCounts=[32], sums=[496], means=[15.5]`.

**Now → wanted.** `producerOverflow: polygonOverflow` guards the zone raster only. Document the one-node OR shown in journey 3 and expose an additive `incompleteSources` input for multi-producer compositions. `KNOWN(RT-137)` already motivated the rasterization guard. The independent cost-value branch is the addition here. **Callers changing:** none. **Value/cost:** raster analysts can distinguish a failed calculation from no reachable cells, S.

### CMP-10 — The zonal example drops polygon row zero

**Where.** K `docs/api/gpu-raster/operations-analysis.md:335` and R copy at `:292`, R `Ras/polygon-rasterization/gpu-polygon-rasterization.ts:62,669`, R `Ras/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:442,891`.

**Evidence — EXECUTED.** One unit-square polygon on the first pixel of a two-pixel grid. Rasterization oracle returned `[0,4294967295]`. With values `[7,42]`, the copied `ignoredZone: 0` gave `means=[NaN], valueCounts=[0], overflow=1`. Omitting that setting gave **`means=[7], valueCounts=[1], overflow=0`**. On a grid wholly inside polygon zero, even the out-of-range background warning disappears.

**Now → wanted.** Change the example to `zoneCapacity: featureCount` and omit `ignoredZone`. Add one sentence reserving `ignoredZone: 0` for a source raster that actually encodes background as zero. **Callers changing:** none. **Value/cost:** the first polygon survives the documented handoff, S.

### CMP-11 — A column stack is accepted as the wrong regression design

**Where.** R `D/key-join/gpu-key-join.ts:79,158`, K `S/spatial-regression/gpu-ordinary-least-squares.ts:108,277,470`, K `S/spatial-regression/cpu-ordinary-least-squares-operation.ts:141`.

**Evidence — EXECUTED.** Columns `x=[0,1,0,1,2]`, `z=[0,0,1,1,1]`, response `[1,3,4,6,8]`. Interleaved predictors recovered `[intercept,x,z]=[1,2,3]`. Concatenating the columns produced `[-5.3333335,4.6666665,8.6666670]`. Both fits returned code `0`. I also declared the two-gather and transpose solution in journey 6.

**Now → wanted.** `predictors.length === N*P` accepts both layouts. Publish a regression-design adapter taking `readonly GraphDataView<'float32'>[]` in coefficient order. Lower it with existing graph primitives. For a single column return that view directly. **Callers changing:** none. **Value/cost:** dataframe users can fit multiple predictors in their existing column representation, S.

### CMP-12 — An aggregate cell table cannot reconstruct a temporal cube

**Where.** R `S/cell-aggregation/cell-table.ts:34`, R `S/recipes/space-time-hot-spots-recipe.ts:361,381`, K `D/value-cube/gpu-value-cube.ts:65,95`, K `S/emerging-hot-spots/gpu-emerging-hot-spots.ts:136`.

**Evidence — COUNT and SOURCE.** The cell table has one row per occupied **key**, sorted by key. The cube requires one value per **fixed cell row and period**. A cell with one stop on Monday and one with one stop on Tuesday produce the same single-cell, count-1 table if period was discarded. Per-period tables retain time but can have different occupied keys at row zero. The fully public literal solution declared successfully with `C=4`, `T=2`, compact stop capacity 3. It needed four contributors per period plus the final transpose.

**Now → wanted.** `values: cellTable.counts` is insufficient, even when its length happens to fit. Put the short `rightRows → GPUValueCube.counts` route on the stop and cell API pages. Offer a stop-to-cube recipe with an explicit time choice (`start` or `finish`), fixed dictionary, weights in dictionary order, and named upstream loss. Document the literal per-slice route for callers who already have those tables. **Callers changing:** none. **Value/cost:** mobility analysts can build a comparable time series without writing their own sparse-to-dense assembly, M.

### CMP-13 — Two inputs named `domainCount` use different units

**Where.** K `D/column-classification/gpu-color-scale.ts:97,478`, R `D/column-classification/color-scale-parameters.ts:43,92`, K `docs/api/gpu-dataframe-analysis.md:351`.

**Evidence — EXECUTED and COUNT.** With two classes, the resident word is `2` and the active edge count is `3`. `getGPUColorScaleParameterValues({scale:'threshold', paletteCount:2})` threw `GPUColorScale domainCount must be a non-negative integer`, even though the resident count overrides that setting. The docs supply both numbers. I found 49 `new GPUColorScale` expressions by AST, 14 with a directly named resident `domainCount` field.

**Now → wanted.** Rename the resident prop to `classCount`. Keep numeric `domainCount` for the number of active domain entries. Provide a threshold-settings overload for the resident-class-count route that does not require a dummy edge count. The expression becomes `domain: breaks, classCount, palette`.

**In-repository callers changing.** **20 executable property-supply sites** in the inspected `modules`, `examples` and `website` TypeScript: 14 direct constructor fields, one conditional spread in `kepler-compute/src/analysis-runtime.ts`, one hot-spot recipe props object, one GPU colour-scale fixture object and three node-test prop overrides. I cross-checked the AST inventory with `rg -n 'domainCount:'` and separated numeric packer calls. Update three additional example strings found in the same scan, plus the API page. **Value/cost:** classification users no longer have to remember an off-by-one convention determined by where a field appears, S. I recommend the rename before publication.

### CMP-14 — Family size is presented as the tested population

**Where.** K `S/multiple-testing/gpu-multiple-testing-adjustment.ts:71,103,444`, R `S/multiple-testing/cpu-multiple-testing-adjustment-operation.ts:27,47`.

**Evidence — EXECUTED.** `computeMultipleTestingAdjustmentOnCPU([0.01,NaN,NaN], 'bonferroni', 3)` returned adjusted `[0.0299999993,NaN,NaN]` and `testedCount=[3]`. Exactly one row had a test. Three is the correct adjustment denominator.

**Now → wanted.** A map legend reading `` `${testedCount} polygons tested` `` overstates the observed support. Add `familySize` and `observedTestCount`, retain the existing field as a documented legacy alias, and use the two explicit names in the permutation-to-correction example. **Callers changing:** none required. **Value/cost:** analysts can report both the actual tested population and the denominator used for inference, S.

## Departures by name

Within the six journeys, the following names identify the implementation boundaries to change or document.

| Departure | Producers and consumers |
|---|---|
| A readable prefix or row identity must be reconstructed | `GPUPointInPolygonJoin`, `GPUNetworkNoding`, `GPUNetworkCostMatrix`, `GPUTrajectoryStaypoints`, `GPUPointToCell`, `GPUCellAggregation`, `GPUKernelDensity`, `GPUClassBreaks` |
| Producer evidence is on a separate branch from the numerical input | `GPUZonalStatistics` → `GPULocalMoran` / `GPULocalPermutationTest`; `GPUNetworkNoding` → `GPUNetworkCostMatrix`; `GPUCostDistance` and `GPUPolygonRasterization` → `GPURasterZonalStatistics`; `GPUGroupStatistics` → `GPUKeyJoin` → `GPUOrdinaryLeastSquares`; stop and cell-table loss → `GPUEmergingHotSpots` |
| Axes, empty-bin policy or count units are lost in a bare view | `GPUNetworkCostMatrix`, `GPUNetworkAccessibility`, `GPULocationAllocation`, `GPUKeyJoin`, `GPUCellAggregation`, `GPUEmergingHotSpots`, `GPUClassBreaks`, `GPUColorScale`, `GPUMultipleTestingAdjustment`, `GPUOrdinaryLeastSquares` |
| Existing patterns worth reusing | `GPUContiguityWeights.outputWeights`; `makeGPURasterBand` for `GPUTerrainDerivatives`, `GPUCostDistance` and KDE prefixes; `GPUValueCube` and `GPUCubeLayoutTranspose` for cell-major data; published `GPUGather`, `GPUMask` and `GPUTranspose` for resident glue |

## Quick wins

- [ ] Fix the polygon-zonal example and pin the two-pixel `[0,NO_ZONE]` case.
- [ ] Show the grid-sized band prefix in the KDE classification example. Prefill the spare pixel in the composition test.
- [ ] Document the difference between `GPUKeyJoin.gather` and `rightRows → GPUGather` for absent unsigned counts.
- [ ] Add the stop-time and live-mask gathers to the movement page. Include spare stop capacity in its fixture.
- [ ] Show the cost-value and polygon-assignment flags combined before zonal statistics.
- [ ] Add `familySize` and `observedTestCount` to correction outputs.
- [ ] Publish the one-column and two-column regression wiring next to the group/key-join example.

## Larger proposals

1. **Start with three result ports.** Add an assigned-feature-row port, a noded topology with live count, and an axis-labelled cost matrix. Each addresses a demonstrated case where equal lengths and formats permit different answers. Add overloads for these ports and keep the caller’s existing buffers.
2. **Make completeness part of composed results.** Keep stage-specific causes and publish one final usable/incomplete signal for the requested result. A solver code, permutation tested count or search convergence flag can then retain its existing local meaning. The first fixtures should be CMP-01, CMP-07 and CMP-09.
3. **Use one breaking rename now.** Rename the resident colour-scale `domainCount` to `classCount` at the 20 sites above. For polygon counts, additive explicitly named outputs are preferable because both exclusive assignment and overlapping membership are legitimate answers.
4. **Publish one stop-to-cube recipe.** It should take the event-time choice, stable cell dictionary and weights explicitly. A finished all-period cell table, a relative timestamp without an epoch, and a truncated group table without the original observations each lack information needed to recover the requested result. No repack can restore it.

## Not read

I read the cube-layout transpose, raster producer-completeness helper, CPU multiple-testing operation and raster zonal oracle in full. I traced the constructor types, relevant validation, publication kernels and CPU paths for the 25 principal classes named in the census, plus `GPUPeriodIndex` and the public graph primitives used by the journeys. I sampled tests by following each suspect handoff and used the synthesis, class-name matches in the area reports, API/catalog report, direction report and decision ledger to remove prior findings.

The runtime results above distinguish public CPU twins, test oracles, graph declarations and shader source. I did not run GPU kernels, `tsc`, Vitest, builds or browsers. I did not audit the rest of the 496 producers, all recipe variants, every raster contour route, or the complete lenses and Kepler consumer trees.