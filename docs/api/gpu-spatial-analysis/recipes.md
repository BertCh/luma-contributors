---
title: "GPU Spatial Analysis: Recipes"
sidebar_label: "Recipes"
description: "Prebuilt GPU spatial analysis recipes, documented by the contributors they compose."
---


# Recipes

Choose a task; the execution form tells you what the entry constructs. All recipe builders
append GPU command-graph work and leave submission to the caller.

| Task | Entries | Execution form |
| --- | --- | --- |
| Map observations and rates | `addHotSpotAnalysisRecipe`, `addRateClusterMapRecipe`, `addPointsInPolygonsChoroplethRecipe`, [excess risk and spatial rates](./rate-mapping.md) | GPU graph recipes; choropleth also has staged GPU/CPU aggregate operations |
| Compare periods and model spatial variation | `addPeriodComparisonRecipe`, `addSpaceTimeHotSpotsRecipe`, `addSpatialRegressionRecipe`, [local variability](./expansion-recipes.md#local-variability-comparison) | GPU graph recipes |
| Summarize zones and transfer population | [Zonal distributions](./expansion-recipes.md#zonal-distributions), `addChangeOfSupportRecipe`, [dasymetric redistribution](./dasymetric-redistribution.md) | GPU graph recipes |
| Allocate access and compare network scenarios | `addStraightLineCatchmentsRecipe`, `addDriveTimeCatchmentRecipe`, [network disruption](./expansion-recipes.md#network-disruption), [facility evaluation](./expansion-recipes.md#facility-scenario-evaluation) | GPU graph recipes; drive-time and scenarios imported from `gpu-network` |
| Analyze movement and cluster footprints | Both fleet dwell recipes, `addStopValueCubeRecipe`, `addClusterAndOutlineRecipe`, [track quality](./expansion-recipes.md#track-quality-and-route-conformity), [home ranges and recursions](./home-ranges-and-recursions.md) | GPU graph recipes; home-range/recursion contributors also have GPU/CPU operations |
| Compare surfaces and trace drainage | [Interpolation](./expansion-recipes.md#interpolation-diagnostics), [exposure](./expansion-recipes.md#exposure-by-administrative-area), [regional change](./expansion-recipes.md#regional-raster-change), [drainage](./expansion-recipes.md#drainage-network-analysis), [surface repair](./expansion-recipes.md#surface-preparation-with-provenance) | GPU graph recipes |
| Measure landscapes, buildings and visibility | [Landscape ecology](./landscape-ecology.md), [urban morphology](./urban-morphology.md) | GPU graph recipes |


See [Geospatial workflow recipes](./expansion-recipes.md) for zonal distributions, network and facility scenarios, interpolation diagnostics, exposure, raster change, drainage, track quality, local variability, cell flows and surface preparation.


The local-variability recipe retains every declared bounded-status view on its intermediate
weights port and named weights stage, including approximation, uncertainty and iteration diagnostics.

Recipes are plain functions, not classes: `addXRecipe(graph, props)` constructs the contributors, calls
`graph.add` in chain order, wires outputs to inputs and returns a `GPURecipeResult`: `{contributors, outputs,
intermediates, status}`. `outputs` holds the named terminal views, `intermediates` the named ports between stages and
`status` the GPU-resident completeness and convergence views, grouped by stage. `contributors`
lists every stage the recipe added, in order: composed contributors, and each raw glue kernel as one named
stage. Published calibration stages and drive-time road/time/band adapters are included in that
same declaration-order list. Status counts over rows or cells (invalid
rows, unresolved cells) run as a row-parallel kernel plus a published `GPUReduction` sum, never as one
invocation looping over the rows. Views you
pass in (`zScores`, `bins`, `colors`, ...) are caller-owned and readable; omitted destinations use
graph transient storage. For hot spots, rate clusters, spatial regression, period comparison,
cluster outlines, change of support, straight-line catchments, both fleet variants, interpolation
diagnostics, surface preparation, local variability and cell-flow topology, `output` supplies the
destination map.
Published ESM and CommonJS graph data/vector views retain their exact identities. Destination
tuples are copied, consumed getters are captured once, and omitted optional status fields stay absent.
Read returned views from `result.outputs`.
The destination map is not reproduced verbatim there: hot-spot/rate/cluster result-level diagnostics
are in `result.status`; regression codes stay in the nested model results; change-of-support raster
status tuples retain source/target order at indices 0/1. Fleet stop `startRows`/`endRows`, optional
rate-cluster `localI` and cluster `coreFlags` are written when supplied but omitted from their
returned output maps, so retain the supplied views themselves. Period aggregation tables remain
in `result.intermediates.before`/`after`. Use each builder's declared output shape for other leaves.
To retain a terminal, create/import a caller-owned view in the same graph and supply its corresponding
`output` leaf, including nested status leaves. Otherwise add an explicit graph copy into a
caller-owned readback buffer before compiling; transient views cannot be read back directly. Inference recipes (hot spot, rate cluster, space-time hot spots, spatial regression) take
one `settings` object and `GPUParameterBuffer`s they fill themselves, derive `result.inference` from the same
object and repack it with `result.setSettings(settings)`; the other recipes pass per-frame parameter views
(`getGPU...ParameterValues`) through. Neither rebuilds the graph. Geometry inputs are the shared ports:
`GPUPointGeometryPort`, `GPUPolygonGeometryPort` (four-level layout, one row per feature) and
`GPUPackedGeometryPort<GPUTrajectoryGeometryPort>`.

The current recipes barrels export 25 spatial `add…Recipe` functions and four network recipes,
including `addStopValueCubeRecipe`. Their builders prepare graph declarations before changing the
caller's graph. Contributor constructors and cross-stage checks
finish before the builders expand and schedule contributors. Preparation uses a separate,
uncompiled graph with the same device limits, preserving buffer aliases, view offsets, texture
subresources and vector chunks. Nested recipes share their enclosing preparation. If validation
or contributor expansion fails, the caller's nodes, resources and parameter-buffer contents stay
unchanged: correct the configuration and retry with the same recipe ID. Temporary contributors
that own resources are destroyed even when a later stage fails. Parameter writes are validated
without upload and applied only after successful scheduling; frozen `GPUParameterBuffer` wrappers
are supported. Later `setSettings` calls still write immediately. Contributors and interleaved
adapter kernels retain their declaration order when scheduling is deferred.

Auxiliary luma resources and devices carried alongside a recipe's ports keep their identities and
ownership. Preparation treats them as opaque references and does not inspect their native state;
graph views are mirrored separately. Each consumed configuration field, including inherited getters
and JavaScript Proxy reads, is captured once and shared by both declarations. Unrelated fields are
not evaluated. Configuration records returned in metadata are captured facades; caller graph views,
parameter buffers and declared prepared-join handles retain their original identities.
Retained committed metadata is detached from the temporary preparation graph after preflight.
Raster recipes return a frozen snapshot of declared grid provenance, including affine/origin arrays
and nested CRS metadata. Retaining only this snapshot retains neither graph nor the caller's props;
later caller metadata edits cannot change the returned provenance.
The original 28-builder rollback matrix covers its explicit fixture set; that tested set is
separate from the current 29-export total. `addStopValueCubeRecipe` is an additional public builder,
not evidence that a historical 28-case run covered it. The public family inventory also lists
13 domain recipes outside that matrix, with their own family contracts.

Preparation repeats host declaration work once and never compiles, encodes, submits or reads data.
Use unused recipe/resource IDs on a mutable graph. The published graph has no mutable node-ID
inspection or transaction API, so this configuration guarantee cannot cover collisions with
pre-existing caller node IDs, externally overridden scheduling methods, a graph already frozen by
compilation, or allocation/device failure during final declaration. Successful recipes retain the
same concrete command nodes, outputs and status meanings.

Population support alignment returns an empty `intermediates` map. Under `status.stages`,
the `alignment` stage exposes the whole-support `status.complete` certificate. The
`missing-targets`, `ambiguous-targets` and `excluded-sources` stages each expose a scalar
`status.invalidCount`. The contributor and operation forms accept packed `output.status`.

The hot-spot, rate-cluster, space-time, fleet (both variants), period-comparison, change-of-support,
cluster-and-outline, spatial-regression, points-in-polygons and straight-line-catchment results type
their `outputs` and `intermediates` maps. The exported
`GPU<Name>RecipeOutputs` and, where present, `GPU<Name>RecipeIntermediates` types describe those
maps; the top-level convenience fields remain available. For example, `spaceTime.outputs.cube`
feeds `GPUEmergingHotSpotsProps.values`, and `fleet.outputs.stops.centroids` is a typed
`float32x2` view. Optional stages remain optional in both paths, so check for
`hotSpots.outputs.permutation` or `spaceTime.outputs.colors` before using them.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {addHotSpotAnalysisRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

// All views and parameter buffers belong to this mutable graph. Polygon offsets use
// the four-level feature/polygon/ring layout; output rows match cellCapacity.
declare const graph: GPUCommandGraph;
declare const positions: GraphDataView<'float32x2'>;
declare const studyPositions: GraphDataView<'float32x2'>;
declare const featureOffsets: GraphDataView<'uint32'>;
declare const polygonOffsets: GraphDataView<'uint32'>;
declare const ringOffsets: GraphDataView<'uint32'>;
declare const autocorrelationParameters: GPUParameterBuffer<'float32'>;
declare const permutationParameters: GPUParameterBuffer<'uint32'>;
declare const zScores: GraphDataView<'float32'>;
declare const bins: GraphDataView<'sint32'>;
declare const classBreaksParameters: GraphDataView<'float32'>;
declare const colorScaleParameters: GraphDataView<'float32'>;
declare const palette: GraphDataView<'uint32'>;

const hotSpots = addHotSpotAnalysisRecipe(graph, {
  source: {kind: 'points', positions, family: 'quadbin', resolution: 8, cellCapacity: 4096,
    studyArea: {kind: 'polygons', positions: studyPositions, featureOffsets, polygonOffsets, ringOffsets,
      candidateCapacity: 16384}},
  settings: {significanceLevel: 0.05, seed: 7, permutationCount: 199},
  parameters: autocorrelationParameters, // GPUParameterBuffers the recipe fills from settings
  output: {zScores, bins},
  permutation: {parameters: permutationParameters, maximumPermutations: 199},
  color: {classBreaksParameters, maximumClassCount: 5, colorScaleParameters, palette, maximumPaletteCount: 5}
});
```

| Recipe | Chain |
| --- | --- |
| `addHotSpotAnalysisRecipe` | GPUCellCover study area -> GPUCellAggregation (sorted unique study cells) + (GPUPointToCell) -> GPUCellAggregation (points) -> join (zeros kept) -> area density (GPUCellMeasures for H3) -> GPUCellTopology disk -> CSR weights (or GPULatticeWeights for raster source) -> GPUHotSpotAnalysis -> GPULocalPermutationTest -> GPUClassBreaks -> GPUColorScale |
| `addRateClusterMapRecipe` | GPUEmpiricalBayesRates -> GPUContiguityWeights -> GPUSpatialWeightsTransform(row) -> GPULocalMoran (`quadrantGating: 'none'` with permutation) -> GPULocalPermutationTest -> quadrant colors |
| `addPointsInPolygonsChoroplethRecipe` | backend 'zonal' (GPUZonalStatistics) or 'group' (GPUPointInPolygonJoin -> dense GPUGroupStatistics) -> GPUClassBreaks -> GPUColorScale |
| `addSpaceTimeHotSpotsRecipe` | GPUCalendarBuckets -> GPUPeriodIndex (absolute time-step slices) -> atomic count into the dense cube (sorted GPUGroupStatistics path when its scratch or overflow outputs are passed) -> GPUEmergingHotSpots -> ordinal GPUColorScale |
| `addClusterAndOutlineRecipe` | GPUDBSCAN -> GPUGroupGeometry + GPUGroupConvexHull -> GPUGeometryMeasures |
| `addSpatialRegressionRecipe` | GPUOrdinaryLeastSquares -> GPUSpatialRegressionDiagnostics -> GPU specification decision (Anselin's rule: 0 OLS, 1 lag, 2 error, 3 both robust, 4 undetermined) + residual GPULocalMoran (+ optional GPUGeographicallyWeightedRegression); `mask` and `restandardizeRows` flow through OLS and diagnostics |
| `addDriveTimeCatchmentRecipe` (from `gpu-network`) | GPUNetworkSnapping -> GPUNetworkServiceAreas -> (GPUNetworkIsochrones raster and/or cell outline with rings -> GPUPointInPolygonJoin demand join) -> snap demand -> band adapters -> GPUGroupStatistics |
| `addStraightLineCatchmentsRecipe` | GPUDistanceField allocation -> GPURasterZonalStatistics |
| `addChangeOfSupportRecipe` | 2x GPUPolygonRasterization -> GPUArealInterpolation (with source conservation diagnostics) -> GPUSpatialLag |
| `addFleetDwellRecipe` | GPUTrajectoryMetrics stops -> active-stop mask and join positions -> GPUPointInPolygonJoin -> GPUGroupStatistics (dense, one row per zone) |
| `addFleetDwellZoneEventsRecipe` | GPUZoneEvents -> GPUGroupStatistics (dense, one row per zone; zone-event variant of the fleet dwell recipe) |
| `addPeriodComparisonRecipe` | 2x GPUCellAggregation -> GPUCellTableCompare (with mask) -> GPUClassBreaks -> GPUColorScale |

Notes:

- Fleet zone-event `zone-statistics.status.candidateOverflow` reports candidate scratch loss;
  nonzero means dense dwell statistics may be incomplete. Per-track and final event-list loss
  remain separate and do not by themselves invalidate dwell totals. Optional `visit-table`
  publishes its own candidate-loss word when provided, otherwise the shared scratch-loss word,
  even when final visit capacity fits. Direct zone-event callers may retain either optional
  `output.status.candidateOverflow` or `candidateCount <= candidateCapacity` as scratch evidence.
- The OLS and spatial diagnostics stages expose the one-word solver `code` and the one-word
  excluded-row `status.invalidCount`. The diagnostics count also includes unusable OLS residuals.
  Optional `geographically-weighted-regression.rowCodes` has one code per observation (0
  successful, 1 singular, 2 excluded). Use the model's status decoder for code meanings.
- Weighted point hot spots expose `sumOverflow` on both `point-aggregation` and the joined table.
  Saturated occupied sums become NaN. Joined `candidateOverflow` carries source study/point-table
  loss; joined `overflow` is destination table truncation. Neither is arithmetic `sumOverflow`.
  A whole-inference gate suppresses every z-score and p-value and sets bins to zero on
  any source loss, destination overflow or saturation, including with caller-fixed moments.
  Public study occupancy and empty-cell zero values remain available. A later frame whose inputs
  fit existing storage and arithmetic range can recover on resubmission. To increase capacity or
  change construction-time scaling, rebuild the producer and rebind affected consumers in a new
  graph; the same compiled graph retains its allocations and cannot grow by resubmission.
- Space-time hot spots default to `occupiedCellsOnly: false`: all unmasked supplied cells enter
  eligibility. Set it to true to restrict inference to cells with at least one eligible event
  in any requested slice. Occupancy comes directly from eligible cell/time events, independently
  of retained sorted groups, and intersects the caller's `cellMask` in
  `intermediates.cellMask`. Zero bins of occupied cells remain eligible. The occupancy and mask
  rebuild on every submission. Sorted `cube-counts` group loss suppresses all inference for both
  occupancy settings while preserving the event-derived occupancy; a complete subsequent
  submission recovers inference. Explicit `null` is rejected for `occupiedCellsOnly`.

- Fleet tables are dense: `table.status.count` counts zone rows, including empty zones;
  `table.counts` counts stops or visiting tracks within each zone. Dense-table overflow is zero.
  Final stop/event overflow instead compares the required compact list with its output capacity.
- Hot-spot `neighborCounts` reports available active neighbors even for rows excluded by the
  minimum-neighbor inference guard. A 3 by 3 queen lattice has counts
  `[3,5,3,5,8,5,3,5,3]`. Space-time `hotSliceCount` has one word for every cell, including
  zero words for cells without significant hot slices.
- Interpolation `standardizedCount` counts finite residuals with strictly positive prediction
  variance. An exact noiseless duplicate can give a valid prediction with zero variance and
  contributes zero to this count. Four distinct unit-square corners with spherical range 0.5
  and unit sill instead have leave-one-out variance 4/3; all four finite observations contribute.
- Rate-cluster weights publish `weightsCandidateOverflow` for `pairCapacity` loss and
  `weightsOverflow` for final `neighborCapacity` loss. Inspect both flags. Grow `pairCapacity` first
  when the candidate flag is set; `weightsRequiredCount` is then only a lower bound for neighbor slots.
  The Kepler rate-cluster binding refuses either loss flag and uses its CPU fallback. Explorer and
  showcase readouts distinguish pair loss from final neighbor loss and mark dependent results unavailable.
- Spatial regression diagnostics suppress statistics and p-values with status code 3 on either
  weights-loss flag; specification selection returns 4 (undetermined). Clean reuse restores inference.
  Caller-provided regression writes cannot alias either weights-loss input; rejection precedes all
  resource registration so correcting the output permits a same-ID retry.
- All caller-provided fleet stop outputs, including durations, row indices and status words, must
  have logical byte ranges disjoint from `scratch.stopMask`. Adjacent views
  that share an aligned storage binding remain supported through separate adapter dispatches.
- Fleet zone-event lenses and website readouts distinguish final event overflow, combined event
  candidate loss (scratch or per-track), and scratch-only loss. Either event-loss flag makes crossing
  counts incomplete; only scratch loss makes dense dwell totals incomplete. Pending readbacks are
  labelled pending. Truncated per-track event lists can retain complete dwell totals.
  Crossing counts require the count and zone-column readbacks; positions are needed only for drawing
  dots. Missing numeric readbacks are pending or unavailable, including missing dwell totals, never zero.
  A failed refresh replaces the previous readback with missing data; a prior successful count or
  dwell total is not presented as current after that read fails. A later successful refresh restores it.

- The space-time recipe counts events into a dense `cell * sliceCount + slice` cube with integer atomics,
  because `GPUCellAggregation` keys must be valid Quadbin or H3 cells. Passing the sorted-path scratch
  views or `output.status.groupOverflow` selects the `GPUGroupStatistics` path instead.
  Group status views need at least one packed uint32 word; spare words are untouched.
  On the supplied-cell-ID route, the `weights` status stage preserves both `overflow` and
  `candidateOverflow`; a nonzero flag means neighborhood inference is incomplete. These input
  words cannot alias recipe writes. The slice interval, including seven days per week, must fit
  the period index's signed divisor limit of 2^31 - 1 before recipe resources are registered.
  The origin's whole calendar-step index must fit [-2^31, 2^31 - 1] too; hourly indices count hours
  from the Unix epoch. This check also precedes allocation. Every declared writable port and every
  published weights status word (including invalidCount) participates in alias validation. Declared
  fields, including inherited/getter-backed status fields, are captured once for validation and scheduling.
  `slices.count` is at most 256; trend autocorrelation correction has the lower limit documented by
  `GPUEmergingHotSpots`. Every contributor, settings object and glue binding is validated before any
  recipe command nodes are scheduled. `contributors` includes `GPUPeriodIndex` and every glue stage
  in execution order on both counting paths.
  Event positions must be packed `float32x2`, and event cell IDs packed `uint32`, with exactly one
  row per timestamp. Strided positions are rejected; pack them before construction. Caller-view
  ownership, shape and binding conflicts, parameter length, settings and palette preconditions are
  checked before registering recipe resources, so correcting those props permits retry on the same
  graph and recipe ID. Settings are packed on the host first and written to the caller's parameter
  buffer only after scheduling succeeds; a rejected second recipe cannot change an existing
  recipe's settings through a shared parameter buffer.
  Initial packing preserves all classification settings, including `persistentFraction`,
  `minimumTrendSliceCount` and `sporadicRequiresFinalSlice`; an identical `setSettings` call writes
  identical parameter words. A six-slice series can carry a trend when its requested minimum is three.
- The drive-time catchment recipe classifies demand in network space (the minimum over the snapped edge
  ends) and also returns the isochrone triangles for drawing. It assumes both directions are in the CSR.
  Supplied `demandTimes` and `demandBands` must each have exactly one logical row per demand point,
  regardless of backing-buffer capacity or byte offset. Caller output/scratch buffers must be distinct
  from original routing inputs and independently retained results, including separate imports of the
  same physical buffer. Rejection leaves the graph unchanged for a corrected same-ID retry.
  Active nested isochrone raster, geometry, table and status outputs participate in this check:
  for example, a cost raster cannot share a buffer with retained demand times.
  Disjoint nested outputs may share a physical buffer when the isochrone contributor's aligned-range
  and dispatch checks permit it. The outer recipe requires whole-buffer separation only across stages.
  Caller destinations use `output`. Returned views are the destinations written by the recipe.
  Declared raster fields, including
  inherited getters and get-only Proxy values, are captured once before adding recipe work.
  Raster mode accepts only `'min'` (the omitted-mode default). Returned isochrone statuses
  describe those same captured stages and views.
  `isochrones` takes `raster` and/or `cellOutline` (at least one). `cellOutline.rings` returns closed
  isochrone rings in `result.isochroneRings`. `isochrones.joinDemand` takes `candidateCapacity`.
  Caller join destinations live under `output.isochroneDemand`:
  `pointFeatureIds` has one uint32 row per demand point; writable status leaves are one-row uint32
  `overflow` (join capacity loss), `uncertainCount` (number of unresolved, unassigned candidate pairs) and
  `candidateOverflow` (inherited geometry loss). Join `unresolvedCount` and `complete` destinations
  reject because they have no writer; the ring stage's positive `complete` certificate remains
  separate. A positive `uncertainCount` reports incomplete classification, not successful recovery.
  Each status destination is optional.
  Join destinations require `isochrones.joinDemand`; band sum/mean destinations require
  `demandValues`; snapping diagnostics require `candidateCapacity`. Disabled-stage destinations
  reject before graph allocation. The join suppresses matches while inherited geometry is incomplete and recovers on
  the next complete submission. It joins demand points to the ring polygon (it needs
  `cellOutline.rings.output.polygons`) with `GPUPointInPolygonJoin`, and `result.isochroneDemand.pointFeatureIds`
  is 0 inside and `GPU_SPATIAL_JOIN_NO_FEATURE` outside. That answers "inside the isochrone at
  `cellCostLimit`" at cell resolution (the union of cells that hold reached nodes), where the band
  classification stays exact at the snapped edge. Isoband triangles cannot feed the join, because their
  topology is GPU-written and has no ring structure; the cell rings can. Sparse networks drop cells, so the
  polygon join is coarser than the network-space bands. With `candidateCapacity`, each snapping reports
  BVH candidate overflow in `result.snapping` and the `snap-facilities` and `snap-demand` status stages,
  so a facility or demand point snapped past a dropped candidate is never silent. The other stages are
  `service-areas`, the caller-owned isochrone outputs (`isochrone-bands`, `isochrone-band-edges`,
  `isochrone-band-rings` on the raster path; `isochrone-cells`, `isochrone-outline`, `isochrone-rings` on
  the cell path) and `band-statistics` (count, overflow, required count). The service-area `assignments`
  are passed to `GPUNetworkIsochrones`, so `raster.output.pixelFacilities`, `triangleFacilities` and
  `cellOutline.byFacility` work through the recipe; facility rows are seed rows (`2 * facility + endpoint`).
  The `isochrone-band-rings` stage exposes intermediate boundary-edge loss as `candidateOverflow`
  (the caller's `raster.rings.output.status.edgeOverflow` view). Its `overflow` reports only final
  ring/vertex capacity loss: a truncated open chain can have count 0 and overflow 0 while
  candidateOverflow is 1. The stage also retains the supplied
  `raster.rings.output.status.complete` view as `complete`; this positive certificate is 1 only
  when the promised geometry is whole. Check both loss flags and any supplied complete certificate
  before treating an empty ring result as complete.
- Every recipe reports each inner contributor's status as a `status.stages` entry (rule 10): the
  cluster recipe ends with `hull-measures` (`invalidCount`), the straight-line recipe has
  `zonal-statistics` (`overflow`), and the table, join and rasterization stages also carry `requiredCount`
  or `uncertainCount` where the inner contributor computes them. Result-level status words sit in
  `result.status` next to `stages` (for example `status.weightsOverflow`, `status.invalidCount`,
  `status.rasterOverflow`), sub-results own a `status` (`result.permutation.status.overflow`,
  `result.table.status.count`), and the `output` overrides mirror those paths (`output.status.weightsOverflow`).
- The drive-time, straight-line catchment and fleet dwell recipes take an optional `spatialContext` and
  return the resolved one as `result.spatialContext`. The straight-line recipe and both fleet dwell recipes
  are planar only (longitude/latitude throws; the join and zone edges are straight lines in the plane, and the
  distance field is Euclidean). The drive-time recipe also accepts longitude/latitude (metric `'none'` or
  `'great-circle'`; the cell-outline isochrones need degrees). Both of its snappings then run great-circle:
  edges are ranked by great-circle distance and `maximumSnapDistance` uses `spatialContext.units`
  (metres by default, or kilometres), with no east-west bias at any latitude. With the default
  sphere radius, a one-degree arc is about `111195` metres or `111.195` kilometres; use the matching
  cutoff and read snapped distances in that same unit. Drive times stay in the units of `network.weights` (the snapped fraction splits the edge
  cost). For the other recipes, project with `createGPULocalFrame` and pass its `spatialContext` (planar,
  meters), so the distance-field `parameters` and the straight-line `distances` output are in meters.
  The fleet dwell recipes used to hand the trajectory stage a native planar context whatever the data;
  they now resolve one context and pass it to the metrics, join and zone stages.
- The rate cluster map recipe runs `GPULocalMoran` with `quadrantGating: 'none'` when `permutation` is
  given and keeps the quadrant where the permutation `significant` mask is set; without `permutation` the
  quadrant is gated by the analytic p-value. `analyticQuadrants` returns the local Moran quadrants (gated
  analytically without `permutation`, ungated with it).
- The points-in-polygons choropleth recipe's `'group'` back-end uses dense `GPUGroupStatistics`
  (`keyCount: featureCount`, so empty polygons keep a row), writing `featureValues` directly. For
  `statistic: 'count'`, `featureValues` is the `uint32` counts view, which `GPUClassBreaks` and
  `GPUColorScale` read as numbers (`integerValues: 'numeric'`); the `featureValues` prop is rejected for
  `'count'`, and the result type is `float32` or `uint32`.
- The spatial regression recipe requires a finite residual on every row; the weights pattern may be asymmetric.
  Optional local fits pass their `spatialContext` to `GPUGeographicallyWeightedRegression`; omitted means
  native planar, and fixed bandwidths use the context's length unit.
  Its `settings.significanceLevel` (default 0.05) is packed once into the autocorrelation `parameters`
  buffer and read per frame by both the specification search and the residual `GPULocalMoran` (it used to
  be a compile-time constant of the search kernel); `result.inference` and `result.setSettings` follow it.
- The fleet dwell recipes take `trajectories` (`GPUPackedGeometryPort<GPUTrajectoryGeometryPort>`) and polygon `zones`
  (`GPUPolygonGeometryPort`, `sourceIds` as stable zone IDs); the zone events variant passes the zones to
  `GPUZoneEvents` directly (no edge lists) with `trackEventCapacity` events kept per track. The
  straight-line catchments recipe takes `grid`, `parameters` and `facilities` (`GPUPointGeometryPort`,
  `sourceIds` as zone IDs); the cluster recipe takes `points` (`GPUPointGeometryPort`).
- For `addFleetDwellZoneEventsRecipe`, inspect both `events.status.overflow` (final capacity) and
  `events.status.candidateOverflow` (scratch or per-track loss) before treating the crossing list as
  complete. The separate recipe `status.candidateOverflow` identifies scratch loss affecting zone
  dwell totals and visits; per-track event-list truncation alone leaves those totals intact.
- The cluster recipe's optional `spatialContext` defaults to native planar and is returned as
  `result.spatialContext`. It is planar only because group summaries and convex hulls are planar:
  longitude/latitude throws, so project first. Clustering measures distance and rejects
  `local-lambert-equal-area` frames; `epsilon` uses the context's length unit.
- The cluster recipe drops clusters at or above `maximumClusterCount`, and the caller supplies the
  geometry parameters. The change-of-support recipe's `status.overflow` covers the overlap list only; the raster
  flags are returned as `status.rasterOverflow` and `status.rasterRequiredCount` (stages `source-rasterization` and
  `target-rasterization`).

### Stop values and contiguity-order diagnostics

`addStopValueCubeRecipe` bins the live compact stop prefix into cell-major counts
`cellRow * periods.count + periodRow`. Choose stop `time: 'start'` or `'finish'`, preserve the
original signed Int64 timestamps and temporal context, and supply the fixed cell dictionary in
the same row order as the weights. Empty observed bins are zero; loss-suppressed bins are also
zero, so retain the `stops`, optional `slice-i` and `cube` stage evidence before interpreting counts.

`addContiguityOrderCorrelogramRecipe` publishes input-weights evidence in the `weights` status
stage, preserving the supplied input status views before building orders. Check that stage's
`overflow` and `candidateOverflow` as well as `order-1` through `order-maximumOrder` stage losses.
`outputs.incompleteOrders` marks affected order rows; a derived order's local capacity flag alone
cannot certify input weights that were already incomplete. The order-major `outputs.correlogram`
columns are named by `GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS`.

### Recipe settings and stage props

These are the direct recipe-to-stage mappings from the spatial workflow inventory, including the
network-owned drive-time recipe. `parameters` denotes a live packed buffer; its packer defines each
word's units. A capacity is an allocation bound, not a live output count. Only a stage publishing a compact result
and its live `status.count` uses that word to bound a readable prefix; tails are unspecified unless
cleared explicitly. A model stage such as OLS publishes its solver outcome in `code` and its excluded-row count in
`status.invalidCount`. Fixed-row outputs use their declared row domain.

| Recipe path | Stage identity and prop | Unit and count/capacity meaning |
| --- | --- | --- |
| `addHotSpotAnalysisRecipe.preparationParameters` | `study-area-preparation`: `GPULonLatPolygonPreparation.parameters` | Live float32 `maximumSegmentDegrees` in degrees, followed by reserved words; does not size output |
| `addPointsInPolygonsChoroplethRecipe.preparationParameters` | `polygon-preparation`: `GPULonLatPolygonPreparation.parameters` | Same live subdivision limit in degrees; feature rows remain stable |
| `addSpatialRegressionRecipe.olsParameters` | `ordinary-least-squares`: `GPUOrdinaryLeastSquares.parameters` | Live float32 ridge lambda in squared predictor units; observations and predictors are separate fixed dimensions |
| `addSpaceTimeHotSpotsRecipe.calendarParameters` | `GPUCalendarBuckets.parameters` | Live sint32 `[utcOffsetMinutes, firstDayOfWeek]`: minutes east of UTC and weekday code 0 (Monday) to 6; not a capacity |
| `addPeriodComparisonRecipe.classBreaksParameters` | `GPUClassBreaks.parameters` | Live class-break layout; edges use the compared value's unit, class count is bounded by `maximumClassCount` |
| `addPeriodComparisonRecipe.colorScaleParameters` | `GPUColorScale.parameters` | Live color mapping layout; domain uses the compared value's unit, palette count is bounded separately |
| `addClusterAndOutlineRecipe.maximumVerticesPerHull` | `convex-hull`: `GPUGroupConvexHull.maximumVerticesPerGroup` | Maximum vertex rows per group, an allocation/route bound |
| `addClusterAndOutlineRecipe.hullCapacity` | `convex-hull`: `GPUGroupConvexHull.vertexCapacity` | Total allocated vertex rows across groups, not written vertex count |
| `addClusterAndOutlineRecipe.geometryParameters` | `GPUGroupGeometry.parameters` | Live geographic-distribution layout: origin and median tolerance use planar length units; standard-deviation/ellipse multipliers and orientation flag are dimensionless |
| `addClusterAndOutlineRecipe.clusteringParameters` | `clustering`: `GPUDBSCAN.parameters` | Live clustering layout; epsilon is in the resolved spatial length unit; neighbor thresholds count rows |
| `addChangeOfSupportRecipe.preparationParameters` | `source-preparation` and `target-preparation`: `GPULonLatPolygonPreparation.parameters` | Live `maximumSegmentDegrees` subdivision limit in degrees, followed by reserved words |
| `addChangeOfSupportRecipe.sourceValues` | transfer `GPUSpatialLag.values` | One value per source feature; extensive totals or intensive values retain the caller's measurement unit |
| `addStraightLineCatchmentsRecipe.facilityCount` | allocation `GPUDistanceField.seedCount` | Live one-row uint32 active facility count, bounded by facility position rows; not output capacity |
| `addStraightLineCatchmentsRecipe.facilities.sourceIds` | allocation `GPUDistanceField.seedIds` | Caller-supplied dense zone IDs below facility capacity, one per seed; omitted IDs default to facility rows |
| `addStraightLineCatchmentsRecipe.facilities.positions` | allocation `GPUDistanceField.seedPositions` | Planar coordinate pairs in the resolved length unit, one per facility |
| `addFleetDwellRecipe.zones` | `zone-join`: `GPUPointInPolygonJoin.polygons` | Polygon geometry with zone feature-row identity; coordinates in the shared planar unit |
| `addZonalDistributionRecipe.probabilities` | `zone-distributions`: `GPUGroupStatistics.probabilities` | Live dimensionless fractions in [0,1] |
| `addRasterZonalDistributionRecipe.probabilities` | nested `zone-distributions`: `GPUGroupStatistics.probabilities` | Same live fractions; `output.quantiles` receives group-major `columns[0].output.quantiles`, in input-value units |
| `addExposureByZoneRecipe.zoneCount` | zonal sums: `GPURasterZonalStatistics.zoneCapacity` | Fixed dense allocation for zone IDs 0..zoneCount-1, including empty zones; not the number of observed zones |
| `addRegionalRasterChangeRecipe.zoneCount` | zonal area/count/mean stages: `GPURasterZonalStatistics.zoneCapacity` | Same dense zone allocation; `validCounts` counts finite cell comparisons per zone |
| `addDasymetricRedistributionRecipe.sourceTotals` | transfer `GPUSpatialLag.values` | Extensive source mass, one value per original source feature |
| `addLandscapeMetricsRecipe.countBoundary` | raster patch measurement: `GPURasterPatchMetrics.countRasterBorder` | Construction-time boolean controlling outer raster-border perimeter; no count or capacity |
| `addDriveTimeCatchmentRecipe.facilities` | `snap-facilities`: `GPUNetworkSnapping.positions` | Facility coordinate pairs; planar length units or longitude/latitude degrees; snap-distance output uses `spatialContext.units` (metres by default, or kilometres) for great-circle mode |
| `addDriveTimeCatchmentRecipe.demand` | `snap-demand`: `GPUNetworkSnapping.positions` | Demand coordinate pairs with the same spatial context; one input row per demand point |

Grouped quantile outputs are values, not percentages. For probabilities `[0.25, 0.5, 0.75]`, read
three quantile values per zone. Recipe destination `output.quantiles`
receives the grouped `quantiles` output.

### Drive-time contributor inventory

The basic drive-time recipe returns 12 contributors. With raster isochrones and no polygon-demand
join (the drive-time GPU fixture), it returns the following 15 in execution order: five analysis
contributors and ten adapters. Preparation instances are never included. A single declaration with
preparation bypassed returns the same list. Previously the result omitted the ten adapters and
reported only five; the larger list completes the documented stage inventory.

Prefix each suffix below with `${id}-`; the default `id` is `drive-time-catchment`. Native classes
have an own `id`. Adapter producers have class `Object` and no own `id`; their stable emitted-stage
ID is shown instead. The oriented-fraction adapter also contains the facility-road sort nodes.

| Order | Contributor ID / emitted-stage suffix | Class |
| --- | --- | --- |
| 1 | `edge-sources-kernel-clear` | `Object` adapter |
| 2 | `edge-sources-kernel` | `Object` adapter |
| 3 | `snap-facilities` | `GPUNetworkSnapping` |
| 4 | `facility-roads-kernel` | `Object` adapter |
| 5 | `facility-starts-kernel` | `Object` adapter |
| 6 | `facility-oriented-fractions-kernel` | `Object` adapter |
| 7 | `service-areas` | `GPUNetworkServiceAreas` |
| 8 | `source-path-geometry` | `Object` adapter |
| 9 | `source-path-topology` | `Object` adapter |
| 10 | `isochrones` | `GPUNetworkIsochrones` |
| 11 | `snap-demand` | `GPUNetworkSnapping` |
| 12 | `demand-roads-kernel` | `Object` adapter |
| 13 | `demand-times-kernel` | `Object` adapter |
| 14 | `demand-bands-kernel` | `Object` adapter |
| 15 | `band-statistics` | `GPUGroupStatistics` |

The basic form omits rows 8–10. The result audit checks all public recipe names, including both
visibility entry points: contributor instances/IDs are unique, returned ports belong to the caller
graph, and contributor counts, node IDs and port layouts match a single-declaration baseline.
Intentional aliases between named output/intermediate ports remain valid. Legacy family results
without a `contributors` property are identified separately rather than counted as empty analyses.

### `addHotSpotAnalysisRecipe`

Bins observations onto Quadbin cells inside a study area and returns Getis-Ord Gi* z-scores and confidence bins per cell.

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=result caption="2,344 observations over 13 community areas; each analysed cell is coloured by its Gi* z-score."}

Points on a study area, or a raster, to Getis-Ord Gi* hot and cold spots.

Points source (breaking change from the occupied-cells design):

- Recipe `spatialContext` describes study-area polygons (default longitude/latitude, metric `'none'`,
  linear edges). For antimeridian crossings or great-circle edges, set `prepare: true` or prepare with
  `GPULonLatPolygonPreparation`. Preparation requires `preparedPositionCapacity` and a per-frame
  `preparationParameters` view from `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`.
  It runs before cell cover and preserves feature rows; `status.stages` adds `study-area-preparation`
  with `overflow`, `requiredCount`, `invalidCount` and `approximation`. Preparation requires a points
  source and a longitude/latitude context. The cover's `invalidCount` counts features with unsplit
  non-pole antimeridian-crossing edges, not wide bounding boxes.
- `studyArea` (a `GPUPolygonGeometryPort` plus `candidateCapacity`, optional `coverCapacity`) and
  `cellCapacity` are required. The recipe covers the polygons with `GPUCellCover` (`'center'` containment),
  deduplicates and sorts the cells with `GPUCellAggregation`, aggregates the points the same way and joins them
  by binary search of each 64-bit key. Every study cell is a row, with value 0 when it holds no point, so the
  null mean is not inflated and clusters of empty cells can be cold spots. Points outside the study area are
  excluded; `result.status.pointsOutsideStudyArea` counts them. `result.table.cells` are the sorted study cells and
  `table.counts` / `table.sumValues` the joined values.
- Neighbors are topological: the cells within `neighborRings` grid steps (`GPUCellTopology` disk) that are also
  study cells, self excluded, binary weights, rows ascending. The defaults are 1 ring for Quadbin (8 neighbors)
  and 2 for H3 (18 neighbors), the smallest disks that reach `minimumNeighbors: 8`. This replaces the degree
  radius and the `neighborSearchParameters`, `gridSize` and `tableCapacity` props. `neighborCapacity` is
  optional (defaults to the largest disk); truncation sets the `weights` status `overflow`.
- `distanceBand` (optional, exclusive with `neighborRings`) uses a metric band instead: `GPUNeighborSearch` in radius
  mode over the cell centers with `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`,
  so the radius in `neighborSearchParameters` is in meters (the context's `units`). Any other context throws: a
  radius in degrees is an ellipse that narrows east-west with latitude. `minimumNeighbors` still applies.
  `distanceBand.rowStandardize` (default `false`, classic binary Gi*) row-standardizes the band through the
  search's compile-time `rowStandardize`, overriding the per-frame field of `neighborSearchParameters`. The
  described weights (`information.transform` `'row'` or `'binary'`, and the search status) feed `GPUHotSpotAnalysis`.
  Every source publishes the unclamped neighbor total as the `weights` stage `requiredCount` (the grid-disk
  adapter included), for `getGPUSpatialWeightsCapacityPlan`.
- `analyze: 'density'` (default) analyses value (or count) per square kilometer. Web Mercator tiles shrink
  with the cosine of latitude, so raw Quadbin counts compare cells of different area; `'count'` is allowed for
  H3 only (cells are approximately equal area) and throws for Quadbin. Quadbin area is the exact spherical
  rectangle `R^2 dLon (sin(lat1) - sin(lat2))` and H3 area comes from `GPUCellMeasures`, both on the authalic
  sphere (`GPU_EARTH_AUTHALIC_RADIUS_METERS`, which is H3's own radius), so the families share one model.
  Quadbin areas are within f32 rounding (about 1e-6 relative); H3 pentagons and edge cells above resolution 5 are
  unreliable (see `GPUCellMeasures`).

Statistics and legend (both sources):

- `minimumNeighbors` defaults to 8 (ArcGIS guidance for analytic Gi*). Rows with fewer neighbors other than themselves
  get NaN `zScores` and `pValues`, bin 0, and are outside the multiple-testing family;
  `status` stage `minimum-neighbors` reports `invalidCount`. Quadbin with one ring has exactly 8 neighbors at
  most, so the cells on the study-area border are NaN; use `neighborRings: 2` or a lower minimum for them.
- `multipleTesting` defaults to `'benjamini-hochberg'` (it replaces `falseDiscoveryRate`). Legend contract,
  stable under any setting: bins `+-1`, `+-2`, `+-3` are 90%, 95% and 99% confidence hot (`+`) or cold (`-`)
  spots; without correction that is `|z| >= 1.645, 1.960, 2.576`, with Benjamini-Hochberg the step-up test at
  q = 0.10, 0.05, 0.01. `result.inference` is a `GPUInferenceMetadata` (alternative `'two-sided'`, analytic so
  `permutationCount: 0`, the method, and `testedRowsDefinition`, for example
  `'included rows with at least 8 active neighbors (islands excluded) and a defined z-score'`);
  `result.permutationInference` describes the optional permutation confirmation. Both come from
  the contributors' `getInferenceMetadata` of the one `settings` object (`GPUHotSpotAnalysisRecipeSettings`:
  `significanceLevel`, optional fixed moments, `seed`, `permutationCount`) that the recipe packs into its
  `parameters` and `permutation.parameters` `GPUParameterBuffer`s; `result.setSettings` repacks them per frame.
- The lattice source keeps its queen/rook weights and gets the same guard and metadata.
  Values 0 and 1 of `minimumNeighbors` both use an effective minimum of one: masked rows, islands
  and undefined scores stay outside the analytic family. The contributor's full eligibility
  description is preserved when settings change.
- Permutation confirmation does not apply `minimumNeighbors`. Its family contains included finite-value
  rows with 1 to `permutation.maximumNeighbors` positive-weight neighbors (default maximum 32), as stated
  by `result.permutationInference.testedRowsDefinition`. The analytic family has its own description in
  `result.inference.testedRowsDefinition`; on a 5×5 queen lattice at the default minimum these families
  contain 25 and 9 rows respectively. Multiple-testing correction uses each stage's actual family.
- The permutation confirms the statistic the Gi stage maps. `GPULocalPermutationTest` has a fixed focal
  weight (1 for `'localGStar'`, none for `'localG'`), so `permutation.statistic` defaults to `'localGStar'`,
  or to `'localG'` when `selfWeight` is 0. `'localGStar'` with a `selfWeight` other than 1, or `'localG'` with
  a `selfWeight` other than 0, throws at construction. `output.permutation.observed` (optional, `float32`
  per row) receives the permutation test's observed statistic in esda scaling, so the two stages can be
  compared.

#### How it computes

{/* lens:addHotSpotAnalysisRecipe */}

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addRateClusterMapRecipe`

For Kepler curated rate publication, BH selects the largest rank `k` with
`p_(k) <= k * alpha / m` and rejects ranks 1 through `k`. Rejection can be possible without
floor ties. `sufficientMinimumPValueTieCount` verifies a sufficient scenario using the selected
engine arithmetic (default `cpu-reference`, optional `gpu`). GPU floors and cutoffs use stored float32 values.

Turns bird and total observation counts per tract into smoothed rates and significant local Moran clusters.

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=result caption="Bird observations over all observations in 199 tracts; quadrants mark significant clusters."}

Event counts and populations per polygon to significant local Moran clusters of empirical-Bayes rates (esda
`Moran_Local_Rate`).

- `analyze` is `'standardized'` (default, Assuncao-Reis) or `'smoothed'` (global empirical-Bayes rate).
  `clampPriorVariance` (default `true`) is passed to `GPUEmpiricalBayesRates` and only changes the `'smoothed'`
  path: the prior variance is clamped at zero (Marshall), so counts no more dispersed than Poisson shrink fully
  to the pooled rate. `false` reproduces esda `Empirical_Bayes`. The standardized path is unaffected.
- `multipleTesting` defaults to `'benjamini-hochberg'` (it replaces `permutation.falseDiscoveryRate`) and applies
  to whichever inference runs: the analytic `GPULocalMoran` p-values without `permutation`, or the permutation
  pseudo p-values with it. The family is every polygon with a valid rate and at least one neighbor, never a viewport.
- Analytic inference is a preview (two-sided normal approximation of the conditional-randomization moments).
  Permutation inference is the reference; esda uses 999 permutations and so should you. The alternative defaults to
  `'two-sided'`. Analytic p-values are exposed as `result.analytic`, permutation ones as `result.permutation`
  (`pseudoPValues`, and `adjustedPValues` after the correction).
- Islands and polygons with a NaN rate (zero population, masked) end at quadrant 0 and have NaN p-values: they are
  outside the test family, not counted as tested but not significant.
- Legend contract, stable under any setting: palette index 0 is "not significant" and 1, 2, 3, 4 are high-high,
  low-high, low-low, high-low. The correction changes which polygons reach significance, never what a color means.
- `palette` must contain at least five packed `uint32` entries. Supplied `output.colors` and
  `output.quadrants` must each have exactly one logical row per polygon feature, even when their
  backing buffers have spare capacity. Independent outputs must use separate buffers and must not
  alias any original input, including the palette and parameter buffers. Active scratch is checked
  with the retained outputs too: for example, `scratch.spatialLag` cannot overwrite `smoothedRates`.
  Invalid ports are rejected
  before scheduling, so corrected views can reuse the same recipe ID.
- `events` and `populations` accept `float32` or `uint32` (for example the `counts` of
  `addPointsInPolygonsChoroplethRecipe`); `uint32` columns are converted by one node (exact below 2^24).
- Rows are features: `polygons` is a `GPUPolygonGeometryPort` (four-level layout, one packed `positions`
  view), so multipart features get one row and the `counts` of `addPointsInPolygonsChoroplethRecipe` on the same
  features chain in directly. The `weights` stage publishes `requiredCount` for `getGPUSpatialWeightsCapacityPlan`.
- Settings have one source of truth. Pass `GPUParameterBuffer`s as `parameters` and `permutation.parameters`
  plus `settings: {significanceLevel?, seed?, permutationCount?}`: the recipe packs the settings into both buffers
  and builds `result.inference` from the same object, and `result.setSettings(settings)` repacks them per frame
  (an invalid setting throws and writes nothing). Caller-packed views are not accepted.
- `result.inference` is the `getInferenceMetadata` of the `GPULocalPermutationTest` the recipe built (or of
  `GPULocalMoran` without `permutation`). It holds the alternative, `permutationCount` (0 for analytic), `seed`,
  `significanceLevel` (default 0.05), `multipleTesting` and `testedRowsDefinition`. An invalid permutation
  count or seed throws when the recipe is built.

#### How it computes

{/* lens:addRateClusterMapRecipe */}

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addPeriodComparisonRecipe`

Counts two periods of observations per Quadbin cell and returns the classified, coloured change.

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=result caption="655 spring and 566 autumn observations compared per cell; classes diverge from no change."}

Two periods of events to a classified, colored change map.

For `measure: 'sum'`, pass one `sumScale` (default 65536) for both aggregation stages and the
comparison. For example, `sumScale: 2 ** 18` preserves a one-cell change from 0 to `2 ** -18`;
the default scale rounds that value to zero. Choose a scale with `getGPUCellSumScale` using the
largest magnitude and group row count across both periods.

The comparison stage separates destination `overflow` from `candidateOverflow` inherited from
either aggregation table. Its `requiredCount` is the retained-input union requirement and becomes
a lower bound after source loss. Inspect `status.stages` to find the capacity that needs growing;
adding union slots cannot recover an earlier aggregation's dropped cells.

- `delta` (after minus before) is always defined. `ratio` and `percentChange` are NaN, not 0, for cells whose earlier
  value is 0; NaN rows are skipped by `GPUClassBreaks` and take the palette no-data color.
- `zScore` is a descriptive score of the change (`'poisson'`: `delta / sqrt(before + after)`, or `'standardized'`
  deltas). It has no null distribution and no multiple-testing correction, so a threshold on it is not a significance
  level; use `addHotSpotAnalysisRecipe` or `addRateClusterMapRecipe` for inference. With `measure: 'count'` every
  union cell has `before + after > 0`, so the Poisson denominator is never zero.

#### How it computes

{/* lens:addPeriodComparisonRecipe */}

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addChangeOfSupportRecipe`

Moves tract observation counts onto community areas by raster overlap area and reports what each tract could not transfer.

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=result caption="199 census tracts rasterized with 13 community areas; counts are redistributed by overlap area."}

Transfers values between two polygon zone systems by area: both systems are rasterized to one grid
(`GPUPolygonRasterization`), `GPUArealInterpolation` builds the overlap areas and both weight sets, and
`GPUSpatialLag` moves `sourceValues` to the targets as `extensiveValues` and `intensiveValues`.

- Props: `grid` (`GPURasterMetadata`), `parameters` (`getGPUPolygonRasterizationParameterValues(grid)`),
  `source` and `target` zone systems (`GPUPolygonGeometryPort` plus `crossingCapacity`),
  `pairCapacity`, optional `cellWeights`, `sourceValues` and `columnCount`, `categories`, `denominator`,
  `sourceAnchorCells`, `spatialContext` and `output` / `scratch` overrides.
- Denominators (default change): extensive `'zone'`, intensive `'overlap'`, so a target only partly
  covered by sources is the mean of what covers it. Pass `denominator: 'zone'` for the old intensive result.
- Conservation: the result always carries `sourceAreas`, `sourceTransferredFractions` and `status.invalidCount`;
  the last one is also in the `areal-interpolation` stage `status`. A source entirely outside the targets,
  or smaller than a cell, shows up there instead of silently losing mass; `sourceAnchorCells` rescues the
  latter.
- Overflow overrides: `output.extensiveWeights.status.overflow` is the destination
  when `output.status.overflow` is absent. If both are supplied, their buffers and byte offsets
  must match. The recipe writes and returns the caller's destination and passes it to both lags;
  a dropped overlap pair therefore produces overflow one and NaN transferred values.
- `output.extensiveWeights.status` supports one-row uint32 `count` (the readable CSR prefix),
  `requiredCount` (unclamped distinct pairs), `candidateOverflow` (reset to zero on this route),
  and `overflow`. A nested `requiredCount` may differ from `output.status.requiredCount`; the
  producer writes both. Distinct destinations must use separate physical buffers. Unsupported
  diagnostics such as `complete`, `unresolvedCount` and `invalidCount` reject with their full paths.
- Coordinates: `spatialContext` omitted is native planar (cell counts are areas). `longitude-latitude`
  requires metric `'great-circle'` and declares a longitude/latitude raster whose `parameters` are in degrees.
  The areal stage weights each cell by its area on the authalic sphere in the context's length unit squared
  (square meters by default; square kilometers with `units: 'kilometers'`). It measures area, so
  `local-azimuthal-equidistant` frames throw. Rasterization itself only
  tests cell centers with straight polygon edges in degrees. For antimeridian crossings or great-circle
  edges, set `prepare: true` or prepare with `GPULonLatPolygonPreparation`.
  The resolved context is returned as `result.spatialContext`.
- `prepare` defaults to false and requires a longitude/latitude context, `preparedPositionCapacity`
  (applied separately to each zone system), and a per-frame `preparationParameters` view from
  `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`. Preparation runs before each
  rasterization and preserves zone IDs. Stages `source-preparation` and `target-preparation` expose
  `overflow`, `requiredCount`, `invalidCount` and `approximation`.
- Accuracy is bounded by the raster resolution (cell centers decide membership), never an exact polygon
  intersection.

#### How it computes

{/* lens:addChangeOfSupportRecipe */}

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addPointsInPolygonsChoroplethRecipe`

Counts points per polygon and classifies the counts into a coloured choropleth.

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=result caption="2,344 observations counted in 199 tracts and shaded in five quantile classes."}

Points to a per-polygon statistic to class breaks to colors, with `backend: 'zonal'` (default,
`GPUZonalStatistics`) or `'group'` (`GPUPointInPolygonJoin` then dense `GPUGroupStatistics`). `statistic`
is `'count'` (default), `'sum'`, `'mean'`, `'minimum'` or `'maximum'`; features with no points keep NaN (mean,
minimum, maximum) or 0 (count, sum).
For multipart points (`featureOffsets`), both backends use the same feature-level membership
reduction: a feature contributes once to every polygon matched by any point part, under
`polygons.boundaryRule`. Values must be packed float32 with one row per input feature; an offset
packed view is accepted, while a strided view is refused. A provided `featureValues` output must
be packed float32 with one row per polygon and use a separate buffer from inputs/counts.
Repeated parts in one
polygon do not multiply a value; parts in different polygons contribute to each, so counts summed
over polygons can exceed the input feature count. Empty features contribute nothing. The join
uses a packed feature-by-polygon mask (four bytes per pair), bounded by the device storage-binding
limit, only for value statistics. Count-only multipart recipes request feature counts directly and
do not allocate that matrix. CPU count aggregation and ordinary point value statistics, including
their operation forms, also reuse sparse row assignments instead of retaining a feature-by-polygon
matrix. Dense CPU membership is reserved for multipart value statistics; explicit membership callers
retain the full matrix. Empty point inputs, with or without `[0]` feature offsets, return zero counts
and sums, NaN means/minima/maxima and zero uncertainty through both backends and operation forms.
Ordinary point vectors preserve their assignment chunk layout on both backends and in zonal
distribution, including nonempty chunks, zero chunks and multiple empty chunks.
Overflow is zero only when polygon leaf/candidate capacities are complete: two polygons with
`leafCapacity: 1` still report overflow `1` even with no point queries. The empty reduction preserves
that join diagnostic. Caller `featureValues` buffers must be separate from both join status words.
The CPU function captures declared geometry and option fields, including inherited getters for
offsets, `boundaryRule` and `candidateCapacity`; property ownership cannot change grouping or boundary semantics.
Options supplied by a Proxy getter are read once before resolving defaults. Both choropleth
backends and `addZonalDistributionRecipe` group by polygon rows (`pointFeatureRows`), independently
of caller identifiers. These routes never read point or polygon `sourceIds`, including unused getters.
The published GPU and CPU choropleth operation factories likewise ignore unused ID descriptors
and do not declare logical ID inputs. Inherited source discriminants retain the point-assignment
stage and its loss status in zonal distribution.
Candidate loss sets recipe `status.overflow`; dependent counts/statistics can then be low.
The `point-in-polygon` stage retains the join's completeness and uncertainty status.
Value reductions use dense `GPUGroupStatistics` over distinct memberships for both backend
choices. Its fixed-point grouped sum preserves representable cancellation residuals and does
not scan all input features serially per polygon. Nonfinite polygon candidates increment
uncertainty while valid polygons retain their aggregates.

`computePointsInPolygonsChoroplethOnCPU` computes the packed planar aggregate stage for all five
statistics and both backend choices. It uses the exact CPU membership classifier, float64
sum/mean arithmetic and one final float32 rounding. Empty polygons have zero count/sum and NaN
mean/minimum/maximum. Candidate/leaf loss propagates to `overflow`, so partial statistics are never
reported complete. On loss, CPU retains unsorted polygon-row prefixes; a spatially sorted GPU
recipe can retain a different incomplete prefix for larger polygon sets. The public staged pilot factories
`createPointsInPolygonsChoroplethOperation` and `createCPUPointsInPolygonsChoroplethOperation`
(from `@luma-contributors/analysis/gpu-spatial-analysis`) share this aggregate-stage contract.
They construct named GPU/CPU aggregate operations; they do not reproduce the full colored map.
Use them in the published graph with `GPU_CONTRIBUTOR_OPERATION_LOWERING` for GPU lowering. Count mode publishes `counts` as its feature values; other modes
also publish `featureValues`. Class breaks, palette mapping and geographic preparation remain
separate optional recipe stages, outside these planar aggregate adapters.

- Coordinates: `spatialContext` omitted is native planar. `longitude-latitude` is accepted because the
  mapped statistics are topological: containment uses straight polygon edges in degrees (not geodesics) and
  for antimeridian crossings or great-circle edges, set `prepare: true` or prepare with
  `GPULonLatPolygonPreparation`. No density is mapped, so no polygon area is needed; for
  a density per square meter use `GPUZonalStatistics` directly with spherical `areas`. Other spaces throw.
  The resolved context is returned as `result.spatialContext` (linear edges after preparation).
- `prepare` defaults to false and requires a longitude/latitude context, `preparedPositionCapacity`,
  and a per-frame `preparationParameters` view from
  `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`. Both backends prepare before
  containment; feature rows and output counts retain their original IDs. Stage `polygon-preparation`
  exposes `overflow`, `requiredCount`, `invalidCount` and `approximation`.

#### How it computes

{/* lens:addPointsInPolygonsChoroplethRecipe */}

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addClusterAndOutlineRecipe`

Clusters points with DBSCAN and outlines each cluster with its convex hull, area and perimeter.

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=result caption="2,344 observations clustered at 250 m with six points minimum; each cluster is outlined."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addClusterAndOutlineRecipe */}

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addFleetDwellRecipe`

Kepler fleet dwell and track quality require distinct string spellings of track IDs before GPU
allocation: numeric `3` and text `"3"` must be assigned distinguishable string IDs.

Finds where vessels stop and totals the dwell time per anchorage zone.

The stops variant forwards `trajectories.temporalContext`, `timestamps` and optional `timestampsLow`
to `GPUTrajectoryMetrics`. Float32-relative, Int64 words and double-single times are supported; the
declared encoding must match the bound views. Durations use the declared tick unit. The zone-events
variant takes `GPUFleetDwellZoneEventsTrajectories`: float32-relative or Int64 words only. It rejects
`timestampsLow` and incompatible declarations before scheduling nodes, and forwards supported contexts
to `GPUZoneEvents`.
Both variants reject incompatible time declarations before registering their recipe outputs, so
the caller can correct the declaration and retry on the same graph with the same recipe ID.

Only the GPU-written `stops.status.count` prefix enters the zone join. A private adapter view gives
unused slots nonfinite coordinates, which the join excludes before counting candidates; the public
centroids keep the metrics contributor's zero-filled padding. An empty stop list therefore cannot
consume `zones.candidateCapacity` or set `status.joinOverflow` because of spare stop slots.
`joinOverflow` still reports capacity loss among real stop candidates, and `joinUncertainCount`
reports unresolved containment among them. Check those words and the metrics stage's overflow before
using a per-zone summary as complete.
When caller-owned centroids and a stop mask occupy disjoint subviews whose aligned storage bindings
overlap, the recipe produces the mask and private join positions in separate dispatches. For example,
two centroids at bytes 0–15 and two mask values at bytes 16–23 of one buffer remain supported.

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=result caption="12 vessel tracks over 13 anchorage zones; zones are shaded by total stop time."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addFleetDwellRecipe */}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addSpatialRegressionRecipe`

Fits observations per tract on income and car-free households and tests the residuals for spatial dependence.

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=result caption="199 tracts; residuals of the fit show where observations exceed or fall short of the model."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addSpatialRegressionRecipe */}

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addStraightLineCatchmentsRecipe`

Allocates every raster cell to its nearest facility and sums a value raster per catchment.

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=result caption="150 facilities split the North Side into nearest-facility catchments on a 250 m grid."}

The chain of this recipe is listed in the chain table above.

`output.statistics` supports partial overrides of exactly six columns: `cellCounts`, `valueCounts`,
`sums`, `means`, `minimums` and `maximums`. Each supplied destination is retained; omitted columns
use graph storage. Counts are cells per facility zone; sums, means and extrema use the value
raster's units. `statistics.status.overflow` is the one-row uint32 out-of-range-zone flag and must
share the outer `status.overflow` destination. Moment and dictionary destinations, including
`populationVariances`, and other unwritten status destinations reject before allocation; use
`GPURasterZonalStatistics` directly to request those columns.

#### How it computes

{/* lens:addStraightLineCatchmentsRecipe */}

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

## Ancillary-weighted redistribution

[Dasymetric redistribution](./dasymetric-redistribution.md) combines fractional source-cell coverage,
an ancillary raster and explicit whole-cell target ownership. Source totals remain traceable through
allocated and unallocated outputs; the recipe does not approximate target polygon intersections by
multiplying marginal coverage fractions.
