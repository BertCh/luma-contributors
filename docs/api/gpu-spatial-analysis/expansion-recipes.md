# Geospatial workflow recipes

Choose by task in the [recipe index](./recipes.md). Every section below is a **GPU graph recipe**,
not a submission or a CPU fallback. Related task guides cover [home ranges and recursions](./home-ranges-and-recursions.md)
(**GPU contributors and GPU/CPU operations**), [landscapes](./landscape-ecology.md) and
[urban morphology](./urban-morphology.md) (**GPU graph recipes**). Historical headings, capability
IDs and lens recording associations remain unchanged.

All recipes exported from the spatial and network recipes entry points use the
[shared preparation and retry contract](./recipes.md): validation and expansion failures leave the
caller graph unchanged, including when a raster wrapper composes another recipe.

These recipes add nodes to a caller-owned `GPUCommandGraph`. They never submit work, read back a
result, fit a model on the CPU, or silently change coordinate systems. Import spatial, raster and
terrain workflows from `@luma-contributors/analysis/gpu-spatial-analysis`; import network scenarios and the track-quality/map-matching composite from
`@luma-contributors/analysis/gpu-network`.

Every result exposes `outputs`, `intermediates`, `contributors` and `status.stages`. Inspect the
stage status as well as the numerical output: a partial routing result or an incomplete geometry
is not a completed analysis. Graph-owned transient outputs can feed another contributor or renderer.
Use a caller-owned output override where exposed, or an explicit graph copy to a caller-owned
readback buffer. Destroy the compiled graph after its last submission has completed.

Catalog kinds describe each recipe's actual ports. Local variability consumes CSR spatial weights,
values and parameters and returns per-row attributes; it has no raster input or network output.
Zonal distribution returns attributes and summaries from values with zones or a point/polygon join;
its raster adapter accepts a surface. Exposure returns zone summaries, regional raster change also
returns a difference surface, and interpolation produces a surface plus sample diagnostics from
point observations. Drainage is a surface-to-network workflow. Cell-flow topology takes H3 cells
and an OD pair table and returns cell/edge attributes, topology and boundary geometry. Surface
preparation returns the repaired surface, provenance attributes and an unresolved count.

`addRateClusterMapRecipe` returns the row-standardized contiguity weights description and carries
the producer's loss words through local Moran and permutation inference. If one of two equal-weight
neighbors has a missing rate, the surviving neighbor is restandardized from weight 0.5 to 1.
The returned weights retain that same description for subsequent consumers.
Rate-cluster settings updates capture each declared setting once for the analytic parameter buffer,
permutation parameter buffer and inference metadata, including inherited getters.

`addSpaceTimeHotSpotsRecipe` captures initial settings and each `setSettings` update once,
including inherited and non-enumerable declared settings. Packed GPU parameters and inference
metadata use that same capture. A rejected update changes neither the buffer nor the metadata.

`addStopValueCubeRecipe` treats a weights `requiredCount` greater than the retained terminal CSR
offset, allocation capacity or supplied live `count` as final loss. Its `overflow` word becomes one
and the cell-major counts are suppressed to zero. A later submission whose requirement fits can
clear that loss and restore counts; candidate loss remains a separate cause.

## Zonal distributions

`addZonalDistributionRecipe` accepts calibrated float32 values and either dense zone IDs or point
and polygon inputs for `GPUPointInPolygonJoin`. It returns dense zone IDs, assigned row counts,
finite-value counts, medians, modes, unique counts and optional quantiles. Empty zones remain in the
output, with zero counts and NaN order statistics. Modes break ties toward the lowest value;
quantiles use linear interpolation. Probabilities are supplied in a live float32 view.

The point path assigns each observation to the join's single deterministic owning feature. Supply
explicitly expanded zone/value rows when one observation should count in multiple overlapping zones.
Source feature IDs are retained in the input; output zone IDs are feature row indices.
The point branch accepts `source.join.prepared` when that `GPUSpatialJoinPrepared` has already been
added to the caller graph. Preparation mirrors its geometry and storage for validation; the
committed join retains the original handle. An undeclared handle is rejected before any recipe
nodes or resources are registered.

`addRasterZonalDistributionRecipe` additionally accepts `grid`, `input`, `zones` and `zoneCount`.
It resolves native-format nodata and validity before applying scale/offset. These are unweighted
whole-cell quantiles. Fractional coverage does not turn them into weighted quantiles.
Every assigned cell contributes to `rowCounts`, including cells whose calibrated value is missing.
Only finite calibrated values contribute to `validCounts` and the distribution statistics: zone IDs
`[0, 0]` with values `[4, NaN]` yield row count `2`, valid count `1` and median `4`.
Caller outputs must not alias the original raster band, its validity storage, zone IDs or
probabilities, even though the nested distribution reads a calibrated intermediate. Distinct
outputs also require separate buffers; disjoint offset views of one buffer still alias.
Only declared zonal fields are forwarded to the nested recipe; auxiliary metadata such as `preview`
is ignored. A genuinely foreign raster input still fails graph ownership validation. The ordered
`contributors` list includes calibration before the zonal statistics stage (and both calibration
stages before exposure statistics in `addExposureByZoneRecipe`).

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {addRasterZonalDistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Caller-imported, aligned whole-cell views, all owned by graph.
declare const graph: GPUCommandGraph;
declare const grid: GPURasterMetadata;
declare const elevationBand: GPURasterBufferBand;
declare const zoneIds: GraphDataView<'uint32'>;
declare const administrativeAreaCount: number;
declare const quantileProbabilities: GraphDataView<'float32'>;
declare const medianOutput: GraphDataView<'float32'>;
declare const quantileOutput: GraphDataView<'float32'>;

const distributions = addRasterZonalDistributionRecipe(graph, {
  grid,
  input: elevationBand,
  zones: zoneIds,
  zoneCount: administrativeAreaCount,
  probabilities: quantileProbabilities, // e.g. [0.1, 0.5, 0.9]
  output: {medians: medianOutput, quantiles: quantileOutput}
});
// distributions.status.stages retains grouping overflow and, for point sources, join status.
```

## Network disruption

Kepler curated publication requires exactly one `baseline-routing` and `scenario-routing`
`converged=1` certificate and one `baseline-accessibility` and `scenario-accessibility`
`complete=1` certificate before publishing network comparisons; any additional supplied
certificate loss also blocks publication.

`addNetworkDisruptionRecipe` accepts a cost-matrix `search`, explicit `rowCount`, opportunity
weights, accessibility parameters, and an optional edge mask or replacement edge costs. Zero mask
words close edges. The adapter retains the original CSR slots, including parallel edges, so IDs
remain stable. Targets, when supplied, define the opportunity-weight column order.

Outputs include baseline/scenario matrices, cumulative accessibility, accessibility differences,
finite pair-cost differences and a pairwise newly-unreachable mask. Cost differences are NaN when
both costs cannot be compared. Newly-unreachable flags are set only after both searches converge.
With a finite search cost cutoff, “unreachable” means outside that routing horizon; it does not
establish global disconnection. Always retain the two routing convergence words.

To avoid rerunning the baseline, retain and import its caller-owned costs and convergence. For a
noded search, also retain that build's `baseline.status.sourceComplete`; current scenario noding
cannot certify the retained build. Pass `{costs, status: {converged, sourceComplete}}` for that route.
`baseline.costLimit` supplied as a view is the retained build horizon; `null` declares an unbounded
baseline. Omitting it reuses `search.costLimit` and promises the identical search configuration.
Keep the effective horizon fixed with those costs: a larger live scenario request cannot extend
old coverage. Retain starts, targets, topology/weights and build configuration for invalidation.
Import the original caller-owned buffers into the new graph; do not reuse another graph's transient
views.

Accessibility is cumulative opportunity weight within the supplied threshold. Routing convergence
alone does not certify it: the effective routing horizon must also reach the scoring threshold.
`status.baselineAccessibilityComplete` and `status.scenarioAccessibilityComplete` are separate
one-word certificates for the two scores. A score whose certificate is zero is an incumbent total,
not a certified threshold score. `accessibilityDifferences` is NaN unless both certificates are one
and both scores are finite. For one unit opportunity at baseline cost 1 and scenario cost 2,
threshold 2 and search cutoff 1, both searches can converge, but both score certificates are zero:
both scorers see the same insufficient horizon, even though the baseline happened to reach its
opportunity. The difference is NaN. With an unbounded completed search, both scores are 1 and the difference is 0.
The result exposes these individual words next to `status.stages` and as the
`baseline-accessibility` and `scenario-accessibility` stages' `status.complete`. The joint
`accessibility` stage publishes their conjunction as a one-word `status.complete` (also
`status.accessibilityComplete`), rewritten on every submission. Look it up with
`status.stages.find(stage => stage.stage === 'accessibility')?.status.complete` and require a
successfully read nonzero word before publishing joint scores. Retain both individual words to
diagnose the cause. The two routing stage entries alone cannot certify a score.

Keep pair evidence separate. Routing completeness combines convergence and, for noded inputs, that
build's source certificate. `costDifferences` requires both routing certificates and finite pair
costs. `newlyUnreachable` uses the common smaller baseline/scenario horizon: the baseline trip must
be finite and within it; the scenario trip may be infinite or finite above it. For baseline cost 3
and scenario cost 7 with common horizon 4, the finite difference is 4 and the newly-unreachable
flag is 1. An unchanged cost-7 trip with horizons 9 and 4 is outside the comparison domain and is
not newly unreachable. These pairwise results do not alone certify accessibility.

## Interpolation diagnostics

`addInterpolationDiagnosticsRecipe` takes the existing `GPUKriging` inputs, including an explicitly
supplied variogram and neighborhood. It returns the prediction raster and variance, leave-one-out
sample predictions/variances, residuals, standardized residuals, bias, MAE, RMSE and finite counts.
Masked samples and unusable predictions are excluded. A standardized residual needs a finite,
strictly positive variance and a finite quotient; zero-variance duplicates remain undefined.

Summary counts and error metrics use row contributions and published parallel reductions. Residuals
are divided by their maximum absolute value and finite count before summing or squaring, so means
and RMSE remain finite even for maximum finite float32 residuals. Normalization uses the shared
scaled quotient (bounded mantissas and integer exponents) to avoid an underflowing reciprocal of a
near-maximum residual scale. All-invalid validation sets return zero valid counts and NaN metrics;
all-zero residuals return zero metrics. Reduction rounding can differ from the former streaming
means (absolute tolerance `1e-6` plus `2e-6` times the maximum residual
magnitude). Work remains O(sampleCount), with hierarchical reductions instead of a serial sample
scan. Separate invalid/uncertain raster statuses and invalid validation counts remain visible. No
variogram fitting, automatic model selection or residual significance is implied.

## Facility scenario evaluation

Independent output buffers must be physically distinct, including when different graph handles
import one buffer. In particular, `nearestCosts` cannot alias `accessibility`; correcting the outputs
allows a retry with the same recipe ID before any caller nodes or resources have been added.

`addFacilityScenarioRecipe` reads a complete, reusable facility-major cost matrix, demand, supply,
a live facility mask and network-accessibility parameters. Outputs are gravity accessibility,
nearest selected facility/cost per demand row, covered/uncovered demand, and invalid-demand count.
Equal nearest costs choose the lowest facility row. Nonfinite or negative demand is omitted and
counted; unreachable valid demand remains uncovered. An empty selected set leaves every nearest
cost infinite and facility ID `0xffffffff`. Covered/uncovered demand and invalid counts use parallel
row contributions and published reductions. Demand totals use float32 tree sums; acceptance tolerance
is `1e-6 + 2e-6 * abs(total)` against a float64 sum.

Run the recipe once per scenario with a distinct ID and the same matrix/demand to compare results.
Changing masks or scoring parameters does not rerun routing. The caller must ensure the matrix's
routing search completed and its horizon covers the scoring threshold. Evaluation does not select
facilities or impose capacities; use the separately documented bounded allocation contributor for
those objectives. Nearest-cost outputs can feed the zonal distribution recipe for travel-cost
quantiles, with unreachable demand counted separately.

## Retaining raster and drainage terminals

Kepler drainage publication uses `unresolvedCount` for downstream-distance and stream-network.
Missing unresolved evidence stays missing.

The following terminals default to graph-owned transient views. These workflows currently require
explicit copies for retained/readback results; supplying an undeclared destination property does
not create a writer. Add copies for each required leaf **before compiling**, into caller-owned
buffers imported in the same graph, with matching formats and logical lengths. Submit and await
the compiled graph, then read the destination buffers. Keep the graph alive through its final
submission; copy/capture the status words from the same submission as the numbers. Readbacks after
destroying a transient allocation cannot retain its result.

| Workflow | Actual `result.outputs` terminals to retain |
| --- | --- |
| Exposure by zone | `totalPopulation`, `observedPopulation`, `exposedPopulation`, `coverage`, `exposedShare` (float32, one row per zone); `thresholdInvalidCount` (one uint32 word) |
| Regional raster change | `difference` (float32, one row per cell); `validCounts` (uint32 per zone), `meanChanges`, `changedArea`, `validArea`, `totalArea`, `coverage` (float32 per zone); `thresholdInvalidCount` (one uint32 word) |
| Drainage network | `flowDirections`, `streams`, `streamOrder`, `catchments` (uint32 per cell), `accumulation` (float32 per cell); `distance.distance` (float32), `distance.validity` and `distance.terminalCells` (uint32 per cell); every required leaf of `network` and `summary`, plus the stage evidence below |

Drainage `network` contains `sourceCells`, `targetCells`, `vertexOffsets`, `positions`, `lengths`,
`reachCount`, `vertexCount`, `unresolvedCount`, and `status.overflow`; only the compact prefixes
bounded by the live counts are meaningful. `summary` contains `catchmentIds`, `cellCounts`,
`meanDownstreamDistance`, `count`, and `overflow`. Retain these nested leaves individually, not the
JavaScript map. Capacity tails beyond compact counts are unspecified. Flow convergence words are
also available in `intermediates.flowStatus`. See each workflow below for numerical units.

## Exposure by administrative area

Kepler zonal exposure withholds statistics for certificate loss or capacity overflow. Summary
`overflow` and `candidateOverflow` report their named flags; `complete` describes the result,
and `recipe-incomplete` reports failed certificates separately from `recipe-overflow`. Omitted
optional certificates remain supported.

`addExposureByZoneRecipe` accepts aligned hazard/population buffer bands, a common grid, whole-cell
zone IDs and a live one-row threshold. Values at or above the threshold are exposed. For
`weightMeaning: 'count'`, population is an extensive cell count. For `'density'`, `cellAreas` is
required and must use the area units of the density denominator.

Outputs are total known population, population with a known hazard, exposed population, observed
population coverage and exposed share **among observed population**. Missing hazards do not become
safe observations. Zero denominators produce NaN shares. Raw nodata, masks and calibration are
resolved independently for each band. A nonfinite threshold sets `thresholdInvalidCount` and
returns NaN exposed totals and shares while preserving observed population coverage. Grid
alignment and polygon-to-cell ownership are explicit upstream operations; this recipe does not
infer fractional boundary membership.

Pass the rasterizer's live loss word with its zone raster. For caller-imported views on the same
graph (the rasterizer writes both `zoneIds` and `zoneRasterizerOverflow` before this recipe):

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addExposureByZoneRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const inputs: Omit<Parameters<typeof addExposureByZoneRecipe>[1], 'zones' | 'producerOverflow'>;
declare const zoneIds: GraphDataView<'uint32'>;
declare const zoneRasterizerOverflow: GraphDataView<'uint32'>;
const exposure = addExposureByZoneRecipe(graph, {
  ...inputs, zones: zoneIds, producerOverflow: zoneRasterizerOverflow
});
const exposureStages = exposure.status.stages;
// Retain total-population, observed-population and exposed-population stage loss with outputs.
```

Retain `candidateOverflow` on the `total-population`, `observed-population` and `exposed-population`
stages as well as their final `overflow`. Their candidate word aliases the zonal contributor's
`incomplete` output: nonzero means the supplied zone raster lost support and the summaries are
suppressed as NaN. Such results are unavailable, not valid zeros. On a complete submission,
empty-zone population totals and exposed totals are valid zeros; zero-denominator shares remain
NaN. A later complete frame can recover within existing storage; increasing rasterizer capacity
requires rebuilding that producer and rebinding affected consumers.

## Regional raster change

Kepler curated regional raster-change publication treats `complete=1` as success and
`complete=0` as loss; informational counts do not enter its loss gate.

`addRegionalRasterChangeRecipe` accepts an aligned, calibrated, cell-major single-band slice stack,
explicit `GPUChangeDetection` comparison parameters, zone IDs, per-cell areas and a live absolute
change threshold. Encode missing samples as NaN and use the optional cell mask. The comparison is
`after - before` on common support.

Outputs include cell differences, per-zone finite comparison counts, unweighted mean differences,
changed area, valid comparison area, total area and area coverage. Areas must be finite and
positive. Means/counts describe finite difference observations; area accounting additionally
requires usable cell areas. Thresholds must be finite and nonnegative; otherwise
`thresholdInvalidCount` is set and changed area is NaN, while observed coverage is retained. These descriptive regional
summaries do not claim significance of a regional mean.

As with exposure, pass zone rasterization loss, rather than only its zone IDs:

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addRegionalRasterChangeRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const inputs: Omit<Parameters<typeof addRegionalRasterChangeRecipe>[1], 'zones' | 'producerOverflow'>;
declare const zoneIds: GraphDataView<'uint32'>;
declare const zoneRasterizerOverflow: GraphDataView<'uint32'>;
const change = addRegionalRasterChangeRecipe(graph, {
  ...inputs, zones: zoneIds, producerOverflow: zoneRasterizerOverflow
});
const changeStages = change.status.stages;
// Retain total-area, valid-area, changed-area and change-summary stage evidence with outputs.
```

Read every `total-area`, `valid-area`, `changed-area` and `change-summary` stage's
`candidateOverflow` and `overflow`. On producer loss, floating summaries are NaN and the zonal
integer counts are `0xffffffff`, not measured zero counts. These live words distinguish
suppressed summaries from valid zeros after a complete submission: an empty zone then has zero
area/count, while its mean and coverage ratio are undefined. Cell differences remain available
independently of the lost zone ownership. Keep status and numeric readbacks from the same frame.

## Cluster footprints

`addClusterAndOutlineRecipe` retains its convex default and exposes the separately documented
concave outline option. Alpha-shape geometry uses a circumradius threshold, with holes and
multiple pieces retained. An alpha radius is not Shapely's concave-hull ratio. Inspect geometry
completion and capacity status before consuming its measures.
Select `outline: {kind: 'alpha', maximumCircumradius}` and consume `concaveHull` mesh and boundary
outputs. Legacy single-ring convex-hull fields are empty in alpha mode. Areas, perimeters and
centroids describe the selected geometry.

## Drainage network analysis

`addDrainageNetworkRecipe` composes D8 flow, stream order, watersheds, downstream flow distance,
connected stream reaches and per-catchment distance summaries on one complete domain. Pass ordinary
`GPUTerrainFlow` inputs in `flow`; its live parameter buffer supplies the stream threshold. Optional
pour points are cell IDs. Depression filling and flat resolution remain explicit flow options.

The result includes directions, accumulation, stream mask/order, catchment labels, downstream
distance/terminal cells, and a compact network with source/target cell IDs, vertex offsets,
positions and lengths. Without pour points, catchments are keyed by terminal-cell IDs. With pour
points, they are keyed by pour-point row. Summary counts and mean distances include resolved valid
cells. The `downstream-distance` stage's scalar `unresolvedCount` counts unresolved valid cells;
`stream-network.unresolvedCount` aliases `network.unresolvedCount` and counts stream cells with
unresolved downstream traversal or junction membership.
These are different domains. Check them alongside fill/flat/accumulation, stream-order
and catchment convergence and final reach/vertex overflow; a finite subset summary does not certify
the unresolved part.

Reach capacity defaults to cell count and vertex capacity to twice cell count; both are explicit
options. Preserve fill, flat, accumulation, order, watershed and network completion evidence.
`maximumSteps` bounds tracing. This recipe does not stitch independent tiles or interpret branching
D-infinity/MFD flow as D8.

`distance.distance`, `network.lengths` and `summary.meanDownstreamDistance` use the grid affine's
units in the default `flow.cellSizeMode: 'uniform'`. With `'geographic'` (a longitude/latitude grid in
degrees) or `'web-mercator'` (an EPSG:3857 grid), all three use WGS84 ground metres. Both modes
require an axis-aligned grid. `network.positions` use the grid affine's coordinates in all modes.

## Track quality and route conformity

`addTrackQualityRecipe` is exported by `@luma-contributors/analysis/gpu-network`, since its optional map-matching stage consumes network topology. It preserves the trajectory's raw observation order and timestamp representation,
including double-single and Int64 words. Its live two-row threshold view holds maximum speed and
maximum gap in trajectory units. The recipe forwards `trajectory.temporalContext` to trajectory
metrics, preserving the declared unit, epoch and encoding; a declaration inconsistent with the
timestamp views is rejected before graph resources or nodes are declared, allowing a corrected
retry on the same graph. Raw flags use bits 1 invalid position, 2 nonpositive/nonfinite time step,
4 excessive/nonfinite step speed, 8 time gap and 16 invalid threshold. First observations
have no step checks but still report an invalid timestamp;
rows outside the track-offset domain carry `0xffffffff`. Per-track counts count flagged observations.

Optional `similarity` compares the raw tracks to explicit reference pairs using Hausdorff, Frechet,
DTW and LCSS. Its densification and dynamic-programming limits remain explicit, and `pairStatus`
retains failure evidence. `pairA` and `pairB` must have equal logical lengths. Invalid pair shapes
leave the graph unchanged, and corrected pairs can retry with the same recipe ID. Supplying
`frechetEpsilon` also returns `frechetWithin`. Optional `mapMatching` returns matched edge IDs, snap distances, breaks
and matched count, plus candidate and route-overflow stages. Route-search overflow must not be
interpreted as a reliable off-route result. Resampling is a separate upstream choice; raw flags do
not silently become flags on a resampled track.

By default bit 4 is the step-speed test, so the valid fix after an outlier is flagged too; set `speedOutlierMode: 'point-centred'` to flag a fix only when the speed test fails on both of its adjacent steps (see [network movement coverage](../gpu-network/network-movement-coverage.md)).

## Local variability comparison

`addLocalVariabilityRecipe` uses one explicit weights graph, values and mask for LOSH and local
Moran. It returns separate statistic and classification columns, per-row statuses and invalid counts.
LOSH is descriptive. Moran's analytic-normal p-values and optional correction apply to the Moran
family only; there is no combined significance score. See the statistical expansion documentation
for residual-power, island, bandwidth and inference contracts.

## Cell flow topology

`addCellFlowTopologyRecipe` reads distinct, same-resolution H3 cells and original OD pair rows.
It returns incoming/outgoing cell totals, canonical directed edges, neighboring-cell edge flows,
shared-boundary geometry and pair validity. It retains the original arbitrary pair table. Self and
nonneighbor pairs contribute to cell totals but do not become H3 edges. Each direct-scan invocation
examines at most `GPU_CELL_FLOW_TOPOLOGY_MAXIMUM_PAIRS_PER_PASS = 4096` pairs per pass, per cell or
directed-edge slot. Larger tables run ordered passes carrying the prior float32 accumulator; there
is no total pair cutoff or discarded tail. Work remains O(cells x pairs), including six edge-slot
scans per cell, so preaggregating pairs reduces the cost when appropriate. Aligned pair subviews
let all full passes of a scan share one shader; initialization runs separately. Each accumulator adds
valid rows in original input order with sequential float32 rounding. Summing cell totals regroups
those additions: outgoing, incoming and valid-pair sums need not be exactly equal. For example,
`16777216 + 1 + 1` rounds sequentially to `16777216`, whereas grouping the ones first gives
`16777218`. Edge-flow totals may be smaller because they exclude self and nonneighbor pairs.
H3 keys are exact; shared-boundary coordinates retain float32 H3 geometry precision.

The fleet zone-events recipe's `zone-statistics` stage exposes scratch-specific
`candidateOverflow`: nonzero means its dwell statistics may be incomplete. Per-track and final
event-list truncation are separate flags and leave dwell exact. The optional `visit-table` stage
publishes its own candidate-loss flag when supplied, otherwise the same scratch flag, even when
visit capacity did not overflow. Direct `GPUZoneEvents` callers can check optional
`output.status.candidateOverflow == 0`, or `output.status.candidateCount <= candidateCapacity`,
to establish candidate scratch completeness. Both diagnostics remain optional.

The spatial regression recipe adds a `geographically-weighted-regression` status stage only when
`localFits` is requested. Its `rowCodes` holds the contributor's per-observation `localStatus`
codes (0 successful, 1 singular, 2 excluded), rather than a one-row `code` or failed-row count.
The OLS and diagnostics stages publish the solver `code` and a separate excluded-row `status.invalidCount`.
`predictorCount` may be omitted when the packed predictor length divided by the nonzero response
length determines it: fourteen predictor values and seven responses mean two predictor columns.
An explicit count must match the row-major design; zero-row layouts require it. Input weights
`count`, `requiredCount`, `overflow`, `candidateOverflow` and positive `complete` evidence must not
overlap the bytes each recipe output actually writes. OLS and diagnostics codes write only word
zero; unused tails of oversized code destinations may contain independent evidence. Destructive aliases reject
before graph mutation, allowing a corrected retry on the same graph and ID. Incomplete weights
suppress diagnostics, specification and residual inference while the independent OLS fit remains
available; clearing the producer loss restores those downstream results on a later submission.
A supplied `complete` certificate is consumed with positive polarity: zero suppresses dependent
spatial inference even when counts match. Local fits support at most eleven predictor columns.
Stage `code` exposes exactly one row, and GWR `rowCodes` exposes exactly one row per observation,
even when the caller supplies larger output buffers. The original output overrides retain
their view lengths and offsets. Each OLS and diagnostics `status.invalidCount` exposes one
excluded-row count.

## Surface preparation with provenance

`addSurfacePreparationRecipe` applies bounded continuous nodata filling and an optional rank filter.
Its independent outputs must use separate physical buffers, including through separate graph imports;
read-only input sharing remains legal. The recipe captures declared output fields once, including
inherited getters, and checks the caller-provided boundary before allocating default outputs or
scheduling nodes. Each supplied output is also checked for packed scalar storage, its expected
format and exact row count before any defaults are registered. `values` is float32; the other
five outputs are uint32. Each output has `grid.width * grid.height` rows except `unresolvedCount`,
which has exactly one row. After a boundary or layout rejection, correct the supplied views and
retry on the same graph with the same recipe ID.
It returns a float32 buffer band, original validity, filled and filtered masks, downstream validity,
and unresolved-gap counts before and after filtering. Original valid calibrated observations are
preserved in fill-only mode; interpolated values never become donors during that fill pass.

`maximumDistance` is in pixels. `method: 'nearest'` chooses the nearest original donor;
`'inverse-distance'` weights all original donors in the disk. This is distinct from GDAL's
four-quadrant donor selection. Rank filtering uses a square footprint, explicit border/nodata rules
and the SciPy rank convention (upper middle for an even median population). Filling missing
measurements is separate from filling drainage depressions. Supply the halo needed by both stages
when preparing tiles and publish only nonoverlapping cores.
