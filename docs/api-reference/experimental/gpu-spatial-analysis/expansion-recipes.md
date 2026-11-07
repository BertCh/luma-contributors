# Geospatial workflow recipes

These recipes add nodes to a caller-owned `GPUCommandGraph`. They never submit work, read back a
result, fit a model on the CPU, or silently change coordinate systems. Import spatial, raster and
terrain workflows from `@luma-contributors/graph/gpu-spatial-analysis`; import network scenarios and the track-quality/map-matching composite from
`@luma-contributors/graph/gpu-network`.

Every result exposes `outputs`, `intermediates`, `contributors` and `status.stages`. Inspect the
stage status as well as the numerical output: a partial routing result or an incomplete geometry
is not a completed analysis. Graph-owned transient outputs can feed another contributor or renderer.
Use a caller-owned output override where exposed, or an explicit graph copy to a caller-owned
readback buffer. Destroy the compiled graph after its last submission has completed.

## Zonal distributions

`addZonalDistributionRecipe` accepts calibrated float32 values and either dense zone IDs or point
and polygon inputs for `GPUPointInPolygonJoin`. It returns dense zone IDs, assigned row counts,
finite-value counts, medians, modes, unique counts and optional quantiles. Empty zones remain in the
output, with zero counts and NaN order statistics. Modes break ties toward the lowest value;
quantiles use linear interpolation. Probabilities are supplied in a live float32 view.

The point path assigns each observation to the join's single deterministic owning feature. Supply
explicitly expanded zone/value rows when one observation should count in multiple overlapping zones.
Source feature IDs are retained in the input; output zone IDs are feature row indices.

`addRasterZonalDistributionRecipe` additionally accepts `grid`, `input`, `zones` and `zoneCount`.
It resolves native-format nodata and validity before applying scale/offset. These are unweighted
whole-cell quantiles. Fractional coverage does not turn them into weighted quantiles.

```ts
import {addRasterZonalDistributionRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

const distributions = addRasterZonalDistributionRecipe(graph, {
  grid,
  input: elevationBand,
  zones: zoneIds,
  zoneCount: administrativeAreaCount,
  probabilities: quantileProbabilities, // e.g. [0.1, 0.5, 0.9]
  outputs: {medians: medianOutput, quantiles: quantileOutput}
});
// distributions.status.stages retains grouping overflow and, for point sources, join status.
```

## Network disruption comparison

`addNetworkDisruptionRecipe` accepts a cost-matrix `search`, explicit `rowCount`, opportunity
weights, accessibility parameters, and an optional edge mask or replacement edge costs. Zero mask
words close edges. The adapter retains the original CSR slots, including parallel edges, so IDs
remain stable. Targets, when supplied, define the opportunity-weight column order.

Outputs include baseline/scenario matrices, cumulative accessibility, accessibility differences,
finite pair-cost differences and a pairwise newly-unreachable mask. Cost differences are NaN when
both costs cannot be compared. Newly-unreachable flags are set only after both searches converge.
With a finite search cost cutoff, “unreachable” means outside that routing horizon; it does not
establish global disconnection. Always retain the two routing convergence words.

To avoid rerunning the baseline, pass `{costs, status: {converged}}` as `baseline`, imported into the scenario
graph from retained caller-owned buffers. The caller must invalidate that baseline when topology,
weights, starts, targets or routing parameters change. Reusing transient storage from another graph
is not supported. Accessibility is cumulative opportunity weight within the supplied threshold.

## Interpolation diagnostics

`addInterpolationDiagnosticsRecipe` takes the existing `GPUKriging` inputs, including an explicitly
supplied variogram and neighborhood. It returns the prediction raster and variance, leave-one-out
sample predictions/variances, residuals, standardized residuals, bias, MAE, RMSE and finite counts.
Masked samples and unusable predictions are excluded. A standardized residual needs a finite,
strictly positive variance and a finite quotient; zero-variance duplicates remain undefined.

Summary means use a deterministic linear pass over samples. RMSE uses a scaled sum of squares to
avoid overflowing squared residuals. Runtime is O(sampleCount) for this summary in addition to the
kriging solve. Separate invalid/uncertain raster statuses and invalid validation counts remain
visible. No variogram fitting, automatic model selection or residual significance is implied.

## Facility scenario evaluation

`addFacilityScenarioRecipe` reads a complete, reusable facility-major cost matrix, demand, supply,
a live facility mask and network-accessibility parameters. Outputs are gravity accessibility,
nearest selected facility/cost per demand row, covered/uncovered demand, and invalid-demand count.
Equal nearest costs choose the lowest facility row. Nonfinite or negative demand is omitted and
counted; unreachable valid demand remains uncovered. An empty selected set leaves every nearest
cost infinite and facility ID `0xffffffff`.

Run the recipe once per scenario with a distinct ID and the same matrix/demand to compare results.
Changing masks or scoring parameters does not rerun routing. The caller must ensure the matrix's
routing search completed and its horizon covers the scoring threshold. Evaluation does not select
facilities or impose capacities; use the separately documented bounded allocation contributor for
those objectives. Nearest-cost outputs can feed the zonal distribution recipe for travel-cost
quantiles, with unreachable demand counted separately.

## Exposure by administrative area

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

## Regional raster change

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
cells; unresolved cells remain available through raster statuses.

Reach capacity defaults to cell count and vertex capacity to twice cell count; both are explicit
options. Preserve fill, flat, accumulation, order, watershed and network completion evidence.
`maximumSteps` bounds tracing. This recipe does not stitch independent tiles or interpret branching
D-infinity/MFD flow as D8. Lengths/distances use the planar grid affine's units.

## Track quality and route conformity

`addTrackQualityRecipe` is exported by `@luma-contributors/graph/gpu-network`, since its optional map-matching stage consumes network topology. It preserves the trajectory's raw observation order and timestamp representation,
including double-single and Int64 words. Its live two-row threshold view holds maximum speed and
maximum gap in trajectory units. Raw flags use bits 1 invalid position, 2 nonpositive/nonfinite time
step, 4 excessive/nonfinite step speed, 8 time gap and 16 invalid threshold. First observations
have no step checks but still report an invalid timestamp;
rows outside the track-offset domain carry `0xffffffff`. Per-track counts count flagged observations.

Optional `similarity` compares the raw tracks to explicit reference pairs using Hausdorff, Frechet,
DTW and LCSS. Its densification and dynamic-programming limits remain explicit, and `pairStatus`
retains failure evidence. Supplying `frechetEpsilon` also returns `frechetWithin`. Optional `mapMatching` returns matched edge IDs, snap distances, breaks
and matched count, plus candidate and route-overflow stages. Route-search overflow must not be
interpreted as a reliable off-route result. Resampling is a separate upstream choice; raw flags do
not silently become flags on a resampled track.

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
nonneighbor pairs contribute to cell totals but do not become H3 edges. Total outgoing flow equals
total incoming flow and the valid-pair sum; edge-flow totals may be smaller. See the
[network expansion reference](../gpu-network/expansion.md) for the bounded direct-scan cost and H3
boundary precision.

## Surface preparation with provenance

`addSurfacePreparationRecipe` applies bounded continuous nodata filling and an optional rank filter.
It returns a float32 buffer band, original validity, filled and filtered masks, downstream validity,
and unresolved-gap counts before and after filtering. Original valid calibrated observations are
preserved in fill-only mode; interpolated values never become donors during that fill pass.

`maximumDistance` is in pixels. `method: 'nearest'` chooses the nearest original donor;
`'inverse-distance'` weights all original donors in the disk. This is distinct from GDAL's
four-quadrant donor selection. Rank filtering uses a square footprint, explicit border/nodata rules
and the SciPy rank convention (upper middle for an even median population). Filling missing
measurements is separate from filling drainage depressions. Supply the halo needed by both stages
when preparing tiles and publish only nonoverlapping cores.
