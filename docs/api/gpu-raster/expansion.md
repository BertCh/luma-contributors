# Raster, terrain and geometry expansion

These experimental command-graph contributors keep inputs and results on the GPU. They add
bounded analytical operations and explicit completeness evidence. They have small independent
reference fixtures; they do not yet establish a speed advantage or large-domain production parity.
All outputs are caller-owned, packed graph views. Input/output buffers must be distinct. Import
raster contributors from `@luma-contributors/analysis/raster-analysis`, terrain contributors from
`@luma-contributors/analysis/gpu-terrain`, and hull/mesh contributors from
`@luma-contributors/analysis/gpu-spatial-analysis`.

## Operations and contracts

| Proposal | Public API | Contract and current boundary |
| --- | --- | --- |
| C04 | `GPURasterFractionalCoverage`, `GPURasterFractionalZonalStatistics` | Triangulates polygons including holes, clips every triangle to affine raster cells, and emits compact `(zoneId, cellId, fraction)` tuples. Overlapping features retain separate tuples. Coverage candidates are explicitly capped at 4,194,304 zone/cell pairs; positive pairs are compacted in parallel (flag, scan, scatter) in zone-major, cell-ascending order. Weighted aggregation distinguishes extensive cell totals from density/intensive area integrals. |
| C05 | `GPURasterFillNoData` | Disk search of original valid donors, with nearest or all-donor inverse-distance interpolation. Radius 1–256 pixels. Preserves observed calibrated values; returns filled mask, validity and the remaining invalid count in `status.invalidCount`. The IDW method is **not** GDAL's four-quadrant method. |
| C06 | `GPURasterRankFilter` | Square footprints with radius 0–8. Quantile selects `min(floor(q*n),n-1)` (upper middle for an even median). Clamp or nodata borders; strict or ignored missing neighbors. Invalid centers stay invalid. |
| C07 | `GPUTerrainFlowDistance` | D8 paths stop at streams or terminal cells. Defaults to planar affine distance; explicit nonnegative transition costs support other units. Cycles, unresolved directions/stream masks, numerical overflow and budget exhaustion remain unresolved. |
| C08 | `GPUTerrainStreamNetwork` | Compact polylines between headwaters, junctions and outlets, with source/target raster cell IDs, vertex offsets, affine cell-center positions and lengths. Valid D8 stream paths only. Reach order is source-cell order. Stream order can be gathered from the source-cell IDs. |
| C10 | `GPUConcaveHull`, `GPUGroupConcaveHull` | Alpha-complex area hulls retain Delaunay triangles by maximum circumradius. Directed boundary edges preserve holes and disconnected components. This is radius-based alpha geometry, not GEOS ratio-based concave hull. Collinear/isolated points emit no area. |
| C21 | `GPURasterWalkingCost` | Directed eight-neighbor movement from elevation and friction. Tobler time or explicitly supplied GRASS-style energy coefficients; nonnegative finite edges only. Whole-grid Jacobi rounds are gated on the GPU and stop once a round changes nothing; the current costs are always published and `status.converged` tells whether they are final. Required `parameters` (packed by `getGPURasterWalkingCostParameterValues`, 16 float32 values) carry the affine, row edges, `frictionWeight` and GRASS coefficients per frame, so a changed georeference does not recompile. Optional `cellSizeMode` (`'uniform'` default, `'web-mercator'`, `'geographic'`) converts steps and slopes to WGS84 ground meters per row. |
| C22 | `GPURasterReconstruction`, `GPURasterFillHoles` | Grayscale geodesic dilation/erosion under a mask; hole filling reconstructs background from the boundary. `connectivity` is 4 or 8: reconstruction defaults to 8 (the 3 x 3 square, the default footprint of `skimage.morphology.reconstruction` and of the published `GPURasterDilation`; it was 4 before 2026-11-10), hole filling defaults to a 4-connected background (the rule of `scipy.ndimage.binary_fill_holes`). The current values are always published; `status.converged=0` marks an unfinished run, whose hole mask reports background the boundary flood has not reached as holes. |
| C23 | `GPURasterReprojection` | Generates destination centers, applies an explicit inverse `CompiledProjection`, inverts a nonsingular source affine and samples nearest/bilinear values. Supports rotated grids. One caller-selected source tile; no tile fetching. Float32 projection path only. Grids in the same CRS are differenced at their origins in float64, so large projected origins (UTM northings) keep sub-cell accuracy; with an inverse projection, keep `destinationGrid` in a local frame and set the projection's `destinationOrigin` near the source grid origin. |
| C24 | `GPUConstrainedDelaunayTriangulation` | Existing shell/hole triangulation followed by bounded constrained Lawson flips. Boundary edges remain fixed. Per-polygon triangulation validity and Delaunay convergence are separate; exhaustion retains a valid initial/refined mesh with `output.status.converged=0`. |
| C25 | `GPURasterWatershedSegmentation` | Default `method: 'fifo'` preserves marker priority flooding by path maximum elevation and discovery age, within 16,384 cells. Explicit `'minimum-spanning-forest'` uses parallel rooted Boruvka contraction with linear scratch and device-sized domains. Both support four/eight connectivity and fixed markers; their plateau tie policies differ. The FIFO order is by path maximum, not by the cell's own value as in `skimage.segmentation.watershed`, so the two can differ where fronts from higher markers meet on lower ground. |
| C26 | `GPURasterSkeleton`, `GPURasterMedialAxis` | Zhang–Suen thinning is separate from maximal Euclidean disk centers. The thinning is the 1984 algorithm unchanged, so an isolated 2 x 2 block is deleted completely and two-pixel-thick strokes can vanish. Medial axis uses an exact distance transform and an integer lattice-step containment test, with linear storage and dimensions up to 32,768 subject to device storage limits. Exterior pixels are background; radii remain in pixel units. Even-width plateaus can retain multiple centers. |

### Watershed methods

The default FIFO method orders pending cells by the maximum elevation along their flood path,
then by discovery age. Seeds enter in row-major order. `maximumSteps` bounds the number of popped
cells; it applies only to this method. Its binary heap uses O(N log N) queue work and O(N) storage.

For larger complete domains, select `method: 'minimum-spanning-forest'`. Grid edges are ordered by
maximum endpoint elevation, then by canonical edge ID: four times the row-major source cell plus
the direction index (right, down-left, down, down-right). Four-connectivity omits diagonal edges.
Signed zero elevations compare equally. Parallel contraction keeps each marker fixed and excludes
invalid pixels and reserved marker words. Unseeded connected components retain label zero.
This static edge order can assign plateaus differently from FIFO; it is an explicit alternative,
not FIFO or scikit-image output parity.

Neither method reproduces `skimage.segmentation.watershed` in general. The FIFO method orders by
the maximum elevation along the flood path; scikit-image orders its queue by the cell's own value.
For the row `5, 2, 2, 2, 2, 2, 5` with markers on the two ends the FIFO method labels
`1 1 1 1 2 2 2` (both fronts hold priority 5 and advance in turn) and scikit-image labels
`1 1 1 1 1 1 2`. The committed scikit-image watershed fixture is a case where the two orders
agree; it checks the labelling of that case, not parity of the queue order.

The forest method accepts `maximumIterations` from 1 through 32, defaulting to
`ceil(log2(width * height)) + 1`. It requires fewer than 2^30 cells and scratch buffers that fit
the device's storage binding limits. Supplying `maximumSteps` with this method, or
`maximumIterations` with FIFO, is rejected. Exhaustion reports `converged = 0` and labels valid
non-marker cells unresolved (`0xfffffffe`), rather than publishing a partial partition as complete.
Neither method assembles tiles; independent tile segmentation does not preserve global ownership.

### Exact medial disk containment

`GPURasterMedialAxis` composes the exact banded distance transform with one candidate check per
pixel. Given a nearest background pixel, any disk containing the current pixel's disk must lie on
the ray directly away from that background pixel. The first lattice point along that ray suffices
to decide containment using integer squared distances. This removes both dense all-pairs searches.

Containment follows the mathematical Euclidean-disk definition, correcting previous float32
comparisons at tangent disks. Output radii are correctly rounded float32 values. The grid affine
does not rescale pixel-unit distances. This operation does not perform thinning and does not
promise a one-pixel-wide or topology-preserving skeleton. Storage is linear in cell count, but the
distance-transform scratch still needs to fit device limits; the dimension bound alone is not a
memory or throughput guarantee.

### Fractional quantities

`GPURasterFractionalCoverage` accepts `grid` and `zones` using the shared four-level polygon
layout. Allocate `zoneIds`, `cellIds`, and `fractions` with a common tuple capacity, plus scalar
`status.count`, `status.requiredCount`, `status.overflow`, and per-feature `zoneValidity`. A failed polygon triangulation
invalidates its feature; that feature contributes no tuples. `status.requiredCount` counts all positive
coverage tuples even when output capacity is insufficient. A vertex whose pixel coordinate is
within four float32 steps of an integer (`4 * 2^-23 * max(|k|, 1)`) is placed on that cell
boundary, so a zone edge that follows a cell boundary leaves no sliver tuple in the neighbouring
cell, also when the cell size is not a float32 number (0.3, a third of a degree). The snap cannot
absorb the rounding of `position - origin`: keep the grid origin near the geometry. The required `parameters` view
(`getGPURasterFractionalCoverageParameterValues(grid)`, 8 float32 values) carries the affine, so a
moved georeference of the same size does not recompile.

`GPURasterFractionalZonalStatistics` consumes those views, a calibrated band, `zoneCount`, and
`quantity`. It returns `totals`, coverage-weighted raw-value `means`, `coveredArea`, `validArea`,
and `validity`. Extensive cell totals use `value * fraction`; density and intensive integrals use
`value * fraction * abs(affine determinant)`. Missing values reduce observed area and remain unknown.
Means divide the value/fraction sum by observed fractions. Empty or wholly unobserved zones have
invalid totals/means; covered/observed areas remain available. Incomplete coverage invalidates all
aggregates. Area units are source-coordinate units squared; project geographic polygons before
interpreting physical area. The determinant is that of `grid.affine`, fixed at compile time, unless
the optional `parameters` prop is given: pass the same per-frame georeference view as the coverage
class, and the cell area is read from it on every encoding. Do so whenever the coverage
`parameters` are rewritten for another cell size; otherwise the fractions follow the new
georeference while `coveredArea`, `validArea` and density `totals` keep the compiled cell area.
`GPURasterFractionalCategoricalStatistics` and `GPURasterFractionalZonalDistribution` take the
same prop.

{/* parameters: GPURasterFractionalCoverage generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterFractionalCoverage` parameters

- Packer: `getGPURasterFractionalCoverageParameterValues(grid)`, writes `GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): Grid whose affine is packed; it must be invertible.

{/* /parameters */}

### Preparing continuous surfaces

```typescript
const preparation = new GPURasterFillNoData({
  grid,
  input: elevationBand,
  maximumDistance: 8,
  method: 'nearest',
  output: {values: preparedValues, validity: preparedValidity, filled, status: {invalidCount}}
});
graph.add(preparation);
```

All-donor interpolation never uses a newly filled cell as a donor. A tile requires a halo at least
`maximumDistance` wide to match the full-domain result on its core. Rank filtering requires its
radius as a halo. These contributors do not assemble tiles or suppress halo counts automatically.
Filling nodata and hydrological depression filling remain different operations. The surface
preparation recipe exposes provenance through both stages.

### Drainage outputs

`GPUTerrainFlowDistance` returns dense `distance`, `validity`, and `terminalCells`.
`0xffffffff` is invalid; `0xfffffffe` is unresolved. `maximumSteps` defaults to the cell count.
A resolved stream cell has distance zero. If no streams are supplied, outlets/pits stop the path.
Distances use the full affine's planar lengths; geographic and Mercator physical corrections are
not implicit.

`GPUTerrainStreamNetwork` takes `flowDirections`, `streams` and the same grid. Allocate
`sourceCells`, `targetCells`, `lengths` to a reach capacity, `vertexOffsets` to capacity plus one,
and `positions` to a separate vertex capacity. It returns scalar reach/vertex counts, `status.overflow` and
unresolved-cell count. Junction cells appear in every incident polyline. Isolated stream cells have
no line reach. Cycles and paths crossing unknown directions or stream status are omitted with
unresolved evidence. Cross-tile drainage and branching D-infinity/MFD are not implied.

### Group outlines and recipes

`GPUGroupConcaveHull` masks points for each requested group before running independent Delaunay
stages. Every group reserves `max(0, 2*pointCount-5)` triangle slots and three boundary slots per
triangle; status and masks distinguish used output. This explicit `groupCount * pointCount` work
and memory should be bounded by the caller. It preserves original point indices without readback.

`addClusterAndOutlineRecipe` accepts `outline: {kind: 'alpha', maximumCircumradius}`. The result's
`outlineType` identifies the selected mode. Alpha geometry lives in `concaveHull`; `areas`,
`perimeters`, and `centroids` describe its retained area. In alpha mode the legacy single-ring
`hullOffsets` and `hullCounts` are empty, because one ring cannot represent holes and disconnected
components. Default convex behavior is unchanged. Empty area has zero area/perimeter and undefined
centroid. Nonconverged groups report invalid measures.

### `GPURasterWalkingCost` georeference

Walking cost reads the grid's linear affine terms, its row edges, `frictionWeight` and the GRASS
coefficients from `parameters`, a `GPUParameterBuffer` view written with
`getGPURasterWalkingCostParameterValues(settings, grid)`. Rewriting it between encodings moves the
same compiled graph to a new georeference, cell size or friction weight without recompiling. Only
`model.kind` (Tobler or GRASS) and `cellSizeMode` change the kernels. `cellSizeMode: 'uniform'` (the
default) treats affine units as meters; `'geographic'` (degrees) and `'web-mercator'` (EPSG:3857
meters) scale each step on the WGS84 ellipsoid at the latitude between the two cell centres, so
slopes divide meters of rise by meters of run.

Pack `parameters` for the same `model.kind` the contributor was compiled with. The packed slots
8 to 12 identify the model (all zero for Tobler, a negative threshold for GRASS), and the kernel
checks them on the GPU: parameters packed for the other model move nothing (sources cost 0, every
other cell stays `+Infinity`) and set the optional `status.invalidCount` to 1. Before this check, a
GRASS kernel given Tobler's zero coefficients charged only friction, silently.

{/* parameters: GPURasterWalkingCost generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterWalkingCost` parameters

- Packer: `getGPURasterWalkingCostParameterValues(settings, grid)`, writes `GPU_RASTER_WALKING_COST_PARAMETER_LENGTH` (16) `float32` values.
  - `grid` (`GPURasterMetadata`): Grid whose georeference is packed. Its width and height must match the compiled one.

Settings of `getGPURasterWalkingCostParameterValues`:

| Field | Type | Description |
|---|---|---|
| `frictionWeight` (optional) | `number` | Coefficient multiplying additive average-friction * ground-distance cost. Default 1. |
| `model` (optional) | `GPURasterWalkingCostModel` | Movement model; only the GRASS coefficients and threshold are packed. Default Tobler. |

{/* /parameters */}

### Projection ownership and incomplete iteration

`GPURasterReprojection` accepts an inverse projection already compiled for destination-to-source
coordinates. Without one, CRS metadata must match. Source pixels outside the provided support and
invalid projection domains remain nodata; strict bilinear sampling requires every participating
sample, while renormalization uses surviving support. A projection-bearing contributor registers
once and owns its projection parameter buffer: destroy it after destroying its compiled graph.

Reconstruction, fill holes, thinning and walking perform an additional check round to establish the
final fixed point. The rounds run as GPU-gated iterations: once a round changes nothing, later rounds
dispatch no work, so a generous `maximumIterations` costs only command nodes (see the contributor's
`estimatedCommandNodeCount`, checked against the 32,768-node budget). The node budget fixes the largest accepted `maximumIterations`: 26,196 for walking cost, 26,209 for reconstruction, 26,207 for fill holes and 10,920 for skeletonization; one more is rejected at construction. The current values are
**always published**, whether or not the run converged. `output.status.converged` is 1 only when a
round changed nothing before the limit. Optional `output.status.iterationLimitReached` (1 when the
limit stopped an unconverged run) and `output.status.iterationCount` (rounds that ran, including the
verifying round; thinning counts full two-phase iterations) are written when supplied. An unconverged
walking cost holds `+Infinity` for cells the front has not reached and upper bounds elsewhere, an
unconverged reconstruction holds the partially spread marker, and an unconverged thinning holds a
partially thinned mask. Earlier releases published all-NaN or `0xfffffffe` output instead; callers
that relied on that must now test `converged`. Watershed segmentation leaves original
markers identifiable but publishes unresolved non-marker pixels when its step budget is exhausted.

## Reference evidence

[`raster-terrain-expansion.py`](../../../modules/analysis/test/oracles/raster-terrain-expansion.py)
records exact package versions and produces committed fixtures using SciPy median/hole filling,
scikit-image reconstruction/watershed, NetworkX path lengths, Shapely intersections/CDT, and a
SciPy-Delaunay/Shapely alpha-complex composition. The matching browser tests execute actual GPU
kernels. Other cases test boundary masks, directed costs, capacities, provenance, cyclic receivers
and forced nonconvergence. The maximal-disk medial axis has a direct SciPy radius reference and
structural center tests; it does not claim direct scikit-image medial-axis parity.

Performance gates remain open: measure compilation, upload, execution, readback and repeated runs
on representative large domains. Bounded priority queues, disk searches, grouped tessellation and
per-path walks are useful complete reference implementations; their existence is not a performance
claim. No hidden CPU analysis, GPU readback or implicit command submission occurs in these APIs.

For explicitly supplied tiles on one shared source lattice, [GPURasterTileReprojection](./tile-reprojection.md) assembles a bounded mosaic on GPU before sampling. [Nodata filling](./fill-nodata.md) documents the optional original-donor four-quadrant method.
