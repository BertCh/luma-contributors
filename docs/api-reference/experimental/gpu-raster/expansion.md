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
| C04 | `GPURasterFractionalCoverage`, `GPURasterFractionalZonalStatistics` | Triangulates polygons including holes, clips every triangle to affine raster cells, and emits compact `(zoneId, cellId, fraction)` tuples. Overlapping features retain separate tuples. Coverage candidates are explicitly capped at 4,194,304 zone/cell pairs. Weighted aggregation distinguishes extensive cell totals from density/intensive area integrals. |
| C05 | `GPURasterFillNoData` | Disk search of original valid donors, with nearest or all-donor inverse-distance interpolation. Radius 1–256 pixels. Preserves observed calibrated values; returns filled mask, validity and unresolved count. The IDW method is **not** GDAL's four-quadrant method. |
| C06 | `GPURasterRankFilter` | Square footprints with radius 0–8. Quantile selects `min(floor(q*n),n-1)` (upper middle for an even median). Clamp or nodata borders; strict or ignored missing neighbors. Invalid centers stay invalid. |
| C07 | `GPUTerrainFlowDistance` | D8 paths stop at streams or terminal cells. Defaults to planar affine distance; explicit nonnegative transition costs support other units. Cycles, unresolved directions/stream masks, numerical overflow and budget exhaustion remain unresolved. |
| C08 | `GPUTerrainStreamNetwork` | Compact polylines between headwaters, junctions and outlets, with source/target raster cell IDs, vertex offsets, affine cell-center positions and lengths. Valid D8 stream paths only. Reach order is source-cell order. Stream order can be gathered from the source-cell IDs. |
| C10 | `GPUConcaveHull`, `GPUGroupConcaveHull` | Alpha-complex area hulls retain Delaunay triangles by maximum circumradius. Directed boundary edges preserve holes and disconnected components. This is radius-based alpha geometry, not GEOS ratio-based concave hull. Collinear/isolated points emit no area. |
| C21 | `GPURasterWalkingCost` | Directed eight-neighbor movement from elevation and friction. Tobler time or explicitly supplied GRASS-style energy coefficients; nonnegative finite edges only. Whole-grid Jacobi rounds publish finite answers only after a fixed-point check. |
| C22 | `GPURasterReconstruction`, `GPURasterFillHoles` | Grayscale geodesic dilation/erosion under a mask; hole filling reconstructs background from the boundary. Four/eight connectivity is explicit. Nonconvergence never publishes a plausible completed mask. |
| C23 | `GPURasterReprojection` | Generates destination centers, applies an explicit inverse `CompiledProjection`, inverts a nonsingular source affine and samples nearest/bilinear values. Supports rotated grids. One caller-selected source tile; no tile fetching. Float32 projection path only. |
| C24 | `GPUConstrainedDelaunayTriangulation` | Existing shell/hole triangulation followed by bounded constrained Lawson flips. Boundary edges remain fixed. Per-polygon triangulation validity and Delaunay convergence are separate; exhaustion retains a valid initial/refined mesh with `output.status.converged=0`. |
| C25 | `GPURasterWatershedSegmentation` | Default `method: 'fifo'` preserves marker priority flooding by path maximum elevation and discovery age, within 16,384 cells. Explicit `'minimum-spanning-forest'` uses parallel rooted Boruvka contraction with linear scratch and device-sized domains. Both support four/eight connectivity and fixed markers; their plateau tie policies differ. |
| C26 | `GPURasterSkeleton`, `GPURasterMedialAxis` | Zhang–Suen thinning is separate from maximal Euclidean disk centers. Medial axis uses an exact distance transform and an integer lattice-step containment test, with linear storage and dimensions up to 32,768 subject to device storage limits. Exterior pixels are background; radii remain in pixel units. Even-width plateaus can retain multiple centers. |

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
coverage tuples even when output capacity is insufficient.

`GPURasterFractionalZonalStatistics` consumes those views, a calibrated band, `zoneCount`, and
`quantity`. It returns `totals`, coverage-weighted raw-value `means`, `coveredArea`, `validArea`,
and `validity`. Extensive cell totals use `value * fraction`; density and intensive integrals use
`value * fraction * abs(affine determinant)`. Missing values reduce observed area and remain unknown.
Means divide the value/fraction sum by observed fractions. Empty or wholly unobserved zones have
invalid totals/means; covered/observed areas remain available. Incomplete coverage invalidates all
aggregates. Area units are source-coordinate units squared; project geographic polygons before
interpreting physical area.

### Preparing continuous surfaces

```typescript
const preparation = new GPURasterFillNoData({
  grid,
  input: elevationBand,
  maximumDistance: 8,
  method: 'nearest',
  output: {values: preparedValues, validity: preparedValidity, filled, unresolvedCount}
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

### Projection ownership and incomplete iteration

`GPURasterReprojection` accepts an inverse projection already compiled for destination-to-source
coordinates. Without one, CRS metadata must match. Source pixels outside the provided support and
invalid projection domains remain nodata; strict bilinear sampling requires every participating
sample, while renormalization uses surviving support. A projection-bearing contributor registers
once and owns its projection parameter buffer: destroy it after destroying its compiled graph.

Reconstruction, thinning and walking perform an additional check round to establish the final
fixed point. When it still changes, they report `output.status.converged=0`; raster floats are NaN and binary
masks are `0xfffffffe`. Consumers must retain that evidence. Watershed segmentation leaves original
markers identifiable but publishes unresolved non-marker pixels when its step budget is exhausted.

## Reference evidence

[`raster-terrain-expansion.py`](../../../../modules/analysis/test/oracles/raster-terrain-expansion.py)
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
