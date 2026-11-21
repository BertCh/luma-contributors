# Raster, terrain and geometry expansion

These experimental command-graph contributors keep inputs and results on the GPU. They add
bounded analytical operations and explicit completeness evidence. They have small independent
reference fixtures; they do not yet establish a speed advantage or large-domain production parity.
All outputs are caller-owned, packed graph views. Input/output buffers must be distinct.
Constructors reject views whose element formats disagree with their named ports, including
status leaves. A buffer band's values must match its declared float32, sint32 or uint32
format; validity is uint32. `makeGPURasterBand` applies the same scalar/validity checks before
borrowing a grid-sized prefix, preserving byte offsets and exact-sized view identities.
Required named ports fail at construction when omitted; optional status words remain optional.
Fractional coverage status accessors are captured once and validated like ordinary fields.
Fractional zonal `quantity` accepts only `'extensive'`, `'density'` or `'intensive'`. Import
raster contributors from `@luma-contributors/analysis/raster-analysis`, terrain contributors from
`@luma-contributors/analysis/gpu-terrain`, and hull/mesh contributors from
`@luma-contributors/analysis/gpu-spatial-analysis`.

`GPURasterRankFilter`, `GPURasterFillNoData` and `GPURasterReprojection` accept the published
`GPURasterBand`: either packed buffer storage or a selected channel of a single-mip, single-layer
2D `r32float`/`rgba32float`, `r32uint`/`rgba32uint`, or `r32sint`/`rgba32sint` texture matching
the band's scalar format. Texture extents must equal the input grid (the source grid for
reprojection). Channels are zero-based; single-channel textures require channel zero. Texture
handles must include `Texture.SAMPLE` usage; `Texture.COPY_DST` alone is insufficient. Supplied
`input.scale` and `input.offset` must remain finite when converted to float32.
The published `GPURasterTextureToBuffer` adds one graph conversion pass, compares nodata in the
raw selected channel, combines it with the band's optional `validity` mask, and applies scale
and offset once. These continuous-value operators produce calibrated float32 values and uint32
validity; integer category/label operators retain their separate exact-integer input contracts.
Texture shape, format, channel and mask are validated at construction; graph ownership is checked
before any conversion scratch or nodes are created. Updating texture contents or mask flags and
resubmitting the compiled graph recomputes the result.
Input band metadata and its storage descriptor are captured at construction, including channel,
calibration, nodata and view/mask identities. Declared fields are read by property access, including
inherited and non-enumerable values on both the band and its storage descriptor. Mutating the caller's
descriptors does not change the measurement read by a wired graph; update the captured GPU contents
or rebuild the contributor.
The surface-preparation recipe validates the texture, target graph and caller-provided output
boundary, including each supplied view's format, packing and exact row count, before allocating
default outputs. A rejected texture or invalid output can be corrected and retried on the same
graph with the same recipe ID.

## Choose a task

- [Morphology and surface preparation](#morphology-and-surface-preparation): fill missing values,
  filter, reconstruct, fill holes, thin or measure medial disks (C05, C06, C22, C26).
- [Costs and travel](#costs-and-travel): directed walking cost (C21).
- [Hydrology](#hydrology): drainage distances/networks and watershed partitions (C07, C08, C25).
- [Geometry and coverage](#geometry-and-coverage): fractional quantities, outlines and constrained
  triangulation (C04, C10, C24). [Reprojection](#gpurasterreprojection) covers C23.

## Operations and contracts

The rank filter's optional `output.changed` is one where a valid result differs from its
calibrated input center, zero for an unchanged or invalid center. Omitting it still computes
the same `output.values` and `output.validity`.
Skeleton `output.status.iterationCount`, when supplied, counts whole thinning rounds, including
the round that confirms no deletion. An isolated center reports one; a solid 3 by 3 block that
thins to its center reports two. Omitting this diagnostic does not change the skeleton.
The required nodata-fill `status` object may be empty when `invalidCount` is not needed.
Focal-pair count and status outputs accompany any nonempty subset of float metrics: requesting
only covariance, correlation, slope, intercept, R-squared or residual sum of squares uses the
same paired windows and normalization as requesting every metric. Scalar-ridge strength and
walking-cost iteration/invalid-count diagnostics can likewise be omitted without changing their
required results.

| Proposal | Public API | Contract and current boundary |
| --- | --- | --- |
| C04 | `GPURasterFractionalCoverage`, `GPURasterFractionalZonalStatistics` | Triangulates polygons including holes, clips every triangle to affine raster cells, and emits compact `(zoneId, cellId, fraction)` tuples. Overlapping features retain separate tuples. Coverage candidates are explicitly capped at 4,194,304 zone/cell pairs; positive pairs are compacted in parallel (flag, scan, scatter) in zone-major, cell-ascending order. Weighted aggregation distinguishes extensive cell totals from density/intensive area integrals. |
| C05 | `GPURasterFillNoData` | Disk search of original valid donors, with nearest or all-donor inverse-distance interpolation. Radius 1–256 pixels. Preserves observed calibrated values; returns filled mask, validity and the remaining invalid count in `status.invalidCount`. The IDW method is **not** GDAL's four-quadrant method. |
| C06 | `GPURasterRankFilter` | Square footprints with radius 0–8. Quantile selects `min(floor(q*n),n-1)` (upper middle for an even median). Clamp or nodata borders; strict or ignored missing neighbors. Invalid centers stay invalid. |
| C07 | `GPUTerrainFlowDistance` | D8 paths stop at streams or terminal cells. Defaults to planar affine distance; explicit nonnegative transition costs support other units. Cycles, unresolved directions/stream masks, numerical overflow and budget exhaustion remain unresolved. |
| C08 | `GPUTerrainStreamNetwork` | Compact polylines between headwaters, junctions and outlets, with source/target raster cell IDs, vertex offsets, affine cell-center positions and lengths. Valid D8 stream paths only. Reach order is source-cell order. Stream order can be gathered from the source-cell IDs. |
| C10 | [`GPUConcaveHull`, `GPUGroupConcaveHull`](../gpu-spatial-analysis/operators-geometry.md#hull-construction) | Alpha-complex area hulls retain Delaunay triangles by maximum circumradius. Directed boundary edges preserve holes and disconnected components. This is radius-based alpha geometry, not GEOS ratio-based concave hull. Collinear/isolated points emit no area. |
| C21 | `GPURasterWalkingCost` | Directed eight-neighbor movement from elevation and friction. Tobler time or explicitly supplied GRASS-style energy coefficients; nonnegative finite edges only. Whole-grid Jacobi rounds are gated on the GPU and stop once a round changes nothing; the current costs are always published and `status.converged` tells whether they are final. Required `parameters` (packed by `getGPURasterWalkingCostParameterValues`, 16 float32 values) carry the affine, row edges, `frictionWeight` and GRASS coefficients per frame, so a changed georeference does not recompile. Optional `cellSizeMode` (`'uniform'` default, `'web-mercator'`, `'geographic'`) converts steps and slopes to WGS84 ground meters per row. |
| C22 | `GPURasterReconstruction`, `GPURasterFillHoles` | Grayscale geodesic dilation/erosion under a mask; hole filling reconstructs background from the boundary. `connectivity` is 4 or 8: reconstruction defaults to 8 (the 3 x 3 square, the default footprint of `skimage.morphology.reconstruction` and of the published `GPURasterDilation`; it was 4 before 2026-11-10), hole filling defaults to a 4-connected background (the rule of `scipy.ndimage.binary_fill_holes`). The current values are always published; `status.converged=0` marks an unfinished run, whose hole mask reports background the boundary flood has not reached as holes. |
| C23 | `GPURasterReprojection` | Generates destination centers, applies an explicit inverse `CompiledProjection`, inverts a nonsingular source affine and samples nearest/bilinear values. Supports rotated grids. One caller-selected source tile; no tile fetching. Same-CRS composition is computed on the host in float64; explicit projections use local float32 or double-single coordinates, subtracting the split source origin before tile-relative float32 sampling. Grids in the same CRS are differenced at their origins in float64, so large projected origins (UTM northings) keep sub-cell accuracy; with an inverse projection, keep `destinationGrid` in a local frame and set the projection's `destinationOrigin` near the source grid origin. |
| C24 | [`GPUConstrainedDelaunayTriangulation`](../gpu-spatial-analysis/operators-geometry.md#constrained-delaunay-triangulation) | Existing shell/hole triangulation followed by bounded constrained Lawson flips. Boundary edges remain fixed. Per-polygon triangulation validity and Delaunay convergence are separate; exhaustion retains a valid initial/refined mesh with `output.status.converged=0`. |
| C25 | `GPURasterWatershedSegmentation` | Default `method: 'fifo'` preserves marker priority flooding by path maximum elevation and discovery age, within 16,384 cells. Explicit `'minimum-spanning-forest'` uses parallel rooted Boruvka contraction with linear scratch and device-sized domains. Both support four/eight connectivity and fixed markers; their plateau tie policies differ. The FIFO order is by path maximum, not by the cell's own value as in `skimage.segmentation.watershed`, so the two can differ where fronts from higher markers meet on lower ground. |
| C26 | `GPURasterSkeleton`, `GPURasterMedialAxis` | Zhang–Suen thinning is separate from maximal Euclidean disk centers. The thinning is the 1984 algorithm unchanged, so an isolated 2 x 2 block is deleted completely and two-pixel-thick strokes can vanish. Medial axis uses an exact distance transform and an integer lattice-step containment test, with linear storage and dimensions up to 32,768 subject to device storage limits. Exterior pixels are background; radii remain in pixel units. Even-width plateaus can retain multiple centers. |

## Hydrology

See also [drainage outputs](#drainage-outputs).

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

The CPU function and CPU/GPU operation forms honor the same explicit budgets: FIFO counts queue
pops and the forest counts synchronous rooted Boruvka contractions. If reachable work remains,
`status.converged` is 0, `status.iterationLimitReached` is 1, and usable non-marker labels are
`0xfffffffe`. For elevations `[5, 0, 0]` and markers `[7, 0, 0]`, one forest round joins cells 1
and 2 but leaves their component unseeded; two rounds finish with labels `[7, 7, 7]`.
Without an explicit CPU forest budget, the complete Kruskal solution retains its existing tie
order. CPU arithmetic remains float64; elevations tied only after float32 rounding can differ.
CPU and GPU watershed operations capture these budgets at construction; changing the original
props object later does not change the operation's budget.
They also capture each band's raw nodata, calibration and logical port names, and every output
name, including inherited and non-enumerable properties. A raw nodata barrier therefore remains
a barrier through either operation form; later nested metadata or output-name edits do not change
the declared operation. Direct GPU contributors capture their budgets, band metadata, storage
and output descriptors too. Both operation forms and the direct contributor copy grid dimensions
and affine coordinates before validation; later caller edits cannot resize a validated raster.
GPU views retain their identities and their contents can still change between runs.
The direct watershed and walking constructors require `output.status.converged` and validate
all captured status views, including inherited properties: each must hold a `uint32` word and
must not overlap raster outputs or inputs. Missing or malformed convergence ports are rejected
at construction, before command nodes are generated.
Both CPU and GPU operation forms require non-empty logical output names. Every raster and
status output must have its own name, distinct from all inputs; optional status names may be
omitted but cannot be empty. Validation precedes keyed declarations so a convergence word cannot
silently replace a raster vector with the same name.
CPU forest relabeling compresses shared parent paths, so a long flat row resolves roots once
instead of repeatedly walking the full chain for each cell.

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

## Morphology and surface preparation

See [preparing continuous surfaces](#preparing-continuous-surfaces) for a fill example.

### Preparing continuous surfaces

```typescript
import {GPURasterFillNoData} from '@luma-contributors/analysis/raster-analysis';
declare const grid: NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['grid']>;
declare const elevationBand: NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['input']>;
declare const preparedValues: NonNullable<NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['output']>['values']>;
declare const preparedValidity: NonNullable<NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['output']>['validity']>;
declare const filled: NonNullable<NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['output']>['filled']>;
declare const invalidCount: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPURasterFillNoData>[0]['output']>['status']>['invalidCount']>;
const preparation = new GPURasterFillNoData({
  grid,
  input: elevationBand,
  maximumDistance: 8,
  method: 'nearest',
  output: {values: preparedValues, validity: preparedValidity, filled, status: {invalidCount}}
});
graph.add(preparation);
```

### Choose a neighborhood statistic

| Task | Public import entry and class | Radius in source pixels | Missing center and physical edge |
| --- | --- | --- | --- |
| Moving mean, sum, extrema, population standard deviation | `GPUFocalStatistics` from `@luma-contributors/analysis/gpu-spatial-analysis` | Compiled `maximumRadius` (0–64); per-frame `radius` in `getGPUFocalStatisticsParameterValues`, clamped to that cap | Clips the window and skips invalid neighbors; fills a missing center unless per-frame `propagateCenterNoData` is true. |
| Covariance, correlation, regression | `GPURasterFocalPairs` from `@luma-contributors/analysis/raster-analysis` | Compiled `radius` (0–8) | Default clipped jointly valid pairs; compiled `propagateCenterNoData` defaults to false. See [paired focal statistics](./focal-pairs.md#missing-pairs-and-borders). |
| Rank or median | `GPURasterRankFilter` from `@luma-contributors/analysis/raster-analysis` | Compiled `radius` (0–8) | Default clamp at physical edges; a missing center stays invalid even with `noDataPolicy: 'ignore'`. |

For `[1, NaN, 3]` at radius 1, the ordinary moving mean at the center is `(1 + 3) / 2 = 2`
when center propagation is disabled. With response `[2, NaN, 6]`, paired statistics use the two
surviving positions: center sample covariance `4`, slope `2`, paired count `2`. Rank filtering
keeps that center invalid (NaN and validity 0). These policies differ even on the same footprint.

At internal tile boundaries, expand the source by each contributor's advertised `requiredHalo`
(`maximumRadius` for ordinary moving statistics, `radius` for paired statistics and rank).
Execute on the expanded grid and crop back to the core before stitching outputs or accumulating
counts. Physical-edge border policies do not replace the halo at an internal boundary.

Nodata-policy spelling is per operation: rank filtering accepts `'propagate'` and its legacy
`'strict'` alias, or `'ignore'`; reprojection accepts `'propagate'` (legacy `'strict'`) or
`'renormalize'`. Propagation rejects a result with a missing participating sample, whereas
renormalization redistributes the remaining interpolation weights. Reprojection methods are
`'nearest'` and `'bilinear'`; fill methods are `'nearest'`, `'inverse-distance'` and
`'gdal-four-quadrant'`. Unknown option strings must be rejected rather than selecting a
default; consult the installed method type before copying settings between these operations.

All-donor interpolation never uses a newly filled cell as a donor. A tile requires a halo at least
`maximumDistance` wide to match the full-domain result on its core. Rank filtering requires its
radius as a halo. These contributors do not assemble tiles or suppress halo counts automatically.
Filling nodata and hydrological depression filling remain different operations. The surface
preparation recipe exposes provenance through both stages.

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

## Geometry and coverage

Use a common local frame for coverage geometry and its raster. The versioned parameter packet
retains the absolute Y placement for ground-area calculations:

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getGPURasterFractionalCoverageParameterValues} from '@luma-contributors/analysis/raster-analysis';
const absoluteGrid: GPURasterMetadata = {
  width: 1, height: 1, affine: [1, 0, 0, 0, 1, 4_600_000], pixelInterpretation: 'area'
};
const localOrigin = [0, 4_600_000] as const;
const originalVertices = new Float64Array([
  0, 4_600_000.25, 1, 4_600_000.25, 1, 4_600_000.75, 0, 4_600_000.75
]);
const localVertices = Float32Array.from(originalVertices, (value, index) => value - localOrigin[index % 2]);
const sharedParameters = getGPURasterFractionalCoverageParameterValues(absoluteGrid, {localOrigin});
const sharedStorage = new GPUParameterBuffer(device, {
  id: 'shared-coverage-area-parameters', format: 'float32', length: 10, values: sharedParameters
});
const sharedView = sharedStorage.importToGraph(graph);
// Upload localVertices as part of the four-level polygon geometry; Y remains .25 and .75.
// Both coverage and area consumers: grid: absoluteGrid, parameters: sharedView.
// Ground-area statistics: cellSizeMode: 'web-mercator' for this meter-coordinate example.
// Destroy sharedStorage after its compiled graph is destroyed.
```


See also [group outlines and recipes](#group-outlines-and-recipes).

### Fractional quantities

`GPURasterFractionalCoverage` accepts `grid` and `zones` using the shared four-level polygon
layout. Allocate `zoneIds`, `cellIds`, and `fractions` with a common tuple capacity, plus scalar
`status.count`, `status.requiredCount`, `status.overflow`, optional `status.candidateOverflow`, and
per-feature `zoneValidity`. A failed polygon triangulation invalidates its feature; that feature
contributes no tuples. On final tuple-capacity overflow, `requiredCount` counts positive tuples:
increase the tuple capacity to that count and rerun. On indexed candidate exhaustion,
`candidateOverflow = 1`, `requiredCount = 0xffffffff`, and `overflow` may remain zero. Increase
the candidate budget and rerun **before interpreting requiredCount as a tuple allocation size**.
The sentinel is not a request for a 0xffffffff-row tuple buffer. Keep the two overflow flags separate.
A vertex whose pixel coordinate is
within `min(1e-5, 4 * 2^-23 * max(|k|, 1))` pixels of a cell boundary is placed on that boundary:
integers for area grids, half-integers for point grids. This removes boundary slivers only when
the inverse-affine error is within that cap. Decimal cells can exceed the cap even with zero
origin: at width 0.3, `[39.3, 39.6]` can leave about 1.53e-5 coverage in column 130 beside the
intended column 131. Every positive sliver occupies a tuple and can exhaust an otherwise exact
tuple capacity; allocate spare tuple capacity, checking candidateOverflow before requiredCount/overflow. Device division can
change these tiny supports. The exact identity
affine `[1, 0, 0, 0, 1, 0]` bypasses snapping, preserving all representable subpixel coordinates.
For other affines, snapping compares displacement from the nearest lattice point and moves a
vertex by at most 1e-5 pixel per axis, independently of pixel index. Clipping uses cell-local
coordinates. The snap cannot
absorb rounding already lost at upload. Subtract a common origin from original binary64 polygon
vertices **before** constructing a Float32Array, and use the same relative origin in coverage
parameters. At northing 4,600,000, absolute `.25`/`.75` offsets round to 0/1 in float32;
subtracting first preserves the half-cell strip. Pack with
`getGPURasterFractionalCoverageParameterValues(absoluteGrid, {localOrigin})` and give the same
ten-word view to clipping and area consumers. Words 0–6 hold
`[relativeOriginX, relativeOriginY, a, b, d, e, determinant]`; word 7 remains ignored legacy
padding, word 8 is the layout discriminator `2`, and word 9 restores absolute Y. Clipping
inverts the actual packed axes, using integer significand products where cancellation would erase
the determinant or a numerator. Normal significands and separate binary exponents keep division
finite when the determinant is subnormal and its reciprocal would overflow. Word 6 remains the separately consumed original-grid area metadata.
The no-options packer returns eight words when it allocates the destination. In a caller-provided
destination it also zeros any present extension words at indices 8 and 9 to clear version-2
metadata. Words at index 10 and above retain their previous values. Repacking either layout moves
the georeference without recompiling when dimensions and pixel interpretation stay fixed.
Public coverage and patch-metrics packets retain the signed physical determinant of the original
affine in word 6. Clipping inverts the packed axes independently. The analysis-core resolver's
coordinate packet instead carries its packed-axes determinant and exposes `affineDeterminant`
separately.
The website coverage runs distinguish pending readback, incomplete polygon coverage and a valid
empty intersection. Pending/incomplete readouts show that state and clear the highlighted support;
only complete coverage may display zero cells and zero area.

`GPURasterFractionalZonalStatistics` consumes those views, a calibrated band, `zoneCount`, and
`quantity`. It returns `totals`, coverage-weighted raw-value `means`, `coveredArea`, `validArea`,
and `validity`. Extensive cell totals use `value * fraction`; density and intensive integrals use
`value * fraction * cellArea`. Missing values reduce observed area and remain unknown.
Means divide the value/fraction sum by observed fractions. Empty or wholly unobserved zones have
invalid totals/means; covered/observed areas remain available. Incomplete coverage invalidates all
aggregates. The default `cellSizeMode: 'uniform'` uses source-coordinate units squared.
The scalar input can be a packed buffer band or supported sampled-texture band. Raw nodata
and mask are tested before calibration; scale/offset are applied exactly once to continuous
float32 results. Native integer labels require nearest reprojection with identity calibration and
matching `output.labels` (see the checked [reprojection example](#gpurasterreprojection));
bilinear interpolation is continuous and cannot preserve arbitrary uint32 identities.
Scalar, categorical and distribution consumers require `count <= tuple capacity`,
`requiredCount == count`, zero `overflow`, zero optional `candidateOverflow`, and nonzero
zone validity. Prepared-coverage operation ports preserve optional candidateOverflow. Missing
candidate evidence is accepted only when that optional input is not supplied. A clean subsequent
submission restores usable results. Means divide by the actual positive fraction sum, including
normal float32 sums below 1e-30; only zero support uses a safe denominator.
`'geographic'` (longitude/latitude degrees) and `'web-mercator'` (EPSG:3857 meters) use WGS84
ground square meters at each row center, with the same approximation as the fractional categorical
and distribution contributors. Geographic cell area is `abs(a*e-b*d)` times WGS84
longitude and latitude meters-per-degree at the row center. Whole-cell zonal statistics instead
uses an authalic latitude strip with `R = 6371007.180918475 m`; a `.001°` square has
fractional/whole-cell ratios `.995530` at 0° and `1.008584` at 80°. Counts, area sums and
value-times-area totals are distinct measures. Both modes require an axis-aligned grid. Alternatively, supply
positive finite `cellAreas`, exactly one per raster cell, in caller-declared area units; this requires
uniform mode. An invalid explicit area suppresses the affected zone's result (NaN areas and
totals/means, validity zero). `coveredArea` and `validArea` use the selected area units; density
and intensive `totals` have input units times area, while extensive totals retain input units.
Means remain coverage-fraction weighted in input units for every area model.
The default planar determinant is that of `grid.affine`, fixed at compile time, unless
an optional `parameters` view supplies the per-frame georeference. Prefer the shared version-2
view shown above for rebased geometry: geographic/Web Mercator consumers restore absolute Y
from word 9 only when word 8 is `2`; word 7 is always ignored. Update that shared packet when
cell size or absolute latitude changes. Separate local coverage and absolute area views remain
a valid alternative for legacy packets, which carry no absolute-Y extension. For a 1 m² EPSG:3857
cell centered at Y `4,600,000.5`, WGS84 ground area is about `.617505 m²`; using a legacy packet
with rebased Y center `.5` instead gives `.993306 m²`.
Ground-mode row centers follow the origin y, signed row step and versioned absolute Y offset
in `parameters`.
Explicit `cellAreas` override the affine area and must be updated by the caller when needed.
With explicit areas the unused affine determinant need not fit float32. Ground modes also check
the per-frame cross-axis words: a repacked non-axis-aligned affine suppresses all scalar results
(NaN aggregates and validity zero). Each zone's published areas and, when observed, total/mean
must fit finite float32; an unrepresentable aggregate suppresses that zone, preserving usable
zones. Empty or wholly missing zones retain their finite area outputs and validity zero.

{/* parameters: GPURasterFractionalCoverage generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterFractionalCoverage` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterFractionalCoverage`.

- Packer: `getGPURasterFractionalCoverageParameterValues(grid)`, writes `GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges
- Packer: `getGPURasterFractionalCoverageParameterValues(grid, options)`, writes `GPU_RASTER_FRACTIONAL_COVERAGE_LOCAL_PARAMETER_LENGTH` (10) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPURasterFractionalCoverageParameterValues(grid, options)` for `GPURasterFractionalCoverageParameterOptions`:

| Field | Type | Description |
|---|---|---|
| `localOrigin` (optional) | `readonly [number, number]` | Subtract this binary64 world origin from vertices before converting them to float32. |

{/* /parameters */}

### Group outlines and recipes

`GPUGroupConcaveHull` uses one group-aware Delaunay build and preserves original point rows without
readback. Its default fixed layout reserves `T = max(0, 2*pointCount-5)` triangle slots and `3*T`
boundary slots per group. With `layout: 'packed'`, those reservations cover all groups together;
retained triangles and boundary edges form separate compact group-major lists. The four views in
`output.groups` (`triangleOffsets`, `triangleCounts`, `edgeOffsets`, `edgeCounts`) each have one row
per group. Packed `status.count` and `requiredCount` count boundary edges and `overflow` is zero
with the required full reservation. `status.converged` has one row per group; incomplete groups
emit no geometry. Validity marks the used compact prefix, with zero padding. The triangle
reservation is bounded by 715,827,882 rows for uint32 addressing and by device binding sizes.

`addClusterAndOutlineRecipe` accepts `outline: {kind: 'alpha', maximumCircumradius}`. The result's
`outlineType` identifies the selected mode. Alpha geometry lives in `concaveHull`; `areas`,
`perimeters`, and `centroids` describe its retained area. In alpha mode the legacy single-ring
`hullOffsets` and `hullCounts` are empty, because one ring cannot represent holes and disconnected
components. Default convex behavior is unchanged. Empty area has zero area/perimeter and undefined
centroid. Nonconverged groups report invalid measures.

## Costs and travel

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

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterWalkingCost`.

- Packer: `getGPURasterWalkingCostParameterValues(settings, grid)`, writes `GPU_RASTER_WALKING_COST_PARAMETER_LENGTH` (16) `float32` values.
  - `grid` (`GPURasterMetadata`): Grid whose georeference is packed. Its width and height must match the compiled one.

Settings of `getGPURasterWalkingCostParameterValues`:

| Field | Type | Description |
|---|---|---|
| `frictionWeight` (optional) | `number` | Coefficient multiplying additive average-friction * ground-distance cost. Default 1. |
| `model` (optional) | `GPURasterWalkingCostModel` | Movement model; only the GRASS coefficients and threshold are packed. Default Tobler. |

{/* /parameters */}

## Reproject rasters

### `GPURasterReprojection`

Nearest interpolation alone does not preserve native integer identity in `output.values`.
Request matching `output.labels` and read it with `output.validity`. This uint32 example
uses identity calibration; sint32 input similarly requires a sint32 labels view:

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPURasterReprojection, makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
const grid = makeGPURasterMetadata({width: 2, height: 1});
const inputStorage = new GPUParameterBuffer(device, {
  id: 'warp-input', format: 'uint32', length: 2, values: Uint32Array.of(33554432, 33554433)
});
const labelStorage = new GPUParameterBuffer(device, {id: 'warp-labels', format: 'uint32', length: 2});
const valueStorage = new GPUParameterBuffer(device, {id: 'warp-values', format: 'float32', length: 2});
const validityStorage = new GPUParameterBuffer(device, {id: 'warp-validity', format: 'uint32', length: 2});
const reprojection = new GPURasterReprojection({
  sourceGrid: grid, destinationGrid: grid, method: 'nearest',
  input: makeGPURasterBand(inputStorage.importToGraph(graph), grid, {scale: 1, offset: 0}),
  output: {
    labels: labelStorage.importToGraph(graph), values: valueStorage.importToGraph(graph),
    validity: validityStorage.importToGraph(graph)
  }
});
graph.add(reprojection);
const compiled = await graph.compileAsync();
const encoder = device.createCommandEncoder();
compiled.encode(encoder, {parameters: undefined});
device.submit(encoder.finish());
const labelBytes = await labelStorage.buffer.readAsync();
const validityBytes = await validityStorage.buffer.readAsync();
const valueBytes = await valueStorage.buffer.readAsync();
const labels = new Uint32Array(labelBytes.buffer, labelBytes.byteOffset, 2);
const validity = new Uint32Array(validityBytes.buffer, validityBytes.byteOffset, 2);
const values = new Float32Array(valueBytes.buffer, valueBytes.byteOffset, 2);
for (let row = 0; row < 2; row++) {
  if (validity[row] !== 0) console.log(labels[row], values[row]);
}
// Native labels: [33554432,33554433]; continuous values: [33554432,33554432].
// Invalid rows have unspecified labels; always check validity before using them.
compiled.destroy();
reprojection.destroy();
inputStorage.destroy(); labelStorage.destroy(); valueStorage.destroy(); validityStorage.destroy();
```

The sampling transform is read from a float32 parameter view packed by
`getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid, inverseProjection})`.
Write new words into a `GPUParameterBuffer` between encodings to change origins, rotation, shear
or resolution. Shapes, CRS, pixel interpretation and the inverse projection stay fixed. Affines
whose packed float32 axes are singular are rejected before encoding. The packer
composes the entire destination-to-source transform in float64 for same-CRS sampling, including
origin differences and area/point sample shifts. Evaluation is centered on a destination pixel
near source support before packing, preserving cancellation between large scales and translations.
Identical grids therefore
preserve sample coordinates on wide sheared strips within float32 sampling-index precision.
With an explicit projection, the source inverse linear affine is computed in float64 before packing;
the packer rejects a source inverse whose packed coefficients form a singular matrix.
Double-single outputs subtract a split
source origin. Projection inputs should still use a local frame; ill-conditioned source affines can
amplify float32 projection error, so choose a source lattice that resolves the required cells.
Without `parameters` the contributor owns a buffer containing the initial affines. Call `destroy()`
after destroying its compiled graph, including for same-CRS sampling. Tile reprojection forwards
this same parameter view to its sampler; source tile bounds stay fixed in source pixel coordinates.

The 20-word packed layout is now **version 3**, exposed as
`GPU_RASTER_REPROJECTION_PARAMETER_VERSION`, with the version stored in word 17. Repack manual
parameter buffers using the current packer. A different version produces NaN values and validity
zero. Versions 1 (separate axes/determinant) and 2 (unanchored composition) must be repacked.

| Words | Same CRS (no inverse projection) | Explicit inverse projection |
|---|---|---|
| 0–3 | Composed sample matrix `[a, b, d, e]` | Destination world matrix `[a, b, d, e]` |
| 4–5 | Source sample position at the destination anchor, including pixel shifts | Destination world origin |
| 6–9 | Reserved zeros | Source inverse matrix `[a, b, d, e]` |
| 10 | Reserved zero | Reserved zero |
| 11–12 | Zeros | Local-f32 projection-origin minus source-origin shift; zeros for double-single |
| 13–16 | Zeros | Source origin `[xHigh, xLow, yHigh, yLow]` for double-single; zeros for local-f32 |
| 17 | Version 3 | Version 3 |
| 18–19 | Destination pixel anchor `[column, row]` | Reserved zeros |

Without an inverse projection, subtract words 18–19 from integer destination column/row,
multiply by words 0–3, and add words 4–5 to obtain source sample column/row directly.
Area centers and the source half-cell shift
are already included. With a projection, generate destination centers as before, project, subtract
the source origin, apply words 6–9, and subtract 0.5 for an area-interpreted source grid.

Nearest sampling compares the coordinate's fractional part to 0.5 before incrementing its floor,
on both axes. Exact ties select the higher index. This avoids rounding a coordinate just below
the half-cell boundary up when adding 0.5 in float32.

{/* parameters: GPURasterReprojection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterReprojection` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterReprojection`.

- Packer: `getGPURasterReprojectionParameterValues(settings)`, writes `GPU_RASTER_REPROJECTION_PARAMETER_LENGTH` (20) `float32` values.

Settings of `getGPURasterReprojectionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `sourceGrid` | `GPURasterMetadata` | - |
| `destinationGrid` | `GPURasterMetadata` | - |
| `inverseProjection` (optional) | `CompiledProjection` | Same compiled inverse projection used by the contributor. |

{/* /parameters */}

### Projection ownership and incomplete iteration

`GPURasterReprojection` accepts an inverse projection already compiled for destination-to-source
coordinates. Without one, CRS metadata must match. Source pixels outside the provided support and
invalid projection domains remain nodata; strict bilinear sampling requires every participating
sample, while renormalization uses surviving support. A projection-bearing contributor registers
once and owns its projection parameter buffer: destroy it after destroying its compiled graph.
Renormalization divides by the actual positive weight, so a lone value of 7 remains 7 even at
normal float32 weight 1e-32. Zero support uses a safe denominator and remains invalid.

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

The CPU walking-cost function and operation honor an explicit `maximumIterations` with the same
`maximumIterations + 1` round limit as the GPU contributor and operation. A flat six-cell unit-cost
line seeded at cell 0 with a budget of one publishes `[0, 1, 2, Infinity, Infinity, Infinity]`,
`iterationCount=2`, `converged=0` and `iterationLimitReached=1`. A complete solve needs five
advances and one unchanged pass. Without an explicit CPU budget the existing complete solve is
retained. `invalidCount` is independently 1 for a parameter block packed for the other model,
and 0 for a matching block; that diagnostic does not count unreachable cells.
Both walking-cost operation forms capture `maximumIterations` at construction.
The direct GPU walking contributor captures it too. Band metadata and declared output/status
names are captured by property access, including inherited names; later nested caller edits do
not rename the graph's outputs. All walking forms also capture grid dimensions/affine. Operation
constructors default only an omitted model to Tobler; supplied models must have kind `'tobler'`
or `'grass'`, checked before capturing model-specific options. Changing the caller's model from
GRASS to Tobler cannot invalidate an already constructed GRASS operation. GPU-resident values,
masks and parameter contents remain dynamic. Both CPU and GPU walking and watershed operation
forms require nonempty selected input and output graph names. Optional validity is selected
whenever it is not `undefined`, so `''` is rejected. Repeated reads of one input are allowed;
outputs must be distinct and cannot overwrite an input. These name checks add no GPU capacities
to CPU admission.

For reconstruction and fill holes, `maximumIterations` bounds the published propagation distance
in neighbor steps. The extra verification pass only checks whether the budgeted result is a fixed
point; it never advances that result. `iterationCount` includes this check and can therefore be
`maximumIterations + 1`. With marker `[1, 0, 0, 0]`, mask `[1, 1, 1, 1]` and a one-round dilation
budget, reconstruction publishes `[1, 1, 0, 0]`, count 2, `converged=0` and
`iterationLimitReached=1`. The CPU reconstruction and hole-fill twins obey the same budget and
default of 64 rounds; they drop only the GPU command-node cap. For an unfinished hole fill,
background farther than the budget from the boundary is still marked as a hole.
CPU reconstruction band calibration follows GPU float32 conversion order: convert the raw
sample, scale and offset to float32 before multiplication/addition, then determine usable
marker/mask pairs. Native integer nodata is compared before conversion. Thus marker 33554433
with offset -33554432 and mask 0 yields usable value 0, because the integer rounds to 33554432
first. Both reconstruction forms read only the grid prefix of padded values/validity arrays.
The CPU hole-fill queue stops at that frontier and only inspects immediate neighbors there to
verify completeness; it does not traverse the remaining connected component. Construction captures
the grid, iteration budget and nested port names in both operation forms and contributors.
Reconstruction and fill-hole operation forms capture documented band/storage and convergence
names by property access, including inherited getters, and retain those names through lowering.
Reconstruction selects marker and mask validity whenever the name is not `undefined`; an
explicitly empty name is rejected rather than dropping the validity input.
The website's fill-hole and reconstruction previews return convergence, label unfinished propagation,
and withhold final fill/cut counts and their derived overlays until convergence.
The showcase's missing-observations scene likewise omits the reconstruction bar from its selected-cell
chart until convergence, matching its map and readout even when the partial value is finite.

Reconstruction band inputs in both operation forms use the first `grid.width * grid.height` samples
and validity flags. Each supplied values/validity view must contain at least that many entries; spare
capacity is ignored. CPU calibration borrows that logical prefix at the supplied typed-array offset,
so a nonzero offset in either values or validity remains meaningful. Undersized bands are rejected.

## Reference evidence

[`raster-terrain-expansion.py`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/oracles/raster-terrain-expansion.py)
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
