# Point cloud analysis

`@luma-contributors/analysis/pointcloud-analysis` provides experimental WebGPU point-cloud
neighborhoods, local shape, terrain surfaces, ground classification, rigid registration, and
surface change as operations in a caller-owned `GPUCommandGraph`. Voxel downsampling, radius
filtering and octree construction (`GPUPointCloudVoxelDownsample`, `GPUPointCloudRadiusFilter`,
`GPUPointCloudOctree`) are owned by luma.gl: see
[`@luma.gl/graph/gpu-pointcloud`](https://luma.gl/docs/api-reference/experimental/gpu-pointcloud).

| Operation | Result | Cost |
| --- | --- | --- |
| `GPUPointCloudNeighborhoods` | Bounded nearest 3D neighbors and actual radius support | With `grid`: sort + visited candidates × capacity. Without: O(queries × points × capacity), refused above 2^27 query-point pairs |
| `GPUPointCloudFeatures` | Oriented normals, covariance eigenvalues, shape measures | O(points × capacity) |
| `GPUPointCloudStatisticalFilter` | Mean neighbor distance and outlier mask | O(points × capacity) + three published reductions |
| `GPUPointCloudSurfaces` / `GPUPointCloudHeight` | Indexed DTM/DSM/CHM and cell, bilinear or bounded-IDW HAG | Sort + O(points + cells); HAG O(points × sampling window) |
| `GPUPointCloudGround` / `GPUPointCloudSMRF` | Progressive morphology / SMRF ground mask/classification | O(cells × sum(windowArea) + points × stages) |
| `GPUPointCloudICP` | Point-to-point / point-to-plane rigid transform, robust weights, status/history | With `targetGrid`: O(iterations × visited candidates + sourcePoints). Without: O(iterations × source × target), refused above 2^27 pairs |
| `GPUPointCloudChange` | Signed cylinder change, confidence interval and support | With `grid`: sort + O(cores × points in the cylinder's cells). Without: O(cores × totalPoints), refused above 2^27 core-point pairs |
| `GPUPointCloudVoxelMerge` | Associative merge of per-part voxel counts, centroids, representatives and attribute means | O(voxels × parts) |

## Runtime parameter updates

For features, statistical filtering, progressive ground filtering, SMRF, ICP and M3C2 change, import a caller-owned
`GPUParameterBuffer` from `@luma-contributors/analysis/analysis-core`, initialize it with the
contributor's `getGPUPointCloud*ParameterValues` packer, and pass its graph view as `parameters`.
Writing new packed values before the next submission changes that encoding without rebuilding the
graph or shaders. Parameter views support nonzero byte offsets. Values must satisfy the packer's
limits; do not write unchecked raw settings. Destroy the compiled graph before destroying the buffer.
Constructor threshold/orientation props remain available for existing callers; a supplied parameter
view takes precedence. This does not change output layouts or status values.

Constructors capture declared props and caller-owned `output` fields, including inherited and
non-enumerable getters. The captured bindings remain fixed after construction; buffer contents
remain per-frame data. This includes voxel-merge output attribute fields and a statistical-filter
`standardDeviations` value of zero, which retains the global-mean threshold instead of the default
two-deviation threshold.

## Grid-index compatibility

The local point-cloud index remains necessary with published luma.gl `10.0.0-alpha.13`:

- Point-cloud cells use float32 `(position - origin) / cellSize`; `GPUGridIndex` normalizes through
  inclusive bounds and gridSize, which need not produce the same float32 cell edges.
- Point-cloud domains are half-open and exclude the upper edge; the published index includes it.
- Point-cloud IDs are ascending within each cell for reproducible sums and tie handling; published
  IDs within a cell have unspecified order after atomic scatter.
- Raster indexing ignores Z while retaining `float32x3` positions and supports north/south row
  order. The published index requires position and grid dimensions to match and has no raster mode.

Published caller-owned `cellOffsets` and `objectIds` already provide the needed output views;
output access is not a blocker. Replacing the local helper would require the four semantic
contracts above to match. Existing half-open, ordering and raster-direction specs pin them.

## Quick start

Follow the [complete GPU walkthrough](./operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for device creation, allocation, imports, submission, typed readback and cleanup. Statistical
filtering consumes neighborhoods rather than raw positions: first run `GPUPointCloudNeighborhoods`
for your `float32x3` points, or supply equivalent packed squared distances and counts.

| Input | Format and rows | Meaning |
| --- | --- | --- |
| `squaredDistances` | `float32`, pointCount × capacity | Row-major retained neighbor distances squared, with `capacity` slots per point. |
| `counts` | `uint32`, pointCount | Radius support per point; may exceed `capacity`. The filter uses `min(count, capacity)` retained slots. |
| `capacity` | constructor number, 1–64 | Slots per point, matching the neighborhood producer. |

Add `GPUPointCloudStatisticalFilter` with these inputs, `standardDeviations: 2`, and separate
output buffers passed as `output: {meanDistances, validity}`:

| Output | Format and rows | Meaning |
| --- | --- | --- |
| `meanDistances` | `float32`, pointCount | Mean retained neighbor distance; -1 when no neighbors exist. |
| `validity` | `uint32`, pointCount | 1 for inliers, 0 for outliers or rows without neighbors. |

For a small filter-only fixture, set `capacity: 1`, squared distances `[1, 1, 100, 0]`, and
counts `[1, 1, 1, 0]`. Mean distances are `[1, 1, 10, -1]`; the last row is excluded from
the global mean and population standard deviation and has validity 0. With multiplier 2,
the first three rows are inliers, giving validity `[1, 1, 1, 0]`. Read both outputs using the
walkthrough's lifecycle, then destroy the compiled graph before the imported buffers. A
neighborhood producer's candidate/overflow conditions must be handled before interpreting the
filter as a complete spatial-neighborhood analysis; the filter operates on the supplied slots.

## Ownership and batching

Every contributor takes its caller-owned result views in `output`, typed as
`GPUPointCloud<Name>Output`. Inputs stay at the top level. For example, neighborhood `counts`
is `output.counts` on the producer and becomes the top-level `counts` input of features:

```ts
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures
} from '@luma-contributors/analysis/pointcloud-analysis';

// All views are allocated by the caller on graph.
graph.add(new GPUPointCloudNeighborhoods({
  positions, radius: 1, capacity: 4,
  output: {indices, squaredDistances, counts, status: neighborhoodStatus}
}));
graph.add(new GPUPointCloudFeatures({
  positions, indices, counts, capacity: 4,
  output: {normals, eigenvalues, shapes, status: featureStatus}
}));
```

The `Outputs` lists below name fields inside `output`; formats, lengths and status values are
unchanged. Update an output override as `{...props, output: {...props.output, normals}}` so the
other result views remain present. Ground filtering still receives `minimumSurface` and `coverage`
as inputs, and height sampling still receives `terrain` and `coverage` as inputs. Voxel merge
keeps each part's statistics in `parts`, and writes merged statistics to `output`, including
`output.attribute` when requested. Statistical filtering publishes flags at `output.validity`
(also available through `contributor.props.output.validity`); the former top-level props and
`contributor.validity` access have been removed.
Authored catalog and showcase examples use the same nested `output` contract; the catalog examples
are typechecked against the public point-cloud package entry point, separately from its generated
inventory. Their parameter lists expose the output object instead of repeating writable fields as inputs.
Status views keep their producer's row domain: SMRF's `smrfStatus` has one word per point, while
voxel merge's separate `mergedStatus` has one word per voxel, even when both run on the same graph.

Inputs and outputs are non-owning `GraphDataView`s with support for nonzero byte offsets. Output
buffers must be separate from input buffers. Writable output storage bindings must remain disjoint
after their start offsets are rounded down to 256-byte alignment; disjoint logical views alone are
not sufficient. Use separate buffers or leave sufficient aligned space between output views. The caller owns
allocation, graph compilation, encoding, submission, rendering and optional readback. Destroying
compiled kernels does not destroy imported buffers.

Initial operations accept one resident packed view per attribute, fitting device storage-binding
limits. They never concatenate streaming chunks. `GPUPointCloudVoxelMerge` explicitly merges disjoint owned-row reductions on an identical grid; it never infers overlap or deduplicates halo copies.


## Contributor reference

One heading per class. All eleven implement `GPUCommandNodeProducer`, take non-owning packed
`GraphDataView`s, and refuse bad props at construction with a message that starts with the
contributor ID (default `gpu-pointcloud-<name>`) and names the prop and its accepted range. Numeric
props use the per-frame parameter views described below where available. Constructor numeric
props remain compatible defaults when no parameter view is supplied; those defaults are compiled
into the shader. Capacities, view shapes, grid geometry and morphology windows are compile-time.
The limits are exported from `pointcloud-analysis/operations/pointcloud-limits.ts` as
`GPU_POINT_CLOUD_MAXIMUM_*`. Status bit and code values are exported as
`GPU_POINT_CLOUD_<NAME>_STATUS_*`.

### GPUPointCloudNeighborhoods

- Props: `positions` (`float32x3`), `radius` (inclusive, positive), `capacity` (1 to 64,
  `GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY`), optional `queryRows`, `strategy` (`'serial'` default or
  `'cooperative'`), `supportBounds` (requires `status`) and `grid`.
- Outputs: `indices` and `squaredDistances` (`capacity` slots per query), `counts` (full radius
  support), optional `status`.
- Status bits: 1 `..._NEIGHBORHOODS_STATUS_INVALID_QUERY` (query row out of range, not finite or outside
  `grid`; the list is empty), 2 `..._TRUNCATED` (support above `capacity`), 4 `..._UNSUPPORTED`
  (the search sphere crosses `supportBounds`). Bits combine; higher bits are zero.
- Limits: `grid` axes at most 2^24 cells (`GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS`, exact float32 cell
  coordinates), total cells below 2^32, offsets within one storage binding. Without `grid`,
  queries × positions at most 2^27 (`GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK`).

{/* parameters: GPUPointCloudNeighborhoods generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudNeighborhoods` parameters

- Packer: `getGPUPointCloudGridParameterValues(settings)`, writes `GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `gridParameters`.
- Packer: `getGPUPointCloudNeighborhoodsParameterValues(settings)`, writes `GPU_POINT_CLOUD_NEIGHBORHOODS_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPUPointCloudGridParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` | `readonly [number, number] \| readonly [number, number, number]` | - |
| `cellSize` | `number` | - |

Settings of `getGPUPointCloudNeighborhoodsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` | `number` | - |
| `supportBounds` (optional) | `GPUPointCloudNeighborhoodsProps['supportBounds']` | - |

{/* /parameters */}

### `GPUPointCloudFeatures`

- Props: `positions`, `indices`, `counts`, `capacity` (1 to 64), optional `viewpoint` (finite).
- Outputs: `normals`, `eigenvalues`, `shapes` (`float32x3`), `status`, optional packed `covariance`
  (6 values per row).
- Status bits: 1 `..._FEATURES_STATUS_DEGENERATE` (fewer than three neighbors, or collinear or
  coincident neighbors: normal and shape are zero), 2 `..._TRUNCATED` (counts above `capacity`),
  4 `..._NOT_CONVERGED` (eight cyclic Jacobi sweeps, at most 24 pair rotations, were not enough).
- Per-frame: optional `parameters: float32[4]` from `getGPUPointCloudFeaturesParameterValues`
  overrides `viewpoint`, including enabling/disabling it. Other input contents are per-frame.
- Compile-time: capacity, view shapes and optional covariance output. Runtime orientation uses a
  second bounded-neighborhood pass so covariance output still fits eight storage bindings.
- Limits: capacity 64, per-lane arrays and the shared eigensolve's eight-sweep budget.

{/* parameters: GPUPointCloudFeatures generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudFeatures` parameters

- Packer: `getGPUPointCloudFeaturesParameterValues(settings?)`, writes `GPU_POINT_CLOUD_FEATURES_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUPointCloudFeaturesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `viewpoint` (optional) | `readonly [number, number, number]` | Orient normals toward this point; omitted chooses nonnegative z, then y, then x. |

{/* /parameters */}

### `GPUPointCloudStatisticalFilter`

- Props: `squaredDistances`, `counts`, `capacity` (1 to 64), `standardDeviations` (default 2).
- Outputs: `meanDistances` (`-1` without support), `validity` (1 inlier, 0 outlier).
- Per-frame: optional `parameters: float32[1]` from
  `getGPUPointCloudStatisticalFilterParameterValues` overrides `standardDeviations`.
- Compile-time: capacity and view shapes.
- Status: none per row. A row without neighbors is an outlier by definition.
- Limits: capacity 64. The global statistics are three `GPUReduction` sums plus one single-row
  threshold kernel; no kernel loops over the rows.

{/* parameters: GPUPointCloudStatisticalFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudStatisticalFilter` parameters

- Packer: `getGPUPointCloudStatisticalFilterParameterValues(settings?)`, writes `GPU_POINT_CLOUD_STATISTICAL_FILTER_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUPointCloudStatisticalFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `standardDeviations` (optional) | `number` | Nonnegative multiplier of the population standard deviation; default 2. |

{/* /parameters */}

### GPUPointCloudSurfaces

- Props: `positions`, `classifications`, `grid` (`PointCloudRasterGrid`), `groundClass` (default 2).
- Outputs: `terrain`, `surface`, `canopy`, `coverage` (one per cell).
- Coverage bits: 1 `GPU_POINT_CLOUD_COVERAGE_ANY_POINT`, 2 `GPU_POINT_CLOUD_COVERAGE_GROUND`.
- Limits: grid cells below 2^32 (`GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS`), 
- Row order: `grid.rowDirection` is `'south'` by default (row 0 at the maximum y, rows run toward decreasing y), the order of the raster and terrain families (GDAL and image order, `GPURasterMetadata`, D8 flow codes), so these rasters feed `GPUTerrainFlow`, the derivatives and the illumination contributors without flipping rows. Pass `'north'` for row 0 at the minimum y. Unlike the terrain classes, `PointCloudRasterGrid` has no affine or signed cell height, so the direction cannot be derived from the grid and a contradiction cannot be detected: give every point-cloud contributor that shares the grid the same direction. `GPUPointCloudSurfaces`, `GPUPointCloudHeight`, `GPUPointCloudGround` and `GPUPointCloudSMRF` honour it; a point on a cell boundary belongs to the cell above in either order. Membership is the same half-open test in both orders, on `o = (y - origin.y) / cellSize`: the point is inside when `0 <= o < height`, in north row `floor(o)` or south row `height - 1 - floor(o)`. Before this change the default was `'north'`, so a caller that left the prop out now gets the rows mirrored.

{/* parameters: GPUPointCloudSurfaces generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudSurfaces` parameters

- Packer: `getGPUPointCloudGridParameterValues(settings)`, writes `GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `gridParameters`.
- Packer: `getGPUPointCloudSurfacesParameterValues(settings?)`, writes `GPU_POINT_CLOUD_SURFACES_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUPointCloudGridParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` | `readonly [number, number] \| readonly [number, number, number]` | - |
| `cellSize` | `number` | - |

Settings of `getGPUPointCloudSurfacesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `groundClass` (optional) | `number` | - |

{/* /parameters */}

### GPUPointCloudHeight

Height queries and aligned outputs accept chunked GraphVectorView input. Height sampling settings
are runtime inputs; grid axes and float32 support bounds are validated at construction.

- Props: `positions`, `grid`, `terrain`, `coverage`, `coverageBit`, `interpolation` (`bilinear` or
  `idw` with `radius`, `power`, `minimumSupport`).
- Outputs: `heights`, `status`.
- Status codes (values, not bits): 0 `GPU_POINT_CLOUD_HEIGHT_STATUS_VALID`, 1 `..._INVALID_POINT`,
  2 `..._UNSUPPORTED_TERRAIN`, 3 `..._NONFINITE`.
- Limits: IDW radius at most 64 cells (`GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS`, each point visits that many cells squared),
  power in (0, 8] (`GPU_POINT_CLOUD_MAXIMUM_IDW_POWER`, higher powers underflow the normalized weights).

{/* parameters: GPUPointCloudHeight generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudHeight` parameters

- Packer: `getGPUPointCloudGridParameterValues(settings)`, writes `GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `gridParameters`.
- Packer: `getGPUPointCloudHeightParameterValues(settings, grid)`, writes `GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH` (5) `uint32` values.
  - `grid` (`PointCloudRasterGrid`)

Settings of `getGPUPointCloudGridParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` | `readonly [number, number] \| readonly [number, number, number]` | - |
| `cellSize` | `number` | - |

Settings of `getGPUPointCloudHeightParameterValues`:

| Field | Type | Description |
|---|---|---|
| `coverageBit` (optional) | `number` | - |
| `interpolation` (optional) | `GPUPointCloudHeightProps['interpolation']` | - |

{/* /parameters */}

### `GPUPointCloudGround`

- Props: `positions`, `classifications`, `grid`, `minimumSurface`, `coverage`, `radii`,
  `initialThreshold`, `slope`, `maximumThreshold`.
- Per-frame: optional `parameters: float32[radii.length]` from
  `getGPUPointCloudGroundParameterValues(settings, grid, radii)` overrides all three threshold props.
  The packer computes each stage threshold in float64 and rounds once, retaining the existing
  float32 boundary behavior. Use the same grid and radii passed to the contributor.
- Compile-time: grid, radii and view shapes.
- Outputs: `groundMask`, `outputClassifications`, `status`.
- Status bits: 1 `..._GROUND_STATUS_UNSUPPORTED` (outside the grid, not finite or uncovered cell),
  2 `..._INCOMPLETE_WINDOW` (the final window reaches past covered cells). Bit 1 excludes bit 2.
- Limits: 1 to 32 radii (`GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_COUNT`, three kernels each), each radius 1 to 32
  cells (`GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_CELLS`, each cell scans (2r+1)² neighbors).

{/* parameters: GPUPointCloudGround generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudGround` parameters

- Packer: `getGPUPointCloudGridParameterValues(settings)`, writes `GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `gridParameters`.
- Packer: `getGPUPointCloudGroundParameterValues(settings, grid, radii)`, writes `getGPUPointCloudGroundParameterLength(...)` `float32` values.
  - `grid` (`PointCloudRasterGrid`): The contributor's compile-time grid.
  - `radii` (`readonly number[]`): The contributor's compile-time radii, in cells.

Settings of `getGPUPointCloudGridParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` | `readonly [number, number] \| readonly [number, number, number]` | - |
| `cellSize` | `number` | - |

Settings of `getGPUPointCloudGroundParameterValues`:

| Field | Type | Description |
|---|---|---|
| `initialThreshold` | `number` | Nonnegative elevation threshold added at every stage, in z units. |
| `slope` | `number` | Nonnegative terrain slope per cell of radius increment. |
| `maximumThreshold` | `number` | Nonnegative upper threshold, at least initialThreshold. |

{/* /parameters */}

### `GPUPointCloudSMRF`

- Props: `positions`, `classifications`, `grid` (at least 2 by 2), `selection`, `slope`, `scalar`,
  `threshold`, `window`, `cut`, `groundClass`, `otherClass`, `onlyGround`.
- Per-frame: optional `parameters: float32[2 + maximumRadius]` from
  `getGPUPointCloudSMRFParameterValues(settings, grid, maximumRadius)` overrides `slope`, `scalar`
  and `threshold`. The packer rounds per-radius slope products once, matching constructor thresholds.
  Pass the contributor's `maximumRadius` and grid.
- Compile-time: grid, window, cut, class labels, onlyGround, optional selection and view shapes.
- Outputs: `groundSurface` (per cell), `groundMask`, `outputClassifications`, `status` (per point).
- Status bits: 1 `..._SMRF_STATUS_INVALID`, 2 `..._NO_TERRAIN`, 4 `..._EXCLUDED`, 8 `..._INPAINTED`
  (inpainted terrain, or within the filter margin of the grid edge). Bits 1, 2 and 4 are exclusive
  and exclude bit 8.
- Limits: window at most 64 cells (`GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS`; the openings are unrolled
  into published `GPURasterErosion` and `GPURasterDilation` passes of radius at most 8, about 50 passes at
  the default 18 and 350 at 64), cut at most 32 cells
  (`GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS`). The inpainting valid counts are published `GPUReduction` sums. The
  inpainting ring search is still a per-cell loop that can reach `max(width, height)` rings on sparse coverage.

{/* parameters: GPUPointCloudSMRF generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudSMRF` parameters

- Packer: `getGPUPointCloudSMRFParameterValues(settings, grid, maximumRadius)`, writes `getGPUPointCloudSMRFParameterLength(...)` `float32` values.
  - `grid` (`PointCloudRasterGrid`): The contributor's compile-time grid.
  - `maximumRadius` (`number`): The contributor's compile-time maximumRadius, in cells.

Settings of `getGPUPointCloudSMRFParameterValues`:

| Field | Type | Description |
|---|---|---|
| `slope` (optional) | `number` | Nonnegative terrain slope; default 0.15. |
| `scalar` (optional) | `number` | Nonnegative multiplier of the terrain gradient; default 1.25. |
| `threshold` (optional) | `number` | Nonnegative elevation threshold in z units; default 0.5. |

{/* /parameters */}

### `GPUPointCloudICP`

- Props: `source`, `target`, `maximumDistance`, `iterations` (1 to 100, default 20,
  `GPU_POINT_CLOUD_MAXIMUM_ICP_ITERATIONS`), `targetGrid`, `metric`, `targetNormals`, `robust`,
  `initialTransform`, `origin`, tolerances.
- Per-frame: optional `parameters: float32[22]` from `getGPUPointCloudICPParameterValues(settings, origin)`
  overrides maximum distance, robust scale, both tolerances and the initial transform. Pass the
  same origin as the contributor so the packer converts absolute transforms to its local frame.
  Set the packer's `robustMethod` to the contributor's compile-time `robust.method` (the packer
  defaults to `'tukey'` to preserve earlier validation). Huber accepts every positive finite
  float32 scale; only Tukey requires a finite, nonzero float32 square. When a Huber square is out
  of range, the unused square slot is 1; the unsquared scale retains its requested float32 value.
- Compile-time: iterations, metric, robust method (including whether it is enabled), targetGrid,
  origin and view shapes.
- Outputs: `transform` (16), `history` (one `float32x3` per iteration), `inlierMask`, `status` (1 row).
- Status codes (values, not bits): 0 `..._ICP_STATUS_CONVERGED`, 1 `..._ITERATION_LIMIT`,
  2 `..._INSUFFICIENT_MATCHES`, 3 `..._DEGENERATE`, 4 `..._EIGENSOLVE_EXHAUSTED`.
- Limits: iterations are unrolled graph nodes. Without `targetGrid`, source × target at most 2^27.

{/* parameters: GPUPointCloudICP generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudICP` parameters

- Packer: `getGPUPointCloudICPParameterValues(settings, origin?)`, writes `GPU_POINT_CLOUD_ICP_PARAMETER_LENGTH` (22) `float32` values.
  - `origin` (`readonly [number, number, number]`): The contributor's compile-time local origin, when present.

Settings of `getGPUPointCloudICPParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumDistance` | `number` | Positive inclusive correspondence distance with a finite, nonzero float32 square. |
| `translationTolerance` (optional) | `number` | Nonnegative translation-step tolerance; default 1e-5. |
| `rotationTolerance` (optional) | `number` | Nonnegative rotation-column-change tolerance; default 1e-5. |
| `robustScale` (optional) | `number` | Positive finite float32 scale; default 1. Only Tukey also requires a finite nonzero square. |
| `robustMethod` (optional) | `'huber' \| 'tukey'` | Match the contributor's compile-time robust.method; default tukey preserves earlier validation. |
| `initialTransform` (optional) | `readonly number[]` | Absolute column-major rigid initial transform; default identity. |

{/* /parameters */}

### `GPUPointCloudChange`

- Props: `reference`, `comparison`, `cores`, `normals`, `alignment`, `radius`, `halfDepth`,
  `minimumSupport`, uncertainty options, optional `grid`.
- Per-frame: optional `parameters: uint32[12]` from `getGPUPointCloudChangeParameterValues`
  overrides minimum support, registration error/covariance, confidence multiplier and the supplied
  sampling covariance values. Word 0 holds exact uint32 support; other words carry float32 bit
  patterns. Covariance model presence and uncertainty combination stay compile-time.
- Compile-time: radius, halfDepth (bounded cylinder candidate loops), grid, optional covariance
  model, uncertainty combination, alignment-view presence and view shapes. Alignment transform
  contents are already per-frame. With parameters, one extra per-core pass projects registration
  covariance and grid support into a `float32x3` scratch view, keeping every stage at eight or fewer
  storage bindings.
- Outputs: `changes`, `roughness` (`float32x3` per core), `counts` (two per core), `status`.
- Status codes (values, not bits): 0 `..._CHANGE_STATUS_VALID`, 1 `..._INVALID_CORE`,
  2 `..._INSUFFICIENT_SUPPORT`, 3 `..._NONFINITE_UNCERTAINTY`. In grid mode the flag
  `GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE` (0x100) is OR-ed onto any code except
  `..._INVALID_CORE` when a cell the core's cylinder needs lies outside the grid, so its counts may
  be incomplete; mask it off to read the code.
- Grid limit: each core visits `(ceil(2 * sqrt(halfDepth^2 + radius^2) / cellSize) + 3)^3` cells, at
  most `GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE` (2^20, about 4,800 times the largest use in
  the tests); the constructor asks for a larger `cellSize` above it.
- Limits: without `grid`, cores × (reference + comparison) at most 2^27; with `samplingCovariance`,
  at most 32768 points per cloud (`GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS`).
- Grid path: pass a `grid` of the same kind `GPUPointCloudNeighborhoods` takes. The reference and
  the aligned comparison are indexed once per encoding, and each core visits only the cells that the
  bounding box of its cylinder touches (half-depth along the normal, radius across it, plus one cell of margin).
  Counts and statuses equal the unindexed scan exactly; changes and roughness equal it to float32
  summation order (the tests compare to four decimals on the PDAL golden fixture and on a random cloud
  with tilted normals, alignment and sampling covariance). Points outside the half-open grid do not
  participate, so the grid must cover both clouds after alignment. The work is one kernel per cloud and a final
  combining kernel in both modes.

{/* parameters: GPUPointCloudChange generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointCloudChange` parameters

- Packer: `getGPUPointCloudChangeParameterValues(settings?)`, writes `GPU_POINT_CLOUD_CHANGE_PARAMETER_LENGTH` (12) `uint32` values.

Settings of `getGPUPointCloudChangeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumSupport` (optional) | `number` | - |
| `registrationError` (optional) | `number` | Nonnegative registration uncertainty in coordinate units, added to the sampling standard error. |
| `registrationCovariance` (optional) | `readonly [number, number, number, number, number, number]` | Optional aligned-frame translation covariance [xx,xy,xz,yy,yz,zz]; projected along each normal. |
| `confidenceMultiplier` (optional) | `number` | Two-sided normal quantile, default1.96 (95%); caller specifies their confidence convention. |
| `samplingCovariance` (optional) | `{ referenceVariance: number; comparisonVariance: number; correlationLength: number; }` | Optional known covariance of normal-direction measurement error: sigma² exp(-distance²/(2 length²)). Uses all within-cylinder pairs exactly; variances are supplied separately from empirical roughness. |

{/* /parameters */}

### GPUPointCloudVoxelMerge

- Props: `parts` (counts, centroids, representatives, `sourceRowOffset`, optional attribute),
  `partitionPolicy: 'disjoint-owned-rows'`, merged outputs and optional `attribute`.
- Outputs: `counts`, `centroids`, `representatives`, `status`, optional attribute means and counts.
- Status bits: 1 `..._VOXEL_MERGE_STATUS_COUNT_OVERFLOW` (merged count above uint32),
  2 `..._REPRESENTATIVE_OVERFLOW` (representative plus offset above the largest row ID). Reject flagged voxels.
- Limits: part `sourceRowOffset` below 2^32 - 1. No published or local contributor produces the
  `centroids` input yet; the tests build the parts by hand.

## 3D neighborhoods, normals and shape

`GPUPointCloudNeighborhoods` accepts packed Cartesian `float32x3` positions. Distances use all
three axes. `radius` is inclusive; each row excludes itself, while separate duplicate rows count.
`capacity` is 1–64. Allocate `indices: uint32[N * capacity]`,
`squaredDistances: float32[N * capacity]`, and `counts: uint32[N]`. Neighbors sort by squared
distance, breaking ties by original row ID. Missing slots have ID `0xffffffff` and maximum finite
float32 distance. `counts` contains the **full radius support**, so `counts > capacity` explicitly
identifies truncation. A finite radius covering the cloud gives bounded k-nearest neighbors.
Nonfinite points do not participate. This fixed-stride representation is not CSR.

`strategy: 'serial'` is the default. For dense candidate sets, `strategy: 'cooperative'` assigns
one workgroup to each query, scans candidates across its lanes and merges local nearest-neighbor
lists. Both strategies preserve full support counts, distance/ID ordering, radius boundaries and
status bits, with or without a grid. Cooperative selection uses at most 8,704 bytes of workgroup
storage and adds no global partial-list buffer. It can cost more on sparse queries; compare the
complete grid-and-query workflow before choosing it. Strategy selection is explicit.

`GPUPointCloudFeatures` consumes those three neighborhood fields and the original positions.
Allocate `normals`, `eigenvalues`, and `shapes` as `float32x3[N]`, and `status` as `uint32[N]`.
Optional `covariance: float32[N * 6]` exposes population moments in xx, xy, xz, yy, yz, zz order.
It centers coordinates relative to each query before accumulating population covariance, then
uses the shared symmetric eigensolver with at most eight cyclic sweeps (24 pair rotations) and
relative tolerance 1e-6. The solver returns ascending eigenpairs internally; this contributor keeps
its descending output order and the existing degeneracy thresholds. Eigenvalues are descending. Shape components are
`(largest-middle)/largest`, `(middle-smallest)/largest`, and `smallest/largest`.
Normals are unit vectors toward the supplied `viewpoint`; without one, z is nonnegative, then y,
then x at exact ties. This is a local orientation convention, not globally consistent orientation.

Feature status is a bitmask: 1 means fewer than three retained neighbors or a rank-deficient
neighborhood; 2 means radius support exceeded capacity; 4 means the eigensolver exhausted its
bound. Degenerate neighborhoods produce a zero normal and zero shape. Check status before use.
Without a `grid` the search costs O(N² × capacity) and uses O(N × capacity) output storage; the
constructor refuses queries × positions above `GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` (2^27) and
asks for a `grid`. The optional
`grid: {origin: [x,y,z], cellSize, dimensions: [width,height,depth]}` adds a GPU sorted 3D grid.
The grid bounds are half-open: points outside them do not participate. Grid key sorting, cell offset
construction, candidate search and features stay in the graph and rebuild on each encoding.
The exact distance test and deterministic nearest-ID order are identical to the reference path.
A conservative extra cell around the query protects inclusive boundaries from float32 bin rounding.
Optional neighborhood `status: uint32[N]` reports bit 1 for invalid/outside-grid queries and bit 2
for truncation. Choose grid bounds that cover the source and any required tile halos.

Indexed work is the GPU sort plus O(cells × log(points)) offset construction and the visited
candidate pairs × capacity; dense/skewed cells can still approach quadratic work. Scratch is
O(points + cells). The octree does not participate in this grid. The opt-in scaled benchmark
measures the complete indexed-neighborhood-plus-features graph rather than the eigensolve alone.

## Statistical filtering and voxel averaging

`GPUPointCloudStatisticalFilter` consumes neighborhood `squaredDistances`, `counts`, and `capacity`.
It writes the mean distance to retained neighbors (`-1` without support), and a row-preserving
`uint32` mask. Its threshold is the population mean of supported rows' mean distances plus
`standardDeviations` (default 2) times their population standard deviation. Zero-support rows are
rejected. Radius truncation remains observable through the input counts. The implementation is
O(N × capacity) for neighbor means; the valid count, mean and variance are three published
`GPUReduction` sums, each term divided by the valid count before it is summed so that a finite
distance cannot overflow the accumulated moment. It performs no compaction or attribute rewriting.
A point exactly at the threshold can classify differently across devices, because the sums are
tree-ordered and the reduction strategy depends on the device.

The published `GPUPointCloudVoxelDownsample` (`@luma.gl/graph/gpu-pointcloud`, 10.0.0-alpha.10)
takes `positions`, `origin`, `voxelSize` and `gridSize` and writes `representatives` (the smallest
source row ID per voxel), `counts` and `output` (the representative's position). It has no
`centroids` and no `attribute` option: an earlier version of this page described both, and they
are not in the published release. Gather sampled attributes using the representative IDs.

`GPUPointCloudVoxelMerge` needs a per-voxel `centroids` input (the arithmetic mean of the member
positions) and, optionally, per-voxel attribute means with their own counts. Nothing published or
local produces them, so today the caller computes them per part: on the CPU while the part is
built, or on the GPU as a keyed mean, with the voxel index
`(z * gridSize[1] + y) * gridSize[0] + x` of `floor((position - origin) / voxelSize)` as a dense
`uint32` key and the coordinates (and attribute components) as columns, for example with
`GPUGroupStatistics` and `keyCount = voxelCount`. Empty voxels must hold count 0. Do not pass the
published `output` as `centroids`: the merge would then return a count-weighted mean of
representative points, which is not the centroid. Voxel centroids and attribute means on
`GPUPointCloudVoxelDownsample` are an upstream gap. `GPUPointCloudVoxelMerge` combines disjoint partition reductions as described below. Applications must supply complete
neighborhood halos for tiled filtering; per-tile statistical thresholds are different from a
global-cloud threshold.

The website voxel-merge example and lidar support showcase compute each tile's arithmetic means
with `computePointCloudVoxelMeans(ownedPositions, voxelSize, gridSize, origin)` from the public
point-cloud entry point. This helper accepts CPU-owned float32 XYZ rows and returns dense float32
voxel means in the same coordinate frame; empty voxels contain zero. The helper rejects trailing
partial XYZ rows and voxel sizes that are not positive and finite after float32 rounding
(including zero, underflow and overflow). The helper also requires three positive safe-integer
grid dimensions, a safe-integer total cell
count below 4,294,967,295, and three origin coordinates finite after float32 rounding, matching
`GPUPointCloudVoxelDownsample`. Invalid grids and origins are rejected before allocation.
Upload those means separately from the downsampler's selected positions. The merge
consumes GPU counts and representative rows. Feature and change displays retain their status,
suppress unusable measurements, and explicitly label truncated or insufficient support.

Height displays accept status 0 only: unsupported terrain (status 2) is shown as unavailable,
distinct from a supported height of zero. ICP displays show aligned geometry only for status 0
(converged); an exhausted budget or failed fit is labelled and the alignment is hidden. Ground
and SMRF displays exclude every nonzero-status classification from ground counts and colored
point geometry, and report the number of incomplete or unsupported points. The SMRF raster is
explicitly provisional, including its inpainted terrain.

## Terrain surfaces and height above ground

`GPUPointCloudSurfaces` uses a caller-supplied planar grid (`origin: [x, y]`, `cellSize`, `width`,
`height`) and one `uint32` classification per point. Cells are row-major with x varying fastest;
bounds include the minimum and exclude the maximum. Each cell produces:

- `terrain`: minimum elevation among `groundClass` points (default LAS class 2).
- `surface`: maximum elevation among all finite points.
- `canopy`: surface minus terrain where both exist.
- `coverage`: bit 1 has any point; bit 2 has classified ground.

Outputs are zero where unsupported, so always consume coverage. A sorted cell index makes the reduction O(points + cells) after sorting.

When using classified-ground Surfaces terrain to seed Ground, first derive a separate nonzero
seed mask from `(coverage & GPU_POINT_CLOUD_COVERAGE_GROUND) != 0`; occupancy bit 1 alone is
insufficient. The lidar-ground showcase uses this mask and suppresses class-2 classifications
with any nonzero Ground status before the second Surfaces pass. Thus absent seeds and incomplete
morphology support cannot become recovered terrain. A filter deliberately initialized from all
returns can instead prepare an unclassified minimum surface with every point eligible.

The showcase uses a radius-one window on its sparse class-2 seed grid so that some recovered
terrain remains fully supported. Its ground colors and legend distinguish supported acceptance,
supported rejection, provisional incomplete windows, and unsupported/invalid rows. In particular,
`groundMask = 1` with status 2 is provisional, never accepted ground for the height pipeline.
Surfaces, Ground and Height capture nested grid settings (including a copy of the origin) at construction;
Ground also copies the radii and Height captures interpolation settings. Mutating caller settings
later does not change the declared passes or sampling frame. Buffer contents and explicit runtime
parameter buffers remain the per-frame inputs.

```ts
import {GPUPointCloudSurfaces, GPUPointCloudHeight, GPU_POINT_CLOUD_COVERAGE_GROUND}
  from '@luma-contributors/analysis/pointcloud-analysis';

graph.add(new GPUPointCloudSurfaces({
  positions, classifications, grid,
  output: {terrain, surface, canopy, coverage}
}));
graph.add(new GPUPointCloudHeight({
  positions, grid, terrain, coverage,
  coverageBit: GPU_POINT_CLOUD_COVERAGE_GROUND,
  output: {heights, status: heightStatus}
}));
```

`GPUPointCloudHeight` samples the containing terrain cell and subtracts its elevation from each
point's z. Pass the same caller-owned `coverage` view as Surfaces' `output.coverage` and Height's
`coverage` input, with `coverageBit: 2` to require classified-ground support; otherwise any
nonzero bit is accepted. Heights remain signed. Status is 0 for a supported sample, 1 for an invalid
or outside point, 2 for unsupported terrain, and 3 for a nonfinite interpolated elevation or height
difference. The displayed lidar pipeline uses ground-only `groundSeedCoverage` for classification
and `coverageBit: 2` for height sampling, matching the live scene: coverage word 1 alone cannot
establish ground support. Every invalid result is zero and must be masked by status. The default is **cell-constant raster sampling**; optional bilinear and bounded IDW sampling are described below. Direct nearest-ground and Delaunay interpolation are not provided. Inputs and
outputs retain the caller's units and vertical datum; no datum transform is performed.

## Progressive morphological ground classification

`GPUPointCloudGround` consumes an **unclassified cell-minimum surface**, its nonzero coverage,
original point positions and original classifications. Use `GPUPointCloudSurfaces` with an
all-equal classification input and that value as `groundClass` to prepare such a minimum surface.
`radii` is a strictly increasing array of 1–32 cell radii. Each stage opens the original minimum
surface using square erosion then dilation, and rejects points whose elevation above the opening
exceeds `min(maximumThreshold, initialThreshold + slope * cellSize * radiusIncrement)`, where
`radiusIncrement` is `radii[k] - radii[k - 1]` and the first stage uses `radii[0]`.
A point rejected at an earlier stage stays rejected. Negative residuals remain ground candidates.

This threshold is a convention of this class, not equation 7 of Zhang et al. (2003). Zhang uses
`initialThreshold` alone while the window is at most 3 cells wide, and
`initialThreshold + slope * (w[k] - w[k - 1]) * cellSize` afterwards, with the full window width
`w = 2 * radius + 1`. So the first stage here already adds `slope * cellSize * radii[0]`, and at
later stages the same `slope` is half as permissive, because a radius increment is half a
window-width increment: pass twice the slope to match Zhang's later stages. With `radii: [1, 2]`,
`cellSize: 1`, `slope: 0.1` and `initialThreshold: 0.2` the thresholds here are 0.3 and 0.3;
Zhang's are 0.2 and 0.4. No PCL or PDAL `filters.pmf` fixture pins this class.

The output `groundMask` identifies survivors. `outputClassifications` sets survivors to LAS ground
class 2, changes rejected former ground points to unclassified class 1, and retains other rejected
classes. The original classification input is never changed. Status bit 1 means outside/unsupported;
bit 2 means incomplete morphological support, including holes or grid edges. Incomplete windows
still use available samples; treat the flag according to the application's edge policy. The
algorithm is a documented progressive square-window morphological filter, **not PDAL SMRF parity**.
SMRF slope/scalar/cut semantics, hole filling, automatic cell sizing, and authoritative ground truth
are outside this implementation. Work is O(cells × sum(windowArea) + points × stages).

## Simple morphological filtering (SMRF)

`GPUPointCloudSMRF` separately implements the SMRF stages described by
[PDAL 2.8.4 SMRFilter](https://github.com/PDAL/PDAL/blob/2.8.4/filters/SMRFilter.cpp), using
[PDAL's diamond morphology and finite differences](https://github.com/PDAL/PDAL/blob/2.8.4/pdal/private/MathUtils.hpp).
It accepts `positions`, original `classifications`, optional `selection`, and a caller-owned grid.
For the PDAL grid alignment, use the selected cloud's minimum XY as origin, `cellSize` as PDAL's
`cell`, and `floor(extent / cellSize) + 1` dimensions. Dimensions must be at least 2 × 2.

The parameters preserve SMRF meanings: `slope` defaults to 0.15; `scalar` to 1.25; `threshold` to
0.5; `window` to 18 times `cellSize`; and `cut` to zero. `window` is a physical radius, converted
with `ceil(window / cellSize)`, bounded to 64 cells. Nonzero `cut` is physical net spacing, rounded
up to at most 32 cells. Unlike the square-window PMF above, this operation:

1. Rasterizes selected points to cell minima and fills empty cells from eight nearest valid cells.
2. Detects low outlier cells by opening the inverted surface with a one-cell diamond and a 5 × cell
   elevation threshold.
3. Optionally cuts a grid net using a diamond opening of radius twice the net spacing.
4. Erodes progressively by one cell and dilates to each radius. It flags differences from the
   **previous opening** above `slope * cellSize * radius`.
5. Removes object, low-outlier and net cells, then fills the provisional DEM from eight nearest
   remaining cells.
6. Computes central interior and one-sided edge gradients, and accepts absolute point-to-DEM
   residuals up to `threshold + scalar * gradientMagnitude`.

Allocate `groundSurface: float32[width * height]`, `groundMask: uint32[N]`,
`outputClassifications: uint32[N]`, and `status: uint32[N]`. `groundClass` defaults to 2 and
`otherClass` to 1. `onlyGround: true` preserves rejected original classes. Original input classes
are always retained separately. Zero `selection` rows preserve their original class and never
contribute to the terrain; construct this mask explicitly for return-number, class-bit and ignored
range policies. No LAS return inference or implicit point deletion occurs.

Status is 0 for supported interior samples, 1 for invalid/outside points, 2 when no terrain anchors
survive, 4 for excluded rows, and 8 for inpainted/edge-supported results. Code 8 still has a valid
classification under clipped-window semantics; it exposes the support limitation. Unsupported
results retain original classifications and have zero ground masks and zero terrain. This is a
classification estimate, not a claim of universal ground truth.

Eight-neighbor fill ties use column-major source-cell order deterministically. PDAL's KD-tree tie
ordering is not a portable public contract; equally distant cutoff ties may select different
anchors. The independent SciPy oracle covers holes, low outliers, slope scaling, scalar zero and
net cutting. Its checked-in generator optionally invokes the actual PDAL CLI for differential
classifications. Arithmetic is float32, so threshold-boundary decisions can differ from PDAL's
double-precision implementation. Inpainting expands square rings and stops only after every cell capable of beating or tying the eighth distance is visited. It is exact and remains quadratic for large unsupported regions; completely anchorless grids are detected once and bypass searching. Diamond morphology delegates to the published `GPURasterErosion` and `GPURasterDilation` (`@luma.gl/graph/gpu-raster`) with the `'cross'` structuring element, a `'nodata'` border and the `'ignore'` policy, so cells outside the grid do not take part. One pass covers a radius of at most 8 and a larger diamond is a sequence of passes, which preserves clipped diamond semantics exactly (a diamond is a Minkowski sum of smaller diamonds). A headless spec compares the windows with a full diamond scan for radii 1 to 37. Cell minima use a sorted index rather than scanning every point per cell. Provide full
halos for tiled use: silently classifying independently cropped tiles is not partition-equivalent.

## Bounded rigid registration

`GPUPointCloudICP` estimates a source-to-target rigid transform using point-to-point ICP. A caller
may supply a column-major rigid `initialTransform`; identity is the default. Every iteration finds
exact nearest targets within inclusive `maximumDistance`, accumulates centered correspondences,
and solves Horn's quaternion system using the shared symmetric eigensolver with at most eight
cyclic sweeps (48 pair rotations) and relative tolerance 1e-6. The largest ascending eigenpair
supplies the quaternion; the existing eigenvalue-gap degeneracy check and status codes remain. Optional `robust: {method: 'huber' | 'tukey', scale}` applies distance-based iteratively reweighted least squares. Tukey rejects residuals at or above its scale; Huber downweights them. Reflection and scale are not estimated.

Allocate `transform: float32[16]`, `history: float32x3[iterations]`, `inlierMask: uint32[sourceCount]`,
and `status: uint32[1]`. Iterations are bounded to 1–100 (default 20) and emitted as graph passes;
there is no intermediate host readback. The GPU freezes updates after convergence or failure.
History records pre-update support, RMSE, and translation step length. Final inlier flags are
recomputed using the returned transform. A zero-support RMSE of zero is **not** success: inspect
status (0 converged, 1 budget exhausted, 2 fewer than three matches, 3 underdetermined rotation,
4 eigensolve exhaustion). Convergence requires both translation and rotation-column change below
their respective tolerances (default `1e-5`). Frozen history entries repeat the current residual
with zero step.

Without `targetGrid`, correspondence work is O(iterations × sourceCount × targetCount). With a supplied exact 3D grid, the target is indexed once per encoding and every iteration searches intersecting cells. Target rows outside the half-open grid are excluded; supply a grid covering the full intended target. Ties retain the lowest source target-row ID. Stable weighted moments are computed in 256-row blocks and merged before the small transform solve; the solve no longer scans every source point serially. Independent scans can be represented by independent operations. Poor initialization,
partial overlap, repeated structures, or high outlier rates can lead to local minima. No global registration guarantee or automatic initialization is claimed. Robust reweighting is not a trimmed objective.

### Local origin and work limits

Positions are float32. At projected magnitudes near 1e6 m the float32 spacing is about 0.06 m, which
destroys millimetre registration. Recenter on the CPU with a shared float64 origin: call
`getPointCloudLocalOrigin(positions)`, upload `makePointCloudLocalPositions(positions, origin)` for the source,
target, cores and any `targetGrid`, and pass the same `origin` to `GPUPointCloudICP`. The GPU runs in the
local frame; `initialTransform` stays absolute and is converted for you. The published `transform` is in the
local frame: convert a readback with `icp.getAbsoluteTransform(transform)`
(`getPointCloudAbsoluteTransform`, or `getPointCloudLocalTransform` for the inverse). Every cloud that is
registered or compared together must use one origin; `GPUPointCloudChange` needs no origin option because it
consumes local-frame clouds and an ICP transform in that same frame.

The origin bounds use only points whose three coordinates are finite. For an audit count, call
`getPointCloudLocalOriginSummary(positions)` to obtain `{origin, skippedPointCount}`; a point with
any non-finite coordinate counts once and contributes none of its axes to the bounds. Empty or
all-invalid clouds return `[0, 0, 0]`. Both origin helpers and `makePointCloudLocalPositions` reject
a trailing partial XYZ triple. Recentering preserves invalid rows and row order; only originally
invalid points remain invalid when finite local coordinates fit float32.

Without `targetGrid`, `GPUPointCloudICP` refuses `sourceCount * targetCount` above
`GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK` (2^27, about 11,585 by 11,585) and asks for `targetGrid`.
`GPUPointCloudChange` does the same for `coreCount * (referenceCount + comparisonCount)` and asks for
`grid`; `GPUPointCloudNeighborhoods` does the same for `queryCount * positionCount`. With a grid there
is no cap on the pair count, because the work follows the points near each query. With
`samplingCovariance` each cloud is also limited to 32768 points because the in-cylinder correlation
sum is quadratic. Subsample or tile larger inputs.

How the limit was chosen: the largest unindexed run in the tests and the website is 5,000 by 5,000
points (25 million pairs, the feature-extraction page), and the largest kernel-audit case is 4,096
squared (16.8 million). 2^27 is about five to eight times those, so no existing run changes. The older
ceiling `GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK` (2^32) is still exported.

## Surface change and confidence

`GPUPointCloudChange` requires explicit alignment. With `alignment: 'already-aligned'`, the caller
certifies that reference, comparison, core points and normals share a coordinate frame. Alternatively
`alignment: {transform}` applies a resident column-major comparison-to-reference rigid transform
in a GPU graph pass. An ICP operation with comparison as source and reference as target can supply
this view directly; the caller must inspect/accept its convergence status. No alignment is guessed. Each core
uses its normalized oriented normal and a cylinder with inclusive `radius` and `halfDepth`.
All finite points in the cylinder contribute, without truncation.

`counts: uint32[coreCount * 2]` contains paired reference/comparison sample support. `changes` and
`roughness` are `float32x3[coreCount]`. Changes contain signed comparison-minus-reference mean
normal distance and its lower/upper 95% confidence bounds. Roughness contains the two sample
standard deviations and the detection limit:

`1.96 * (sqrt(referenceVariance/referenceCount + comparisonVariance/comparisonCount) + registrationError)`.

Registration uncertainty is nonnegative, defaults to zero, and has coordinate units. Reversing
normal orientation reverses the signed change. `minimumSupport` defaults to 3 and must be at least
2. Status is 0 for valid results, 1 for an invalid core/normal, 2 for insufficient support, and
3 for nonfinite accumulated uncertainty or confidence endpoints. Invalid result values stay zero
and must be masked; support counts remain available. This M3C2-style estimator uses caller-selected
projection scales and independent-sample Gaussian intervals by default. Compose `GPUPointCloudNormalScale` below for automatic normal-scale selection. Optional known spatial covariance is described below. Work is O(coreCount × totalCloudPoints) without `grid` and O(coreCount × points in the cells the cylinder touches) with it.



## Numerical evidence and current limits

The numerical browser suite checks translated tilted planes against NumPy 2.2.6 `eigh` fixtures,
rigid registration against independent NumPy SVD Kabsch, analytic signed surface change/roughness,
3D-only neighbor separation, truncation and ties, collinear degeneracy, distinct samples/centroids,
statistical outliers, signed HAG, missing coverage, morphological edge support and no-match failure.
The fixture generator is checked in beside the tests. All kernels use baseline float32 arithmetic:
choose a local coordinate origin and scale that avoids distance/covariance overflow. No CPU fallback,
implicit concatenation, or hidden data-dependent readback is used. The opt-in benchmark records full indexed-pipeline timing phases, storage limits and scratch memory.
PDAL differential fixture coverage is recorded explicitly by the SMRF generator; these contracts do not establish PDAL
algorithm equivalence or global streaming/partition equivalence.

### Measured indexed-neighborhood and feature scope

The October 23, 2026 native Apple WebGPU run completed the entire sorted 3D index rebuild,
nearest-eight radius query and feature graph at all four tested sizes. Radius was 1.5 on a unit
Cartesian lattice, with cell size 1. Each warm number below is the median of three full
encode/submit/queue-fence measurements; it includes host overhead and is not an isolated GPU timestamp.

| Points | Allocate/upload | Graph compile | Cold encode/execute | Warm resident median | Physical scratch | Input + output + scratch |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 1.7 ms | 1.6 ms | 153.7 ms | 5.1 ms | 20,068 bytes | 140,068 bytes |
| 10,000 | 1.4 ms | 0.9 ms | 151.8 ms | 3.2 ms | 202,916 bytes | 1,402,916 bytes |
| 100,000 | 7.0 ms | 1.1 ms | 151.9 ms | 3.2 ms | 2,018,448 bytes | 14,018,448 bytes |
| 1,000,000 | 65.3 ms | 1.1 ms | 167.9 ms | 19.1 ms | 20,031,424 bytes | 140,031,424 bytes |

Cold execution includes lazy pipeline creation. Two four-byte result reads cost 0.7–4.1 ms after
the resident samples. Graphs used 28 nodes for the two smaller sizes and 44 for the two larger sizes;
each measurement made four submissions. The million-point case used 108 MB of output and 12 MB of
input; its largest output binding was 32 MB. The native device advertised approximately 4 GiB
storage-binding/buffer limits, though these individual outputs fit the baseline 128 MiB binding limit.

**Which path these numbers measured.** The neighborhood run used the indexed path: the benchmark
passes `grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [side, side, side]}`
(`gpu-pointcloud-benchmark.spec.ts`). It did not measure the default path without a grid, which tests
every point for every query and is now refused above 2^27 pairs. The advanced table below used
`targetGrid` for ICP, and for SMRF a grid capped at 256 by 256 cells with `window: 4` (the default
window is 18 cells), so its cost follows the raster and the window, not the point count. No
one-million-point measurement of the unindexed path, of `GPUPointCloudChange` or of `grid`-indexed
change exists.

These numbers establish this lattice workload and device only. Dense coincident points, highly
uneven cells, larger capacities, wide radii, and other GPUs can behave differently. The scaled
canary checks known first-point support and numerical status; the separate numerical suites compare
full small-case outputs to independent NumPy/SciPy and all-pairs references. Raw timing samples,
device metadata, phase definitions and invocation are recorded in
`dev-docs/research/evidence/pointcloud-coverage-2026-10-23.json` in the repository.


## Owned rows, halos and explicit partition merges

`GPUPointCloudNeighborhoods.queryRows` selects owned query rows while `positions` contains both owned and halo points. Output row order follows `queryRows`; neighbor IDs always index the supplied local position view, so callers retain the local-to-global mapping. Invalid query IDs set status bit 1. `supportBounds` certifies the extent whose point population is complete; status bit4 marks a radius sphere that crosses that extent. Truncation remains bit 2 and full support counts remain available. Status bits combine. Supply halos at least as wide as the query radius, keep grid and support extents consistent, and compare only owned outputs when stitching tiles. The operator cannot detect an omitted point inside a falsely certified extent.

`GPUPointCloudFeatures` accepts a neighborhood row count independent of the number of candidate positions, so the owned-row neighborhoods compose directly. Covariance is centered on each neighborhood; viewpoint orientation uses its centroid. Read neighborhood halo-status separately from feature degeneracy/truncation status.

The `attribute` of `GPUPointCloudVoxelMerge` accepts matching float32 scalar or x2/x3/x4 means for the parts and the merged output. Its finite-support count is separate from the geometric population, so the caller that builds a part's means decides which vectors contribute (count only vectors whose components are all finite). The merge itself is linear in voxels times parts.

`GPUPointCloudVoxelMerge` combines counts, centroids, minimum global representative IDs and optional vector means from aligned per-grid reductions. Every part declares `sourceRowOffset`; the caller certifies `partitionPolicy: 'disjoint-owned-rows'`. Exclude halo replicas before voxel reduction. Geometric and attribute means use their own population weights. Status bit 1 reports uint32 population overflow and bit 2 reports representative-ID overflow; reject flagged cells. Floating-point merge order can affect the last bits. Sampled positions can be gathered from the returned global representative IDs through the caller's global source mapping.

## Interpolated height sampling

`GPUPointCloudHeight.interpolation` defaults to the containing cell. `{method: 'bilinear'}` interpolates cell-centre samples and requires every positive-weight corner to be finite and covered; it does not extrapolate across the grid edge. `{method: 'idw', radius, power, minimumSupport}` samples supported cell centres inside an inclusive physical radius, with power default 2, minimumSupport default 1 and radius at most 64 cells. An exact supported centre is reproduced even when its other neighbors are missing. IDW fills a query's missing terrain support, not the input raster. Both modes preserve negative heights and return status 2 if their declared support is unavailable. Status 3 reports a nonfinite interpolated elevation or height difference and leaves the output zero, including overflow from finite input values. Their parameters are the interpolation provenance; all heights retain the input vertical datum. Datum conversion, direct Delaunay ground interpolation and automatic extrapolation are separate operations.

## Point-to-plane registration

Set `metric: 'point-to-plane'` and supply target-row `targetNormals`. Nearest correspondence still uses Euclidean distance; the update minimizes weighted normal residuals through a six-parameter Gauss–Newton system. Normals are normalized, zero/nonfinite normals do not contribute, and the final mask includes normal validity. Six independent constraints and a full-rank system are required. The moments are taken about the centroid of the matched source points (256-row blocks, each about its own centre, moved to the common centroid before the solve), and the normal matrix is scaled to a unit diagonal before elimination, so the rank test does not depend on the coordinate unit or on how far the cloud lies from the origin: a room corner 500 m from the origin, a scene 10 km across and a scene in millimetres all solve. A pivot at or below 1e-5 of that scaled matrix is rank loss. Rank-deficient configurations, including an isolated flat plane, return status 3 rather than an arbitrary translation/rotation. The rotational increment uses Rodrigues' formula about the centroid and is composed back into the transform about the origin; the history's translation step is the change of that origin-frame translation, as in point-to-point mode. Float32 coordinates still limit the result to their own spacing, so use `origin` for large projected coordinates. History RMSE is the normal residual for this mode, versus Euclidean residual for point-to-point. Both remain local algorithms whose initialization matters.

## Declared uncertainty models

`registrationCovariance: [xx,xy,xz,yy,yz,zz]` supplies positive-semidefinite translation covariance in the aligned frame. Each core projects it along its oriented normal. Its projected variance is added to `registrationError²`. The default `uncertaintyCombination: 'linear'` retains the M3C2-style addition of sampling and registration standard errors. `'quadrature'` combines their independent variances. `confidenceMultiplier` defaults to 1.96 and explicitly sets the two-sided Gaussian quantile.

Optional `samplingCovariance: {referenceVariance, comparisonVariance, correlationLength}` uses a supplied Gaussian measurement-error model, `Cov(i,j) = variance × exp(-distance(i,j)² / (2 × correlationLength²))`. The mean variance is the exact within-cylinder double covariance sum divided by support squared. These supplied normal-direction variances are distinct from observed roughness; the GPU contributor does not estimate a covariance model from the sample. The explicit CPU helper `fitPointCloudSamplingCovariance` below fits detrended residuals. The two clouds are assumed independent. This mode costs O(cores × cloudPoints²), targets bounded supports, and is tested against independent SciPy distance-matrix integration. It is not a correction for an unknown correlation process.

## Actual external references and advanced scaling

The pinned generators now execute PDAL 2.8.4 directly. Four SMRF fixtures (holes, low outliers, slope sensitivity, scalar zero and net cutting) agree on every classification. Planar normals agree with PDAL `filters.normal`; that fixture uses PDAL's self-inclusive support and compares orientation only, not covariance eigenvalues across differing supports. Point-to-point ICP agrees with PDAL's composed source-to-target matrix, including its centroid adjustment. Outlier classifications agree on a separated-cluster fixture; PDAL uses sample standard deviation while this API explicitly uses population standard deviation, so exact threshold parity is not claimed. Raster HAG agrees with `hag_nn(count=1)` where queries share ground samples' cell-centre XY, including negative heights; this is not general nearest-ground interpolation parity. `pdal-pointcloud.json` and `smrf.json` retain the version and actual reference outputs. Independent NumPy/SciPy fixtures still check covariance, interpolation and uncertainty conventions.

The advanced native Apple WebGPU measurements rebuild their indexes every encoding. ICP ran with `targetGrid` and SMRF with a 256 by 256 grid and `window: 4` (see the note above); neither measures the default window or an unindexed search. Warm values are medians of three full encode/submit/queue-fence wall measurements, including host overhead. No individual binding exceeds 128 MiB.

| Workload | 10,000 points | 100,000 points | 1,000,000 points |
| --- | --- | --- | --- |
| SMRF, window4, small raised objects, maximum256×256 cells | 3.1ms | 4.6ms | 12.8ms |
| Indexed point-to-point ICP, four iterations, Huber weights | 4.7ms | 6.4ms | 27.6ms |
| Coincident cloud,64 owned exact queries, nearest4 plus full counts | 5.0ms | 43.9ms | 439.4ms |

The skew test deliberately exposes occupancy-dependent cost: it does not establish that querying every point in a million-point coincident cloud is practical. SMRF evidence covers the stated grid/window/object pattern, not every sparse-hole morphology. ICP evidence checks known rigid translation and status; the small numerical cases separately cover rotation, outliers, point-to-plane rank loss and invalid support. Raw records, native device limits, cold timings, readback timing, scratch memory and exact workload definitions are in `dev-docs/research/evidence/pointcloud-advanced-2026-10-23.json`.

The executed PDAL 2.8.4 and isolated 2.9.0/2.9.2 cores did not include `filters.m3c2`. The attempted 2.10.2 core required a newer GDAL ABI unavailable in the isolated reference environment. Temporary reference installations were removed after fixture generation to reclaim disk space; the pinned generators and outputs remain in the repository. M3C2 known correlated-noise fixtures use independent NumPy/SciPy references. A separate executed
GPU comparison now covers the actual upstream PDAL M3C2 golden fixture, without claiming a local
PDAL binary execution: tree `0e64af327cba0a4c54e501db0d61b62390b69a79`,
`M3C2FilterTest.cpp` / `verifyPoints`, two Autzen BMX LAS epochs (829 and 687 points), normal radius
1.390432, cylinder radius 2.890432 and half-depth 5.5. All 91 golden rows having at least two
samples in both epochs pass signed-distance error below 0.01 coordinate units and exact support
counts. The remaining 12 rows have no reported reference solution and are explicitly listed;
they do not establish GPU behavior outside its minimum sample-variance support contract.

The caller-supplied normals are independently prepared with NumPy 2.2.6 eigendecomposition of
reference neighbors at the upstream radius and oriented upward. Coordinates are recentered before
f32 upload. Core row zero runs separately with the first source point omitted, reproducing PDAL's
explicit first-point exclusion. PDAL reports population standard deviations; the GPU reports
sample standard deviations. Converting the GPU values back to the population convention agrees
with the rounded golden roughness and detection limits within 0.002 coordinate units, and all
significance classifications agree. This is explicit convention-aware comparison, not identical
uncertainty APIs or automatic normal-scale selection. The repository fixture record at `modules/analysis/test/oracles/upstream-golden/README.md`
retains unmodified source data, expected outputs, filter/test sources, copyright/license and SHA-256
hashes. Two GPU batches and three provenance-integrity node tests passed.


## `GPUPointCloudNormalScale`

Minimum population plane roughness over supplied radii, excluding the query itself. Each scale uses
four compute stages plus parameter uploads, with at most 32 scales. Truncated scales are rejected;
grid coverage must include intended support.

`GPUPointCloudNormalScale` selects the minimum-roughness plane over 1–32 supplied spherical
`radii`. Radii are strictly increasing **radii**, in local Cartesian coordinate units;
CloudCompare's UI normal scale uses diameters. The selection rule is the smallest population
covariance eigenvalue (squared orthogonal roughness), with exact ties resolved by the smaller radius.
This implements the minimum-roughness option, not Brodu–Lague dimensionality classification.
See the [CloudCompare M3C2 multiscale description](https://www.cloudcompare.org/doc/wiki/index.php/PluginM3C2).

Supply `positions: float32x3`, optional owned `queryRows: uint32`, `capacity` (3–64), and
`parameters: float32[5]` packed by `getGPUPointCloudNormalScaleParameterValues`. The settings
`minimumSupport` (3–64, default 3) and `viewpoint` are per-frame; support above capacity gives
no usable scale. Each query excludes its own source row; other coincident
rows remain separate samples. `viewpoint` orients from each core toward a sensor, otherwise normals face positive
z (then y, then x). `spatialContext` defaults to native planar Cartesian coordinates; geographic
coordinates are refused. Recenter large absolute coordinates before upload.

Caller-owned outputs have one row per query: `normals: float32x3`, `normalScales: float32`
(selected radius), `normalVariances: float32` (squared population roughness), `status: uint32`.
Status bits are 1 no usable scale, 2 `candidateOverflow` at any scale, and 4 eigensolve exhaustion
at any scale. Only complete, converged, nondegenerate supports participate. If bit 1 is clear, a
normal is published even when another scale overflowed; inspect the other bits to see rejected
candidates. When bit 1 is set every other output is zero. Invalid IDs, empty supports and
collinear supports cannot produce a normal. Exact flat-plane ties select the smallest eligible radius.

The largest-radius nearest list is reused. If it overflows, a smaller radius is still complete
when its boundary precedes the last retained neighbor; a boundary tied with that neighbor is
conservatively rejected. Missing slots never contribute. The optional `grid` is built once per
encoding and must cover the full intended support; outside-grid points are excluded, and missing
halos cannot be inferred. Without a grid, query-count times position-count is limited to 2^27.
Scratch is checked against the storage binding limit before nodes are constructed. The exposed
`estimatedCommandNodeCount` includes a conservative 512-node allowance for the published index.

Radii, capacity, grid and view shapes are compile-time. Position, query-row and parameter contents
can change per encoding. Write packed settings into a caller-owned `GPUParameterBuffer` to change
support eligibility and orientation without recompiling. Each scale emits three kernels with no
readback. Pass `normals` directly to `GPUPointCloudChange`; its `cores` must be the same owned
positions, in the same row order. Inspect selector status before accepting change results.
`gpu-pointcloud-normal-scale.spec.ts` exercises this full GPU path against NumPy eigendecomposition
and cylinder means, sample variances and confidence intervals on planes with known offsets.

{/* parameters: GPUPointCloudNormalScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUPointCloudNormalScale` parameters

- Packer: `getGPUPointCloudGridParameterValues(settings)`, writes `GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `gridParameters`.
- Packer: `getGPUPointCloudNormalScaleParameterValues(settings?)`, writes `GPU_POINT_CLOUD_NORMAL_SCALE_PARAMETER_LENGTH` (5) `float32` values.

Settings of `getGPUPointCloudGridParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` | `readonly [number, number] \| readonly [number, number, number]` | - |
| `cellSize` | `number` | - |

Settings of `getGPUPointCloudNormalScaleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumSupport` (optional) | `number` | Minimum other points for a candidate plane, 3–64, default 3. Above capacity no scale qualifies. |
| `viewpoint` (optional) | `readonly [number, number, number]` | Orient toward this viewpoint; omitted uses positive z, then y, then x. |

{/* /parameters */}

## Fitting sampling covariance

`fitPointCloudSamplingCovariance(positions, residuals, correlationLengths)` is an explicit CPU
fit of the Gaussian model already accepted by `GPUPointCloudChange.samplingCovariance`.
Inputs are packed XYZ positions and detrended normal-direction residuals. For every unordered
pair, its observed semivariance is half the squared residual difference. At each supplied length,
the helper fits a nonnegative variance to `variance * (1 - exp(-distance² / (2 * length²)))`
by ordinary least squares. It returns `variance`, `correlationLength` and `meanSquaredError`;
exact ties choose the smaller length. With two distinct points, every candidate fits the single
semivariance exactly: the error is analytically zero and the smallest length wins. Fits with more
pairs retain direct residual errors without an error tolerance. This is a finite candidate search,
not a continuous optimizer.

The float64 reference fit is bounded to 4,096 points and 32 candidate lengths because its work
is quadratic in point count. Fit epochs separately with a common fixed length to populate
`referenceVariance`, `comparisonVariance`, and `correlationLength`. The helper does not infer
surface detrending, fit a nugget, or combine registration uncertainty. Constant residuals give
zero variance; coincident-only positions cannot identify a length and are refused. The NumPy
fixture uses an independent vectorized least-squares solve.

### Executable-backed M3C2 check

`gpu-pointcloud-m3c2-binary.node.spec.ts` runs a supplied PDAL executable through CSV readers,
`filters.m3c2` and a CSV writer, comparing distance, population roughness, support, detection limit
and significance against independent NumPy synthetic planes. Run with:

```sh
PDAL_M3C2_BINARY=/path/to/pdal yarn vitest run --project node --maxWorkers=2 modules/analysis/test/gpu-pointcloud/gpu-pointcloud-m3c2-binary.node.spec.ts
```

The test is explicitly skipped when no executable is configured; an explicitly configured binary
that lacks the stage fails. This staging environment has no PDAL/CloudCompare executable, so the
new binary test has **not been executed** here. The existing upstream golden evidence above remains
distinct from local binary execution. GPU confidence uses sample variance; PDAL reports population
variance, and the executable test checks PDAL's convention directly.

## Kepler catalog bindings

The Kepler contributor catalog binds all eleven classes on this page through
`modules/kepler/src/analysis/catalog-pointcloud-bindings.ts`. Operation IDs use the `pointcloud-`
prefix: `ground`, `smrf`, `surfaces`, `height`, `neighborhoods`, `features`, `statistical-filter`,
`normal-scale`, `icp`, `change` and `voxel-merge`. Import `GPUPointCloudHeight` from the public
`@luma-contributors/analysis/pointcloud-analysis` entry point; it computes height above explicit
class-2 terrain support. The ground
binding first runs GPUPointCloudSurfaces with every point eligible to prepare the unclassified
cell minima and coverage consumed by GPUPointCloudGround; original LAS classes are kept separate
for the final classification pass. The catalog preserves finite covariance and converged
eigenvalues for adequately supported collinear neighborhoods while publishing null normals and
shape descriptors. Voxel-merge centering ignores zero-count entries, retaining their slots with
local zero coordinates so arbitrary empty-cell placeholders cannot reduce occupied-centroid precision.

These GPU-only host adapters consume bounded XYZ Point inputs, preserve source row identity and
altitude, and publish Arrow columns through native Sources for point color/size/altitude styling.
Geographic inputs use the catalog local metric frame and a shared binary64 origin; planar inputs
keep their Cartesian unit. The host form exposes a bounded subset of each contributor's options,
not a new numerical implementation. Voxel merge requires sufficient statistics, identical voxel
IDs/grid and explicit disjoint ownership; it is not raw-point voxel downsampling. See the
[Kepler example contract and output table](../../examples/deck/kepler-compute/README.md#point-cloud-contributor-catalog)
for input aliases, capacities, output columns, publication steps and remaining form limitations.
