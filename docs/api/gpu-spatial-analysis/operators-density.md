---
title: "GPU Spatial Analysis: Density & interpolation"
sidebar_label: "Density & interpolation"
description: "Density & interpolation contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Density & interpolation


### `GPUPointDensity`

Chunked positions and weights use fixed tile scratch for packed hexagon/masked routes, controlled by
scratchTileSize. Shared grid outputs remain packed; fixed-point scale and counts span all chunks.

Bins points into grid or hexagon cells and writes a per-cell count, sum or mean raster.

::lens{scenario=gpu-spatial-analysis.point-density rung=result caption="2,344 observations (grey) binned into 125 m cells (blue); counts per cell, not per km²."}

Bins points into a square grid (`GPUGridBinning`, `GPUGridAggregation`) or pointy-top hexagons
(keys kernel plus `GPUGroupAggregation`) and writes per-cell counts, sums, means, a heatmap field,
its `[min, max]` extent, a histogram, and an optional `r32float` texture. Bounds and hexagon radius
may be GPU views; the field can be smoothed with a per-frame `GPUConvolution` kernel. Instead of a full
`kernel`, pass a separable `separableKernel: {horizontal, vertical}` (one of the two is required), which smooths
in two 1D passes.

Grid bounds accept every validated finite float32 literal, including scientific notation such as
`[-3e38, 0, 3e38, 1]`. For scientific-notation integer bounds, one initialization pass writes a
`float32x4` bounds view consumed by the published grid builders, avoiding their literal formatter.
The inclusive maximum belongs to the final column/row in both sum accumulation modes.
Known limit (rectangular grid, every frame): when the float32 width or height of the bounds
reaches `2^126` or overflows (for example x from `-3e38` to `3e38`), the counts and atomic sums come from the published
luma.gl `GPUGridBinning` and `GPUGridAggregation`, which on Apple Metal with luma.gl
`10.0.0-alpha.16` put `x = 0` in column 0 of 360 instead of column 180, while workgroup sums use
this package's own grid keys. Counts, sums and means can then disagree. Keep each positive float32 bounds extent below `2^126` (about `8.51e37`) to keep its reciprocal
from becoming subnormal. Rescale positions and bounds together if either extent reaches that limit.
Bounds are captured once at construction. Literal tuples are copied, so later caller mutation
does not change validation, node estimation or the emitted domain. A GPU bounds view retains
its identity and its contents remain per-frame.

**This is binned counting, not a kernel density estimate.** `output.values` is the count (or weight
sum or mean) per cell, so it changes with `gridSize` and is not an intensity per unit area or a
probability density. Optional `smoothing` convolves that field with a kernel whose radius and sigma
are in **cells**, not ground units, using a zero boundary: kernel mass leaks out of the domain, so the
field is underestimated within about two sigma of the border, and a grid whose cells are not square on
the ground smooths anisotropically. For intensity per unit area, a bandwidth in ground units, a
choice of kernels, and edge correction use [`GPUKernelDensity`](#gpukerneldensity). `spatialContext`
is planar/native only (omitted means planar/native); longitude/latitude throws, so project first.
It measures area, so `local-azimuthal-equidistant` frames throw.

Weighted sums default to `sumAccumulation: 'auto'`, which always selects the deterministic
`'workgroup'` path, because reproducible sums are the contract. `'atomic'` (direct float
compare-exchange atomics, no key pass) is an explicit opt-in for known diffuse input and is not
deterministic: its last float bits depend on the atomic interleaving. Each 256-row
workgroup pre-aggregates in shared memory before writing to the grid, so a hotspot no longer
serializes on one cell (90% of 1M points in one cell drops from seconds to about 4 ms, at about 2 ms
extra on uniform data). `'workgroup'` sums are fixed point: each weight is quantized to one
power-of-two unit derived from the largest weight and the row count, then added as an exact 96-bit
integer, so the sums are bitwise identical across runs, row orders and chunkings and are the f32
nearest to the exact sum of the quantized weights. Weights smaller than about `2^(h - 71)` times
the largest (`h` is the bit length of the row count, so about `1e-15` at a million rows) round to
the unit. `'atomic'` sums can differ from a CPU sum, and between runs, in the last bits.

```ts
import {GPUPointDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUPointDensity>[0];
const {positions, weights} = inputs;
const {values, extent, histogram, texture} = inputs.output;
declare const viewport: GPUParameterBuffer<'float32'>;
declare const kernel: GPUParameterBuffer<'float32'>;

graph.add(new GPUPointDensity({
  positions, weights, bounds: viewport.importToGraph(graph), gridSize: [256, 256],
  statistic: 'mean',
  smoothing: {kernel: kernel.importToGraph(graph), kernelWidth: 5, kernelHeight: 5},
  output: {values, extent, histogram, texture}
}));
```

`mask` is an optional per-row `uint32` view with the same length as `positions`. Rows whose mask
is `0` are excluded from count, sum, mean, smoothing input, extent and histogram; any nonzero value
includes the row, like the mask inputs of `GPURegionStatistics` and `GPUFlowAggregation`. Contents
are per-frame and rewriting them never recompiles. Hexagon binning reads the mask in its keys
kernel; grid binning adds one pass that writes masked positions as NaN into a transient copy.
Without `mask` the graph is unchanged. Pass a residency arena `liveMask`, or
`GPUTimeWindowFilter.outputMask`, to get time- or tile-gated density without poisoning position
columns.

`getGPUPointDensityHexagonGridSize` sizes a hexagon lattice that covers given bounds, and
`GPU_POINT_DENSITY_HEXAGON_WGSL` lets render shaders place hexagon bins.

#### How it computes

{/* lens:GPUPointDensity */}

::lens{scenario=gpu-spatial-analysis.point-density rung=datum datum=3882 caption="Inspect cell 3882, containing the first iNaturalist observation."}

::lens{scenario=gpu-spatial-analysis.point-density rung=formula caption="Count accepted observations in each grid cell."}

::lens{scenario=gpu-spatial-analysis.point-density rung=kernel caption="Read the count-to-float finalization kernel."}

::lens{scenario=gpu-spatial-analysis.point-density rung=graph caption="Follow grid binning, aggregation and count publication."}


### `GPUStudyWindowPreparation`

Prepares a reusable polygon study window: oriented edges, bounds, area, perimeter and sampling cells for window-aware statistics, replayed each encoding.

::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=result caption="13 community areas prepared as one study window (blue outline) in an equal-area frame."}

Prepares one valid polygon or multipolygon, including holes and disconnected islands, as a reusable
`studyWindow` handle for point-pattern statistics and kernel-density edge correction. Add the
preparation to the graph before adding its consumers. One handle belongs to one graph; declaring
it a second time throws, including in the same graph. `isDeclaredIn(graph)` reports its declaration.

The first submitted encoding captures a persistent geometry snapshot. `invalidate()` or a true
CPU `rebuildWhen(parameters)` requests a new generation. Its applied generation is committed only
after the complete preparation executes, so discarded encodings and abandoned prefixes retry
automatically. Published preparation stages replay against that snapshot every encoding; geometry
refresh alone is GPU-gated. This repeats preparation work and adds one copy of each geometry view
plus 1,072 fixed generation bytes per handle (1,024-byte immutable byte lookup,
32-byte request and 16-byte state), excluding geometry copies and dispatch-gate storage.
Each recorded encoding captures its own generation request without a queue write or a new allocation.
Ring summaries initially process at most two rings per invocation, followed by compensated pairwise
reductions. Scratch holds 11 float32 words per rounded-up ring pair:
`11 * ceil(ringCount / 2) * 4` bytes. Each geometry copy and scratch plane must fit
`min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)`; exceeding the
binding limit is rejected before allocation. These costs retain the same geometry view lengths.
Changed geometry needs invalidation, otherwise consumers deliberately reuse the previous geometry.
Consumers evaluate their
current inputs on every encoding. Changing view lengths requires a new handle and graph.
`destroy()` frees the handle's buffers and allows re-adding it; destroy compiled graphs using it
first. `getViews(graph)` throws until this handle has been declared in that graph.

If preparation node construction/allocation fails, owned allocations are cleaned up and
`isDeclaredIn(graph)` stays false. A later failure while `graph.add(studyWindow)` registers the
returned nodes cannot be observed by that callback: call
`studyWindow.recoverFailedDeclaration()`, discard the partially registered graph, and retry on a
fresh graph. Do not recover a successfully submitted graph. Failed graphs may retain imported
IDs even after construction cleanup. Reuse persistent caller buffers through the fresh graph's
public canonical `importGPUData` imports with `GPUData` wrappers covering the same original
format/range, and pass equivalent fresh views to consumers. Recreate transient producers, views
and preparation handles in the fresh graph. Caller-owned storage is retained; only owned
allocations are freed. This failure/retry path is distinct from discarded encodings, whose
unapplied generations retry automatically.

The first encoding and first encoding after explicit invalidation bypass `rebuildWhen`; other
encodings call it once. `encodedBuildCount` counts CPU refresh-request encodings, including
discarded requests; automatic GPU retries and replayed builds do not increment it.

The props and handle accept a frame type parameter, for example
`GPUStudyWindowPreparation<{geometryChanged: boolean}>`; use that type on `GPUCommandGraph` too.
The default is `unknown`; `invalidate()` alone also works with a void graph. See the
[execution guide](./concepts.md) for a typed `rebuildWhen` example. No additional invalidation is
needed after discarding an encoding. Use one active resumable execution per handle/graph.

```ts
import {GPUStudyWindowPreparation} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUStudyWindowPreparation>[0];
const {positions: windowPositions, featureOffsets: windowFeatureOffsets, polygonOffsets: windowPolygonOffsets, ringOffsets: windowRingOffsets, spatialContext} = inputs;
declare const windowSummary: import('@luma.gl/graph/gpu-core').GraphDataView<'float32'>;

const studyWindow = new GPUStudyWindowPreparation({
  positions: windowPositions,
  featureOffsets: windowFeatureOffsets, // exactly two entries: one feature
  polygonOffsets: windowPolygonOffsets,
  ringOffsets: windowRingOffsets,
  spatialContext,
  samplingGrid: [256, 256],
  output: {summary: windowSummary} // optional caller-owned float32[8] readback copy
});
graph.add(studyWindow);
// Pass studyWindow to consumers added after this preparation.
```

The geometry uses the GeoArrow polygon layout: the first ring of each polygon is its shell,
regardless of winding. Only planar frames are supported. Project longitude/latitude positions first
with `GPULocalFrame.projectArray`, using the same frame as the observations. The polygon must be
valid; consider `GPUGeometryValidity`. Self-intersections or overlapping polygons invalidate the
fan identity used by consumers.

`GPURectangleClip` polygon output can feed the preparation directly, including empty polygon and
ring slots. A ring with fewer than three vertices, or exactly zero orientation for every triangle
`(first, vertex, next)` in its fan, is degenerate. Such rings contribute no area, perimeter, edge
count, bounding box or uncertainty flag; their oriented edges are NaN. The constructor still needs
at least three vertex rows and one ring slot. An all-degenerate window is legal and produces EMPTY
at encode time. The raster sampler reads raw positions, so degenerate rings can only add boundary
flags (extra exact inclusion tests), never change coverage.

The GPU copies oriented edges with the interior on their left and computes the eight-float
`GPU_STUDY_WINDOW_SUMMARY` row: `[minX, minY, maxX, maxY, area, perimeter, edgeCount, flags]`.
Area subtracts holes; perimeter includes them. Flag 1
(`GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN`) indicates a ring area within a conservative f32
rounding bound of zero, or non-finite orientation; flag 2 (`GPU_STUDY_WINDOW_FLAG_EMPTY`) indicates
non-finite or non-positive total area. Ring areas use a fixed pairwise reduction of determinants
relative to the first vertex. The orientation bound is
`8 * epsilon_f32 * (1 + ceil(log2(vertexCount))) * sum(abs(products))` on doubled area.

`samplingGrid` is compile-time, defaults to `[256, 256]`, and must contain positive integers with a
product no larger than 1048576. The sampler compacts inside or boundary cells in row-major order;
bit 31 marks a boundary cell that needs an inclusion test. `crossingCapacity` is forwarded to
`GPUPolygonRasterization` and defaults to `4 * samplingGrid[1] * ringCount`; size it for polygons
with many crossings per row. On crossing overflow, a slower exact GPU center test preserves
sampler coverage. `slabEntryCapacity` controls the optional edge index, whose overflow
falls back to walking edges. `getViews(graph)` exposes the edge, summary, slab and sampler views;
its packed view is internal to consumers.

All preparation runs on the GPU. Each rebuild's fixed ring tree costs O(V log V); sampler cost
also depends on the grid and crossing capacity.

#### How it computes

{/* lens:GPUStudyWindowPreparation */}

::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=datum datum=4 caption="Inspect summary field 4: the prepared window area in square metres."}

::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=kernel caption="Read edge orientation and summary kernels."}

::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=graph caption="Follow edge preparation, slab indexing and sampling-cell compaction."}


### `GPULineDensity`

Sums line length per grid cell, optionally dividing by cell area. Grid placement and cell sizes
come from `getGPULineDensityParameterValues`; lengths use the spatial context's length unit.
In longitude/latitude, segments follow straight lines in degree coordinates and each cell piece
is measured by its great-circle endpoint distance. A segment from `(179, 0)` to `(-179, 0)`
therefore follows the literal 358-degree route. Optional one-row uint32
`output.status.antimeridianUncertainCount` counts segments with an absolute longitude jump
larger than 180 degrees, including segments outside the grid, and resets on every encoding.
The lengths retain that literal route; split or unwrap seam crossings before use when a short
seam-crossing path is intended. Densify long segments along their intended path because total
piece length can depend on the grid. `output.status.overflow` still reports dropped cell pieces.

### `GPUKernelDensity`

Explicit method binned accepts aligned chunked positions and weights. Corrected-weight scalar
scratch splits when a global binding cannot fit; split totals use fixed-point accumulation. Exact
and auto retain packed indexed inputs, and shared grid outputs must fit.

Estimates event intensity on a raster from points with a planar kernel, a ground-unit bandwidth and edge correction.

::lens{scenario=gpu-spatial-analysis.kernel-density rung=result caption="2,344 iNaturalist observations (grey) and their 400 m gaussian density on 125 m cells."}

Maximum: `indexGridSize` has at most `getGPUGridIndexMaximumCellCount(device)` (`2^25 - 1` on a default device: `cells + 1` uint32 cell offsets in one 128 MiB binding) cells.

A true planar kernel density estimate on a raster, with the bandwidth in ground units. For each output
cell center `u` it evaluates `lambda(u) = sum_i w_i K_h(|u - x_i|)` over the points inside the kernel
support, where `K_h(d) = K(d / h) / h^2` is radially symmetric and integrates to one over the plane
(Silverman 1986, Density Estimation, Table 3.1):

| `kernel` | `K_h(d)` for `t = d^2 / h^2` | Support |
| --- | --- | --- |
| `'gaussian'` (default) | `exp(-t / 2) / (2 pi h^2)` | truncated at `5h`; lost mass `exp(-12.5)`, 3.7e-6 |
| `'quartic'` (biweight), alias `'bisquare'` | `3 / pi (1 - t)^2 / h^2` | `t < 1` |
| `'epanechnikov'` | `2 / pi (1 - t) / h^2` | `t < 1` |

`'bisquare'` is the name `GPUNeighborSearch` and `GPUSpatialWeightsTransform` use for the same shape (their weights are
normalised in one dimension, `15/16 (1 - z^2)^2`, where the density kernel is normalised over the plane). KDE accepts
either name for `kernel`; both run the same kernel and `kernel` reads back as `'quartic'`.

```ts
import {GPUKernelDensity, getGPUKernelDensityParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUKernelDensity>[0];
const {positions, weights, grid} = inputs;
const {values, weightSum} = inputs.output;
declare const minX: number, minY: number, maxX: number, maxY: number;
const {invalidCount} = inputs.output.status ?? {};

const kdeParameters = new GPUParameterBuffer(device, {id: 'kde', format: 'float32', length: 9});
graph.add(
  new GPUKernelDensity({
    positions, // float32x2, planar frame with equal units on both axes
    weights, // optional float32 (populations, counts); NaN or negative excludes the point
    parameters: kdeParameters.importToGraph(graph),
    grid, // axis-aligned GPURasterMetadata; its width and height size the raster
    indexGridSize: [64, 64], // internal GPUGridIndex over the points
    indexBounds: [minX, minY, maxX, maxY], // must cover the points that should count
    kernel: 'quartic',
    edgeCorrection: 'uniform', // default
    normalization: 'intensity', // default
    // bandwidthRule: 'scott', // optional automatic rule
    // bandwidthPopulation: 'included', // opt in; omitted defaults to 'all-finite'
    // Optional output.bandwidth: float32[1], output.bandwidthPopulationCount: uint32[1].
    output: {values, weightSum, status: {invalidCount}} // weightSum and status are optional
  })
);
kdeParameters.write(
  getGPUKernelDensityParameterValues({bandwidth: 250 /* meters */}, grid) // no recompile
);
```

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUKernelDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUClassBreaks,
  GPUColorScale,
  type GPUClassBreaksProps,
  type GPUColorScaleProps
} from '@luma-contributors/analysis/dataframe-analysis';

declare const graph: GPUCommandGraph;
declare const kde: GPUKernelDensity; // add this producer before the classification stages
declare const breaksInputs: GPUClassBreaksProps;
declare const colorInputs: GPUColorScaleProps;

const values = kde.outputBand.storage.values;
graph.add(new GPUClassBreaks({...breaksInputs, values}));
graph.add(new GPUColorScale({
  ...colorInputs,
  values, // exactly the same bounded view used by GPUClassBreaks
  domain: breaksInputs.output.breaks,
  classCount: breaksInputs.output.classCount
}));
```

The readable pixel count is `grid.width * grid.height`. `kde.outputBand.storage.values` borrows that exact prefix, preserving its byte offset and buffer identity. Pass the same bounded view to both classification stages; size row-aligned color outputs to that pixel count. Spare allocation tails in `output.values` are unspecified and must not enter classification. For example, two equal-interval classes over `[2,4,6,8]` have edges `[2,5,8]`; reading an extra allocation sentinel of 1000 changes the legend to `[2,501,1000]`. Fixed/adaptive KDE capture ordinary options, tuples and nested output/status routing at construction while preserving GPU resource identities; input buffer contents remain live between submissions.

- **Grid.** `grid` is a `GPURasterMetadata`: its `width` and `height` size the kernels (compile-time)
  and the packer reads the placement from the same object, so pass the frame's grid to
  `getGPUKernelDensityParameterValues(settings, grid)`. It must be axis-aligned (a rotated or sheared grid
  throws). Output rows follow the grid's row order: row 0 is the grid's first row, the top row of a
  north-up grid (`makeGPURasterMetadata` default, `affine[4] < 0`) and `minY` when rows advance north
  (`rowDirection: 'north'`). Cell `(x, y)` is evaluated at its center, from the outer corner of
  cell (0, 0), half a cell in with `pixelInterpretation: 'point'`.
- **Units and normalization.** `normalization: 'intensity'` (default) sums event weights per unit area: points per square meter in a meter frame. `'probability'` divides by total included event weight (`output.weightSum`); this normalization alone does not guarantee a window integral of one. Values sample density at cell centers. Multiplying by cell area is midpoint quadrature, not an exact cell integral: one centered unit event with quartic bandwidth `.1` in a unit-area cell gives midpoint mass `3 / (pi * .1^2) ≈ 95.493`, although the continuous cell integral is 1. Refine cells relative to bandwidth to approximate integrals.
- **Edge correction** repairs the downward bias within one bandwidth of the study window, where kernel
  mass falls outside the data region. `'uniform'` (default) divides each cell by `e(u)`, the kernel
  mass inside the window (Diggle 1985; spatstat `density.ppp(edge = TRUE)`). `'diggle'` instead weights each point by `1 / e(x_i)` (Jones 1993; spatstat `diggle = TRUE`): event-wise correction conserves continuous window mass for the untruncated kernel. Uniform correction does not conserve mass for arbitrary event locations. For a single corner Gaussian event with `h = 1` in `[0,10]^2`, its untruncated integral is about `(ln 2)^2 ≈ .480453`; the infinite-quadrant limit is exactly `(ln 2)^2`. A 400 × 400 midpoint raster with radial truncation at `5h` gives about `.480429` for uniform and `.999996` for Diggle. Gaussian radial truncation and midpoint quadrature introduce separate errors; allowing for both, prefer fixed KDE's event-wise `'diggle'` when conserving window mass matters. `'none'` is the plain sum. The window is
  `window` in `getGPUKernelDensityParameterValues` and defaults to the rectangle of the grid; with correction, points
  outside the window are excluded and cells whose centers are outside are NaN. `e(u)` is closed form,
  with no quadrature: normal-CDF differences for the gaussian, and an exact inclusion-exclusion of
  four disk-clipped quadrant integrals for quartic and epanechnikov. [`getGPUKernelDensityWindowOverlap`](./space-time-density-and-k-function.md#encoded-coordinates-time-slices-and-overlap) gives float64 rectangle overlap. For temporal overlap use [`getGPUSpaceTimeTemporalWindowOverlap`](./space-time-density-and-k-function.md#encoded-coordinates-time-slices-and-overlap). For a polygon, pass a `GPUStudyWindowPreparation` as the
  compile-time `studyWindow` prop (default `'rectangle'`) and add it to the graph before the KDE.
  Set `window: 'study-window'` in the parameter packer; slots 5–8 become NaN and polygon kernels
  read the prepared window instead. A rectangle-mode operator given this sentinel with edge
  correction produces NaN cells. Polygon windows work with `'exact'`, `'binned'` and `'auto'`.
  Outside points are excluded and cells with centers outside W are NaN. Polygon `e(u)` uses
  oriented fan decomposition, with analytic compact-kernel integrals and Gaussian quadrature
  (2e-5 relative accuracy); only edges within the kernel support contribute corrections.
  Uniform correction divides by `e(u)`; Diggle correction uses `w_i / e(x_i)`. Both retain the
  rectangle small-mass guard (`mass > 1e-6`). The window must be valid (consider
  `GPUGeometryValidity`) and in the same planar frame as the KDE. Window queries use absolute
  frame coordinates, without subtracting `gridStart`. Polygon plus `edgeCorrection: 'none'`
  throws `${id} studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'`.
- **Bandwidth.** Per-frame `h` is in position units: Gaussian standard deviation for `'gaussian'`, support radius for `'epanechnikov'` and `'quartic'` / `'bisquare'`. With `bandwidthRule: 'scott'` or `'silverman'`, each encoding computes `h = kernelFactor * n_eff^(-1/6) * sqrt((varianceX + varianceY) / 2)`, with unbiased weighted covariance and `n_eff = (sum w)^2 / sum(w^2)`. The two rules coincide in two dimensions; the IQR-based one-dimensional Silverman rule is not used. `kernelFactor` is 1 for Gaussian, `2.40187391` for Epanechnikov and `2.77919324` for quartic/bisquare.
  `bandwidthPopulation: 'all-finite'` is the default: moments use finite positions and finite nonnegative weights even outside `indexBounds` or the active edge window. Set `bandwidthPopulation: 'included'` to use the density's inclusion population instead, applying index bounds and the active rectangular/polygon edge window. Two centered reduction passes use a shared per-axis minimum among eligible positive-weight rows as their numerical origin, independent of grid placement. Zero-weight rows are eligible for the population count but do not choose the origin or add moment weight.
  `parameters` slot 4 is ignored with an automatic rule. Optional `output.bandwidth` is one float32 row containing the resolved h; optional `output.bandwidthPopulationCount` is one uint32 row counting selected eligible rows, including zero weights, and requires `bandwidthRule`. These differ from `output.weightSum` (density's total included weight, in original weight units) and `output.status.invalidCount` (excluded density rows). The Gaussian unit triangle `(0,0),(1,0),(0,1)` gives h≈`.4807498568`. Adding `(1000,1000)` outside the index leaves included h unchanged, but default all-finite h becomes `396.718156`; density weightSum and included population count remain 3, while invalidCount is 1. With an automatic rule, insufficient effective sample size or zero spread yields NaN h and an all-NaN raster. Normal-reference rules can oversmooth clustered data; likelihood cross-validation and Diggle's selector are not provided.
- **Exclusions.** A non-finite position or weight, a negative weight, or a position outside `indexBounds` excludes the row. With active edge correction, a position outside the active rectangular or polygon study window is also excluded, even when finite, positively weighted and inside the index. `output.status.invalidCount` counts all these excluded density rows. Zero-weight rows satisfying the spatial inclusion rules remain included. These density diagnostics do not count the default all-finite bandwidth population: that population may still include a finite out-of-window event; choose `bandwidthPopulation: 'included'` to align the populations.
- **Determinism and precision.** The index is rebuilt every encoding, IDs are put back in ascending
  row order inside each index cell by a stable sort of the rows by cell (linear in the row count, however many rows share a cell), and each cell sums in a fixed order (index row, column, ascending row), with no float
  atomics. Distances use offsets from the grid's start corner to limit f32 cancellation; the f32 inputs
  themselves are not recentered, so use local coordinates for fine resolution far from the origin.
- **Large bandwidths (`method`).** The default `'exact'` gather costs `(5h / cell)^2` cells per point and
  explodes when the bandwidth spans many cells. `method: 'binned'` (gaussian only; other kernels throw)
  splits each included weight over the four nearest cell centers (linear, cloud-in-cell binning, Wand
  1994) with the shared deterministic fixed-point accumulator, then blurs the bins along x and y with
  three running-sum box blurs and one two-sided exponential pass per axis, variance-matched to
  `(h / cell)^2` (Wells 1986; Kovesi 2010; the exponential cancels the boxes' fourth cumulant). The
  cost is a few passes over the grid whatever the bandwidth; every bandwidth-dependent filter parameter
  is computed on the GPU, so changing `h` never recompiles. Edge correction, normalization and NaN
  rules are the same as `'exact'`. `method: 'auto'` builds both and the GPU picks one every encoding:
  exact while `5h` is at most 32 cells on the larger axis, binned above when padding suffices.
  The binned error bound requires the frame's effective bandwidth `h` to span at least **3 cells
  on both axes**, including unequal cell dimensions. Optional one-row uint32
  `output.status.bandwidthBelowThreeCells` is 1 when a binned frame violates that condition and
  0 otherwise (including an exact frame selected by `'auto'`). It is recomputed every encoding
  from the GPU-selected bandwidth when `bandwidthRule` is supplied. Error versus `'exact'`, in
  units of `W / (2 pi h^2)` (`W` the included weight, the height of one point's kernel): at most
  `r^2 / 4 + 0.016 + 1.4e-5 + 6e-8 (13 / r + 210)` with `r = cell / h` (`getGPUKernelDensityBinnedErrorBound`):
  `r^2 / 4` from the linear binning, 0.016 from the filter shape, the rest from truncation and float32.
  For example 1.9% at `h = 10` cells and 1.6% at `h = 64`; smooth fields err far less than one point. The
  blur drops the mass that leaves its array, so it needs a margin of `2.5h` around the cells you read:
  `binnedPadding` (default: the larger grid side, so `h` up to 40% of it) adds that margin and also
  extends the region whose points are counted. `output.status.insufficientPadding` flags a `'binned'`
  run that lacked it; `'auto'` falls back to the exact gather instead. The array costs
  `(width + 2 binnedPadding + 2) * (height + 2 binnedPadding + 2)` cells of 20 bytes.
- Pinned to `sklearn.neighbors.KernelDensity` (gaussian, epanechnikov), numpy (quartic), and
  `scipy.integrate.dblquad` (window mass, edge-corrected fields); bandwidth rules to `scipy.stats.gaussian_kde`.
- `spatialContext` is planar/native only (omitted means planar/native). It measures area, so `local-azimuthal-equidistant` frames throw.

Not provided: adaptive or space-time bandwidths, network KDE (see the network contributors), reflection
boundaries.

```ts
import {GPUKernelDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';

const deviceLimits = {
  maxBufferSize: 128 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024
};
const shape = {
  pointCount: 3,
  gridSize: [16, 16] as const,
  indexGridSize: [1, 1] as const,
  method: 'auto' as const,
  cellSize: [1, 1] as const,
  binnedPadding: 32
};
const exactPlan = GPUKernelDensity.getExecutionPlan({...shape, bandwidth: 6.4}, deviceLimits);
const binnedPlan = GPUKernelDensity.getExecutionPlan({...shape, bandwidth: 6.40001}, deviceLimits);
// exactPlan.route === 'exact'; binnedPlan.route === 'binned'.
// Inspect .feasible and .constraints; physical view layouts are checked on construction/declaration.
```

#### How it computes

{/* lens:GPUKernelDensity */}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=datum datum=4887 caption="One cell near Montrose Harbor: its kernel sum divided by the window mass."}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=formula datum=4887 caption="Gaussian kernel truncated at 5h, divided by the in-window kernel mass e_h(u)."}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=kernel caption="The gather kernel: one invocation per output cell visits grid-index rows within 5h."}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=graph caption="Grid index build, then the gather pass; no readback between them."}

{/* parameters: GPUKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUKernelDensity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUKernelDensity`.

- Packer: `getGPUKernelDensityParameterValues(settings, grid)`, writes `GPU_KERNEL_DENSITY_PARAMETER_LENGTH` (9) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned, with a nonzero finite cell size; its width and height must match the contributor's.

Settings of `getGPUKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU computes it) when `GPUKernelDensityProps.bandwidthRule` is set; pass any positive value. |
| `window` (optional) | `GPUKernelDensityWindow \| 'study-window'` | Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells. |

{/* /parameters */}

### `GPUInverseDistanceWeighting`

Interpolates sample values onto a raster as inverse-distance-weighted means of the nearest samples.

::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=result caption="Helene rain gauges (grey) to a rainfall surface: power 2, up to eight gauges within 150 km."}

Maximum: `indexGridSize` has at most `getGPUGridIndexMaximumCellCount(device)` (`2^25 - 1` on a default device: `cells + 1` uint32 cell offsets in one 128 MiB binding) cells.

Interpolates scattered samples onto a raster with inverse distance weighting (`w = 1 / d^p`),
the GPU counterpart of turf `interpolate`, GDAL `invdist`, and GRASS `v.surf.idw`. The output
placement (the `grid` passed to the packer), search radius, power, nearest-neighbor limit, and
minimum neighbor count are per-frame; the grid's size is compile-time.

```ts
import {GPUInverseDistanceWeighting} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUInverseDistanceWeighting>[0];
const {positions, values, mask, grid} = inputs;
const {values: surface, counts} = inputs.output;
declare const minX: number, minY: number, maxX: number, maxY: number;
const idwParameters = new GPUParameterBuffer(device, {id: 'idw', format: 'float32', length: 9});

const idw = new GPUInverseDistanceWeighting({
    positions, // float32x2 per sample
    values, // float32 per sample, NaN skips
    mask, // optional uint32, 0 skips
    parameters: idwParameters.importToGraph(graph), // 9 float32
    grid, // axis-aligned GPURasterMetadata; its width and height size the raster
    indexGridSize: [128, 128], // internal GPUGridIndex over the samples
    indexBounds: [minX, minY, maxX, maxY], // samples outside are ignored
    maximumNeighborCount: 16, // compile-time capacity for k; 0 compiles radius-only
    output: {values: surface, counts} // float32 (NaN = nodata), optional uint32
  });
graph.add(idw);
idwParameters.write(
  idw.getParameterValues(
    {
      searchRadius: 250,
      power: 2,
      taper: 'none', // or 'modified-shepard' (Franke-Nielson, needs a finite radius)
      neighborCount: 8, // 0 = every sample within the radius
      minimumNeighborCount: 3
    },
    grid // the frame's placement: pan or zoom by passing a different grid of the same size
  )
); // no recompile
```

- Gather form: one invocation per output cell samples the cell center
  `gridStart + (x + 0.5, y + 0.5) * cellSize` and visits the grid-index rows that overlap the
  search square. No atomics touch values. `grid` must be axis-aligned (the packer and constructor
  throw otherwise); `gridStart` is the outer corner of cell (0, 0) and `cellSize` is signed, so
  output rows follow the grid's row order: row 0 is the top row of a north-up grid
  (`affine[4] < 0`) and `minY` when rows advance north.
- Samples with `d^2 <= radius^2` contribute. With `neighborCount = k > 0` the `k` nearest are kept
  by `(d^2, row)`, so equal distances keep the smallest row. `idw.getParameterValues(settings, grid)` rejects `neighborCount` above the instance's
  `maximumNeighborCount` (at most 64) and grid dimensions differing from construction.
  `maximumNeighborCount: 0` compiles radius-only gathering and accepts only `neighborCount: 0`.
  The free `getGPUInverseDistanceWeightingParameterValues(settings, grid)` cannot check instance
  capacity; direct raw GPU words retain their clamp to the compiled capacity.
- An exact hit (`d == 0`) returns that sample's value, the smallest row among several, regardless
  of `minimumNeighborCount`. Otherwise a cell with fewer than `max(minimumNeighborCount, 1)`
  contributors is nodata (NaN). `counts` holds the contributors after the `k` limit.
- Negative or NaN radius, or a negative or non-finite power, produces an all-NaN raster.
  `searchRadius: Infinity` searches the whole index domain.
- Units and coordinates: by default positions, grid and radius share one planar unit and
  distances are differences of offsets from the grid's start corner, but float32 input quantization is already present at large absolute coordinates. Subtract the
  same local origin from positions, grid placement and index bounds before upload when fine
  resolution requires it; also recenter the window for KDE. Corner-based raster packers reject
  collapsed float32 endpoints/spans/steps; apply a common origin to every spatial input. With `spatialContext: {coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'}` positions are absolute lon/lat degrees, the index and output grid stay
  in degrees, and distances and `searchRadius` use the context's length unit (meters by default;
  kilometers with `units: 'kilometers'`), on a sphere of `sphereRadius` or the mean Earth radius in that unit. Neighbors and radius
  tests order by the haversine term. The conservative query box widens longitude coverage with latitude. `longitudeSearch: 'clipped'` is the default and clips at the antimeridian. Opt into `longitudeSearch: 'wrapped'` to search both seam sides and polar caps; use `indexBounds: [-180, -90, 180, 90]` to include the full canonical longitude/latitude domain. Domain edges and radius tests are inclusive: candidates exactly on the seam, pole or radius are included when covered by the bounds. Wrapped mode requires index longitudes within `[-180,180]`; bounds still determine which observations are indexed. Nearest-k scans the whole query box without early stopping.
  Pass optional `spatialContext` to `getGPUInverseDistanceWeightingParameterValues(settings, grid)` to warn on short `searchRadius` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
  Other longitude/latitude metrics
  (rhumb, ellipsoidal, none) throw; project to a planar frame (for example `createGPULocalFrame`)
  for them. It measures distance, so `local-lambert-equal-area` frames throw.
- `taper: 'none'` (default) is `w = 1 / d^p` inside the radius, a hard cutoff: a sample entering
  or leaving the radius changes the surface discontinuously. `taper: 'modified-shepard'` is the
  Franke and Nielson (1980) weight `w = ((R - d)+ / (R d))^p` with `R = searchRadius`, which falls
  to zero at the radius (a sample exactly at `R` has weight zero). It needs a finite radius
  (`getGPUInverseDistanceWeightingParameterValues` throws otherwise). `counts` still counts every
  sample inside the radius. Default stays `'none'` for backward compatibility; prefer the taper
  when a finite radius is a modelling choice rather than just a speed limit.
- Parameter guidance: `power` 2 is conventional, not optimal; higher powers flatten the surface
  around samples. `minimumNeighborCount` defaults to 1, so a lone distant sample inside the radius
  paints a constant plateau of its value; use 3 or more to leave sparse cells as nodata. IDW
  has no direction or sector search and no leave-one-out output; use `GPUKriging`'s
  cross-validation to compare models on the same samples.
- Determinism: the grid index's in-cell order depends on atomics, so the contributor re-sorts IDs inside
  each index cell every encoding; sums then run in a fixed order (index row, column, ascending
  row; or `(d^2, row)` with `k`). Weights are accumulated in log space against a running maximum,
  so tiny distances with large powers do not overflow f32.
- The index is rebuilt every encoding, so positions and values may change per frame. Keep a few
  samples per index cell: the in-cell sort is quadratic in cell population.
- `output.values` is a row-major float32 raster that `GPUTerrainContours` accepts directly as
  `elevation: {id, format: 'float32', storage: {kind: 'buffer', values: surface}}`; NaN cells
  clear their marching-squares cells.

#### How it computes

{/* lens:GPUInverseDistanceWeighting */}

::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=datum datum=1659 caption="Inspect cell 1659, containing station USW00013872, Asheville, and its contributing-station count."}

::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=formula caption="Normalize inverse-distance weights; the lowest-row exact hit takes precedence."}

::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=kernel caption="Read nearest-neighbor selection and log-space accumulation."}

::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=graph caption="Follow station indexing, sorting and rainfall gathering."}

{/* parameters: GPUInverseDistanceWeighting generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUInverseDistanceWeighting` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUInverseDistanceWeighting`.

- Packer: `getGPUInverseDistanceWeightingParameterValues(settings, grid)`, writes `GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH` (9) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned; its width and height must match the contributor's.
- Packer: `instance.getParameterValues(settings, grid?)`, writes the `float32` parameter view.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUInverseDistanceWeightingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `searchRadius` | `number` | Search radius in position units. `Infinity` searches the whole index domain, except with the `'modified-shepard'` taper, which needs a finite radius. |
| `power` (optional) | `number` | Distance power `p` in `w = 1 / d^p`. Zero gives an unweighted mean. Defaults to 2. |
| `taper` (optional) | `GPUInverseDistanceWeightingTaper` | Weight taper at the search radius. Defaults to `'none'` (hard cutoff). |
| `neighborCount` (optional) | `number` | Nearest-neighbor limit `k`; zero (default) uses every sample within the radius. Values above the contributor's compile-time `maximumNeighborCount` are clamped by raw GPU parameters; use the instance's `getParameterValues` method to reject requests beyond its capacity. |
| `minimumNeighborCount` (optional) | `number` | Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 1. |

{/* /parameters */}


### `GPUKriging`

Predicts values and kriging variance on a raster by solving an ordinary kriging system per cell from nearby samples.

::lens{scenario=gpu-spatial-analysis.kriging rung=result caption="Helene rain gauges to rainfall: spherical variogram, 150 km range; nearer gauges get larger weights."}

Maximum: `indexGridSize` has at most `getGPUGridIndexMaximumCellCount(device)` (`2^25 - 1` on a default device: `cells + 1` uint32 cell offsets in one 128 MiB binding) cells.

Local ordinary kriging on a raster (PyKrige `OrdinaryKriging` with `n_closest_points`, GSTAT, ArcGIS Kriging).
For every output cell the `k <= 16` nearest samples within the radius form a `(k + 1) x (k + 1)` system
under a fitted variogram, solved per invocation by pivoted Gaussian elimination in f32 (the variogram is
normalized by its total sill and the prediction is relative to the nearest value). The samples come from
the same internal `GPUGridIndex` as `GPUInverseDistanceWeighting`.

```ts
import {GPUKriging, getGPUKrigingParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUKriging>[0];
const {positions, values, grid, indexGridSize, indexBounds} = inputs;
const {values: surface, variance, crossValidationPrediction, crossValidationVariance} = inputs.output;
const {invalidCount, uncertainCount} = inputs.output.status ?? {};
const krigingParameters = new GPUParameterBuffer(device, {
  id: 'kriging', format: 'float32', length: 12,
  values: getGPUKrigingParameterValues({searchRadius: 250, neighborCount: 16, minimumNeighborCount: 3,
    variogram: {model: 'spherical', nugget: 0, sill: 1, range: 250}}, grid)
});

graph.add(
  new GPUKriging({
    positions, values, parameters: krigingParameters.importToGraph(graph),
    grid, // axis-aligned GPURasterMetadata; its width and height size the raster
    indexGridSize, indexBounds, maximumNeighborCount: 16,
    output: {
      values: surface, variance, // float32 rasters
      crossValidationPrediction, crossValidationVariance, // optional float32 per SAMPLE row
      status: {invalidCount, uncertainCount} // optional uint32 scalars
    }
  })
);
```

- Props: `positions`, `values` (NaN skips), optional `mask`, `parameters` (`getGPUKrigingParameterValues`,
  at least 12 float32), `grid`, `indexGridSize`, `indexBounds`, `maximumNeighborCount` (1 to 16,
  compile-time) and `output: {values, variance?}`.
- Per frame: radius, `k`, minimum neighbors, the placement (the `grid` passed to
  `getGPUKrigingParameterValues(settings, grid)`, axis-aligned, same width and height as the props; output
  rows follow its row order, row 0 at the top of a north-up grid and at `minY` when rows advance north), model (`'spherical'`, `'exponential'`, `'gaussian'`, `'matern-3/2'` or `'matern-5/2'`,
  the `fitVariogramModel` convention) and nugget, sill and range. `range` is the effective range: the
  exponential model is `1 - exp(-3 h / range)` and the Gaussian model `1 - exp(-3 (h / range)^2)`, both
  at 95 percent of the sill at `range`. Converting to another library:
  - PyKrige: pass `sill + nugget` as its "sill" (PyKrige's value is the total sill). Its exponential and
    spherical models take the same `range`. Its built-in Gaussian model is
    `1 - exp(-(h / (range * 4 / 7))^2)`, exponent 3.0625 instead of 3, so pass
    `range * 7 / (4 * sqrt(3))` (about `1.0104 * range`) to get the same surface.
  - gstat `vgm`: the range parameter is `range / 3` for `"Exp"`, `range / sqrt(3)` for `"Gau"` and
    `range` for `"Sph"`.
- `values` is a row-major raster and nodata is NaN: too few distinct neighbors, an ill-conditioned
  system, or an unreliable variance. `variance` is the kriging variance in squared value units, 0 at exact
  hits, NaN at nodata.
- Units and coordinates: by default positions, grid, radius and variogram range share one planar
  unit; distances are differences of offsets from the grid's start corner (cross-validation: from the
  index domain minimum), but uploaded float32 absolute coordinates may already have lost local detail. Subtract a common
  local origin from positions, grid placement and index bounds before upload for fine
  neighborhoods. Corner-based raster packers reject collapsed float32 endpoints/spans/steps;
  apply a common origin to every spatial input. With `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
  'great-circle'}` positions are absolute lon/lat degrees, the index and grid stay in degrees,
  and the search radius, variogram range and all sample-sample and sample-cell distances are
  haversine meters (`sphereRadius`, else 6371008.8 m; kilometers with `units: 'kilometers'`).
  Neighbors are ordered by `(haversine term, row)`; exact hits and duplicate coalescing compare
  degree positions exactly; leave-one-out uses the same metric. The conservative search defaults to `longitudeSearch: 'clipped'`, clipping at the antimeridian. Set `longitudeSearch: 'wrapped'` with `indexBounds: [-180, -90, 180, 90]` for both seam sides and polar caps, including covered domain/radius boundaries. Wrapped mode requires index-bound longitudes within `[-180,180]`; narrower latitude bounds still exclude observations outside them. The whole query box is scanned without the planar early stop, so unbounded radius is `O(samples)` per cell.
  Pass optional `spatialContext` to `getGPUKrigingParameterValues(settings, grid)` to warn on short `searchRadius` and `range` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
  Other longitude/latitude metrics throw. It measures distance, so `local-lambert-equal-area` frames throw.
- Status: `invalidCount` counts cells that are NaN for lack of data or invalid settings (fewer than
  `minimumNeighborCount` distinct positions, no sample in the radius, outside the index domain).
  `uncertainCount` counts cells rejected by the numerical checks below. Both are computed over the
  raster of the last encoding, so a nonzero `uncertainCount` means the surface has holes that are
  not "no data" but "do not trust the solve here".
- Singularity is a relative test, not an absolute one: a pivot at or below
  `KRIGING_RELATIVE_PIVOT_TOLERANCE` (1e-6) of the system scale is zero, and a system whose smallest
  pivot is below `KRIGING_PIVOT_RATIO_TOLERANCE` (1e-4) of its largest is rejected as ill conditioned
  (the same limit `GPULocalKriging` and `GPULocalRBF` apply),
  since an f32 solve loses about `cond * 6e-8` of the weights' accuracy and the ratio tracks `1 / cond`.
  The usual cause is a Gaussian model with a near-zero nugget and closely spaced samples: add a nugget
  (0.001 of the sill moves the ratio from 1e-6 to 5e-5 for 12 clustered samples) or thin the samples. The
  constants were calibrated against emulated f32 solves compared with float64.
- Variance is `mu + sum(w * gamma)` in units of the total sill and is never negative for a valid
  model. A negative value above `-KRIGING_NEGATIVE_VARIANCE_TOLERANCE` (1e-4) is f32 rounding and is
  clamped to zero; a larger negative variance makes the cell NaN (value and variance) and counts it in
  `uncertainCount`. It is never silently clamped.
- Duplicate sample positions inside a neighborhood are coalesced into one kriging point whose value is
  the mean of their values (summed in `(d^2, row)` order, so the result is deterministic), instead of
  making the system singular. Duplicates still occupy neighbor slots, so a neighborhood of `k` rows can
  hold fewer than `k` points, and `minimumNeighborCount` counts distinct positions. An exact hit returns the
  mean of the samples at that position with variance 0.
- The nugget is treated as micro-scale variation: the predictor is exact at the data even with a
  positive nugget (zero variance at exact hits). Filtered kriging with a measurement-error nugget is not
  provided.
- Leave-one-out cross-validation (`crossValidationPrediction`, optional `crossValidationVariance`, one
  float32 row per sample) krigs each sample from its `k` nearest OTHER samples with the same variogram,
  radius, `k` and minimum-neighbor settings and the same solver code. Rows that are NaN, masked or
  unestimable are NaN. Another sample at exactly the same position is an exact hit (its mean, variance 0).
  Compare models with the standardized error `(value - prediction) / sqrt(variance)`, which should have
  mean near 0 and variance near 1, and the root mean squared error of `value - prediction`.
- A sample `mask` is folded into a transient value copy first (a one-pass kernel), and the per-cell status
  goes to a transient before the counters are accumulated, so every kernel stays within 8 storage buffers.
- Pinned to PyKrige 1.7.3 with 8 closest points: exponential, spherical and Gaussian within 2e-5 of
  the data scale on values (about 1.4e-3 on values near 70) and 1e-4 on variance relative to
  `max(1, variance)`. The Gaussian fixture uses a custom PyKrige variogram with exponent 3, because
  PyKrige's built-in Gaussian has the other range convention (see above); the pivot-ratio, duplicate-coalescing, mask, leave-one-out and
  translation fixtures are float64 numpy ordinary kriging (Cressie 1993, eq. 3.2.12 to 3.2.16) that
  reproduces the PyKrige numbers to 5e-7. Global and universal kriging, anisotropy, octant search and
  filtered (measurement-error) kriging are not provided.

#### How it computes

{/* lens:GPUKriging */}

::lens{scenario=gpu-spatial-analysis.kriging rung=datum datum=1659 caption="Inspect prediction and variance in cell 1659, containing station USW00013872, Asheville."}

::lens{scenario=gpu-spatial-analysis.kriging rung=formula caption="Solve variogram weights with a Lagrange multiplier; range controls decorrelation, not the search cutoff."}

::lens{scenario=gpu-spatial-analysis.kriging rung=kernel caption="Read duplicate coalescing, pivot checks and variance rejection."}

::lens{scenario=gpu-spatial-analysis.kriging rung=graph caption="Follow station indexing and eight-neighbor ordinary kriging."}

{/* parameters: GPUKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUKriging` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUKriging`.

- Packer: `getGPUKrigingParameterValues(settings, grid)`, writes `GPU_KRIGING_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned; its width and height must match the contributor's.

Settings of `getGPUKrigingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `searchRadius` | `number` | Search radius in position units. `Infinity` searches the whole index domain. |
| `neighborCount` (optional) | `number` | Neighborhood size `k`. Clamped to the contributor's compile-time `maximumNeighborCount`. Defaults to that capacity. |
| `minimumNeighborCount` (optional) | `number` | Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 3. |
| `variogram` | `{ model: VariogramModelType; nugget: number; sill: number; range: number; }` | Fitted variogram, for example the result of `fitVariogramModel`: `gamma(h) = nugget + sill * shape(h / range)` for `h > 0`, with the same effective-range convention. The nugget plus the partial sill must be positive. |

{/* /parameters */}


### `GPULocalKriging`

Predicts values and latent variance at query points or quadrature supports by solving a local covariance system per query.

Local kriging in simple, ordinary, universal, external-drift, block and co-kriging forms, with spherical, exponential, Gaussian or Matérn (smoothness 1/2, 3/2, 5/2) covariances. Unlike `GPUKriging`, which fills a raster from an indexed grid, it takes explicit query rows and scans the samples or a supplied neighborhood. Full reference: [Interpolation and simulation](./interpolation-expansion.md).

- `observations.w` includes a sample only when it is greater than zero; zero, negative and NaN
  validity values skip the sample.
- With convenience `covariance`, `coregionalization` or `coregionalizationModels` props, call `kriging.setCovariance(settings)` between submitted frames to change values without recompiling; preserve `covarianceStructureCount` and `variableCount`. With caller-owned `parameters`, write its owning `GPUParameterBuffer` using `getGPULocalKrigingParameterValues(settings)` instead. The parameter view consumes one storage binding. Submit preceding encodings before updating either route.

Conditional Gaussian simulations capture covariance at construction. By default, preparation is rebuilt every encoding. With `reusePreparation: true`, write changed geometry/input buffers and call `simulation.invalidatePreparation()` before the next encoding; retained preparation otherwise replays. Both routes require a new simulation and graph for new covariance. See [Interpolation and simulation](./interpolation-expansion.md) for prediction and conditional realization examples.

- Without a supplied neighborhood, `sampleIndex: 'brute-force'` scans every sample for every query.
  Construction checks the total sample count times the total query count against
  `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK` (2³², inclusive), before query chunking.
  The default `'auto'` scans when there are at most 4,096 samples and the work fits that limit.
  Otherwise it selects the grid if it fits the device’s binding limits, with a scan fallback
  subject to the same work cap. Use `'grid'` to require indexing.
- Matérn `range` is the effective range (correlation `exp(-3)` at `range`, like the exponential and Gaussian models); a Matérn 1/2 model equals the exponential model. Pinned to GSTools 1.7.0 correlation values and simple and ordinary kriging, predictions and variances, with `len_scale = sqrt(nu) * range / factor`.
- A system whose pivot ratio is below `KRIGING_PIVOT_RATIO_TOLERANCE` (1e-4) reports status 2 and writes its pivot ratio to `values.z`. No nugget floor is applied; see [Conditioning of smooth covariances](./interpolation-expansion.md#conditioning-of-smooth-covariances).

{/* parameters: GPULocalKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalKriging` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalKriging`.

- Packer: `getGPULocalKrigingParameterValues(settings)`, writes `GPU_LOCAL_KRIGING_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `getGPULocalKrigingParameterValues(settings)`, writes `GPU_LOCAL_KRIGING_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPULocalKrigingParameterValues(settings)` for `LocalKrigingSettingsOptions & { covariance: GPUPredictionCovariance; }`:

| Field | Type | Description |
|---|---|---|
| `covariance` | `GPUPredictionCovariance` | Covariance of the univariate modes and of the two-by-two `coregionalization` shortcut. |
| `coregionalization` (optional) | `readonly [number, number, number, number]` | Two-by-two row-major loading matrix `L`, with `B = L L^T`, as in the contributor props. |
| `coregionalizationModels` (optional) | `readonly { covariance: GPUPredictionCovariance; loadings: readonly number[]; }[]` | One to four covariance structures with row-major square loading factors. |
| `variableCount` (optional) | `number` | Co-kriging variables, 2..4. Defaults to 2 when loadings are supplied. |

Settings of `getGPULocalKrigingParameterValues(settings)` for `LocalKrigingSettingsOptions & { coregionalizationModels: readonly { covariance: GPUPredictionCovariance; loadings: readonly number[]; }[]; }`:

| Field | Type | Description |
|---|---|---|
| `covariance` (optional) | `GPUPredictionCovariance` | Covariance of the univariate modes and of the two-by-two `coregionalization` shortcut. |
| `coregionalization` (optional) | `readonly [number, number, number, number]` | Two-by-two row-major loading matrix `L`, with `B = L L^T`, as in the contributor props. |
| `coregionalizationModels` | `readonly { covariance: GPUPredictionCovariance; loadings: readonly number[]; }[]` | One to four covariance structures with row-major square loading factors. |
| `variableCount` (optional) | `number` | Co-kriging variables, 2..4. Defaults to 2 when loadings are supplied. |

{/* /parameters */}

### `GPUSpaceTimeKriging`

Predicts values and latent variance at space-time query points by solving a local space-time covariance system per query.

Local simple or ordinary space-time kriging over the separable, product-sum, metric, sum-metric and Gneiting covariance families (gstat 2.1-6 definitions, and Gneiting 2002), with metric, octant or staged neighbor search. Full reference: [Space-time kriging](./interpolation-expansion.md#space-time-kriging).

- `observations.w` includes a sample only when it is greater than zero; zero, negative and NaN
  validity values skip the sample.
- Every covariance parameter and the search radius, time window and anisotropy change per frame through `parameters` without a recompile.
- Pass `temporalContext`, and `queryTemporalContext` for the query times; both must be `float32-relative`, with the same unit and epoch.
- A system below `KRIGING_PIVOT_RATIO_TOLERANCE` (1e-4) reports status 2 and writes its pivot ratio to `values.z`.
- `sampleIndex` (construction-time, default `'auto'`): auto scans every sample through 4,096 samples, then uses the per-frame metric grid. Forced `'brute-force'` scans at most 1,048,576 samples; `'grid'` always indexes, with a structural sample-row ceiling of `GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT` (1,073,741,808). The usable indexed row count is constrained separately by the device and by every shared packed sample binding. For a 128 MiB storage binding, `observations` as `float32x4` (16 bytes per observation) holds at most 8,388,608 observations at byte offset zero, or 8,388,607 at byte offset 16. Query/output chunking does not split this shared observation binding. Other sample bindings, dispatch limits and scratch must also fit. Coincident samples, wide ties or restrictive filters can make an indexed query visit the whole index.
- Verified against independent binary64 solves for every family, mode and search; GSTools 1.7.0 for the metric family. **gstat fixtures pending** (R not available).

{/* parameters: GPUSpaceTimeKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpaceTimeKriging` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpaceTimeKriging`.

- Packer: `getGPUSpaceTimeKrigingParameterValues(settings)`, writes `GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH` (28) `float32` values.

Settings of `getGPUSpaceTimeKrigingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `covariance` | `SpaceTimeVariogramModel` | Space-time covariance, for example the result of `fitSpaceTimeVariogramModel`. |
| `searchAnisotropy` (optional) | `number` | Spatial units per unit of time of the neighbor metric `sqrt(h^2 + (anisotropy u)^2)`. Defaults to the `anisotropy` of a metric or sum-metric covariance; required otherwise. |
| `searchRadius` (optional) | `number` | Largest spatial lag of a neighbor. Defaults to infinity. |
| `maximumTimeLag` (optional) | `number` | Largest absolute time lag of a neighbor. Defaults to infinity. |
| `mean` (optional) | `number` | Known mean of `simple` mode. Defaults to 0. |

{/* /parameters */}

### `GPUDotDensity` and `GPURandomPointsInPolygon`

`GPUDotDensity`: Normalized features are bounded by 16,777,215. The triangle CDF uses segmented
saturating counts; packed outputs must satisfy logical and co-bound aligned-range checks.

**`GPUDotDensity`**: Places seeded random dots inside polygons in proportion to a count column, one dot per stated number of units.

::lens{scenario=gpu-spatial-analysis.dot-density rung=result caption="199 tracts (grey) to one dot per 500 residents (blue); dots are not where people live."}

**`GPURandomPointsInPolygon`**: Samples a requested number of seeded, uniformly distributed points inside each polygon.

::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=result caption="13 community areas (grey) with 20 seeded uniform points each (blue)."}

**`GPURandomPointsOnLine`**: Samples seeded points along lines, uniformly per unit of arc length.

::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=result caption="721 street edges (grey) with two seeded points each (blue), uniform along length."}

Maximum: `GPUDotDensity` holds at most `getGPUDotDensityMaximumCapacity(device)` (8388608 on a default device) dots.

Dasymetric dot-density maps: `value * dotsPerUnit` random dots inside each polygon, per category,
optionally concentrated by a weight raster such as built-up area. Changing the dot value with zoom
only adds or removes dots; existing dots never move, so there is no flicker.
`GPURandomPointsInPolygon` places an integer count of uniform random points per feature. Its internal triangulation limits the polygon layout to `getGPUPolygonTriangulationMaximumNodeCount(device)` scratch nodes (see `GPUPolygonTriangulation`).

```ts
import {GPUDotDensity, getGPUDotDensityParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUDotDensity>[0];
const {positions, featureOffsets, polygonOffsets, ringOffsets, values} = inputs;
const {positions: dotPositions, categories} = inputs.output;
const {ids, status: {count, overflow, requiredCount}} = inputs.output.dots;
const {failedCount} = inputs.output.status ?? {};
declare const peoplePerDot: number, x0: number, y0: number, cw: number, ch: number;

const parameters = new GPUParameterBuffer(device, {id: 'dots', format: 'uint32', length: 8});
graph.add(
  new GPUDotDensity({
    positions, featureOffsets, polygonOffsets, ringOffsets, // four-level polygon layout
    values, // float32, featureCount * categoryCount rows, feature-major
    categoryCount: 3,
    parameters: parameters.importToGraph(graph),
    ...(inputs.mask ? {mask: inputs.mask} : {}), // optional, weights in [0, 1]
    output: {positions: dotPositions, dots: {ids, status: {count, overflow, requiredCount}}, categories, status: {failedCount}}
  })
);
parameters.write(
  getGPUDotDensityParameterValues({seed: 1, dotsPerUnit: 1 / peoplePerDot,
    maskPlacement: {origin: [x0, y0], cellSize: [cw, ch]}})
);
```

- A slot (feature, category) draws the integer part of float32 `x = value * dotsPerUnit`, plus
  one if its seeded Philox uniform `u` is below `frac(x)`, monotone in `dotsPerUnit`.
  Every exactly representable integer request remains exact before per-slot clamping and
  empty-feature rejection; subtracting the random draw from a large integer is not used.
  The fractional round-up probability is quantized by the 24-bit uniform draw, within one
  `2^-24` quantum of the requested fraction (for example `2^-25` rounds up with probability
  `2^-24`). This does not weaken the exact representable-integer guarantee.
- Dot `j` of a slot is placed exactly, keyed by Philox `(seed, slot, j, attempt)` and independent of the
  dot value and of other slots, so coarser dot values give a stable prefix of the finer dots. Every
  polygon is triangulated on the GPU (`GPUPolygonTriangulation`; the first ring of a polygon is the
  shell, later rings are holes), a triangle is picked in proportion to its area (binary search of the
  feature's cumulative areas, 32-bit fixed point) and the point is uniform in it. Thin or concave
  polygons in a large bounding box lose no dots. With a mask a candidate is also kept with probability
  equal to its cell weight, and a masked dot is retried up to `maximumAttempts` times (default 32;
  ignored without a mask).
- Dots are ordered by feature, category and rank (offsets from a prefix scan). `dots.ids` holds the
  feature row of each dot and `dots.status.count` is safe as an instance count; rows past it are NaN /
  `0xffffffff`.
- Both polygon samplers expose optional `slotStarts` and `slotCounts`, one uint32 row per
  (feature, category) slot for dot density, or per feature for random polygon points.
  `slotStarts` contains pre-clamp starts without a terminal. Read a slot's stored dot
  interval as `[min(start, writtenCount), min(start + slotCounts[slot], writtenCount))`, where
  `writtenCount = dots.status.count`. Multiply both ends by two for a flat XY array.
  Requests `[2, 2]` with capacity three therefore delimit `[0, 2)` and `[2, 3)`.
- `maskPlacement` names `{origin: [x, y], cellSize: [dx, dy]}` in polygon position units.
  Signed steps are preserved, including negative Y; defaults are origin `[0,0]` and steps `[1,1]`.
- Compact `status.overflow` reports final output capacity only. Optional `candidateOverflow`
  reports a per-slot/per-line count clamped before sampling; `requiredCount` counts retained
  requests and is then a lower bound. The uint32 prefix bound limits each slot to
  `min(capacity + 1, floor(0xffffffff / slotCount), 2^31)` requests. With 65,536 slots and capacity
  65,536, a slot requesting 65,536 retains 65,535 and sets candidateOverflow, while final overflow
  stays zero. Growing the output alone cannot remove the prefix bound.
  For one line requesting three points at capacity 1, the retained request count is 2:
  `[count, requiredCount, overflow, candidateOverflow] = [1, 2, 1, 1]`. Requesting
  two instead gives `[1, 2, 1, 0]`: final overflow can occur without candidate loss.
  When that per-slot limit is at or below final capacity, construction requires
  `dots.status.candidateOverflow` (or `points.status.candidateOverflow` for line sampling),
  because clamping could otherwise leave a short result with no visible loss flag.
- Empty polygons and degenerate rings from clipping are skipped on the GPU. A ring is degenerate
  when it has fewer than three vertices or every first-vertex fan triangle has exact orientation zero,
  matching `GPUStudyWindowPreparation`; an uncertified orientation is retained. A degenerate shell
  makes its entire polygon empty. Removing these entries while retaining feature rows preserves
  triangle selection and Philox draws bitwise.
- Features with zero total area get zero slots, so they consume no capacity and contribute zero to
  `slotCounts`, `dots.status.count` and `dots.status.requiredCount`. Their clamped requested dot counts
  are reported in **`status.failedCount`**, without an exception.
- **`status.failedCount`** also counts other placement failures, which keep a NaN slot: untriangulable
  polygons, exceeded triangulation budgets, non-finite areas and masked dots whose attempts all missed
  (probability `(1 - w)^maximumAttempts` for mean mask weight `w`). A feature with no requested dots
  reports nothing.
- **Coordinates.** Dots are uniform in coordinate space, so `spatialContext` is planar/native only (omitted
  means planar/native), on `GPUDotDensity`, `GPURandomPointsInPolygon` and `GPURandomPointsOnLine`.
  Longitude/latitude throws. `GPUDotDensity` and `GPURandomPointsInPolygon` declare `measure: 'area'`:
  an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
  `GPURandomPointsOnLine` declares `measure: 'distance'`: a Lambert equal-area local frame throws;
  use an azimuthal-equidistant frame.
- Ring roles follow GeoArrow position, not an even-odd test: ring 0 of each polygon is the shell and
  later rings are holes (either winding). The triangulation scratch is about 56 bytes per input vertex.

**`GPURandomPointsOnLine`** places `counts` points per line uniformly in arc length (`GeoSeries.sample_points`
on lines). The Philox key is `(line, rank, 0, 0)` with key `(seed, 3)`, so the prefix of points is stable
as counts grow. Outputs are a bounded `positions` buffer and a compact `points` record (`ids` = line row,
`status` with `count`, `overflow`, optional `candidateOverflow`, `requiredCount`), plus optional `fractions`. Parameters reuse
`getGPUDotDensityParameterValues` (`seed`). Points matched `LineString.interpolate` at the pinned Philox
fractions to 2e-5 relative. Not provided: `cluster_poisson` sampling.

#### How it computes

{/* lens:GPUDotDensity */}

::lens{scenario=gpu-spatial-analysis.dot-density rung=datum datum=0 caption="Inspect tract 17031010100: population input and allocated dot count."}

::lens{scenario=gpu-spatial-analysis.dot-density rung=kernel caption="Read stochastic rounding and area-weighted triangle sampling."}

::lens{scenario=gpu-spatial-analysis.dot-density rung=graph caption="Follow triangulation, allocation, sampling and failure reporting."}

#### How it computes

{/* lens:GPURandomPointsInPolygon */}

::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=datum datum=0 caption="Inspect Rogers Park: requested points and allocated slots."}

::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=kernel caption="Read seeded triangle selection and point placement."}

::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=graph caption="Follow triangulation, prefix allocation and point publication."}

#### How it computes

{/* lens:GPURandomPointsOnLine */}

::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=datum datum=0 caption="Inspect the first sample slot: road row and arc-length fraction."}

::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=kernel caption="Read Philox draws and segment interpolation."}

::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=graph caption="Follow segment lengths, prefix allocation and point publication."}

{/* parameters: GPUDotDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUDotDensity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUDotDensity`.

- Packer: `getGPUDotDensityParameterValues(settings)`, writes `GPU_DOT_DENSITY_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUDotDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed. Every dot position is a pure function of the seed and its (slot, rank). |
| `dotsPerUnit` (optional) | `number` | Dots per unit of value (the reciprocal of "people per dot"), usually zoom dependent. Ignored by GPURandomPointsInPolygon . Zero or a non-finite value draws no dots. Defaults to 1. |
| `maskPlacement` (optional) | `{ origin: readonly [number, number]; cellSize: readonly [number, number]; }` | Mask origin and signed cell steps in polygon position units. Defaults to origin [0,0], cellSize [1,1]. |

{/* /parameters */}

{/* parameters: GPURandomPointsOnLine generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURandomPointsOnLine` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPURandomPointsOnLine`.

- Packer: `getGPUDotDensityParameterValues(settings)`, writes `GPU_DOT_DENSITY_PARAMETER_LENGTH` (8) `uint32` values.
  - Input port: `parameters`.
  - This consumer reads only `seed`; other settings in the shared layout are ignored.

Settings of `getGPUDotDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed. Every dot position is a pure function of the seed and its (slot, rank). |

{/* /parameters */}

