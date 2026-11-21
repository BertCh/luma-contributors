# Space-time kernel density, relative risk, K function and Diggle test

STX-04 and STX-05. `GPUSpaceTimeKernelDensity` and `GPUSpaceTimeRelativeRisk` import from
`@luma-contributors/analysis/gpu-spatial-analysis`. `GPUSpaceTimeKFunction` and
`GPUSpaceTimeDiggleTest` import from `@luma-contributors/analysis/dataframe-analysis`, beside
`GPURipley` and `GPUSpaceTimeVariogram`, because they run on the indexed pair histogram.

Times are float32 relative to a nearby origin, in one unit (see
[Temporal context](./temporal-context.md) and `getGPUSpaceTimeGuardedRelativeTimes`). A
`temporalContext` prop is optional, must be `float32-relative`, and is checked and exposed, never
converted. Every time length below (`temporalBandwidth`, `sliceInterval`, `maximumTimeLag`, the time
window) is in that unit.

Density constructors capture their used construction options, nested grids, contexts and
output/status routing once. Mutating the original options before graph declaration does not change
the graph. Input and parameter graph views keep their identity; their buffer contents remain live
and can be rewritten between submissions without recompilation.

## References and status of the oracles

The references are R-only and GPL. R is not installed on the development machines, so there are no
sparr, stpp or stopp fixtures yet, and the splancs fixture is a NumPy port of the splancs 2.01-45
source, not a run of R (`modules/analysis/test/oracles/space-time-diggle-splancs-port.py`; see the
Diggle test below). Every spec compares the GPU result with a binary64 CPU
brute force written from the definitions on this page. The reference code shares nothing with the
source under test: the temporal window mass is a Simpson integral, the circle fraction of the edge
weight comes from circle and side-line crossing angles, and the time permutations use the
TypeScript twin of the shuffle. The definitions follow the cited papers and package manuals; where
this page says a convention is unverified, it was not checked against the package.

| Operator | Definition | Package pinned for the pending fixture |
| --- | --- | --- |
| `GPUSpaceTimeKernelDensity` | Fernando and Hazelton (2014), fixed bandwidths | sparr 2.3-16 `spattemp.density` |
| `GPUSpaceTimeRelativeRisk` | the same, time-varying controls | sparr 2.3-16 `spattemp.risk` |
| `GPUSpaceTimeKFunction` | Gabriel and Diggle (2009) | stpp 2.0-8 `STIKhat`, `PCFhat`, `LISTAhat`; stopp 1.0.0 |
| `GPUSpaceTimeDiggleTest` | Diggle, Chetwynd, Haggkvist and Morris (1995) | splancs 2.01-45 `stkhat`, `stmctest` (a port of the source is committed; the R run is pending), surveillance `stKtest` |

## Space-time kernel density: `GPUSpaceTimeKernelDensity`

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {getGPUSpaceTimeKernelDensityParameterValues, GPUSpaceTimeKernelDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

declare const graph: GPUCommandGraph;
declare const parameterBuffer: GPUParameterBuffer<'float32'>;
const parameters = parameterBuffer.importToGraph(graph);
declare const positions: GraphDataView<'float32x2'>;
declare const times: GraphDataView<'float32'>;
declare const weights: GraphDataView<'float32'>;
declare const values: GraphDataView<'float32'>;
declare const weightSum: GraphDataView<'float32'>;
declare const invalidCount: GraphDataView<'uint32'>;
const [minX, minY, maxX, maxY] = [0, 0, 256_000, 256_000];

const grid = makeGPURasterMetadata({width: 256, height: 256, cellSize: [1000, 1000], origin: [minX, maxY]});
const parameterValues = getGPUSpaceTimeKernelDensityParameterValues(
  {bandwidth: 500, temporalBandwidth: 3, firstSliceTime: 0, sliceInterval: 1, timeWindow: [0, 63]},
  grid,
  64 // sliceCount
);
parameterBuffer.write(parameterValues); // a GPUParameterBuffer, imported as `parameters`
graph.add(
  new GPUSpaceTimeKernelDensity({
    positions, // float32x2
    times, // float32, relative
    weights, // optional float32
    parameters, // float32, GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH (14) values
    grid,
    sliceCount: 64,
    indexGridSize: [512, 512],
    indexBounds: [minX, minY, maxX, maxY],
    spatialKernel: 'gaussian', // or 'quartic' (alias 'bisquare'), 'epanechnikov'
    temporalKernel: 'gaussian', // or 'quartic', 'epanechnikov'
    spatialEdgeCorrection: 'uniform', // or 'none'
    temporalEdgeCorrection: 'uniform', // or 'none'
    normalization: 'intensity', // or 'probability'
    output: {values, weightSum, status: {invalidCount}}
  })
);
```

For the voxel at cell center `u` and slice time `t_k = firstSliceTime + k * sliceInterval`:

`f(u, t) = sum_i w_i Ks_h(|u - x_i|) Kt_l(t - t_i) / (es(u) et(t))`

- `Ks_h` is the radially symmetric 2D kernel of `GPUKernelDensity` (gaussian truncated at 5
  bandwidths, quartic, epanechnikov). `Kt_l(d) = K(d / l) / l` is a 1D kernel integrating to one:
  gaussian `exp(-s^2 / 2) / sqrt(2 pi)` truncated at `s = 5`, epanechnikov `3/4 (1 - s^2)`, quartic
  `15/16 (1 - s^2)^2`, with `s = |d| / l`.
- `es(u)` and `et(t)` are the kernel masses inside the spatial window (`parameters.window`, the
  grid rectangle by default) and the time window (`parameters.timeWindow`, first to last slice time
  by default) under `'uniform'` correction, and 1 under `'none'`. Points outside a corrected window
  are excluded and counted in `status.invalidCount`; voxels whose center or slice time is outside a
  corrected window are NaN. This is the `sedge` / `tedge = "uniform"` of sparr.
- `'probability'` divides by the sum of included weights. With uniform edge correction the
  integral over the observation window depends on event locations. For spatial mass conservation,
  fixed `GPUKernelDensity` offers event-wise `'diggle'` correction, allowing for truncation and
  raster quadrature; the space-time contributors currently offer only `'none'` and `'uniform'`.

Both axes accept `'quartic'` and its old synonym `'bisquare'`, with the same bandwidth convention
and normalization on that axis. Spatial quartic is `3/pi (1-r²)²` per unit area; temporal quartic
is `15/16 (1-s²)²` per time unit. An unrecognized kernel name throws at construction, including
through `GPUSpaceTimeRelativeRisk`, and the host kernel/overlap helpers reject it too.

### Encoded coordinates, time slices and overlap

All three density packers validate the float32 values before writing the destination. Grid X and
Y endpoints must remain finite, retain their direction, and yield finite nonzero cell steps.
Bandwidths must remain finite and positive, and explicit spatial and temporal windows must remain
strictly ordered after encoding. A positive binary64 value such as `Number.MIN_VALUE` is too small
for a float32 bandwidth. A grid at projected origin `[500000, 6000000]` with `cellSize: 0.1` can lose
its Y extent. Subtract one common origin from positions, grid placement, index bounds and spatial
window before converting any of them to float32; subtracting it after conversion cannot restore
the lost extent. Errors identify the axis and encoded endpoints.

The slice expression uses encoded controls and explicit float32 `fma(slice, interval, first)`, with
one rounding after the multiply-add, including on Metal. The
default temporal window uses those same first and last slice values, shared by space-time density
and relative risk. An exponent-bin representability check avoids walking every slice while still
locating the first adjacent time that does not advance. For example, eight slices with first `.1`
and interval `.05` end at `0.45000001788139343`, including that final voxel under uniform correction.
For example, `firstSliceTime: 0.7, sliceInterval: 1e-9, sliceCount: 3` repeats its first encoded
instant; move the time origin to `0.7` and use `firstSliceTime: 0` with correspondingly recentered
event times and explicit window. A single slice still defaults to a zero-duration window at its
encoded instant: give it an explicit window when requesting uniform temporal correction.

`getGPUKernelDensityWindowOverlap(kernel, bandwidth, point, window)` and
`getGPUSpaceTimeTemporalWindowOverlap(kernel, bandwidth, time, window)` return dimensionless kernel
overlap, including when the query lies outside the window. Gaussian overlap uses stable narrow
interval integration and survival differences in the tails. Compact support caps are evaluated in
remaining support width rather than by subtracting nearly equal cumulative masses. Partially
clipped chords split at rectangle-edge crossings; centered compact temporal intervals use their
centered polynomial instead of subtracting two half-masses. Gaussian shader overlaps retain the
original encoded window width before translating either endpoint to the query origin. The density
estimator separately rejects a correction mass at or below
`1e-6`; the overlap helpers return the smaller positive mass itself.

### Fixed-density bandwidth and downstream rasters

For fixed `GPUKernelDensity`, `bandwidthPopulation: 'all-finite'` is the default and preserves the
existing automatic-bandwidth population: finite positions with finite nonnegative weights,
including events outside the index domain and active correction window. Set
`bandwidthPopulation: 'included'` to apply those same inclusion filters used by the density.
With `bandwidthRule`, optional `output.bandwidthPopulationCount` is one uint32 live row count in
the chosen population, including zero-weight rows; `output.bandwidth` is one float32 bandwidth in
position units. `output.weightSum` retains the original included weight units under either
population. Weight moments use a common power-of-two scale internally, so switching equal weight
units between `1`, `1e20` and `1e-24` preserves Scott/Silverman bandwidth. The triangle
`(0,0), (1,0), (0,1)` has Gaussian bandwidth `0.4807498568`; adding excluded `(1000,1000)` changes
only the all-finite population's bandwidth to `396.718156`. Compact support bandwidths multiply
the Gaussian-scale rule by `2.40187391` for Epanechnikov and `2.77919324` for quartic/bisquare.

A fixed density's `outputBand.storage.values` borrows exactly `grid.width * grid.height` rows,
preserving the allocation's byte offset without copying. Pass this **same view** as `values` to
both `GPUClassBreaks` and `GPUColorScale`; using the larger output allocation can include spare
pixels in the legend. Rows beyond the grid prefix are unspecified.

`GPUPointDensity.limits` and `GPUKernelDensity.limits` expose source-backed numeric route metadata;
their static `getExecutionPlan(shape, deviceLimits)` methods inspect rows, grid dimensions, chunk
bytes, scratch bytes and command budgets without allocating GPU resources. Point density defaults
to deterministic workgroup sums; atomic sums require explicit selection and have order-dependent
float32 rounding. Gaussian KDE auto mode selects binned evaluation only above 32 support cells
and with at least 2.5 bandwidths of padding; otherwise it gathers exactly. The planner and
parallel shader gate compare single-rounded products `5*h > 32*minimumCell` and
`5*h <= 2*margin*minimumCell`, avoiding reciprocal-dependent decisions at the boundary. A frame
with zero/nonfinite encoded cell area is invalid. Plans include the packed mask binding, requested
sum/mean scratch and weight chunk topology; count-only outputs do not allocate sum scratch. The binned error bound
requires at least three cells per bandwidth. Supply encoded `bandwidth` and `cellSize` in the
KDE shape to inspect one frame's route; without them the planner reports `'auto'`. Construction
and graph declaration additionally validate port formats, layouts and graph ownership.
Both density classes cap total input rows at 4,294,967,295. Point density's weighted
workgroup route, when sum/mean results are requested, caps the whole grid at 11,184,810 cells; larger weighted grids require explicit
atomic sums. KDE's binned and Gaussian auto declarations cap the entire padded grid at
11,184,810 cells. Both limits reserve three words per cell within 128 MiB; actual scratch and
each input chunk must also fit the device binding and buffer byte limits. Point density and the
binned KDE route cap their estimated command budget at 32,768 across all chunks and tiles.

The output is cell-major, row `(y * width + x) * sliceCount + slice`, the layout
`GPUCubeLayoutTranspose` calls `'cell-major'`.

{/* lens:GPUSpaceTimeKernelDensity */}

### How it runs

One thread per voxel over a grid index with a time-sorted run per cell. Rows are stably sorted by
time (radix sort of the order-preserving float key; ignored rows sort last), copied in that order
into a `GPUGridIndex`, and the row order inside every index cell is restored with
`createGridCellRowOrderNodes`, which leaves every cell a run in ascending time. A voxel visits the
index cells under its spatial support in a fixed order, binary-searches each run for the first
point inside the temporal support and walks the run until its time leaves the support. There are no
float atomics, so the result is deterministic for a device.

Cost per voxel is the cells under the spatial support times a logarithm, plus the points inside the
space-time cylinder. The number of points per cell is not capped; a very dense cell costs its
points inside the window for every voxel. The voxel count is limited by the 4-byte output binding
(33554432 on a default device) and the row count by the 16-byte cell-ordered point table (8388608).

Per-frame: `parameters` (both bandwidths, grid placement, both windows, slice times) and every input
buffer. Compile-time: grid size, `sliceCount`, point count, `indexGridSize`, `indexBounds`, both
kernels, both edge corrections, `normalization` and the optional outputs.

### Not implemented

Diggle-style per-point edge reweighting (`GPUKernelDensity` has it, the space-time density does
not), adaptive bandwidths, the bandwidth selectors (`OS`, `NS`, `LSCV`, `LIK`, `BOOT`),
tolerance contours, and a polygon study window. Choose both bandwidths and the windows explicitly.
The sparr convention for the temporal edge window of the default (`tlim`) was not verified; this
operator takes the first and last slice time unless `timeWindow` is given. With one slice that
default is a single instant, which holds no kernel mass: under `temporalEdgeCorrection: 'uniform'`
every voxel is then NaN, so pass `timeWindow` for a single slice (or use `'none'`).

{/* parameters: GPUSpaceTimeKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeKernelDensity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpaceTimeKernelDensity`.

- Packer: `getGPUSpaceTimeKernelDensityParameterValues(settings, grid, sliceCount)`, writes `GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH` (14) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid, as for `getGPUKernelDensityParameterValues`.
  - `sliceCount` (`number`): Number of slices (a positive integer), to default the time window.

Settings of `getGPUSpaceTimeKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU computes it) when `GPUKernelDensityProps.bandwidthRule` is set; pass any positive value. |
| `window` (optional) | `GPUKernelDensityWindow \| 'study-window'` | Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells. |
| `temporalBandwidth` | `number` | Temporal kernel bandwidth, positive and finite, in the unit of the relative `times`. |
| `firstSliceTime` | `number` | Time of slice 0 in the unit and origin of `times`. Slice `k` uses one float32 fused multiply-add, `fma(float32(k), sliceInterval, firstSliceTime)`. |
| `sliceInterval` | `number` | Time between slices, positive and finite. |
| `timeWindow` (optional) | `readonly [number, number]` | `[start, end]` of the temporal edge correction window, `start < end`. Defaults to the first and last slice time. Ignored without `temporalEdgeCorrection: 'uniform'`. With one slice the default is the instant of that slice, which holds no kernel mass: under the uniform temporal correction every voxel is then NaN, so give one slice an explicit window. |

{/* /parameters */}

## Space-time relative risk: `GPUSpaceTimeRelativeRisk`

`GPUSpaceTimeRelativeRisk` builds the case and control densities as two `GPUSpaceTimeKernelDensity`
pipelines with `normalization: 'probability'`, the same bandwidths, kernels, windows, grid and
index, and writes one ratio kernel:

- `scale: 'log'` (default, sparr `log = TRUE`): `log(f / g)`, `-Infinity` where the case density is
  zero.
- `scale: 'ratio'`: `f / g`.
- NaN where either density is NaN or the control density is not positive.

Under equal distributions the risk is 1 (log 0). The risk `output.values` must hold at least `grid.width * grid.height * sliceCount` rows,
including when both intermediate density outputs are supplied; construction rejects a short view.
Optional outputs `caseDensity` and `controlDensity`
expose the two densities, and `status.caseInvalidCount` / `controlInvalidCount` the excluded rows.

Not implemented: time-static controls (a spatial-only control density with the conditional case
density), asymptotic tolerance contours and p-values (they need the asymptotic variance of Fernando
and Hazelton, which was not verified), and separate case and control bandwidths.

{/* lens:GPUSpaceTimeRelativeRisk */}

{/* parameters: GPUSpaceTimeRelativeRisk generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeRelativeRisk` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpaceTimeRelativeRisk`.

- Packer: `getGPUSpaceTimeKernelDensityParameterValues(settings, grid, sliceCount)`, writes `GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH` (14) `float32` values.
  - Input port: `parameters`.
  - `grid` (`GPURasterMetadata`): The output grid, as for `getGPUKernelDensityParameterValues`.
  - `sliceCount` (`number`): Number of slices (a positive integer), to default the time window.

Settings of `getGPUSpaceTimeKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU computes it) when `GPUKernelDensityProps.bandwidthRule` is set; pass any positive value. |
| `window` (optional) | `GPUKernelDensityWindow \| 'study-window'` | Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells. |
| `temporalBandwidth` | `number` | Temporal kernel bandwidth, positive and finite, in the unit of the relative `times`. |
| `firstSliceTime` | `number` | Time of slice 0 in the unit and origin of `times`. Slice `k` uses one float32 fused multiply-add, `fma(float32(k), sliceInterval, firstSliceTime)`. |
| `sliceInterval` | `number` | Time between slices, positive and finite. |
| `timeWindow` (optional) | `readonly [number, number]` | `[start, end]` of the temporal edge correction window, `start < end`. Defaults to the first and last slice time. Ignored without `temporalEdgeCorrection: 'uniform'`. With one slice the default is the instant of that slice, which holds no kernel mass: under the uniform temporal correction every voxel is then NaN, so give one slice an explicit window. |

{/* /parameters */}

## Space-time K, pair correlation, local statistics: `GPUSpaceTimeKFunction`

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPUSpaceTimeKFunctionParameterValues, GPUSpaceTimeKFunction} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

declare const graph: GPUCommandGraph;
declare const parameterBuffer: GPUParameterBuffer<'float32'>;
const parameters = parameterBuffer.importToGraph(graph);
declare const positions: GraphDataView<'float32x2'>;
declare const times: GraphDataView<'float32'>;
declare const mask: GraphDataView<'uint32'>;
declare const k: GraphDataView<'float32'>;
declare const spatialK: GraphDataView<'float32'>;
declare const temporalK: GraphDataView<'float32'>;
declare const d: GraphDataView<'float32'>;
declare const pairCorrelation: GraphDataView<'float32'>;
declare const local: GraphDataView<'float32'>;
declare const pairCounts: GraphDataView<'uint32'>;
declare const statusView: GraphDataView<'uint32'>;

const parameterValues = getGPUSpaceTimeKFunctionParameterValues({
  bounds: [0, 0, 10_000, 8_000],
  maximumDistance: 1_500,
  maximumTimeLag: 14,
  timeWindow: [0, 365],
  spatialBandwidth: 150, // pair correlation only
  temporalBandwidth: 2
});
parameterBuffer.write(parameterValues); // a GPUParameterBuffer, imported as `parameters`
graph.add(
  new GPUSpaceTimeKFunction({
    positions,
    times,
    // Homogeneous estimator: omit intensities when requesting marginals and d.
    mask, // optional
    parameters, // GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH (11) values
    gridSize: [64, 64],
    distanceClassCount: 8,
    timeClassCount: 7,
    edgeCorrection: 'isotropic', // or 'none'
    output: {k, spatialK, temporalK, d, pairCorrelation, pairCounts, local, status: statusView}
  })
);
```

Notation: `W` the rectangle `bounds` (area `|W|`), `T = [timeStart, timeEnd]` (length `|T|`), `n` the
included events, ordered pairs `i != j`, `d_ij` the distance and `t_ij = |t_i - t_j|`.

- Included events have a nonzero mask, a finite position inside `W`, a finite time inside `T`, and
  with intensities a finite positive one.
- `getGPUSpaceTimeKFunctionParameterValues` checks the settings after rounding to float32, the
  values the shaders read. It throws when the bounds or the time window lose their strict order,
  when a value overflows, or when a positive maximum, bandwidth or search radius rounds to 0. For
  example, bounds `[1e7, 0, 1e7 + 0.25, 1]` pack to a zero X extent. Recenter such coordinates and
  times on a local origin. On this space-time K route, the derived float32 width, height,
  duration, area (`width * height`) and space-time volume (`area * duration`) must each remain
  finite and normal, at least `2^-126`. These checks apply after each subtraction and product
  is rounded to float32. Individually finite, ordered endpoints can still produce an overflowing
  difference, or an area or volume that underflows, becomes subnormal or overflows. Rescale
  coordinates or time units so every derived quantity meets that limit; usable endpoints alone
  do not ensure usable normalizers.
- Grid: `u_a = maximumDistance (a + 1) / distanceClassCount`, `v_b = maximumTimeLag (b + 1) /
  timeClassCount`. A pair is in distance class `a` when `d / maximumDistance * U` is in `(a, a +
  1]` (distance 0 is class 0), the right-closed convention of `GPURipley`, and likewise for lags.
  All outputs are distance-major, row `a * timeClassCount + b`.
- Exact ties follow stpp 2.0-8 (`src/stikfunction.f`, read from the CRAN source): a pair at
  exactly the largest distance or lag is counted (`hij <= s`, `tij <= t`), and a lag that exactly
  reaches a limit of the time window keeps temporal weight 1 (the window is closed). splancs
  decides both the other way and `GPUSpaceTimeDiggleTest` follows splancs, so on times recorded
  as whole days the two classes can return different `K`, `K_s`, `K_t` and `D`.
- `K(u, v) = |W||T| / (n (n - 1)) sum 1(d_ij <= u) 1(t_ij <= v) w_ij` (homogeneous). With
  `intensities`, `lambda'_i = lambda_i / (n / (|W||T|))` and `K = |W||T| / n^2 sum 1(...) w_ij /
  (lambda'_i lambda'_j)`, which is `(1 / (|W||T|)) sum 1(...) w_ij / (lambda_i lambda_j)`. The
  homogeneous `n (n - 1)` and the inhomogeneous `n^2` differ by the factor `(n - 1) / n`; whether
  stpp's homogeneous default uses `n (n - 1)` was not verified.
- Spatial marginal `K_s(u) = |W| / (n (n - 1)) sum 1(d_ij <= u) w^s_ij` over all lags, temporal
  marginal `K_t(v) = |T| / (n (n - 1)) sum 1(t_ij <= v) w^t_ij` over all distances, and Diggle
  `D(u, v) = K(u, v) - K_s(u) K_t(v)`. Homogeneous only.
- Pair correlation `g(u, v) = |W||T| / (n (n - 1)) / (4 pi u) sum ks(u - d_ij) kt(v - t_ij) w_ij`
  at the same grid (with the `n^2` weights when inhomogeneous), `ks` and `kt` normalised
  `'box'` (default, stpp's) or `'epanechnikov'` kernels of half widths `spatialBandwidth` and
  `temporalBandwidth`.
- Local statistics (LISTA) `K_i(u, v) = n s sum_{j != i} 1(...) w_ij / (lambda'_i lambda'_j)` with
  `s` the global scale, so the mean of `K_i` over the included events is `K`. Output row `event *
  bins + a * timeClassCount + b`, NaN for excluded rows, at most 64 classes.

### Edge corrections implemented

| `edgeCorrection` | Spatial weight `w^s_ij` | Temporal weight `w^t_ij` |
| --- | --- | --- |
| `'none'` | 1 | 1 |
| `'isotropic'` (default) | Ripley (1977): the reciprocal of the fraction of the circle about `x_i` through `x_j` that lies inside `W`, exact for the rectangle, floored at 1/100 as in `GPURipley` | 1 when both `t_i - t_ij` and `t_i + t_ij` lie in `T`, otherwise 2 (the one-dimensional Ripley weight) |

Not implemented: stpp's `border`, `modified.border` and `translate` corrections, polygon study
windows (a rectangle only), and `infectious = TRUE` (pairs ordered in time). Whether stpp's exact
temporal weight for `isotropic` equals the one-dimensional Ripley weight above was not verified
against the package; the fixture will decide.

### How it runs

The row filter, the 3 x 3 lattice neighborhood of the shared pair histogram and the temporal
marginal. Pair counts are accumulated as exact multiword integers within accumulator capacity;
the published `uint32` pair-count port saturates at 4,294,967,295 (`2^32 - 1`). For example,
65,537 events in one class have 4,295,032,832 ordered nonself pairs, so that port returns the cap.
The temporal marginal starts from integer counts; its normalized estimates are float32.
Weighted sums use fixed point in the deterministic pair histogram; results are bitwise reproducible. Homogeneous terms use
2^-16 units. Inhomogeneous terms (K and the pair correlation with `intensities`) use a per-frame
power-of-two scale `S` set from the smallest included `1 / lambda'`. For untapered
inhomogeneous **K** pair terms (edge weights at least 1), an unclamped scale keeps the smallest
term at least 2^16 accumulator units: integer quantization contributes at most 2^-17 relative
error, in addition to float32 arithmetic error. The supported **K scale exponent** is `[-100, 100]`.
The scale is chosen from `e = floor(log2(minimum inverse intensity))`, with exponent `16 - 2e`.
Admission is checked for **every included row**, separate from the minimum used to choose that scale:
its inverse relative intensity must be positive and finite and its own exponent `e_i` must satisfy
`16 - 2e_i` in `[-100, 100]` (`e_i` in `[-42, 58]`). Any row failing that condition reports loss
and suppresses dependent estimates. For example, inverse intensities `[1, 2^59]` have a supported
minimum-derived exponent of 16, but the second row requires -102 and makes the frame incomplete.

Pair-correlation terms also multiply the spatial and temporal kernel tapers. A positive tapered
term may be arbitrarily small. PCF therefore uses an additional `2^24` scale: its effective scale
is `S_PCF = S * 2^24`. Its **absolute quantization** error in the small channel is at most
`0.5 / (S * 2^24)`, plus float32 rounding of the scaled term and addition of 0.5. The coarse
(large) channel quantizes in 4096-unit steps, with absolute error at most
`2048 / (S * 2^24)` plus float32 rounding. These are bounds per raw weighted contribution before
final output normalization, assuming no accumulator or term overflow. For example, at `S = 2^16`, these absolute bounds are `2^-41` (small) and `2^-29`
(coarse). They are not a relative error guarantee for arbitrarily small tapers.

A disappearing positive product (including a positive tapered product below half a scaled unit),
an unsupported intensity scale, or accumulator overflow marks the frame incomplete. Optional
`output.status` reports **3** for this loss on an otherwise usable frame. Dependent estimates are
suppressed as NaN, including K, pair correlation and local estimates, even when the optional status
view is omitted. Published pair counts remain exact only below their uint32 saturation cap,
including when a weighted estimate is suppressed. Count-port saturation alone does not set status 3;
weighted-accumulator overflow and unsupported/disappearing weighted terms do. An unusable frame
reports status 1 with precedence over loss.

The former fixed 2^-16 unit gave +6.8 % at a relative intensity `lambda'` of 100, +37 % at 300
and lost every term from 363. LISTA sums add
in float32 in the fixed neighbor order. The temporal marginal uses the sorted times and a binary
search per event and class, with the same class test as the pair pass, so the two agree on every
boundary; its cost is `n V log n`. The pair search radius is `maximumDistance`, plus
`spatialBandwidth` for the pair correlation.

Limits: at most 64 distance and 64 time classes, and `(classes + distanceClassCount) * channels <=
2048` accumulators (3 channels, 5 with the pair correlation); 8388608 rows; local output at most 64
classes and `rows * classes` at most 33554432. Per-frame: window, maximum distance and lag, time
window, bandwidths and the buffers. Compile-time: row count, `gridSize`, class counts,
`edgeCorrection`, kernel and the optional views.

{/* parameters: GPUSpaceTimeKFunction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeKFunction` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUSpaceTimeKFunction`.

- Packer: `getGPUSpaceTimeKFunctionParameterValues(settings)`, writes `GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH` (11) `float32` values.

Settings of `getGPUSpaceTimeKFunctionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Spatial study window `[minX, minY, maxX, maxY]`, a rectangle with positive area. Rows outside are excluded. |
| `maximumDistance` | `number` | Largest distance `u` of the K grid, in position units. Class `a` has radius `u * (a + 1) / distanceClassCount`. |
| `maximumTimeLag` | `number` | Largest time lag `v` of the K grid, in the unit of `times`. Class `b` has lag `v * (b + 1) / timeClassCount`. |
| `timeWindow` | `readonly [number, number]` | Observation interval `[start, end]` of the times, `start < end`. Rows whose time is outside it are excluded. Its length is `\|T\|`. |
| `spatialBandwidth` (optional) | `number` | Spatial bandwidth `hs` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. |
| `temporalBandwidth` (optional) | `number` | Temporal bandwidth `ht` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. |

{/* /parameters */}

## Diggle test of space-time interaction: `GPUSpaceTimeDiggleTest`

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSpaceTimeDiggleTest} from '@luma-contributors/analysis/dataframe-analysis';

declare const graph: GPUCommandGraph;
declare const positions: GraphDataView<'float32x2'>;
declare const times: GraphDataView<'float32'>;
declare const mask: GraphDataView<'uint32'>;
declare const parameters: GraphDataView<'float32'>;
declare const randomParameters: GraphDataView<'uint32'>;
declare const d: GraphDataView<'float32'>;
declare const statistics: GraphDataView<'float32'>;
declare const summary: GraphDataView<'float32'>;
declare const permutationMean: GraphDataView<'float32'>;
declare const permutationStandardDeviation: GraphDataView<'float32'>;

graph.add(
  new GPUSpaceTimeDiggleTest({
    positions, times, mask,
    parameters, // as for GPUSpaceTimeKFunction (bandwidths are ignored)
    randomParameters, // getGPUSpaceTimeDiggleRandomParameterValues(seed)
    gridSize: [64, 64],
    distanceClassCount: 8,
    timeClassCount: 7,
    permutationCount: 99,
    output: {d, statistics, summary, permutationMean, permutationStandardDeviation}
  })
);
```

Under random relabelling of the times over the locations, `K_s` and `K_t` do not change, so only `K`
is recomputed. Each permutation is a swap-or-not shuffle (the shuffle of `GPUKnoxTest`, on its own
Philox stream) of the included events, ranked in row order, so the permutations depend on the seed
and the rows only, not on `gridSize`. The statistic is `U = sum D(u_a, v_b)`, the plain sum of `D`
over every class, for the observed pattern and for each permutation: the statistic of splancs
`stmctest` (`sum(kst - outer(ks, kt))`) and of surveillance `stKtest`. It is not standardised, so
classes with a large `K` weigh more; splancs divides by the `stsecal` standard error only for the
residual plot of `stdiagn`. The pseudo p-value for interaction is `(1 + #{U_p >= U}) / (P + 1)`
(`summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]`), with the regularity and two-sided
p-values and the z-score beside it. The optional outputs `permutationMean` and
`permutationStandardDeviation` are the per-class moments of `D` over the permutations, for a
standardised residual `(d - permutationMean) / permutationStandardDeviation`; they are diagnostics
and not part of `U`. (Until the math review of 2026-11-10 `U` was the sum of those standardised
residuals, which no reference computes.)

Agreement with splancs: `K_s`, `K_t`, `K`, `D` and `U` equal a NumPy port of the splancs 2.01-45
source (`stkhat.f`, `stmctest.S`) to about 2e-6 of the largest `K`, for the observed pattern and
six relabellings of a 72-event scene, and the null distribution of 199 permutations has the
spread of the port's. The fixture is a port of the source, not a run of R; the port's edge weight
is checked against Shapely and its `stsecal` variance against all 720 relabellings of six events.

Exact ties follow splancs `stkhat.f`, because times recorded as whole days make them ordinary: a
lag that exactly reaches a limit of the time window has temporal weight 2; a pair at exactly
`maximumDistance` or `maximumTimeLag` is dropped (the last class is open above, the others are
closed above); and the class is found with products, so a whole-number lag on a whole-number
break stays in the class the break closes. The fixture has a scene of 60 events on days 0 to 30
with three lag grids (8.5 in 5 classes, 7 in 7, 13 in 13), where these rules act on 128 to 201
ordered pairs at the time limits and up to 166 at the largest lag; the kernel equals the port on
it for the observed pattern, three relabellings and its own permutations. `GPUSpaceTimeKFunction`
follows stpp at the same ties, so its outputs can differ from the `k`, `spatialK`, `temporalK`
and `d` of this class on such data (the review measured `U` 43.95 against 52.26 on 60 whole-day
events).

Differences from splancs that remain: the spatial weight is capped at 100 here; at a distance
exactly equal to the largest, `stkhat.f` drops the pair from `ks` but adds a stale term to `kst`,
which is not reproduced (the pair is dropped from both). splancs also takes a polygon window and
unequal class breaks, and has an analytic standard error (`stsecal`); this class has none of the
three.

Permutations run in batches of `floor(1024 / classes)` (and as many as the time table of `batch *
rows` floats fits one binding): a visited pair adds the weight of every permutation of the batch
into its own `(permutation, distance, lag)` slot, so the sums are exact integers. The command node
count grows with `ceil(P / batch)` and is exposed as `estimatedCommandNodeCount`, checked against
the node budget. At most 999 permutations.

Not implemented: the inhomogeneous variant, the other stpp corrections, sequential stopping, and a
permutation test of the local statistics (the local statistics are observed values only).

{/* parameters: GPUSpaceTimeDiggleTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeDiggleTest` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUSpaceTimeDiggleTest`.

- Packer: `getGPUSpaceTimeDiggleRandomParameterValues(seed)`, writes `GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH` (2) `uint32` values.
  - Input port: `randomParameters`.
  - `seed` (`number`): Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and the inputs.
- Packer: `getGPUSpaceTimeKFunctionParameterValues(settings)`, writes `GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH` (11) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeKFunctionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Spatial study window `[minX, minY, maxX, maxY]`, a rectangle with positive area. Rows outside are excluded. |
| `maximumDistance` | `number` | Largest distance `u` of the K grid, in position units. Class `a` has radius `u * (a + 1) / distanceClassCount`. |
| `maximumTimeLag` | `number` | Largest time lag `v` of the K grid, in the unit of `times`. Class `b` has lag `v * (b + 1) / timeClassCount`. |
| `timeWindow` | `readonly [number, number]` | Observation interval `[start, end]` of the times, `start < end`. Rows whose time is outside it are excluded. Its length is `\|T\|`. |
| `spatialBandwidth` (optional) | `number` | Spatial bandwidth `hs` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. |
| `temporalBandwidth` (optional) | `number` | Temporal bandwidth `ht` of the pair correlation kernel, positive. Needed with `output.pairCorrelation`. |

{/* /parameters */}

## Checks

- `modules/analysis/test/gpu-spatial-analysis/point-density/space-time-kernel-density.spec.ts`
  (GPU against the binary64 reference, row-order invariance, bitwise reproducibility, relative risk)
  and `.node.spec.ts` (kernels, window mass, parameters, validation, WGSL audit).
- `modules/analysis/test/gpu-dataframe/pair-statistics/space-time-k-function.spec.ts` (every
  output, both corrections, intensities, both pair correlation kernels, class-edge ties, row-order
  invariance), `space-time-diggle-test.spec.ts` (statistics, p-values, three batch shapes, seeds),
  `space-time-diggle-splancs.spec.ts` (the kernel against the port of the splancs source: `K_s`,
  `K_t`, `K`, `D`, the `stmctest` statistic, the null distribution, the tie convention),
  `space-time-diggle-splancs.node.spec.ts` (the binary64 K reference against the same port) and
  `space-time-k-function.node.spec.ts` (validation, WGSL audit, node count against the estimate).
- Kernel-audit cases and refusals in
  `modules/analysis/test/analysis-core/kernel-audit/cases-stx04-stx05-space-time-density-and-k.ts`.
