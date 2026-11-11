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
const grid = makeGPURasterMetadata({width: 256, height: 256, /* cellSize, origin */});
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
- `'probability'` divides by the sum of included weights.

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

#### `GPUSpaceTimeKernelDensity` parameters

- Packer: `getGPUSpaceTimeKernelDensityParameterValues(settings, grid, sliceCount)`, writes `GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH` (14) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid, as for `getGPUKernelDensityParameterValues`.
  - `sliceCount` (`number`): Number of slices (a positive integer), to default the time window.

Settings of `getGPUSpaceTimeKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU computes it) when `GPUKernelDensityProps.bandwidthRule` is set; pass any positive value. |
| `window` (optional) | `GPUKernelDensityWindow \| 'study-window'` | Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells. |
| `temporalBandwidth` | `number` | Temporal kernel bandwidth, positive and finite, in the unit of the relative `times`. |
| `firstSliceTime` | `number` | Time of slice 0 in the unit and origin of `times`. Slice `k` is evaluated at `firstSliceTime + k * sliceInterval`. |
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

Under equal distributions the risk is 1 (log 0). Optional outputs `caseDensity` and `controlDensity`
expose the two densities, and `status.caseInvalidCount` / `controlInvalidCount` the excluded rows.

Not implemented: time-static controls (a spatial-only control density with the conditional case
density), asymptotic tolerance contours and p-values (they need the asymptotic variance of Fernando
and Hazelton, which was not verified), and separate case and control bandwidths.

{/* lens:GPUSpaceTimeRelativeRisk */}

## Space-time K, pair correlation, local statistics: `GPUSpaceTimeKFunction`

```ts
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
    intensities, // optional: inhomogeneous estimator
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
marginal. Pair counts and the temporal marginal are exact integers; weighted sums are fixed point
(2^-16 units) in the deterministic pair histogram; results are bitwise reproducible. LISTA sums add
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

#### `GPUSpaceTimeKFunction` parameters

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
