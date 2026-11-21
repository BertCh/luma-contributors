# Flow-map and particle diagnostics

Import from `@luma-contributors/analysis/raster-analysis`.

Particles integrated through a time-varying vector field, with FTLE, raster ridge extraction,
connectivity and residence time. Planar inputs use the field's own distance units. Geographic
inputs are projected internally through published luma.gl projection programs. Tests include
NumPy-generated ridge and geographic-flow fixtures; OceanParcels was not run.

## Choose a flow diagnostic

- [Advect persistent particles](./operations-analysis.md#gpuparticleadvection) for frame-by-frame trajectories and trails.
- [Integrate a two-slice flow map](#gpulagrangianflowmap) for fixed-duration endpoints and residence times.
- [Compute FTLE](#gpufinitetimelyapunovexponent) from a regular seed lattice; extract [scalar ridges](#gpuscalarfieldridges).
- [Count region connectivity](#connectivity-matrix) using start/end region classes.
- [Inspect flow recordings](./expansion-lenses.md#flow-diagnostics-recordings) and their operation contracts.

## `GPULagrangianFlowMap`

Takes one second-order Runge-Kutta (midpoint) step `p' = p + h v(p + h/2 v(p, t), t + h/2)`,
`stepCount` times, for every seed. The field has two slices, `velocitiesStart` valid at
`sliceStartTime` and `velocitiesEnd` at `sliceEndTime`; the velocity at time `t` is their linear
blend with weight `(t - sliceStartTime) / (sliceEndTime - sliceStartTime)` clamped to `[0, 1]`, so
the field is held constant outside the interval. To integrate a longer record, swap the slices and
slice times between encodings. A negative `timeStep` integrates backward. Spatial sampling is the
bilinear sampler of `GPUParticleAdvection`.

There is no respawn, no randomness and no atomic; a run is bitwise reproducible.

Every RK2 landing, including the last step, must have velocity support at the end time.
A landing inside the rectangle on a missing velocity cell is lost immediately: both position
components become NaN and `output.validity` is 0. A supported landing retains its coordinates
and validity 1. This rule also applies with signed cell sizes and backward time steps.

Set `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}` for geographic
fields. Seeds, `fieldPlacement` and final `output.positions` are longitude/latitude **degrees**.
The two velocity components are **Web Mercator projected metres per time unit**. The contributor
projects each seed into EPSG:3857, integrates there, and inversely projects each RK2 sampling
position before sampling the degree grid. Ground east/north velocities must be transformed to
that projected frame by the caller. There is no implicit geodesic motion, longitude wrapping or
pole clamping: projection-rejected positions are lost. Continuous longitudes must stay inside
[-180, 180] and latitudes inside the Web Mercator square-world domain (about ±85.05113°).
Absolute projected float32 positions lose metre-scale detail at continental offsets; this mode
is unsuitable for sub-metre movement there.

`spatialContext`, field dimensions, step count and optional outputs are compile-time. Input values
and the parameter view are per-frame. `output.validity` optionally writes one flag per seed,
including zero for projection rejection. Lost particles also retain the existing NaN positions.

`fieldPlacement.cellSize` is signed. With a positive `cellHeight` the field
slices place row 0 at the smallest y; a north-up field (row 0 at the top) is uploaded as is with
`originY` at its top edge and a negative `cellHeight`. `v` is always the velocity toward larger y.
The parameter packer rejects a zero or non-finite float32 cell size. The constructor validates
view layouts and dimensions, but no raster affine is inferred: supply the placement explicitly. The seed grid that `GPUFiniteTimeLyapunovExponent` reads needs
no reordering: output row `r` belongs to seed row `r`, and the exponent is the same whether seed
rows advance toward larger or smaller y.

| Prop | Meaning |
| --- | --- |
| `seeds` | `float32x2` start positions |
| `velocitiesStart`, `velocitiesEnd` | Packed `(u, v)` rows, `fieldWidth * fieldHeight`, row 0 at `originY` (the smallest y with a positive `cellHeight`) |
| `parameters` | Per-frame, `getGPULagrangianFlowMapParameterValues`: field extent, start time, signed time step, slice times |
| `stepCount` | Compile-time, 1 to 8192: the bounded per-thread loop |
| `regions` | Optional `uint32` region of each field cell, `GPU_LAGRANGIAN_NO_REGION` outside every region |
| `spatialContext` | Planar by default, or longitude-latitude with metric `none` |
| `output.validity` | Optional explicit flag: 1 for a valid final position, otherwise 0 |
| `output.positions` | Final positions. NaN for a particle that left the field or reached a NaN cell |
| `output.regionClasses` | Start regions then end regions, period-major (needs `regions`) |
| `output.residenceTimes` | Time in the start region before its first exit (needs `regions`) |

Residence time is `|timeStep|` times the number of steps that began inside the start region, so its
resolution is one step. A particle that never leaves reports `stepCount * |timeStep|` (censored); a
particle that leaves the field counts as having exited; a seed outside every region reports 0.

### Preserve local displacement and slice times

Use signed `fieldPlacement.cellSize` for the field. Here time is seconds and
planar distance is meters; positive x velocity means east even with a negative row cell size.
The packer subtracts the origins in binary64 before float32 conversion. Seeds must be uploaded
in that same local frame before narrowing, and outputs remain local.

```ts
import {getGPULagrangianFlowMapParameterValues} from '@luma-contributors/analysis/raster-analysis';

const localOrigin: readonly [number, number] = [10_000_000, 0];
const sliceStartTime = 1_700_000_000;
const sliceEndTime = sliceStartTime + 10;
const parameters = getGPULagrangianFlowMapParameterValues({
  fieldPlacement: {origin: [10_000_000, 100], cellSize: [100, -100]},
  localOrigin, timeOrigin: sliceStartTime,
  startTime: sliceStartTime, timeStep: 10, sliceStartTime, sliceEndTime
});
// Convert world seeds in binary64, THEN create the float32 upload.
const seedWorld: readonly [number, number] = [10_000_050, 50];
const seedLocal = Float32Array.of(seedWorld[0] - localOrigin[0], seedWorld[1] - localOrigin[1]);
// After GPU readback, reconstruct in binary64; retain local values for further GPU integration.
export function reconstructWorldPosition(position: Float32Array): readonly [number, number] {
  return [position[0] + localOrigin[0], position[1] + localOrigin[1]];
}
```

With a one-cell field, velocities `[0,0]` then `[2,0]`, and one 10-second midpoint step,
the local endpoint is `[60,50]`, validity 1, and reconstructed world position `[10000060,50]`.
For a constant 0.25 m/s field, four one-second steps retain a total 1 m displacement even at
world x=10,000,000. Keep `GPUParticleAdvection` persistent positions and trails local across
encodings too; its packer accepts `localOrigin` and `fieldPlacement` and rebases world
`spawnBounds`. `GPUStreamlines` likewise accepts those packer settings and publishes local points; its separate
`gridPlacement` defines the seed lattice and clipping rectangle in the same world frame.
For LIC, use its separate `fieldPlacement` and `outputPlacement` with signed cell sizes.

`timeOrigin: sliceStartTime` rebases `startTime`, `sliceStartTime` and `sliceEndTime` together;
`timeStep` and residence durations are unchanged. Distinct slices that collapse to the same
float32 instant are rejected with guidance to set `timeOrigin`. Intentionally equal source
slice times retain the constant-field convention. Local origins are for planar integration;
do not combine them
with the internal longitude-latitude projection route.

{/* parameters: GPULagrangianFlowMap generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPULagrangianFlowMap` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPULagrangianFlowMap`.

- Packer: `getGPULagrangianFlowMapParameterValues(settings)`, writes `GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPULagrangianFlowMapParameterValues`:

| Field | Type | Description |
|---|---|---|
| `localOrigin` (optional) | `readonly [number, number]` | Planar origin subtracted before float32 packing. Upload seeds as world minus this origin before narrowing; positions are local too. Add it in binary64 on readback. Do not use with longitude-latitude projection mode. Defaults to [0, 0]. |
| `fieldPlacement` | `FlowRasterPlacement` | Field raster placement with an outer-corner `origin` and `cellSize`, the placement of `GPUParticleAdvection`: cell `(i, j)` is centred at `origin + (i + 0.5, j + 0.5) * cellSize`. Cell sizes are signed: a positive cellHeight puts row 0 at the smallest y; a negative one (north-up, origin at the top edge) puts row 0 at the largest y. A cell size that is zero or not finite in float32 throws. |
| `timeOrigin` (optional) | `number` | Absolute time origin subtracted from startTime, sliceStartTime and sliceEndTime in binary64. Durations (timeStep and residenceTimes) are unchanged. For epoch timestamps, use timeOrigin: sliceStartTime. Distinct slices that still alias in float32 throw. Defaults to 0. |
| `startTime` | `number` | Time at which the particles are released, in the time unit of the field velocities. |
| `timeStep` | `number` | Signed time step. A negative step integrates backward in time, which is what backward FTLE and backward connectivity need. The integration time is `stepCount * \|timeStep\|`. |
| `sliceStartTime` | `number` | Time at which the field `velocitiesStart` is valid. |
| `sliceEndTime` | `number` | Time at which the field `velocitiesEnd` is valid. |

{/* /parameters */}

## `GPUFiniteTimeLyapunovExponent`

`FTLE = ln(sqrt(lambdaMax(J^T J))) / |T|`, where `J` is the gradient of the final positions with
respect to the seeds on a regular seed grid. Central differences are used in the interior and
one-sided differences on the border; `lambdaMax` is the closed-form larger eigenvalue of the 2 by 2
matrix. The result is NaN where a particle in the stencil is lost. Integrate with a negative time
step for the backward exponent.

For geographic final positions, pass the same longitude-latitude spatial context. The stencil
projects them before differencing, so the original seeds must form a **regular grid in projected
metres**, with `seedSpacingX/Y` in those metres. A regular degree grid does not meet that
requirement. Optional input `validity` rejects the center and every stencil that uses it;
optional `output.validity` identifies finite FTLE results.

Verified on a linear strain field `u = e x, v = -e y`, where the exponent is `e` everywhere, on a
one-directional stretch (forward `e`, backward 0), and on a strain rate that ramps linearly across
the two slices, where the exponent is the time mean of the rate (a frozen first slice gives 0).

{/* parameters: GPUFiniteTimeLyapunovExponent generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUFiniteTimeLyapunovExponent` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPUFiniteTimeLyapunovExponent`.

- Packer: `getGPUFiniteTimeLyapunovExponentParameterValues(settings)`, writes `GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUFiniteTimeLyapunovExponentParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seedSpacingX` | `number` | Distance between neighbouring seeds along x, in field units. |
| `seedSpacingY` | `number` | Distance between neighbouring seeds along y, in field units. |
| `integrationTime` | `number` | Integration time `T` of the flow map. The exponent is divided by `\|T\|`, so a backward flow map (negative time step) gives the backward FTLE, whose ridges are attracting structures. |

{/* /parameters */}

## `GPUScalarFieldRidges`

{/* lens:GPUScalarFieldRidges */}

::lens{scenario=gpu-raster.scalar-field-ridges rung=result caption="Quadratic field: the half-cell tie selects column 3; missing samples invalidate their stencils"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=formula caption="Negative Hessian curvature and a normal Newton step inside the half-open pixel box"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=datum datum=24 caption="Cell 24: column 3 owns the exact half-cell ridge tie"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=kernel caption="The central difference stencil, smallest eigenpair and subcell ridge test"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=graph caption="A fixed-size ridge mask with stencil validity and optional curvature strength"}

The synthetic lens uses `f = 20 - (column - 2.5)^2` on a 7 × 7 metre grid. Its normal curvature
is 2. The half-open pixel rule selects column 3 in interior rows 1 through 5. Missing cell 40
(column 5, row 5) invalidates every stencil touching it; border outputs are also zero.

Extracts a fixed-sized ridge mask from a scalar `GPURasterBand`, including Brownian-bridge
utilization densities, cumulative utilization volumes and FTLE. The grid must have orthogonal
pixel axes and at least 3 × 3 cells. Rotated uniform grids work; geographic and Web Mercator
ground-distance modes require axis-aligned grids. The parameter packer reads the grid affine,
so update the parameter view when the affine changes. Dimensions and `cellSizeMode` are
compile-time. Geographic grids require a longitude-latitude spatial context with metric `none`
and `cellSizeMode: 'geographic'`; otherwise the context is planar.

A central 3 × 3 difference stencil gives the gradient and Hessian in the orthogonal pixel basis.
The smallest Hessian eigenvalue defines downward curvature and its eigenvector the ridge normal.
Spacing quotients combine mantissas and integer exponents without forming a spacing product,
so a representable curvature survives even when the squared cell spacing would overflow.
The shared 2 × 2 eigenpair helper retains the smaller diagonal in its original units and subtracts
a rationalized Jacobi correction; only its bounded direction calculation is scaled. This avoids
half-trace cancellation and preserves a small normal negative eigenvalue beside a much larger
positive diagonal. Subnormal inputs and results may still be flushed to zero by the GPU.
A cell is selected when curvature is strictly positive and meets `minimumCurvature`, its calibrated
value meets `minimumValue`, and the normal Newton displacement falls inside its pixel box
[-0.5, 0.5) along both axes. Exact half-cell ties go to the larger column/row; equal Hessian
eigenvalues choose the column normal. This is a local raster approximation of the
[height-ridge criterion of Peikert and Sadlo (2008)](https://cgl.ethz.ch/Downloads/Publications/Papers/2008/Pei08/Pei08.pdf),
with no curve tracing, smoothing, or claim that every detected ridge is a material LCS.

`output.ridges` and `output.validity` are required `uint32` views, one row per cell.
`output.strength` optionally holds `max(0, -smallestEigenvalue)` at every valid stencil.
Borders and stencils touching missing or nonfinite samples write zero to every output.
Band calibration, no-data and validity are honoured. Flat plateaus and valleys have no ridges.
Geographic and Web Mercator modes use the existing row-local WGS84 ground scale; their Hessian
is a local tangent approximation. Outputs preserve the source cell index and never truncate,
so no compact-output capacity or overflow status is needed. There is no CPU operation twin for
these raster contributors.

{/* parameters: GPUScalarFieldRidges generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUScalarFieldRidges` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPUScalarFieldRidges`.

- Packer: `getGPUScalarFieldRidgesParameterValues(settings, grid)`, writes `GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): Grid affine; update these parameters when the affine changes.

Settings of `getGPUScalarFieldRidgesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumValue` (optional) | `number` | Lowest accepted calibrated field value. Defaults to 0. |
| `minimumCurvature` (optional) | `number` | Lowest accepted downward curvature, in field units per squared distance. Defaults to 0; flat cells are always rejected. |

{/* /parameters */}

## Connectivity matrix

Compose `GPULagrangianFlowMap` (with `regions` and `output.regionClasses`) with the spatial
contributor imported below. Set `periodCount: 2` and `classCount` equal to the number of regions.
Basic counts and probabilities accept **1 to 65,535 classes**, subject to dense buffer/device
bounds: each counts/probabilities plane has `classCount * classCount` rows (four bytes per row).
Optional transition diagnostics retain the **64-class** limit; request them only within that
bound. A large declared class count is not a promise that its dense matrices fit the device.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTransitionMatrix} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addFlowConnectivity(
  graph: GPUCommandGraph, regionClasses: GraphDataView<'uint32'>, classCount: number,
  counts: GraphDataView<'uint32'>, probabilities: GraphDataView<'float32'>,
  ignored: GraphDataView<'uint32'>
): void {
  graph.add(new GPUTransitionMatrix({
    classes: regionClasses, periodCount: 2, classCount,
    output: {counts, probabilities, ignored}
  }));
}
```

`regionClasses` is period-major (all start regions, then all end regions). `counts[from * K + to]`
is the connectivity matrix and `probabilities` is row-normalized. Particles ending outside every
region, including lost ones, are not counted; `output.ignored` counts these rejected transitions.
See the [transition-matrix contract](../gpu-spatial-analysis/operators-dynamics.md#gputransitionmatrix)
for diagnostic ports and device bounds.
