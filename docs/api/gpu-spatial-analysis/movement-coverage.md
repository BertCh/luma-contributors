# Trajectory estimation, restriction and search

These executable examples use small projected Cartesian trajectories and relative times. Full
parameter, status and limitation contracts are in
[Bounded network optimization and trajectory analysis](../gpu-network/network-movement-coverage.md).
The examples declare real GPU work and capture caller-visible results; they are not performance
benchmarks or claims of complete upstream-library parity.

## Constant-velocity filtering and smoothing

{/* lens:GPUTrajectoryKalman */}

::lens{scenario=gpu-spatial-analysis.trajectory-kalman rung=graph}

Two observations at x=0 and x=2 give a forward estimate x=4/3 at the second observation.
Backward RTS smoothing revises the first estimate to x=2/3. The record preserves raw observations,
filtered states, smoothed states, both covariance arrays and status independently. Position and
velocity covariance are shared by the two independent isotropic axes.

RTS smoothing sets status bit `8` when the predicted covariance is singular or when the backward
correction produces a nonfinite state **or covariance**, including covariance-only overflow after a
successful inversion. For that row, the smoother retains both the filtered state and the filtered
covariance; it publishes neither part of the rejected correction. The next backward step consumes
that filtered fallback. This applies to both isotropic and full covariance and to the operation
forms.

Filtering retains position high parts and small residuals through prediction, measurement updates
and smoothing, then publishes absolute positions. This preserves small velocity increments at
projected coordinates and small measured positions after movement far from the first fix. Velocity
and covariance retain their existing units; segment resets and RTS fallback retain their meanings.

`prepareGPUTrajectoryObservations` returns planar metre positions and elapsed **seconds** in
`times`. Its `timesMetadata` retains the exact source `epochTicks` and `epochUnitsPerSecond`;
the legacy top-level `epoch` and `unitsPerSecond` also describe the source ticks.
`timeWords` contains absolute source ticks, with `timeWordsTemporalContext` supplied for second,
millisecond, microsecond and nanosecond rates. `timesTemporalContext` is available when the source
epoch is an integral second. A fractional-second origin remains exact in `timesMetadata` even
though it cannot be expressed as an integer-second temporal context. `spatialContext` uses
`units: 'meters'`, and the projection origin is a frozen copy of the supplied longitude/latitude
degrees. Use the seconds stream directly; supply the word context when using `timeWords`.
The spatial frame identity includes the WGS84 tangent-plane projection and copied origin: equal
origins compose, while different origins fail coordinate compatibility.

## Closed time and rectangle restriction

{/* lens:GPUTrajectoryRestriction */}

::lens{scenario=gpu-spatial-analysis.trajectory-restriction rung=graph}

A segment from x=-2 to x=2 over four seconds crosses the closed rectangle from time one to three.
Its ever relationship is true and always relationship false. A separate observation at time twenty
has no common temporal support and therefore has unknown relationships, represented by 0xffffffff.
Output intervals retain source-segment rows instead of hiding absent support through compaction.

The pure CPU restriction function accepts the same `timestamps`, `trackOffsets` pair as the graph:

```ts
import {computeTrajectoryRestrictionOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';

const restricted = computeTrajectoryRestrictionOnCPU({
  positions: Float32Array.of(-2, 0, 2, 0), // planar metres
  timestamps: Float32Array.of(0, 4), // elapsed seconds
  trackOffsets: Uint32Array.of(0, 2), // observation row indices
  parameters: Float32Array.of(0, 4, -1, -1, 1, 1, 0) // time window, rectangle, no gap limit
});
```

The returned interval endpoints use the
timestamp unit and output positions use the input position unit.

## Exact batched similarity search

{/* lens:GPUTrackSimilaritySearch */}

::lens{scenario=gpu-spatial-analysis.track-similarity-search rung=graph}

The first candidate exactly matches the query. The second has endpoints ten units away and is
safely pruned before the existing DTW program runs. Captured lower bounds, distances, match flags
and status distinguish the evaluated zero distance from the pruned NaN distance. The contributor
also supports discrete Fréchet and normalized LCSS, with metric-specific admissible bounds.

## Operation forms

These public classes take logical graph-value names in place of buffer views. Declare lengths
and capacities at construction, then compile GPU stages with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`. The CPU classes use the corresponding input/output names;
their binary64 reference calculations need not reproduce GPU rounding or bounded searches.

| GPU operation | CPU operation | Inputs and status outputs |
| --- | --- | --- |
| `GPUTrajectoryKalmanOperation` | `CPUTrajectoryKalmanOperation` | Planar `float32x2` observations, track offsets, relative `float32` or signed Int64 `uint32x2` timestamps, and parameters. `rowCount` sizes the filtered and optional smoothed states/covariances. Per-observation `status` accompanies the estimates; bit `8` retains both filtered moments on a singular prediction or nonfinite backward correction, and the next backward step uses that fallback. `covarianceMode` selects isotropic or full covariance. |
| `GPUTrajectoryRestrictionOperation` | `CPUTrajectoryRestrictionOperation` | Planar positions, track offsets, relative `float32` or signed Int64 `uint32x2` timestamps, parameter vector, and optional track-pair columns/count. Word timestamps require a logical `timeOrigin` input containing at least two uint32 rows (low/high words); omit it for float32 time. Outputs retain per-interval `status` and per-track/pair `relationships`; optional compact outputs retain `compactStatus` and `compactSummary`. |
| `GPUTrajectoryThinningOperation` | `CPUTrajectoryThinningOperation` | Bind `output.status.candidateOverflow` to a one-row uint32 value for the bounded forward search. The pure CPU function returns the flag. |
| `GPUTrajectoryStaypointsOperation` | `CPUTrajectoryStaypointsOperation` | Bind `output.status.candidateOverflow` to a one-row uint32 value for the bounded staypoint search. The pure CPU function returns the flag. |
| `GPUTrajectoryLocationsOperation` | `CPUTrajectoryLocationsOperation` | Bind `output.status.iterationLimitReached` to a one-row uint32 value. Forward both `staypointStatus` loss flags when the input comes from Staypoints. The pure CPU function returns the iteration flag. |
| `GPUMobilityIndicatorsOperation` | `CPUMobilityIndicatorsOperation` | Timestamps and a counted staypoint table (tracks, centroids, start rows, durations and optional location IDs), with optional triplegs. Forward stay/tripleg overflow and the global location-iteration flag. Optional `output.status.excludedLocationCount` counts excluded sentinel or unsettled keys. The three entropies and home/work use only retained keys. Optional `output.status.candidateOverflow` reports supplied loss or bounded entropy work. |
| `GPUModalSplitOperation` | `CPUModalSplitOperation` | Positions, timestamps, track offsets and a counted tripleg table. Supply observed `modes` or `modeSpeedUpperBoundsFloat32` for prediction; outputs are `modalSplit` and optional `predictedModes`. There is no separate output status port: forward available tripleg loss flags; supplied tripleg loss makes all shares NaN. |
| `GPUTrackSimilarityOperation` | `CPUTrackSimilarityOperation` | Planar or longitude/latitude great-circle line positions/offsets, pair-index columns, optional `activePairCount` and Fréchet epsilon. Requested metric outputs have one row per `pairCount`; `pairStatus` distinguishes empty, invalid and capped results. Inactive pairs share the invalid bit with invalid indices: use `activePairCount` to identify the inactive suffix. The CPU honors an explicit `maximumFrechetVertexCount`; omitting it removes that CPU cap. This is the base distance operation, with the metrics of `GPUTrackSimilarity`. |

For example, this maps Kalman's input `positions` to the logical value `observations` and its
output `filteredState` to `filtered`. The graph supplies those values as vectors:

```ts
import {GPUTrajectoryKalmanOperation} from '@luma-contributors/analysis/gpu-spatial-analysis';

const kalman = new GPUTrajectoryKalmanOperation({
  positions: 'observations',
  timestamps: 'observationTimes',
  trackOffsets: 'trackOffsets',
  parameters: 'kalmanParameters',
  rowCount: 2,
  output: {
    filteredState: 'filtered',
    filteredCovariance: 'filteredCovariance',
    status: 'kalmanStatus'
  }
});
```

Each table's count and completeness flags describe its usable prefix. Preserve those flags when
connecting staypoints, locations and triplegs to mobility or modal aggregation; allocated capacity
alone does not describe the amount of valid input.
