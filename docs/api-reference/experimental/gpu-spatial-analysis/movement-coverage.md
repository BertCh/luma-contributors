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

## Closed time and rectangle restriction

{/* lens:GPUTrajectoryRestriction */}

::lens{scenario=gpu-spatial-analysis.trajectory-restriction rung=graph}

A segment from x=-2 to x=2 over four seconds crosses the closed rectangle from time one to three.
Its ever relationship is true and always relationship false. A separate observation at time twenty
has no common temporal support and therefore has unknown relationships, represented by 0xffffffff.
Output intervals retain source-segment rows instead of hiding absent support through compaction.

## Exact batched similarity search

{/* lens:GPUTrackSimilaritySearch */}

::lens{scenario=gpu-spatial-analysis.track-similarity-search rung=graph}

The first candidate exactly matches the query. The second has endpoints ten units away and is
safely pruned before the existing DTW program runs. Captured lower bounds, distances, match flags
and status distinguish the evaluated zero distance from the pruned NaN distance. The contributor
also supports discrete Fréchet and normalized LCSS, with metric-specific admissible bounds.
