var e={slug:`api/gpu-spatial-analysis/movement-coverage`,title:`Trajectory estimation, restriction and search`,html:`<h1 id="trajectory-estimation-restriction-and-search">Trajectory estimation, restriction and search</h1>
<p>These executable examples use small projected Cartesian trajectories and relative times. Full
parameter, status and limitation contracts are in
<a href="/luma-contributors/docs/api/gpu-network/network-movement-coverage">Bounded network optimization and trajectory analysis</a>.
The examples declare real GPU work and capture caller-visible results; they are not performance
benchmarks or claims of complete upstream-library parity.</p>
<h2 id="constant-velocity-filtering-and-smoothing">Constant-velocity filtering and smoothing</h2>
<p>{/* lens:GPUTrajectoryKalman */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-kalman rung=graph}</p>
<p>Two observations at x=0 and x=2 give a forward estimate x=4/3 at the second observation.
Backward RTS smoothing revises the first estimate to x=2/3. The record preserves raw observations,
filtered states, smoothed states, both covariance arrays and status independently. Position and
velocity covariance are shared by the two independent isotropic axes.</p>
<h2 id="closed-time-and-rectangle-restriction">Closed time and rectangle restriction</h2>
<p>{/* lens:GPUTrajectoryRestriction */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-restriction rung=graph}</p>
<p>A segment from x=-2 to x=2 over four seconds crosses the closed rectangle from time one to three.
Its ever relationship is true and always relationship false. A separate observation at time twenty
has no common temporal support and therefore has unknown relationships, represented by 0xffffffff.
Output intervals retain source-segment rows instead of hiding absent support through compaction.</p>
<h2 id="exact-batched-similarity-search">Exact batched similarity search</h2>
<p>{/* lens:GPUTrackSimilaritySearch */}</p>
<p>::lens{scenario=gpu-spatial-analysis.track-similarity-search rung=graph}</p>
<p>The first candidate exactly matches the query. The second has endpoints ten units away and is
safely pruned before the existing DTW program runs. Captured lower bounds, distances, match flags
and status distinguish the evaluated zero distance from the pruned NaN distance. The contributor
also supports discrete Fréchet and normalized LCSS, with metric-specific admissible bounds.</p>`};export{e as default};