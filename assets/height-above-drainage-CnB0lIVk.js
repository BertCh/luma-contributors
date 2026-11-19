var e={slug:`api/gpu-terrain/height-above-drainage`,title:`Height above drainage over D-infinity flow`,html:`<h1 id="height-above-drainage-over-d-infinity-flow">Height above drainage over D-infinity flow</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices">terrain hydrology API</a>.</p>
<h3 id="gputerrainheightabovedrainage">GPUTerrainHeightAboveDrainage</h3>
<p>The existing contributor accepts either <code>flowDirections</code> (D8) or <code>flowFractions</code>
(eight float32 weights per cell, east first clockwise). Exactly one is required.
For Tarboton D-infinity routing, generate <code>flowFractions</code> with
<code>GPUTerrainFlowFractions({routing: 'd-infinity', ...})</code> using the same routing
surface, validity, grid and ground-distance model as the flow computation. Pass
its <code>output.fractions</code> to HAND together with the elevation raster band and stream
mask. D8 callers keep their current behavior and direction encodings.</p>
<p>The fractional result is the downslope weighted average vertical drop to the
first stream along each branch:</p>
<p><code>H[c] = sum(p[c,n] * (z[c] - z[n] + H[n])) / sum(p[c,n])</code>, with <code>H[stream] = 0</code>.</p>
<p>This implements the <a href="https://hydrology.usu.edu/taudem/taudem5/help53/DInfinityDistanceDown.html">TauDEM DinfDistDown vertical average method</a>
with its default edge-contamination checking: every positive-weight branch must
reach a stream. A branch ending without drainage or touching an invalid elevation
makes HAND NaN. Zero-weight branches are ignored. Scale, offset, nodata and band
validity are applied before computing drops. Stream cells stop routing before
reading their fraction row; unknown stream words remain unresolved. Fraction rows
must be finite and nonnegative and sum to one within 1e-5, or be all zero for an
outlet. Malformed rows and edges outside the grid are unresolved. MFD fractions
are also accepted with the same average recurrence.</p>
<p><code>drainageCells</code> is the unique first stream index if every branch reaches that
stream, including branches that split and later merge. It is
<code>GPU_TERRAIN_DRAINAGE_MULTIPLE</code> (0xfffffffd) when several first streams contribute,
<code>GPU_TERRAIN_DRAINAGE_NONE</code> (0xffffffff) for no drainage or invalid elevations, and
<code>GPU_TERRAIN_DRAINAGE_UNRESOLVED</code> (0xfffffffe) for unknown routing, unknown stream
membership, a cycle or an unfinished dependency. Multiple drainage cells still
have a valid averaged HAND value. If both missing and unknown branches occur,
missing drainage takes precedence.</p>
<p>Fractional routing uses synchronous GPU dependency sweeps with eight neighbors
per invocation. <code>maximumIterations</code> defaults to 128 in fractional mode and 32 in
D8 mode, and must be in [1, 1024]. A path with D edges needs D sweeps plus one to
certify convergence. Completed cells remain available at the budget; unfinished
cells are NaN/unresolved. Cycles exhaust the budget. Optional scalar status ports
are <code>converged</code>, <code>iterationLimitReached</code> and <code>iterationCount</code>; convergence means
the dependency computation finished, and does not certify that all input routing
or drainage is known. No status is read back inside the contributor.</p>
<p>The routing input choice, grid size, iteration budget, elevation format and
calibration, and optional output presence are compile-time. View contents change
per frame. HAND is in elevation units; horizontal ground-distance corrections
belong to the producer of the fractions. <code>estimatedCommandNodeCount</code> bounds the
unrolled schedule and is checked at construction. Both fractional snapshots (12
bytes per cell each) and fraction input (32 bytes per cell) must fit an individual
storage binding. Inputs and outputs must be packed and use separate buffers.</p>
<p>Validation: the committed NumPy oracle fits planes to Tarboton facets and solves
an absorbing-flow linear system on small DEMs. The headless spec compares both
supplied oracle fractions and GPU-generated D-infinity fractions, and covers
splits, merges, ties, padding, calibration, missing data, cycles and budgets.
There is no HAND CPU operation twin.</p>`};export{e as default};