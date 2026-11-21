# Height above drainage over D-infinity flow

Import from `@luma-contributors/analysis/gpu-terrain`.

See also the [terrain hydrology API](operators.md#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices).

### GPUTerrainHeightAboveDrainage

The existing contributor accepts either `flowDirections` (D8) or `flowFractions`
(eight float32 weights per cell, east first clockwise). Exactly one is required.
For Tarboton D-infinity routing, generate `flowFractions` with
`GPUTerrainFlowFractions({routing: 'd-infinity', ...})` using the same routing
surface, validity, grid and ground-distance model as the flow computation. Pass
its `output.fractions` to HAND together with the elevation raster band and stream
mask. D8 callers keep their current behavior and direction encodings.

D8 publication checks that a stable pointer reaches a genuine terminal or stream. A receiver
cycle can collapse to self-pointers; it remains `GPU_TERRAIN_DRAINAGE_UNRESOLVED`, with NaN
HAND and convergence 0. For directions `[1,16]`, streams `[0,0]` leave both cells unresolved;
streams `[1,0]` stop the cycle at cell 0 and give drainage cells `[0,0]` and convergence 1.
The pointer loop can stop unchanged before its budget, so a cycle can have iteration-limit flag 0.

The fractional result is the downslope weighted average vertical drop to the
first stream along each branch:

`H[c] = sum(p[c,n] * (z[c] - z[n] + H[n])) / sum(p[c,n])`, with `H[stream] = 0`.

This implements the [TauDEM DinfDistDown vertical average method](https://hydrology.usu.edu/taudem/taudem5/help53/DInfinityDistanceDown.html)
with its default edge-contamination checking: every positive-weight branch must
reach a stream. A branch ending without drainage or touching an invalid elevation
makes HAND NaN. Zero-weight branches are ignored. Scale, offset, nodata and band
validity are applied before computing drops. Stream cells stop routing before
reading their fraction row; unknown stream words remain unresolved. Fraction rows
must be finite and nonnegative and sum to one within 1e-5, or be all zero for an
outlet. Malformed rows and edges outside the grid are unresolved. MFD fractions
are also accepted with the same average recurrence.

`drainageCells` is the unique first stream index if every branch reaches that
stream, including branches that split and later merge. It is
`GPU_TERRAIN_DRAINAGE_MULTIPLE` (0xfffffffd) when several first streams contribute,
`GPU_TERRAIN_DRAINAGE_NONE` (0xffffffff) for no drainage or invalid elevations, and
`GPU_TERRAIN_DRAINAGE_UNRESOLVED` (0xfffffffe) for unknown routing, unknown stream
membership, a cycle or an unfinished dependency. Multiple drainage cells still
have a valid averaged HAND value. If both missing and unknown branches occur,
missing drainage takes precedence.

Fractional routing uses synchronous GPU dependency sweeps with eight neighbors
per invocation. `maximumIterations` defaults to 128 in fractional mode and 32 in
D8 mode, and must be in [1, 1024]. A path with D edges needs D sweeps plus one to
certify convergence. Completed cells remain available at the budget; unfinished
cells are NaN/unresolved. Fractional-routing cycles exhaust the budget. In D8 mode a path that enters a receiver cycle (possible only with caller-supplied directions) is also `GPU_TERRAIN_DRAINAGE_UNRESOLVED`, and `status.converged` is 0. A D8 cycle can stabilize during pointer doubling and then fail the original-receiver check with `iterationLimitReached = 0`; it need not exhaust the budget. Optional scalar status ports
are `converged`, `iterationLimitReached` and `iterationCount`; convergence means
the dependency computation finished without unresolved drainage. Unknown stream
membership (`0xffffffff`) clears convergence in both D8 and fractional routing,
including every dependent cell whose drainage remains unknown. A known outlet
without a stream remains complete (NaN height, no drainage); unknown membership
alone does not set `iterationLimitReached`. Fractional cycles and unfinished
dependencies still exhaust the budget, and `iterationCount` still counts executed
rounds. Missing drainage retains precedence over an unknown branch. No status is
read back inside the contributor.

The routing input choice, grid size, iteration budget, elevation format and
calibration, and optional output presence are compile-time. View contents change
per frame. HAND is in elevation units; horizontal ground-distance corrections
belong to the producer of the fractions. `estimatedCommandNodeCount` bounds the
unrolled schedule and is checked at construction. Both fractional snapshots (12
bytes per cell each) and fraction input (32 bytes per cell) must fit an individual
storage binding. Inputs and outputs must be packed and use separate buffers.

Validation: the committed NumPy oracle fits planes to Tarboton facets and solves
an absorbing-flow linear system on small DEMs. The headless spec compares both
supplied oracle fractions and GPU-generated D-infinity fractions, and covers
splits, merges, ties, padding, calibration, missing data, cycles and budgets.
There is no HAND CPU operation twin.
