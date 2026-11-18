# Bounded network optimization and trajectory analysis

These graph contributors implement the supported modes below. They accept explicit packed views;
the application owns preparation, uploads, graph compilation, submission and optional readback.
No contributor creates a CPU numerical fallback or combines source batches implicitly. Build one
contributor per independently prepared batch when source partition boundaries must remain distinct.
All outputs are caller-owned. Graph scratch is transient and graph-owned.

| Backlog | Contributor | Implemented contract | Limits and exclusions |
| --- | --- | --- | --- |
| NW01 | `GPUNetworkPointRouting` | Batched static paths between fractional positions on identified directed edges; legal incoming/outgoing turn states, per-turn penalties, same-edge forward travel, parallel-edge identity, edge predecessor provenance | Reference mode: 128 edges, 4096 explicit turns. Scalable mode: device binding and command-budget limits. Fractions retain the selected directed-edge state even at endpoints. Reverse direction requires a separate directed edge. Snapping, reverse-edge discovery, and arbitrary restriction history are caller preparation. |
| NW02 | `GPUNetworkFlow` | Integral maximum flow, residual source cut, edge-aligned feasible flow, flow/cut equality certificate and optional path decomposition; unit capacities provide edge-disjoint paths | Reference mode: 128 nodes, 1024 original directed edges and total capacity at most 16,777,215. Parallel mode: up to 30 million nodes/edges and total capacity at most 4,294,967,295, subject to device binding and command-budget limits. Optional decomposition requires at most 128 expanded vertices and 1024 expanded edges with either solver. Explicit reverse residuals preserve antiparallel edges. |
| NW03 | `GPUNetworkFlow` with `costs`, `prepareGPUNetworkFlowDemands` | Minimum-cost requested/maximum flow, reverse cancellation, explicit balanced-demand preparation and optional exact 64-bit integer objective | Nonnegative original costs ≤32767; no negative-cost circulation or lower-bound constraints. Summary cost is f32; `exactCost` preserves the integer objective. |
| NW04 | `GPUNetworkLouvain` | Sparse parallel weighted directed/undirected modularity, quotient coarsening, objective trace and convergence; legacy dense reference | See [Louvain limits](louvain.md#parameters-and-limits) for sparse addressing, per-binding storage, dense-reference and command-budget bounds. This is a modularity heuristic, never a global optimum certificate. |
| NW05 | `GPUNetworkFacilityOptimization` | Exhaustive backup coverage, p-dispersion and unsplit capacitated p-median; feasibility and exhaustive completion certificates | At most 16 facilities, 64 demand rows and 1,048,576 (2^20) candidates, checked at construction. Backup coverage and p-dispersion enumerate `2 ** facilityCount` candidates, so 64 rows is their real demand limit. Capacitated p-median enumerates `facilityCount ** demandRows` candidates, so the candidate limit, not the 64-row ceiling, bounds its demand rows: 20 rows at 2 facilities, 10 at 4, 5 at 16. For larger problems use `GPULocationAllocation`. Candidates run independently in GPU batches of at most 65,536, with deterministic hierarchical winner reductions. Exhaustive search can still be expensive within the limit; there is no branch-and-bound or scalable mixed-integer solver. |
| NW06 | `GPUNetworkTimeRouting`, `prepareGPUNetworkRestrictionStates` | FIFO affine or piecewise-arrival routing, first-sample waiting, reachable forbidden-sequence states and original-edge provenance | ≤256 states,4096 transitions,256 samples/transition. Non-FIFO schedules and arbitrary time windows remain unsupported. CPU topology preparation never solves routes. |
| NW07 | `GPUNetworkRationalAccess` | Congestion equalization with integral/fractional demand, fractional/absolute shift caps, unserved demand and explicit equilibrium gap | ≤64 origins,32 facilities,4096 cycles; f32 demand total ≤2^24. Integral rounding stalls remain unconverged. Fractional conservation has f32 rounding tolerance. |
| MV01 | `GPUTrajectoryKalman`, `prepareGPUTrajectoryObservations` | Isotropic or full correlated 4x4 Kalman/Joseph covariance and RTS smoothing; explicit WGS84 local-plane/bigint epoch preparation | Linear planar constant velocity; sequential per track, parallel across tracks. Local tangent projection is explicit and drops height/curvature, not an automatic CRS choice. |
| MV02 | `GPUTrajectoryRestriction` | Rectangle/polygon-ring restriction, holes, open/closed/half-open endpoints, compact/merged intervals and analytic pairwise within-distance intervals | Planar linear interpolation; ≤128 polygon vertices, ≤256 slots/segment. Pairwise static query topology, ≤65536 slots/pair. Gaps/invalidity/capacity are explicit; no geodesic interpolation or general temporal SQL algebra. |
| MV03 | `GPUTrackSimilaritySearch` | Admissible DTW/Fréchet/LCSS bounds, stable GPU survivor compaction, bounded DP scratch and exact threshold top-k with pair provenance | Explicit candidate index/pair batches; no implicit all-pairs/spatial-index construction. Top-k is among supplied threshold-matching pairs, not unseen trajectories. No densification or geographic metric. |

## Completion, validity and provenance

`GPUNetworkFlow.output.summary` has eight uint32 words: valid input, complete search, flow value,
unserved requested flow, augmentation count, cost as float32 bits, residual cut capacity, and cut
certificate. `complete=0` means the feasible flow is an incumbent. An incomplete positive unserved
amount does not prove infeasibility. `complete=1` means either the requested value was attained or
no residual source/sink path remains. Only `cutCertified=1` proves the reported maximum-flow value.
Minimum-cost optimality is conditional on valid/complete input and the supported nonnegative
original-cost model. Original-edge output order preserves capacities and provenance without
merging parallel or antiparallel edges.

The explicit/reference route uses one serial invocation and is construction-limited to 128 nodes,
1024 original edges and 65536 augmentations (default 1024). Its reviewed contract allowance names
all three serial axes at that kernel's invocation count; it does not apply to parallel kernels or
relax any constructor limit. Reaching the augmentation budget still publishes the feasible flow
and its existing incomplete/iteration-limit status.

The contributor captures compile-time counts, solver choices and nested input/output/status port
maps before validation. The GPU operation and CPU twin likewise capture counts and graph value
names before deferred execution. Later mutation of the caller's props cannot enlarge the reference
kernel or redirect its writes. Graph views and buffers keep their identities: callers still change
per-frame values by updating their contents, rather than replacing entries in a props object.

Point routing reports 0 for a complete query, 1 for invalid data and 2 for an iteration-limited
query. The query's `predecessorEdges` row preserves the winning route, including a legal cycle
when the source and destination are the same edge and the destination fraction is behind the
source fraction. For that returning route, the source entry records the final incoming edge.
To reconstruct it, begin at the destination, follow at least one predecessor, and stop on reaching
the source. Direct same-edge travel and unreachable labels retain `0xffffffff`; use the cost
and status to distinguish them. All other source entries retain the sentinel.
Time routing reports `[valid, converged, chosenTargetState, arrivalTimeBits]`. Only a valid
converged search can establish that an infinite distance is unreachable. Transition provenance
from `prepareGPUNetworkRestrictionStates().originalEdges` maps predecessor transitions back to
the original directed edges. Initial states are the empty restriction prefix at each vertex.
The topology helper does no searching, consumes explicit JavaScript topology arrays, and returns
typed arrays for explicit upload; it does not read GPU data.

The [Louvain guide](louvain.md) defines summary/trace layouts, completion status and the
differences between sparse parallel, dense reference and CPU execution. Compare objective and
partition invariants rather than label IDs. A budget-limited partition remains a heuristic incumbent.

Facility optimization reports `[valid, feasibleIncumbent, enumerationComplete, examinedCandidates]`.
When enumeration is complete, a feasible incumbent is the exhaustive optimum for the supplied f32
objective comparisons; no incumbent then certifies infeasibility for this bounded model. When
incomplete, neither conclusion follows. Objective is NaN without a feasible incumbent. Backup
coverage without `selectCount` first minimizes the facilities needed to cover every demand row,
then maximizes the demand with at least two covering facilities at that count. Supplying a count
instead fixes that count. P-dispersion maximizes the minimum selected pair distance and requires
symmetric costs and at least two facilities. Capacitated p-median assigns whole demand rows,
minimizing demand-weighted distance with individual capacity constraints. Assignments are published
only for capacitated p-median; other modes leave the assignment sentinel intact.

Candidates are scored independently in batches that reuse one score buffer. Each candidate
retains input row order for floating-point sums and capacity checks. Hierarchical reductions select
the lowest original candidate index on objective ties, including ties across batches. A partial
`maximumCandidates` still examines exactly the original enumeration prefix; batch padding does
not count as examined work. Every encoding resets the cursor, incumbent and validation status.
The number of assignments remains exponential in demand count for capacitated p-median; batching
bounds scratch storage rather than changing the search complexity.

RAAM parameters are `[tau, rho, initialStep, minimumStep, halfLife, gapTolerance]`. Travel costs are
divided by tau, and supplies are multiplied by rho. To use global mean demand/supply normalization,
compute and supply rho explicitly. Inputs require positive finite supplies; unreachable costs are
positive infinity. Missing/negative costs are invalid. Nonintegral demand requires `demandMode: 'fractional'`. Demand conservation
is `sum(assignments[origin,*]) + unservedDemand[origin] = demand[origin]`. Scores are NaN for
unserved and zero-demand origins. Final scores use final loads rather than the preceding cycle's
loads; this deliberately avoids the stale final cost in the reference `iterate_raam` implementation.
The reported gap is maximum used-facility cost minus minimum reachable cost. Equilibrium status
does not hide unserved rows; inspect both outputs. `stepMode: 'fraction'` caps a fraction of each origin's demand; `'absolute'` caps a demand amount.
Integral mode floors equalizing shifts and uses round-to-even early inflow scaling. Fractional mode
retains continuous shifts and does not round early inflow scaling.

## Trajectory conventions

Kalman parameters are acceleration spectral density, positive measurement variance, nonnegative
initial velocity variance and maximum gap, where 0 means no limit. Process covariance is
`q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]`. A new segment starts with the first measured position and zero
velocity. Missing positions produce prediction-only steps after initialization; no state is
invented before the first measurement. Nonfinite timestamps or backward time produce invalid
estimates and reset initialization. Equal timestamps are supported as repeated measurements.
The raw observations, filtered outputs and optional smoothed outputs remain separate. For both
estimation and restriction, offsets must form a complete nondecreasing partition: first offset zero,
last offset observation count, and every intermediate offset inside that range. Overlapping or
incomplete partitions are outside the input contract; outputs outside supplied track spans are not
implicitly initialized.

State rows are `[x,y,vx,vy]`. Covariance rows are `[positionVariance,covariance,covariance,velocityVariance]`,
shared by the independent x and y axes. Status bits are reset=1, prediction-only=2, invalid=4,
singular backward solve=8. A singular backward prediction retains the forward estimate and marks
bit 8; it is not represented as a successful RTS update. Noise units must match coordinates and
relative seconds. Subtract a nearby epoch before converting timestamps to f32. Far from one epoch,
pass `times` as `uint32x2` Int64 `(low, high)` words instead: each step is differenced exactly and
only then converted to f32, so 10 Hz samples stay 0.1 s apart however far they are from the epoch.
Set `timeUnitsPerSecond` (default 1, for example 1000 for epoch milliseconds) so the model keeps
running in seconds. Restriction word `times` need a per-frame `timeOrigin` (`[low, high]` uint32):
`minimumTime`, `maximumTime` and the output intervals are f32 relative to it, so keep the origin
near the window. `maximumGap` is always in the timestamps' unit: relative seconds for f32 Kalman
times, the word unit (for example milliseconds) for word times. 0 means no limit in both contributors.
Kalman and restriction also read the trajectory port (`timestamps`, `trackOffsets`) with an optional
`temporalContext`; Kalman then takes `timeUnitsPerSecond` from `context.unit`, for float32 times too.
Both also take an optional `spatialContext` that resolves to planar coordinates (the default); a
longitude-latitude, grid or topology context throws at construction. Kalman and a restriction with
a distance predicate additionally need the native metric and reject `local-lambert-equal-area` frames.
The per-observation `status` words document every bit in `GPU_TRAJECTORY_KALMAN_STATUS` and
`GPU_TRAJECTORY_RESTRICTION_STATUS`.

Restriction parameters are `[minTime,maxTime,minX,minY,maxX,maxY,maxGap]` (`maxGap` 0 means no limit). Nonstationary segments
need strictly increasing timestamps; equal-time equal-position rows are singletons. Crossing
intervals preserve their original segment row, so adjacent endpoints can appear in both rows.
Status bits are interval=1, singleton=2, invalid=4, unsupported gap=8, full temporal support inside=16,
temporal support=32. Per-track relationships are `[ever,always,supportRows,unknownRows]` with
`0xffffffff` for unknown. Zero common temporal support gives unknown for both predicates.
A known inside witness establishes ever=true despite other unknown rows; a known outside witness
establishes always=false. Unknown gaps are never treated as straight-line observations.

Similarity search returns evaluated distance, endpoint lower bound, threshold match and status.
Status 0 means the existing dynamic program ran, 1 means safely pruned, 2 means invalid input and
4 means dynamic-program capacity exhaustion and 8 means survivor-capacity exhaustion. Pruned distances are NaN. The lower bound is the
maximum of first/last endpoint distances: every admissible DTW or Fréchet coupling includes both
endpoints, including with a band. A conservative f32 margin retains threshold-neighbor cases.
Band-disconnected DTW results can be infinity with evaluated status and match=false. LCSS instead accepts a minimum normalized similarity threshold and reports `similarity` and
`upperBound`. Every matched element must have at least one eligible counterpart, so the minimum
of eligible-element counts divided by the shorter track length bounds normalized LCSS from above.
The bound considers tolerance and ignores the band conservatively, and costs O(n*m); it saves dynamic-program work and
scratch for rejected pairs. An exclusive GPU prefix scan compacts surviving pairs in original order;
`survivorCapacity` bounds the dynamic-program dispatch/scratch. Overflow pairs have status 8 and NaN,
never a false nonmatch. Optional `survivorSummary` is `[required,evaluated,overflow]`.
`topK` returns original pair row IDs ordered by distance (or descending LCSS), then row ID for ties.
`topKSummary` is `[written,complete]` per query; invalid/overflow pairs make that query incomplete.
The supplied pair arrays are the candidate index and stable provenance; batching is explicit.

## Evidence and remaining gates

The independent generator is
[`network-movement-coverage.py`](../../../modules/analysis/test/oracles/network-movement-coverage.py),
with committed
[`network-movement-coverage.json`](../../../modules/analysis/test/oracles/fixtures/network-movement-coverage.json).
It pins NetworkX 3.5 and NumPy 2.2.6. Network flow and modularity fixtures call NetworkX directly;
Kalman fixtures are independent dense matrix compositions, not MovingPandas calls. Tests also
check conservation, reverse residual cancellation, self-loops, a greedy-allocation counterexample,
iteration exhaustion, missing measurements, covariance symmetry, gaps, zero support, threshold
ties, and mutable input updates. Node tests audit shader parsing, reserved identifiers and the
portable eight-storage-binding limit.

The selected RAAM source is
[`access` v1.1.9 `raam.py`](https://github.com/pysal/access/blob/v1.1.9/access/raam.py);
the implementation here is independent and exposes final-load and convergence semantics explicitly.
The additional [advanced generator](../../../modules/analysis/test/oracles/network-movement-advanced.py)
calls MovingPandas 0.22.4/StoneSoup 1.8 smoothing and polygon clipping, tslearn 0.6.4 Euclidean-cost
DTW/LCSS, similaritymeasures 1.3.0 Fréchet, access 1.1.9 RAAM, and spopt 0.7.0/PuLP 3.3.0 CBC facility models directly. It also includes an
independent full-covariance NumPy solve, pyproj ECEF projection, and NetworkX virtual-split/time-expanded
routing equivalents. A separate [server generator](../../../modules/analysis/test/oracles/network-movement-postgres.py)
executes PostgreSQL 16.6, PostGIS 3.5.3, pgRouting 3.8.0 and MobilityDB 1.2.1 directly.
Its [saved results](../../../modules/analysis/test/oracles/fixtures/network-movement-postgres.json)
cover directed fractional same-edge, parallel-edge and unreachable routes; polygon holes;
all four time endpoint policies; continuous encounter intervals; and absent common support.
SQL NULL unreachable costs map to infinity, and NULL temporal relationships map to `0xffffffff`.
The crossing fixture exposes a MobilityDB 1.2.1 difference: `aDwithin` returns true while
`always_eq(tDwithin(first, second, 1), true)` returns false. The GPU's full-common-support
always semantics match the latter. Both raw server results are preserved; blanket compatibility
with `aDwithin` is not claimed. Valhalla parity and broad-device coverage remain untested.

The isolated native reference stack used the official PostgreSQL 16.6 and PostGIS 3.5.3 source
archives, pgRouting tag `v3.8.0`, and MobilityDB tag `v1.2.1`. PostgreSQL configured with
`--without-readline --without-icu --without-zlib`; PostGIS omitted raster, topology, protobuf
and address standardizer. GEOS 3.14.1, PROJ 9.9.0, JSON-C 0.18 and GSL 2.7 came from an
existing isolated environment, with Boost 1.87 headers for pgRouting. Core PostGIS library,
control and new-install SQL targets avoided an unrelated raster upgrade-script generation
failure. On macOS, pgRouting's installed `.so` bundle needed a `.dylib` symlink for PostgreSQL's
loader. No reference algorithm source was changed. The local socket-only stack peaked at
505 MB and was removed after fixture capture. Reproduction needs these extensions installed;
the generator accepts `--psql` and standard `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` settings.

The [benchmark fixture](../../../modules/analysis/test/oracles/fixtures/network-movement-benchmark.json)
records 30 workloads on Apple Metal3 WebGPU, including all nine contributors and 4096 trajectory
pairs. Upload/preparation, compilation, first execution, three warm executions and readback are
separate. Full-survivor 4096-pair Fréchet took median 5.2 ms; 205 survivors took 1.1 ms with bounded
scratch. This selectivity comparison is not a CPU speedup claim. Dense Louvain at 32 vertices took
28.9 ms and remains a correctness-oriented bounded solver. See the
[measurement notes](network-movement-benchmarks.md) for workload-specific limits.

## Advanced input and output layouts

For piecewise routing set `costModel: 'piecewise-arrival'` and `samplesPerTransition >= 2`.
Each transition has a fixed-width sequence of `[departure,arrival]` pairs. Departure samples must
strictly increase, arrival samples must not decrease, and arrival must be at least departure.
The earliest label before the first sample waits until that sample; inside the sampled domain,
arrival is linearly interpolated; after the last sample, the final travel duration remains constant.
`waitingPolicy: 'wait-until-first-sample'` makes that behavior explicit. The entire cost curve is FIFO,
so later labels cannot improve a state and the earliest-label relaxation is sound.

`prepareGPUNetworkFlowDemands` takes balanced integer net demands (positive=inflow, negative=supply),
adds zero-cost super-source/sink arcs and returns explicit upload arrays. The first `originalEdgeCount`
flow outputs retain original-edge provenance. Optional `exactCost` contains `[low32,high32]` with no
integer rounding inside the documented capacity/cost envelope.
The adapter returns `solver: 'reference' | 'parallel'`; forward it with the expanded arrays to the
contributor. The default `solver: 'auto'` chooses reference only when the expanded problem has at
most 128 vertices, 1024 edges and total capacity 16,777,215, including both super arcs. Thus 126
input vertices can use reference; 127 require parallel. Explicit `solver: 'reference'` enforces
those expanded limits, and `'parallel'` opts in even for small inputs. Parallel expanded shape is
bounded at 30 million vertices/edges and total capacity 4,294,967,295; device bindings and command
budgets still apply at contributor construction. Original costs must be nonnegative integers no
larger than 32767 for reference or `min(32767, floor(2147483646 / expandedNodeCount))` for parallel.

Optional `GPUNetworkFlow.output.paths` decomposes the **final feasible flow**, including after
reverse cancellation, into simple source-to-sink paths with integral volumes. This reference
pass requires at most 128 expanded vertices and 1024 expanded edges, including super arcs,
with either flow solver. The path capacity is `paths.volumes.length` in the range 1..1024.
Allocate `lengths` with that capacity, `edgeRows` with `capacity * (nodeCount - 1)` words, and
each `status` field (including `status.count`) with one word, all in distinct uint32 buffers. Each path occupies
`nodeCount - 1` edge slots; only its first `lengths[row]` slots are used. Unused edge slots are
`0xffffffff`; unused lengths and volumes are zero. Edge rows refer to the expanded input edge
list, so retain rows below `originalEdgeCount` to recover the original demand paths. Paths
retain their supply/demand endpoints via the adjacent super arcs. Circulating flow is omitted.

The deterministic tie rule applies to a fixed final flow: repeatedly run breadth-first search
over positive remaining flow, visiting queued vertices in FIFO order and each vertex's outgoing
edges in ascending input row order. Keep the first predecessor, emit the discovered path with
its bottleneck volume, subtract that volume and repeat. Thus the unit diamond emits both paths
with volume 1; where two branches merge and split again, the earliest source branch is paired
with the earliest sink branch. Equal-cost solvers can allocate different final edge flows; the
tie rule does not promise the same decomposition when their final edge flows differ.
Breadth-first order matters when capacity drops paths: with unit edges `0→1`, `1→2`, `0→2`
in that input order and a request of two units, capacity 1 retains edge row `[2]` (length 1,
volume 1), with `requiredCount = 2` and `overflow = 1`. An edge-ordered depth-first walk would
retain `[0,1]` instead and does not implement this tie rule.

`paths.status.requiredCount` counts all decomposed paths before capacity truncation;
`paths.status.count = min(requiredCount, capacity)`, and `overflow = 1` when paths were dropped. Counting
continues after the output fills, so this count is exact for the incumbent flow. The required
`invalidCount`, `converged` and `iterationLimitReached` fields propagate the solver's status:
an exhausted augmentation/phase/search budget publishes the partial feasible paths with
`converged = 0`, `iterationLimitReached = 1`. Path capacity loss sets `overflow` independently
of convergence. A solved infeasible request can converge with fewer paths than requested;
check `summary.unserved` as well. Zero demand produces a complete empty decomposition even
with a zero augmentation budget. Invalid problems publish no paths and `invalidCount = 1`.
Every encoding clears all path slots and status words before publishing the current result.

Both graph operations use `pathCapacity` together with the same `output.paths` port map.
The pure `computeNetworkFlowOnCPU` enables the matching `result.paths` with `pathCapacity`.
Both operations accept `edgeCount = 0`, matching the contributor and pure CPU solver; `flow`
then has zero rows. For two distinct vertices, no edges and a request of one unit, the solve
converges with value 0 and unserved demand 1. Optional paths have `status.count = 0`, zero
lengths and volumes, `0xffffffff` edge padding, `requiredCount = 0` and `overflow = 0`.
An unmet request after residual exhaustion is complete, so `iterationLimitReached = 0`.
All explicit solver budgets retain their solver-specific meaning: `maximumAugmentations` binds
reference and costed successive-shortest-path; `maximumRelabels` binds uncosted parallel flow;
`maximumCostPhases` binds primal-dual. `maximumRelaxationRounds` binds parallel searches.
Decomposition exhausts the incumbent in at most one path per expanded edge; it has no separate
iteration budget. Path capacity truncates published rows and does not limit the solve.
On both backends, each uncosted/primal-dual phase recomputes sink-rooted labels, performs
bounded forward discharges, and returns stranded excess before committing the feasible flow.
Returned source excess is available to the next phase. For the demand-2 diamond with
capacities `[2,2,1,1]`, one phase sends one unit and reports an incomplete incumbent; two phases
send both units and converge. Either branch can carry the first unit under GPU atomic ordering.
`maximumRelaxationRounds` separately bounds the sink-rooted search, forward discharge, excess
return and final source-rooted cut search, including an unchanged sweep to certify each search.
Serving a finite request still converges when the final cut search exhausts its budget;
`summary.cutCertified` stays zero in that case.

`solver: 'reference'` rejects `maximumRelabels`, `maximumRelaxationRounds`, `maximumCostPhases`
and `mode` in the contributor, both operation constructors and the pure CPU entry point.
Errors identify the contributor/function ID and incompatible prop. All `output.paths` columns
and its `status` object/fields are required when paths are enabled; missing fields produce
ID-prefixed errors naming the full path (for example, `output.paths.status` or
`output.paths.volumes`).

Point routing accepts optional `turnPenalties: float32[turnCount]`, aligned with its allowed
`turns: uint32x2[turnCount]`. Keep `costs` as the separate base-edge costs. Omitting penalties
retains zero-penalty routing. Every penalty must be finite and nonnegative; invalid contents set
per-query status 1. Traversing a turn pays its penalty plus the following edge cost. Entering a
partial destination pays the full penalty plus only `targetFraction * destinationBaseCost`.
For two incoming edges with base cost 2, source fraction 0.5, and penalties 7 and 1 into a base-cost
10 destination at fraction 0.25, the query costs are 10.5 and 4.5. This applies to both GPU modes,
the graph operation and the CPU twin.
Negative or non-finite base costs close individual edges. Turns touching closed edges are skipped;
a query with a closed source or destination has status 1, while queries between open edges remain
usable. Both forms publish the cost-attaining parent of the partial destination, including when
large full-edge costs round competing parents to a tie. An explicit CPU `maximumRelaxations`
binds the solve: reference counts turn scans, scalable counts rounds of up to 64 synchronous hops.
Status 2 reports exhaustion; a zero budget retains direct same-edge travel and leaves every parent
unset. Without an explicit budget the CPU solve continues to a fixpoint.

`prepareGPUNetworkPointRoutingTurns({offsets, neighbors, weights, costs, overflow})` adapts explicit
read-back arrays from `GPUNetworkLineGraph` to `{turns, turnPenalties}` for upload. Pass the line
graph's overflow word: incomplete graphs throw. The adapter reads only active CSR slots, omits
closed arcs (`weights=-1`), and separates each full arc weight into its following base cost and
penalty. Prefer requesting the producer's optional `output.turnPenalties: float32[arcCapacity]`
and passing that read-back column as the adapter's optional `turnPenalties` input: it preserves
small turn costs before adding a large base-edge cost. Without it, subtraction is a fallback and
the recovered penalty retains only the precision of the supplied float32 arc weight. This
host helper performs no implicit submission or readback; applications can also prepare pair and
penalty arrays directly.

For full Kalman covariance set `covarianceMode: 'full'`; parameters are
`[qxx,qxy,qyy,rxx,rxy,ryy,vxx,vxy,vyy,maxGap]`. Process/velocity covariance must be positive semidefinite;
measurement covariance must be positive definite. Covariance outputs contain **four float32x4 rows
per observation**, the column-major 4x4 covariance in `[x,y,vx,vy]` state order. Full RTS uses a pivoted
4x4 inverse; singular/nonfinite backward solves retain forward values and set bit 8.
`prepareGPUTrajectoryObservations` subtracts a bigint epoch before f32 conversion (and also returns
exact `timeWords`, the absolute timestamps as a `uint32x2` Kalman `times` column) and transforms
WGS84 longitude/latitude through ellipsoidal ECEF into an explicitly chosen local east/north plane.
It returns projection/epoch metadata and performs no filtering or GPU readback.

For polygons, append flattened ring coordinates after the seven restriction parameters and set
`polygonRingVertexCounts`. Rings close implicitly; even-odd parity supports holes regardless of ring
orientation. Valid simple rings are caller-owned topology. A point is on the polygon boundary only
when its exact orientation against an edge is zero inside the edge's bounding box, the same exact
predicate `GPUZoneEvents` uses; there is no tolerance band, so a fix 5 cm from a 100 km edge is
strictly inside or outside. A point where the segment crosses an edge, and a piece of the segment
that runs along an edge, are on the boundary by construction and follow `geometryBoundary`. Whether
a segment meets an edge is decided from the exact sides of the two fixes against the edge and of the
edge's two ends against the segment, so a segment that passes a vertex without touching it reports
nothing and one that runs through a vertex reports one touch. Only the position of a crossing that
exists is f32, so interval end times carry f32 rounding. The two
contributors still differ for a fix exactly on the boundary: here `geometryBoundary` decides,
while `GPUZoneEvents` breaks the tie by its perturbation rule. `timeBoundary` selects closed, open, left-closed or right-closed;
`geometryBoundary` selects closed/open. Status adds start-open=64,end-open=128,capacity-exceeded=256.
`intervalsPerSegment` controls fixed output slots. Absent slots contain NaN/zero status. Relationships
are computed from all classified support even if interval slots overflow.

Optional compact outputs retain interval positions and `[firstSourceRow,lastSourceRow]` provenance.
`mergeIntervals` unions connected intervals across segments of the same track, preserving open joints
and track boundaries. `compactSummary` reports `[required,written,overflow]`; when source slots already
overflowed, required is a lower bound and overflow includes the number of truncated source segments.
Compaction/union is a bounded sequential GPU stream, not an implicit CPU filter.

For temporal within-distance predicates set `pairwiseTracks` to explicit track-ID pairs and
`intervalsPerPair` to output capacity; parameters[2] becomes maximum distance. Output rows are
pair-major compact interval slots and relationships are per pair. The kernel analytically solves
squared relative distance on each overlapping linear segment; output positions belong to the first
track. Repeated segment endpoints may appear in adjacent intervals. Zero common support is unknown;
unsupported gap interiors are unknown, never bridged. This is planar within-distance (distance zero
also supports equality), not the full set of MobilityDB temporal predicates.

`addTrackQualityRecipe` bit 4 (`excessiveSpeed`) is by default the step-speed test documented in the
recipe contract, so the valid fix after an outlier is also flagged. Set
`speedOutlierMode: 'point-centred'` for an outlier test on the fix: a fix is flagged only when the
speed test fails on both adjacent steps, and a track-end fix when its single step fails and the next
step inward does not. A single pass does not catch two consecutive displaced fixes.

## Directed edge-position routing example

{/* lens:GPUNetworkPointRouting */}

::lens{scenario=gpu-network.network-point-routing rung=graph}

The three queries show same-edge partial travel, a route through two legal turns, and an
unreachable reverse journey. Inspect the costs alongside completion status and edge predecessors.

## Residual flow example

`GPUNetworkFlow`: Legacy solver selection and successive-shortest-path defaults remain. Opt-in mode
primal-dual uses three phases by default, with bounded relaxation/phase budgets; incomplete feasible
flows require completion and cut-certification status checks. Device bindings and command budgets
constrain all routes.

{/* lens:GPUNetworkFlow */}

::lens{scenario=gpu-network.network-flow rung=graph}

The four-vertex directed network includes antiparallel arcs. The final edge flows conserve mass,
and the source-side cut certifies total flow five. The summary retains the certificate separately
from the value.

`output.status` optionally exposes one-row `uint32` ports `invalidCount`, `converged`,
`iterationCount` and `iterationLimitReached`. `iterationCount` counts committed batches:
augmentations for reference/successive-shortest-path and phases for uncosted/primal-dual,
including when the solver publishes an incumbent at its budget. A fully solved finite request with unserved
flow still converged; inspect `summary[3]` for that shortfall. Invalid problems report
`invalidCount = 1` and neither stopping flag. Every encoding clears and rewrites the ports.
The graph operation and CPU twin accept the same named ports and honor explicit solver budgets.
Omitted CPU budgets are uncapped. Outputs, including named ports and path columns, need distinct buffers.
When optional `topology` is supplied, the CPU twin reads both CSR ports and checks zero-based,
monotonic offsets ending at the edge count, source-row membership, and destination identity against
the aligned edge list. Inconsistent connectivity produces zero flow, cut, summary and exact cost,
with `invalidCount=1`, `converged=0`, and `iterationLimitReached=0` on reference GPU, parallel GPU and CPU forms.

## `GPUNetworkLouvain`

Supply exactly one of weighted CSR `topology` or row-major dense `weights`. Sparse sorting,
segmented reduction and quotient aggregation use published graph primitives. The [Louvain
guide](louvain.md) is the canonical input/output and numerical contract, including shared
`GPUNetworkLouvainOperation`/`CPUNetworkLouvainOperation` properties, per-frame parameters,
device-dependent limits, and the upstream single-level optimizer comparison.

{/* lens:GPUNetworkLouvain */}

::lens{scenario=gpu-network.network-louvain.sparse rung=graph}

The sparse CSR lens uses two disconnected weighted pairs, which contract to two communities.
The docs-sized sparse example budgets two levels with four rounds per level: one contraction
and a second level that confirms convergence. Its nine-slot objective trace starts at the
singleton partition, reaches modularity 0.5, and retains unused slots as NaN. The graph includes
quotient coarsening and all declared rounds; no shown graph nodes or readbacks are discarded.
The dense reference remains available for comparison:

::lens{scenario=gpu-network.network-louvain rung=graph}

## Exact facility assignment example

{/* lens:GPUNetworkFacilityOptimization */}

::lens{scenario=gpu-network.network-facility-optimization rung=graph}

One origin can reach either facility, while the other can reach only the first. Exhaustive unsplit
assignment allocates the flexible origin to the second facility, certifying a feasible objective
of three where demand-order nearest allocation fails.

The packed four-word result is `output.summary` (formerly `output.status`):
`[valid, feasibleIncumbent, enumerationComplete, examinedCandidates]`. Optional `output.status`
ports are `invalidCount`, `converged`, `iterationCount` and `iterationLimitReached`, each one
`uint32` row. `iterationCount` counts nonempty candidate-scoring batches (at most 16); it is
zero for a zero-candidate budget or inputs rejected before scoring. An overflowing objective
invalidates the result but retains the number of batches already scored. This executed-work
counter is independent of `summary[3]`, which retains the examined candidate count (including
infeasible candidates) for accepted results and is zero for rejected results. `converged` certifies exhaustive
enumeration independently of whether any feasible assignment exists. A valid budget exit sets
`iterationLimitReached` and publishes any incumbent. Invalid input or an overflowing feasible
objective sets `invalidCount = 1` and clears both stopping flags. No claim of infeasibility is
made before complete enumeration.

Compile-time controls are the operation, view lengths, `facilityCount`, `selectCount`,
`maximumCandidates` and which status ports exist. Costs, demand, capacities and threshold are
per-frame buffer contents. Every encoding rewrites all outputs, including empty-demand results.
All four optimization contributors reject unsupported own fields in `output.status` at
construction, even when an extended object is structurally assignable in TypeScript. Only
`invalidCount`, `iterationCount`, `converged` and `iterationLimitReached` are supported.
The flow, time-routing and rational-access GPU and CPU operation forms apply the same field
validation before declaring outputs, so an extra status key cannot replace a primary port.
Each output and named port needs a distinct buffer. `estimatedCommandNodeCount` includes the
candidate batches, hierarchical reductions and optional status pass, at most 83 commands at the
largest search. Construction enforces the shared command budget; encoding checks storage
binding sizes before creating scratch or kernels.

## Departure-time routing example

{/* lens:GPUNetworkTimeRouting */}

::lens{scenario=gpu-network.network-time-routing rung=graph}

The middle edge becomes more expensive as time increases. Departing at three reaches the
intermediate state at four and the destination at nine; the example captures the full arrival
and predecessor tables so the time-dependent evaluation remains inspectable.

Optional `output.status` has one-row `uint32` ports `invalidCount`, `converged`, `iterationCount`
and `iterationLimitReached`. The iteration count includes the final no-change sweep. A valid
budget exit publishes the current arrival labels and predecessors and sets
`iterationLimitReached = 1`; unreachable labels are conclusive only after convergence. Invalid
inputs or nonfinite arrival arithmetic set `invalidCount = 1` and clear both stopping flags.
The existing four-word `summary` layout is unchanged. Outputs and ports need distinct buffers.
The GPU and CPU operation forms accept the same named graph values and honor the same
`maximumRelaxations` sweep budget (default: state count). Zero sweeps retain only the initial
label and report a budget exit. CPU labels use float64 internally, so rounding can affect ties
and the sweep in which convergence is certified.

## Congestion allocation example

{/* lens:GPUNetworkRationalAccess */}

::lens{scenario=gpu-network.network-rational-access rung=graph}

A demand of ten initially chooses the first of two tied facilities, then balances to five each.
A second unreachable origin retains three units of explicitly unserved demand and a NaN score.


The packed gap/cycle result is `output.summary` (formerly `output.status`):
`[valid, equilibriumGapWithinTolerance, completedCycles, maximumGapBits]`. Optional named
`output.status` ports expose `invalidCount`, `converged`, `iterationCount` and
`iterationLimitReached`, each one `uint32` row. The iteration count is completed simultaneous
allocation shifts; the initial gap evaluation is cycle zero. Convergence requires the final gap
within tolerance, so an integer rounding stall still exhausts the cycle budget. A valid budget
exit publishes the allocation, final scores and unserved demand. Invalid inputs or nonfinite
arithmetic set `invalidCount = 1` and clear both stopping flags. Completed shifts remain counted
when the next gap evaluation fails. Ports reset on every encoding
and require separate output buffers. Graph and CPU operations expose the same port names; the
pure CPU result now has `summary` and named `status` arrays, and retains the cycle budget.

Rational-access cycles use one 64-lane workgroup within the existing 64-origin and 32-facility
limits. Each origin scores and shifts independently; facility lanes sum loads and simultaneous
inflow in ascending origin order, preserving float32 addition order and earliest-facility ties.
Uniform workgroup exits stop at equilibrium or invalid arithmetic without unrolling command nodes
per cycle. Ordered initialization and score publication preserve partial outputs on rejection.
The graph uses two kernels plus optional named-status publication. This remains a bounded reference
solver; it does not extend to larger allocations.
