# Bounded network optimization and trajectory analysis

These graph contributors implement the supported modes below. They accept explicit packed views;
the application owns preparation, uploads, graph compilation, submission and optional readback.
No contributor creates a CPU numerical fallback or combines source batches implicitly. Build one
contributor per independently prepared batch when source partition boundaries must remain distinct.
All outputs are caller-owned. Graph scratch is transient and graph-owned.

| Backlog | Contributor | Implemented contract | Limits and exclusions |
| --- | --- | --- | --- |
| NW01 | `GPUNetworkPointRouting` | Batched static paths between fractional positions on identified directed edges; legal incoming/outgoing turn states, same-edge forward travel, parallel-edge identity, edge predecessor provenance | 128 edges, 4096 explicit turns. Fractions retain the selected directed-edge state even at endpoints. Reverse direction requires a separate directed edge. Snapping, reverse-edge discovery, and arbitrary restriction history are caller preparation. |
| NW02 | `GPUNetworkFlow` | Integral maximum flow, residual source cut, edge-aligned feasible flow and flow/cut equality certificate; unit capacities provide edge-disjoint path capacity | 128 nodes, 1024 original directed edges; total capacity at most 16,777,215. Explicit reverse residuals preserve antiparallel edges. Does not emit a path decomposition. |
| NW03 | `GPUNetworkFlow` with `costs`, `prepareGPUNetworkFlowDemands` | Minimum-cost requested/maximum flow, reverse cancellation, explicit balanced-demand preparation and optional exact 64-bit integer objective | Nonnegative original costs ≤32767; no negative-cost circulation or lower-bound constraints. Summary cost is f32; `exactCost` preserves the integer objective. |
| NW04 | `GPUNetworkLouvain` | Weighted directed/undirected modularity, deterministic or seeded Fisher–Yates visitation, quotient coarsening, objective trace and convergence | Dense adjacency, ≤32 vertices. Undirected loop diagonals are twice loop weight; directed arcs/loops occur once. This is a modularity heuristic, never a global optimum certificate. |
| NW05 | `GPUNetworkFacilityOptimization` | Exhaustive backup coverage, p-dispersion and unsplit capacitated p-median; feasibility and exhaustive completion certificates | At most 16 facilities, 64 demand rows and 1,048,576 candidates. Capacitated enumeration has `facilityCount ** demandCount` candidates. Candidates run independently in GPU batches of at most 65,536, with deterministic hierarchical winner reductions. Exhaustive search can still be expensive within the limit; there is no branch-and-bound or scalable mixed-integer solver. |
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

Louvain reports `[valid, converged, completedLevels, completedSweeps]`; its trace includes the
singleton objective followed by each completed sweep. Unused trace rows are NaN. Zero-weight
graphs return singleton labels, objective zero and convergence. Symmetry is an exact input requirement only in undirected mode. Directed modularity uses
community outgoing-strength times incoming-strength as its null-model penalty. An optional uint32
`seed` chooses reproducible Fisher–Yates visitation; it does not promise the same random ordering
as NetworkX. Compare objective and partition invariants, not label IDs. A budget-limited partition remains a heuristic incumbent.

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
initial velocity variance and positive maximum gap. Process covariance is
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
relative seconds. Subtract a nearby epoch before converting timestamps to f32.

Restriction parameters are `[minTime,maxTime,minX,minY,maxX,maxY,maxGap]`. Nonstationary segments
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
[`network-movement-coverage.py`](../../../../modules/graph/test/oracles/network-movement-coverage.py),
with committed
[`network-movement-coverage.json`](../../../../modules/graph/test/oracles/fixtures/network-movement-coverage.json).
It pins NetworkX 3.5 and NumPy 2.2.6. Network flow and modularity fixtures call NetworkX directly;
Kalman fixtures are independent dense matrix compositions, not MovingPandas calls. Tests also
check conservation, reverse residual cancellation, self-loops, a greedy-allocation counterexample,
iteration exhaustion, missing measurements, covariance symmetry, gaps, zero support, threshold
ties, and mutable input updates. Node tests audit shader parsing, reserved identifiers and the
portable eight-storage-binding limit.

The selected RAAM source is
[`access` v1.1.9 `raam.py`](https://github.com/pysal/access/blob/v1.1.9/access/raam.py);
the implementation here is independent and exposes final-load and convergence semantics explicitly.
The additional [advanced generator](../../../../modules/graph/test/oracles/network-movement-advanced.py)
calls MovingPandas 0.22.4/StoneSoup 1.8 smoothing and polygon clipping, tslearn 0.6.4 Euclidean-cost
DTW/LCSS, similaritymeasures 1.3.0 Fréchet, access 1.1.9 RAAM, and spopt 0.7.0/PuLP 3.3.0 CBC facility models directly. It also includes an
independent full-covariance NumPy solve, pyproj ECEF projection, and NetworkX virtual-split/time-expanded
routing equivalents. A separate [server generator](../../../../modules/graph/test/oracles/network-movement-postgres.py)
executes PostgreSQL 16.6, PostGIS 3.5.3, pgRouting 3.8.0 and MobilityDB 1.2.1 directly.
Its [saved results](../../../../modules/graph/test/oracles/fixtures/network-movement-postgres.json)
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

The [benchmark fixture](../../../../modules/graph/test/oracles/fixtures/network-movement-benchmark.json)
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

For full Kalman covariance set `covarianceMode: 'full'`; parameters are
`[qxx,qxy,qyy,rxx,rxy,ryy,vxx,vxy,vyy,maxGap]`. Process/velocity covariance must be positive semidefinite;
measurement covariance must be positive definite. Covariance outputs contain **four float32x4 rows
per observation**, the column-major 4x4 covariance in `[x,y,vx,vy]` state order. Full RTS uses a pivoted
4x4 inverse; singular/nonfinite backward solves retain forward values and set bit 8.
`prepareGPUTrajectoryObservations` subtracts a bigint epoch before f32 conversion and transforms
WGS84 longitude/latitude through ellipsoidal ECEF into an explicitly chosen local east/north plane.
It returns projection/epoch metadata and performs no filtering or GPU readback.

For polygons, append flattened ring coordinates after the seven restriction parameters and set
`polygonRingVertexCounts`. Rings close implicitly; even-odd parity supports holes regardless of ring
orientation. Valid simple rings are caller-owned topology. Geometry boundary classification uses an
f32 cross-product tolerance of `1e-6 * max(1,edgeLengthSquared)` and coordinate bounds expanded by
1e-6 coordinate units; this is not an exact-topology predicate. Use suitable local units and avoid
features below this precision. `timeBoundary` selects closed, open, left-closed or right-closed;
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

## Directed edge-position routing example

{/* lens:GPUNetworkPointRouting */}

::lens{scenario=gpu-network.network-point-routing rung=graph}

The three queries show same-edge partial travel, a route through two legal turns, and an
unreachable reverse journey. Inspect the costs alongside completion status and edge predecessors.

## Residual flow example

{/* lens:GPUNetworkFlow */}

::lens{scenario=gpu-network.network-flow rung=graph}

The four-vertex directed network includes antiparallel arcs. The final edge flows conserve mass,
and the source-side cut certifies total flow five. The summary retains the certificate separately
from the value.

## Modularity optimization example

{/* lens:GPUNetworkLouvain */}

::lens{scenario=gpu-network.network-louvain rung=graph}

Two disconnected weighted pairs contract to two communities. The objective trace starts at the
singleton partition and records accepted sweeps up to modularity 0.5.

## Exact facility assignment example

{/* lens:GPUNetworkFacilityOptimization */}

::lens{scenario=gpu-network.network-facility-optimization rung=graph}

One origin can reach either facility, while the other can reach only the first. Exhaustive unsplit
assignment allocates the flexible origin to the second facility, certifying a feasible objective
of three where demand-order nearest allocation fails.

## Departure-time routing example

{/* lens:GPUNetworkTimeRouting */}

::lens{scenario=gpu-network.network-time-routing rung=graph}

The middle edge becomes more expensive as time increases. Departing at three reaches the
intermediate state at four and the destination at nine; the example captures the full arrival
and predecessor tables so the time-dependent evaluation remains inspectable.

## Congestion allocation example

{/* lens:GPUNetworkRationalAccess */}

::lens{scenario=gpu-network.network-rational-access rung=graph}

A demand of ten initially chooses the first of two tied facilities, then balances to five each.
A second unreachable origin retains three units of explicitly unserved demand and a NaN score.
