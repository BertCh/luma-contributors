var e={slug:`api/gpu-network/network-movement-coverage`,title:`Bounded network optimization and trajectory analysis`,html:`<h1 id="bounded-network-optimization-and-trajectory-analysis">Bounded network optimization and trajectory analysis</h1>
<p>These graph contributors implement the supported modes below. They accept explicit packed views;
the application owns preparation, uploads, graph compilation, submission and optional readback.
No contributor creates a CPU numerical fallback or combines source batches implicitly. Build one
contributor per independently prepared batch when source partition boundaries must remain distinct.
All outputs are caller-owned. Graph scratch is transient and graph-owned.</p>
<table>
<thead>
<tr>
<th>Backlog</th>
<th>Contributor</th>
<th>Implemented contract</th>
<th>Limits and exclusions</th>
</tr>
</thead>
<tbody>
<tr>
<td>NW01</td>
<td><code>GPUNetworkPointRouting</code></td>
<td>Batched static paths between fractional positions on identified directed edges; legal incoming/outgoing turn states, per-turn penalties, same-edge forward travel, parallel-edge identity, edge predecessor provenance</td>
<td>Reference mode: 128 edges, 4096 explicit turns. Scalable mode: device binding and command-budget limits. Fractions retain the selected directed-edge state even at endpoints. Reverse direction requires a separate directed edge. Snapping, reverse-edge discovery, and arbitrary restriction history are caller preparation.</td>
</tr>
<tr>
<td>NW02</td>
<td><code>GPUNetworkFlow</code></td>
<td>Integral maximum flow, residual source cut, edge-aligned feasible flow, flow/cut equality certificate and optional path decomposition; unit capacities provide edge-disjoint paths</td>
<td>Reference mode: 128 nodes, 1024 original directed edges and total capacity at most 16,777,215. Parallel mode: up to 30 million nodes/edges and total capacity at most 4,294,967,295, subject to device binding and command-budget limits. Optional decomposition requires at most 128 expanded vertices and 1024 expanded edges with either solver. Explicit reverse residuals preserve antiparallel edges.</td>
</tr>
<tr>
<td>NW03</td>
<td><code>GPUNetworkFlow</code> with <code>costs</code>, <code>prepareGPUNetworkFlowDemands</code></td>
<td>Minimum-cost requested/maximum flow, reverse cancellation, explicit balanced-demand preparation and optional exact 64-bit integer objective</td>
<td>Nonnegative original costs ≤32767; no negative-cost circulation or lower-bound constraints. Summary cost is f32; <code>exactCost</code> preserves the integer objective.</td>
</tr>
<tr>
<td>NW04</td>
<td><code>GPUNetworkLouvain</code></td>
<td>Sparse parallel weighted directed/undirected modularity, quotient coarsening, objective trace and convergence; legacy dense reference</td>
<td>See <a href="/luma-contributors/docs/api/gpu-network/louvain#parameters-and-limits">Louvain limits</a> for sparse addressing, per-binding storage, dense-reference and command-budget bounds. This is a modularity heuristic, never a global optimum certificate.</td>
</tr>
<tr>
<td>NW05</td>
<td><code>GPUNetworkFacilityOptimization</code></td>
<td>Exhaustive backup coverage, p-dispersion and unsplit capacitated p-median; feasibility and exhaustive completion certificates</td>
<td>At most 16 facilities, 64 demand rows and 1,048,576 (2^20) candidates, checked at construction. Backup coverage and p-dispersion enumerate <code>2 ** facilityCount</code> candidates, so 64 rows is their real demand limit. Capacitated p-median enumerates <code>facilityCount ** demandRows</code> candidates, so the candidate limit, not the 64-row ceiling, bounds its demand rows: 20 rows at 2 facilities, 10 at 4, 5 at 16. For larger problems use <code>GPULocationAllocation</code>. Candidates run independently in GPU batches of at most 65,536, with deterministic hierarchical winner reductions. Exhaustive search can still be expensive within the limit; there is no branch-and-bound or scalable mixed-integer solver.</td>
</tr>
<tr>
<td>NW06</td>
<td><code>GPUNetworkTimeRouting</code>, <code>prepareGPUNetworkRestrictionStates</code></td>
<td>FIFO affine or piecewise-arrival routing, first-sample waiting, reachable forbidden-sequence states and original-edge provenance</td>
<td>≤256 states,4096 transitions,256 samples/transition. Non-FIFO schedules and arbitrary time windows remain unsupported. CPU topology preparation never solves routes.</td>
</tr>
<tr>
<td>NW07</td>
<td><code>GPUNetworkRationalAccess</code></td>
<td>Congestion equalization with integral/fractional demand, fractional/absolute shift caps, unserved demand and explicit equilibrium gap</td>
<td>≤64 origins,32 facilities,4096 cycles; f32 demand total ≤2^24. Integral rounding stalls remain unconverged. Fractional conservation has f32 rounding tolerance.</td>
</tr>
<tr>
<td>MV01</td>
<td><code>GPUTrajectoryKalman</code>, <code>prepareGPUTrajectoryObservations</code></td>
<td>Isotropic or full correlated 4x4 Kalman/Joseph covariance and RTS smoothing; explicit WGS84 local-plane/bigint epoch preparation</td>
<td>Linear planar constant velocity; sequential per track, parallel across tracks. Local tangent projection is explicit and drops height/curvature, not an automatic CRS choice.</td>
</tr>
<tr>
<td>MV02</td>
<td><code>GPUTrajectoryRestriction</code></td>
<td>Rectangle/polygon-ring restriction, holes, open/closed/half-open endpoints, compact/merged intervals and analytic pairwise within-distance intervals</td>
<td>Planar linear interpolation; ≤128 polygon vertices, ≤256 slots/segment. Pairwise static query topology, ≤65536 slots/pair. Gaps/invalidity/capacity are explicit; no geodesic interpolation or general temporal SQL algebra.</td>
</tr>
<tr>
<td>MV03</td>
<td><code>GPUTrackSimilaritySearch</code></td>
<td>Admissible DTW/Fréchet/LCSS bounds, stable GPU survivor compaction, bounded DP scratch and exact threshold top-k with pair provenance</td>
<td>Explicit candidate index/pair batches; no implicit all-pairs/spatial-index construction. Top-k is among supplied threshold-matching pairs, not unseen trajectories. No densification or geographic metric.</td>
</tr>
</tbody>
</table>
<h2 id="completion-validity-and-provenance">Completion, validity and provenance</h2>
<p><code>GPUNetworkFlow.output.summary</code> has eight uint32 words: valid input, complete search, flow value,
unserved requested flow, augmentation count, cost as float32 bits, residual cut capacity, and cut
certificate. <code>complete=0</code> means the feasible flow is an incumbent. An incomplete positive unserved
amount does not prove infeasibility. <code>complete=1</code> means either the requested value was attained or
no residual source/sink path remains. Only <code>cutCertified=1</code> proves the reported maximum-flow value.
Minimum-cost optimality is conditional on valid/complete input and the supported nonnegative
original-cost model. Original-edge output order preserves capacities and provenance without
merging parallel or antiparallel edges.</p>
<p>The explicit/reference route uses one serial invocation and is construction-limited to 128 nodes,
1024 original edges and 65536 augmentations (default 1024). Its reviewed contract allowance names
all three serial axes at that kernel's invocation count; it does not apply to parallel kernels or
relax any constructor limit. Reaching the augmentation budget still publishes the feasible flow
and its existing incomplete/iteration-limit status.</p>
<p>The contributor captures compile-time counts, solver choices and nested input/output/status port
maps before validation. The GPU operation and CPU twin likewise capture counts and graph value
names before deferred execution. Later mutation of the caller's props cannot enlarge the reference
kernel or redirect its writes. Graph views and buffers keep their identities: callers still change
per-frame values by updating their contents, rather than replacing entries in a props object.</p>
<p>Point routing reports 0 for a complete query, 1 for invalid data and 2 for an iteration-limited
query. The query's <code>predecessorEdges</code> row preserves the winning route, including a legal cycle
when the source and destination are the same edge and the destination fraction is behind the
source fraction. For that returning route, the source entry records the final incoming edge.
To reconstruct it, begin at the destination, follow at least one predecessor, and stop on reaching
the source. Direct same-edge travel and unreachable labels retain <code>0xffffffff</code>; use the cost
and status to distinguish them. All other source entries retain the sentinel.
Time routing reports <code>[valid, converged, chosenTargetState, arrivalTimeBits]</code>. Only a valid
converged search can establish that an infinite distance is unreachable. Transition provenance
from <code>prepareGPUNetworkRestrictionStates().originalEdges</code> maps predecessor transitions back to
the original directed edges. Initial states are the empty restriction prefix at each vertex.
The topology helper does no searching, consumes explicit JavaScript topology arrays, and returns
typed arrays for explicit upload; it does not read GPU data.</p>
<p>The <a href="/luma-contributors/docs/api/gpu-network/louvain">Louvain guide</a> defines summary/trace layouts, completion status and the
differences between sparse parallel, dense reference and CPU execution. Compare objective and
partition invariants rather than label IDs. A budget-limited partition remains a heuristic incumbent.</p>
<p>Facility optimization reports <code>[valid, feasibleIncumbent, enumerationComplete, examinedCandidates]</code>.
When enumeration is complete, a feasible incumbent is the exhaustive optimum for the supplied f32
objective comparisons; no incumbent then certifies infeasibility for this bounded model. When
incomplete, neither conclusion follows. Objective is NaN without a feasible incumbent. Backup
coverage without <code>selectCount</code> first minimizes the facilities needed to cover every demand row,
then maximizes the demand with at least two covering facilities at that count. Supplying a count
instead fixes that count. P-dispersion maximizes the minimum selected pair distance and requires
symmetric costs and at least two facilities. Capacitated p-median assigns whole demand rows,
minimizing demand-weighted distance with individual capacity constraints. Assignments are published
only for capacitated p-median; other modes leave the assignment sentinel intact.</p>
<p>Candidates are scored independently in batches that reuse one score buffer. Each candidate
retains input row order for floating-point sums and capacity checks. Hierarchical reductions select
the lowest original candidate index on objective ties, including ties across batches. A partial
<code>maximumCandidates</code> still examines exactly the original enumeration prefix; batch padding does
not count as examined work. Every encoding resets the cursor, incumbent and validation status.
The number of assignments remains exponential in demand count for capacitated p-median; batching
bounds scratch storage rather than changing the search complexity.</p>
<p>RAAM parameters are <code>[tau, rho, initialStep, minimumStep, halfLife, gapTolerance]</code>. Travel costs are
divided by tau, and supplies are multiplied by rho. To use global mean demand/supply normalization,
compute and supply rho explicitly. Inputs require positive finite supplies; unreachable costs are
positive infinity. Missing/negative costs are invalid. Nonintegral demand requires <code>demandMode: 'fractional'</code>. Demand conservation
is <code>sum(assignments[origin,*]) + unservedDemand[origin] = demand[origin]</code>. Scores are NaN for
unserved and zero-demand origins. Final scores use final loads rather than the preceding cycle's
loads; this deliberately avoids the stale final cost in the reference <code>iterate_raam</code> implementation.
The reported gap is maximum used-facility cost minus minimum reachable cost. Equilibrium status
does not hide unserved rows; inspect both outputs. <code>stepMode: 'fraction'</code> caps a fraction of each origin's demand; <code>'absolute'</code> caps a demand amount.
Integral mode floors equalizing shifts and uses round-to-even early inflow scaling. Fractional mode
retains continuous shifts and does not round early inflow scaling.</p>
<h2 id="trajectory-conventions">Trajectory conventions</h2>
<p>Kalman parameters are acceleration spectral density, positive measurement variance, nonnegative
initial velocity variance and maximum gap, where 0 means no limit. Process covariance is
<code>q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]</code>. A new segment starts with the first measured position and zero
velocity. Missing positions produce prediction-only steps after initialization; no state is
invented before the first measurement. Nonfinite timestamps or backward time produce invalid
estimates and reset initialization. Equal timestamps are supported as repeated measurements.
The raw observations, filtered outputs and optional smoothed outputs remain separate. For both
estimation and restriction, offsets must form a complete nondecreasing partition: first offset zero,
last offset observation count, and every intermediate offset inside that range. Overlapping or
incomplete partitions are outside the input contract; outputs outside supplied track spans are not
implicitly initialized.</p>
<p>State rows are <code>[x,y,vx,vy]</code>. Covariance rows are <code>[positionVariance,covariance,covariance,velocityVariance]</code>,
shared by the independent x and y axes. Status bits are reset=1, prediction-only=2, invalid=4,
singular backward solve=8. A singular backward prediction retains the forward estimate and marks
bit 8; it is not represented as a successful RTS update. Noise units must match coordinates and
relative seconds. Subtract a nearby epoch before converting timestamps to f32. Far from one epoch,
pass <code>times</code> as <code>uint32x2</code> Int64 <code>(low, high)</code> words instead: each step is differenced exactly and
only then converted to f32, so 10 Hz samples stay 0.1 s apart however far they are from the epoch.
Set <code>timeUnitsPerSecond</code> (default 1, for example 1000 for epoch milliseconds) so the model keeps
running in seconds. Restriction word <code>times</code> need a per-frame <code>timeOrigin</code> (<code>[low, high]</code> uint32):
<code>minimumTime</code>, <code>maximumTime</code> and the output intervals are f32 relative to it, so keep the origin
near the window. <code>maximumGap</code> is always in the timestamps' unit: relative seconds for f32 Kalman
times, the word unit (for example milliseconds) for word times. 0 means no limit in both contributors.
Kalman and restriction also read the trajectory port (<code>timestamps</code>, <code>trackOffsets</code>) with an optional
<code>temporalContext</code>; Kalman then takes <code>timeUnitsPerSecond</code> from <code>context.unit</code>, for float32 times too.
Both also take an optional <code>spatialContext</code> that resolves to planar coordinates (the default); a
longitude-latitude, grid or topology context throws at construction. Kalman and a restriction with
a distance predicate additionally need the native metric and reject <code>local-lambert-equal-area</code> frames.
The per-observation <code>status</code> words document every bit in <code>GPU_TRAJECTORY_KALMAN_STATUS</code> and
<code>GPU_TRAJECTORY_RESTRICTION_STATUS</code>.</p>
<p>Restriction parameters are <code>[minTime,maxTime,minX,minY,maxX,maxY,maxGap]</code> (<code>maxGap</code> 0 means no limit). Nonstationary segments
need strictly increasing timestamps; equal-time equal-position rows are singletons. Crossing
intervals preserve their original segment row, so adjacent endpoints can appear in both rows.
Status bits are interval=1, singleton=2, invalid=4, unsupported gap=8, full temporal support inside=16,
temporal support=32. Per-track relationships are <code>[ever,always,supportRows,unknownRows]</code> with
<code>0xffffffff</code> for unknown. Zero common temporal support gives unknown for both predicates.
A known inside witness establishes ever=true despite other unknown rows; a known outside witness
establishes always=false. Unknown gaps are never treated as straight-line observations.</p>
<p>Similarity search returns evaluated distance, endpoint lower bound, threshold match and status.
Status 0 means the existing dynamic program ran, 1 means safely pruned, 2 means invalid input and
4 means dynamic-program capacity exhaustion and 8 means survivor-capacity exhaustion. Pruned distances are NaN. The lower bound is the
maximum of first/last endpoint distances: every admissible DTW or Fréchet coupling includes both
endpoints, including with a band. A conservative f32 margin retains threshold-neighbor cases.
Band-disconnected DTW results can be infinity with evaluated status and match=false. LCSS instead accepts a minimum normalized similarity threshold and reports <code>similarity</code> and
<code>upperBound</code>. Every matched element must have at least one eligible counterpart, so the minimum
of eligible-element counts divided by the shorter track length bounds normalized LCSS from above.
The bound considers tolerance and ignores the band conservatively, and costs O(n*m); it saves dynamic-program work and
scratch for rejected pairs. An exclusive GPU prefix scan compacts surviving pairs in original order;
<code>survivorCapacity</code> bounds the dynamic-program dispatch/scratch. Overflow pairs have status 8 and NaN,
never a false nonmatch. Optional <code>survivorSummary</code> is <code>[required,evaluated,overflow]</code>.
<code>topK</code> returns original pair row IDs ordered by distance (or descending LCSS), then row ID for ties.
<code>topKSummary</code> is <code>[written,complete]</code> per query; invalid/overflow pairs make that query incomplete.
The supplied pair arrays are the candidate index and stable provenance; batching is explicit.</p>
<h2 id="evidence-and-remaining-gates">Evidence and remaining gates</h2>
<p>The independent generator is
<a href="../../../modules/analysis/test/oracles/network-movement-coverage.py"><code>network-movement-coverage.py</code></a>,
with committed
<a href="../../../modules/analysis/test/oracles/fixtures/network-movement-coverage.json"><code>network-movement-coverage.json</code></a>.
It pins NetworkX 3.5 and NumPy 2.2.6. Network flow and modularity fixtures call NetworkX directly;
Kalman fixtures are independent dense matrix compositions, not MovingPandas calls. Tests also
check conservation, reverse residual cancellation, self-loops, a greedy-allocation counterexample,
iteration exhaustion, missing measurements, covariance symmetry, gaps, zero support, threshold
ties, and mutable input updates. Node tests audit shader parsing, reserved identifiers and the
portable eight-storage-binding limit.</p>
<p>The selected RAAM source is
<a href="https://github.com/pysal/access/blob/v1.1.9/access/raam.py"><code>access</code> v1.1.9 <code>raam.py</code></a>;
the implementation here is independent and exposes final-load and convergence semantics explicitly.
The additional <a href="../../../modules/analysis/test/oracles/network-movement-advanced.py">advanced generator</a>
calls MovingPandas 0.22.4/StoneSoup 1.8 smoothing and polygon clipping, tslearn 0.6.4 Euclidean-cost
DTW/LCSS, similaritymeasures 1.3.0 Fréchet, access 1.1.9 RAAM, and spopt 0.7.0/PuLP 3.3.0 CBC facility models directly. It also includes an
independent full-covariance NumPy solve, pyproj ECEF projection, and NetworkX virtual-split/time-expanded
routing equivalents. A separate <a href="../../../modules/analysis/test/oracles/network-movement-postgres.py">server generator</a>
executes PostgreSQL 16.6, PostGIS 3.5.3, pgRouting 3.8.0 and MobilityDB 1.2.1 directly.
Its <a href="../../../modules/analysis/test/oracles/fixtures/network-movement-postgres.json">saved results</a>
cover directed fractional same-edge, parallel-edge and unreachable routes; polygon holes;
all four time endpoint policies; continuous encounter intervals; and absent common support.
SQL NULL unreachable costs map to infinity, and NULL temporal relationships map to <code>0xffffffff</code>.
The crossing fixture exposes a MobilityDB 1.2.1 difference: <code>aDwithin</code> returns true while
<code>always_eq(tDwithin(first, second, 1), true)</code> returns false. The GPU's full-common-support
always semantics match the latter. Both raw server results are preserved; blanket compatibility
with <code>aDwithin</code> is not claimed. Valhalla parity and broad-device coverage remain untested.</p>
<p>The isolated native reference stack used the official PostgreSQL 16.6 and PostGIS 3.5.3 source
archives, pgRouting tag <code>v3.8.0</code>, and MobilityDB tag <code>v1.2.1</code>. PostgreSQL configured with
<code>--without-readline --without-icu --without-zlib</code>; PostGIS omitted raster, topology, protobuf
and address standardizer. GEOS 3.14.1, PROJ 9.9.0, JSON-C 0.18 and GSL 2.7 came from an
existing isolated environment, with Boost 1.87 headers for pgRouting. Core PostGIS library,
control and new-install SQL targets avoided an unrelated raster upgrade-script generation
failure. On macOS, pgRouting's installed <code>.so</code> bundle needed a <code>.dylib</code> symlink for PostgreSQL's
loader. No reference algorithm source was changed. The local socket-only stack peaked at
505 MB and was removed after fixture capture. Reproduction needs these extensions installed;
the generator accepts <code>--psql</code> and standard <code>PGHOST</code>, <code>PGPORT</code>, <code>PGDATABASE</code>, <code>PGUSER</code> settings.</p>
<p>The <a href="../../../modules/analysis/test/oracles/fixtures/network-movement-benchmark.json">benchmark fixture</a>
records 30 workloads on Apple Metal3 WebGPU, including all nine contributors and 4096 trajectory
pairs. Upload/preparation, compilation, first execution, three warm executions and readback are
separate. Full-survivor 4096-pair Fréchet took median 5.2 ms; 205 survivors took 1.1 ms with bounded
scratch. This selectivity comparison is not a CPU speedup claim. Dense Louvain at 32 vertices took
28.9 ms and remains a correctness-oriented bounded solver. See the
<a href="/luma-contributors/docs/api/gpu-network/network-movement-benchmarks">measurement notes</a> for workload-specific limits.</p>
<h2 id="advanced-input-and-output-layouts">Advanced input and output layouts</h2>
<p>For piecewise routing set <code>costModel: 'piecewise-arrival'</code> and <code>samplesPerTransition >= 2</code>.
Each transition has a fixed-width sequence of <code>[departure,arrival]</code> pairs. Departure samples must
strictly increase, arrival samples must not decrease, and arrival must be at least departure.
The earliest label before the first sample waits until that sample; inside the sampled domain,
arrival is linearly interpolated; after the last sample, the final travel duration remains constant.
<code>waitingPolicy: 'wait-until-first-sample'</code> makes that behavior explicit. The entire cost curve is FIFO,
so later labels cannot improve a state and the earliest-label relaxation is sound.</p>
<p><code>prepareGPUNetworkFlowDemands</code> takes balanced integer net demands (positive=inflow, negative=supply),
adds zero-cost super-source/sink arcs and returns explicit upload arrays. The first <code>originalEdgeCount</code>
flow outputs retain original-edge provenance. Optional <code>exactCost</code> contains <code>[low32,high32]</code> with no
integer rounding inside the documented capacity/cost envelope.
The adapter returns <code>solver: 'reference' | 'parallel'</code>; forward it with the expanded arrays to the
contributor. The default <code>solver: 'auto'</code> chooses reference only when the expanded problem has at
most 128 vertices, 1024 edges and total capacity 16,777,215, including both super arcs. Thus 126
input vertices can use reference; 127 require parallel. Explicit <code>solver: 'reference'</code> enforces
those expanded limits, and <code>'parallel'</code> opts in even for small inputs. Parallel expanded shape is
bounded at 30 million vertices/edges and total capacity 4,294,967,295; device bindings and command
budgets still apply at contributor construction. Original costs must be nonnegative integers no
larger than 32767 for reference or <code>min(32767, floor(2147483646 / expandedNodeCount))</code> for parallel.</p>
<p>Optional <code>GPUNetworkFlow.output.paths</code> decomposes the <strong>final feasible flow</strong>, including after
reverse cancellation, into simple source-to-sink paths with integral volumes. This reference
pass requires at most 128 expanded vertices and 1024 expanded edges, including super arcs,
with either flow solver. The path capacity is <code>paths.volumes.length</code> in the range 1..1024.
Allocate <code>lengths</code> with that capacity, <code>edgeRows</code> with <code>capacity * (nodeCount - 1)</code> words, and
each <code>status</code> field (including <code>status.count</code>) with one word, all in distinct uint32 buffers. Each path occupies
<code>nodeCount - 1</code> edge slots; only its first <code>lengths[row]</code> slots are used. Unused edge slots are
<code>0xffffffff</code>; unused lengths and volumes are zero. Edge rows refer to the expanded input edge
list, so retain rows below <code>originalEdgeCount</code> to recover the original demand paths. Paths
retain their supply/demand endpoints via the adjacent super arcs. Circulating flow is omitted.</p>
<p>The deterministic tie rule applies to a fixed final flow: repeatedly run breadth-first search
over positive remaining flow, visiting queued vertices in FIFO order and each vertex's outgoing
edges in ascending input row order. Keep the first predecessor, emit the discovered path with
its bottleneck volume, subtract that volume and repeat. Thus the unit diamond emits both paths
with volume 1; where two branches merge and split again, the earliest source branch is paired
with the earliest sink branch. Equal-cost solvers can allocate different final edge flows; the
tie rule does not promise the same decomposition when their final edge flows differ.
Breadth-first order matters when capacity drops paths: with unit edges <code>0→1</code>, <code>1→2</code>, <code>0→2</code>
in that input order and a request of two units, capacity 1 retains edge row <code>[2]</code> (length 1,
volume 1), with <code>requiredCount = 2</code> and <code>overflow = 1</code>. An edge-ordered depth-first walk would
retain <code>[0,1]</code> instead and does not implement this tie rule.</p>
<p><code>paths.status.requiredCount</code> counts all decomposed paths before capacity truncation;
<code>paths.status.count = min(requiredCount, capacity)</code>, and <code>overflow = 1</code> when paths were dropped. Counting
continues after the output fills, so this count is exact for the incumbent flow. The required
<code>invalidCount</code>, <code>converged</code> and <code>iterationLimitReached</code> fields propagate the solver's status:
an exhausted augmentation/phase/search budget publishes the partial feasible paths with
<code>converged = 0</code>, <code>iterationLimitReached = 1</code>. Path capacity loss sets <code>overflow</code> independently
of convergence. A solved infeasible request can converge with fewer paths than requested;
check <code>summary.unserved</code> as well. Zero demand produces a complete empty decomposition even
with a zero augmentation budget. Invalid problems publish no paths and <code>invalidCount = 1</code>.
Every encoding clears all path slots and status words before publishing the current result.</p>
<p>Both graph operations use <code>pathCapacity</code> together with the same <code>output.paths</code> port map.
The pure <code>computeNetworkFlowOnCPU</code> enables the matching <code>result.paths</code> with <code>pathCapacity</code>.
Both operations accept <code>edgeCount = 0</code>, matching the contributor and pure CPU solver; <code>flow</code>
then has zero rows. For two distinct vertices, no edges and a request of one unit, the solve
converges with value 0 and unserved demand 1. Optional paths have <code>status.count = 0</code>, zero
lengths and volumes, <code>0xffffffff</code> edge padding, <code>requiredCount = 0</code> and <code>overflow = 0</code>.
An unmet request after residual exhaustion is complete, so <code>iterationLimitReached = 0</code>.
All explicit solver budgets retain their solver-specific meaning: <code>maximumAugmentations</code> binds
reference and costed successive-shortest-path; <code>maximumRelabels</code> binds uncosted parallel flow;
<code>maximumCostPhases</code> binds primal-dual. <code>maximumRelaxationRounds</code> binds parallel searches.
Decomposition exhausts the incumbent in at most one path per expanded edge; it has no separate
iteration budget. Path capacity truncates published rows and does not limit the solve.
On both backends, each uncosted/primal-dual phase recomputes sink-rooted labels, performs
bounded forward discharges, and returns stranded excess before committing the feasible flow.
Returned source excess is available to the next phase. For the demand-2 diamond with
capacities <code>[2,2,1,1]</code>, one phase sends one unit and reports an incomplete incumbent; two phases
send both units and converge. Either branch can carry the first unit under GPU atomic ordering.
<code>maximumRelaxationRounds</code> separately bounds the sink-rooted search, forward discharge, excess
return and final source-rooted cut search, including an unchanged sweep to certify each search.
Serving a finite request still converges when the final cut search exhausts its budget;
<code>summary.cutCertified</code> stays zero in that case.</p>
<p><code>solver: 'reference'</code> rejects <code>maximumRelabels</code>, <code>maximumRelaxationRounds</code>, <code>maximumCostPhases</code>
and <code>mode</code> in the contributor, both operation constructors and the pure CPU entry point.
Errors identify the contributor/function ID and incompatible prop. All <code>output.paths</code> columns
and its <code>status</code> object/fields are required when paths are enabled; missing fields produce
ID-prefixed errors naming the full path (for example, <code>output.paths.status</code> or
<code>output.paths.volumes</code>).</p>
<p>Point routing accepts optional <code>turnPenalties: float32[turnCount]</code>, aligned with its allowed
<code>turns: uint32x2[turnCount]</code>. Keep <code>costs</code> as the separate base-edge costs. Omitting penalties
retains zero-penalty routing. Every penalty must be finite and nonnegative; invalid contents set
per-query status 1. Traversing a turn pays its penalty plus the following edge cost. Entering a
partial destination pays the full penalty plus only <code>targetFraction * destinationBaseCost</code>.
For two incoming edges with base cost 2, source fraction 0.5, and penalties 7 and 1 into a base-cost
10 destination at fraction 0.25, the query costs are 10.5 and 4.5. This applies to both GPU modes,
the graph operation and the CPU twin.
Negative or non-finite base costs close individual edges. Turns touching closed edges are skipped;
a query with a closed source or destination has status 1, while queries between open edges remain
usable. Both forms publish the cost-attaining parent of the partial destination, including when
large full-edge costs round competing parents to a tie. An explicit CPU <code>maximumRelaxations</code>
binds the solve: reference counts turn scans, scalable counts rounds of up to 64 synchronous hops.
Status 2 reports exhaustion; a zero budget retains direct same-edge travel and leaves every parent
unset. Without an explicit budget the CPU solve continues to a fixpoint.</p>
<p><code>prepareGPUNetworkPointRoutingTurns({offsets, neighbors, weights, costs, overflow})</code> adapts explicit
read-back arrays from <code>GPUNetworkLineGraph</code> to <code>{turns, turnPenalties}</code> for upload. Pass the line
graph's overflow word: incomplete graphs throw. The adapter reads only active CSR slots, omits
closed arcs (<code>weights=-1</code>), and separates each full arc weight into its following base cost and
penalty. Prefer requesting the producer's optional <code>output.turnPenalties: float32[arcCapacity]</code>
and passing that read-back column as the adapter's optional <code>turnPenalties</code> input: it preserves
small turn costs before adding a large base-edge cost. Without it, subtraction is a fallback and
the recovered penalty retains only the precision of the supplied float32 arc weight. This
host helper performs no implicit submission or readback; applications can also prepare pair and
penalty arrays directly.</p>
<p>For full Kalman covariance set <code>covarianceMode: 'full'</code>; parameters are
<code>[qxx,qxy,qyy,rxx,rxy,ryy,vxx,vxy,vyy,maxGap]</code>. Process/velocity covariance must be positive semidefinite;
measurement covariance must be positive definite. Covariance outputs contain <strong>four float32x4 rows
per observation</strong>, the column-major 4x4 covariance in <code>[x,y,vx,vy]</code> state order. Full RTS uses a pivoted
4x4 inverse; singular/nonfinite backward solves retain forward values and set bit 8.
<code>prepareGPUTrajectoryObservations</code> subtracts a bigint epoch before f32 conversion (and also returns
exact <code>timeWords</code>, the absolute timestamps as a <code>uint32x2</code> Kalman <code>times</code> column) and transforms
WGS84 longitude/latitude through ellipsoidal ECEF into an explicitly chosen local east/north plane.
It returns projection/epoch metadata and performs no filtering or GPU readback.</p>
<p>For polygons, append flattened ring coordinates after the seven restriction parameters and set
<code>polygonRingVertexCounts</code>. Rings close implicitly; even-odd parity supports holes regardless of ring
orientation. Valid simple rings are caller-owned topology. A point is on the polygon boundary only
when its exact orientation against an edge is zero inside the edge's bounding box, the same exact
predicate <code>GPUZoneEvents</code> uses; there is no tolerance band, so a fix 5 cm from a 100 km edge is
strictly inside or outside. A point where the segment crosses an edge, and a piece of the segment
that runs along an edge, are on the boundary by construction and follow <code>geometryBoundary</code>. Whether
a segment meets an edge is decided from the exact sides of the two fixes against the edge and of the
edge's two ends against the segment, so a segment that passes a vertex without touching it reports
nothing and one that runs through a vertex reports one touch. Only the position of a crossing that
exists is f32, so interval end times carry f32 rounding. The two
contributors still differ for a fix exactly on the boundary: here <code>geometryBoundary</code> decides,
while <code>GPUZoneEvents</code> breaks the tie by its perturbation rule. <code>timeBoundary</code> selects closed, open, left-closed or right-closed;
<code>geometryBoundary</code> selects closed/open. Status adds start-open=64,end-open=128,capacity-exceeded=256.
<code>intervalsPerSegment</code> controls fixed output slots. Absent slots contain NaN/zero status. Relationships
are computed from all classified support even if interval slots overflow.</p>
<p>Optional compact outputs retain interval positions and <code>[firstSourceRow,lastSourceRow]</code> provenance.
<code>mergeIntervals</code> unions connected intervals across segments of the same track, preserving open joints
and track boundaries. <code>compactSummary</code> reports <code>[required,written,overflow]</code>; when source slots already
overflowed, required is a lower bound and overflow includes the number of truncated source segments.
Compaction/union is a bounded sequential GPU stream, not an implicit CPU filter.</p>
<p>For temporal within-distance predicates set <code>pairwiseTracks</code> to explicit track-ID pairs and
<code>intervalsPerPair</code> to output capacity; parameters[2] becomes maximum distance. Output rows are
pair-major compact interval slots and relationships are per pair. The kernel analytically solves
squared relative distance on each overlapping linear segment; output positions belong to the first
track. Repeated segment endpoints may appear in adjacent intervals. Zero common support is unknown;
unsupported gap interiors are unknown, never bridged. This is planar within-distance (distance zero
also supports equality), not the full set of MobilityDB temporal predicates.</p>
<p><code>addTrackQualityRecipe</code> bit 4 (<code>excessiveSpeed</code>) is by default the step-speed test documented in the
recipe contract, so the valid fix after an outlier is also flagged. Set
<code>speedOutlierMode: 'point-centred'</code> for an outlier test on the fix: a fix is flagged only when the
speed test fails on both adjacent steps, and a track-end fix when its single step fails and the next
step inward does not. A single pass does not catch two consecutive displaced fixes.</p>
<h2 id="directed-edge-position-routing-example">Directed edge-position routing example</h2>
<p>{/* lens:GPUNetworkPointRouting */}</p>
<p>::lens{scenario=gpu-network.network-point-routing rung=graph}</p>
<p>The three queries show same-edge partial travel, a route through two legal turns, and an
unreachable reverse journey. Inspect the costs alongside completion status and edge predecessors.</p>
<h2 id="residual-flow-example">Residual flow example</h2>
<p><code>GPUNetworkFlow</code>: Legacy solver selection and successive-shortest-path defaults remain. Opt-in mode
primal-dual uses three phases by default, with bounded relaxation/phase budgets; incomplete feasible
flows require completion and cut-certification status checks. Device bindings and command budgets
constrain all routes.</p>
<p>{/* lens:GPUNetworkFlow */}</p>
<p>::lens{scenario=gpu-network.network-flow rung=graph}</p>
<p>The four-vertex directed network includes antiparallel arcs. The final edge flows conserve mass,
and the source-side cut certifies total flow five. The summary retains the certificate separately
from the value.</p>
<p><code>output.status</code> optionally exposes one-row <code>uint32</code> ports <code>invalidCount</code>, <code>converged</code>,
<code>iterationCount</code> and <code>iterationLimitReached</code>. <code>iterationCount</code> counts committed batches:
augmentations for reference/successive-shortest-path and phases for uncosted/primal-dual,
including when the solver publishes an incumbent at its budget. A fully solved finite request with unserved
flow still converged; inspect <code>summary[3]</code> for that shortfall. Invalid problems report
<code>invalidCount = 1</code> and neither stopping flag. Every encoding clears and rewrites the ports.
The graph operation and CPU twin accept the same named ports and honor explicit solver budgets.
Omitted CPU budgets are uncapped. Outputs, including named ports and path columns, need distinct buffers.
When optional <code>topology</code> is supplied, the CPU twin reads both CSR ports and checks zero-based,
monotonic offsets ending at the edge count, source-row membership, and destination identity against
the aligned edge list. Inconsistent connectivity produces zero flow, cut, summary and exact cost,
with <code>invalidCount=1</code>, <code>converged=0</code>, and <code>iterationLimitReached=0</code> on reference GPU, parallel GPU and CPU forms.</p>
<h2 id="gpunetworklouvain"><code>GPUNetworkLouvain</code></h2>
<p>Supply exactly one of weighted CSR <code>topology</code> or row-major dense <code>weights</code>. Sparse sorting,
segmented reduction and quotient aggregation use published graph primitives. The <a href="/luma-contributors/docs/api/gpu-network/louvain">Louvain
guide</a> is the canonical input/output and numerical contract, including shared
<code>GPUNetworkLouvainOperation</code>/<code>CPUNetworkLouvainOperation</code> properties, per-frame parameters,
device-dependent limits, and the upstream single-level optimizer comparison.</p>
<p>{/* lens:GPUNetworkLouvain */}</p>
<p>::lens{scenario=gpu-network.network-louvain.sparse rung=graph}</p>
<p>The sparse CSR lens uses two disconnected weighted pairs, which contract to two communities.
The docs-sized sparse example budgets two levels with four rounds per level: one contraction
and a second level that confirms convergence. Its nine-slot objective trace starts at the
singleton partition, reaches modularity 0.5, and retains unused slots as NaN. The graph includes
quotient coarsening and all declared rounds; no shown graph nodes or readbacks are discarded.
The dense reference remains available for comparison:</p>
<p>::lens{scenario=gpu-network.network-louvain rung=graph}</p>
<h2 id="exact-facility-assignment-example">Exact facility assignment example</h2>
<p>{/* lens:GPUNetworkFacilityOptimization */}</p>
<p>::lens{scenario=gpu-network.network-facility-optimization rung=graph}</p>
<p>One origin can reach either facility, while the other can reach only the first. Exhaustive unsplit
assignment allocates the flexible origin to the second facility, certifying a feasible objective
of three where demand-order nearest allocation fails.</p>
<p>The packed four-word result is <code>output.summary</code> (formerly <code>output.status</code>):
<code>[valid, feasibleIncumbent, enumerationComplete, examinedCandidates]</code>. Optional <code>output.status</code>
ports are <code>invalidCount</code>, <code>converged</code>, <code>iterationCount</code> and <code>iterationLimitReached</code>, each one
<code>uint32</code> row. <code>iterationCount</code> counts nonempty candidate-scoring batches (at most 16); it is
zero for a zero-candidate budget or inputs rejected before scoring. An overflowing objective
invalidates the result but retains the number of batches already scored. This executed-work
counter is independent of <code>summary[3]</code>, which retains the examined candidate count (including
infeasible candidates) for accepted results and is zero for rejected results. <code>converged</code> certifies exhaustive
enumeration independently of whether any feasible assignment exists. A valid budget exit sets
<code>iterationLimitReached</code> and publishes any incumbent. Invalid input or an overflowing feasible
objective sets <code>invalidCount = 1</code> and clears both stopping flags. No claim of infeasibility is
made before complete enumeration.</p>
<p>Compile-time controls are the operation, view lengths, <code>facilityCount</code>, <code>selectCount</code>,
<code>maximumCandidates</code> and which status ports exist. Costs, demand, capacities and threshold are
per-frame buffer contents. Every encoding rewrites all outputs, including empty-demand results.
All four optimization contributors reject unsupported own fields in <code>output.status</code> at
construction, even when an extended object is structurally assignable in TypeScript. Only
<code>invalidCount</code>, <code>iterationCount</code>, <code>converged</code> and <code>iterationLimitReached</code> are supported.
The flow, time-routing and rational-access GPU and CPU operation forms apply the same field
validation before declaring outputs, so an extra status key cannot replace a primary port.
Each output and named port needs a distinct buffer. <code>estimatedCommandNodeCount</code> includes the
candidate batches, hierarchical reductions and optional status pass, at most 83 commands at the
largest search. Construction enforces the shared command budget; encoding checks storage
binding sizes before creating scratch or kernels.</p>
<h2 id="departure-time-routing-example">Departure-time routing example</h2>
<p>{/* lens:GPUNetworkTimeRouting */}</p>
<p>::lens{scenario=gpu-network.network-time-routing rung=graph}</p>
<p>The middle edge becomes more expensive as time increases. Departing at three reaches the
intermediate state at four and the destination at nine; the example captures the full arrival
and predecessor tables so the time-dependent evaluation remains inspectable.</p>
<p>Optional <code>output.status</code> has one-row <code>uint32</code> ports <code>invalidCount</code>, <code>converged</code>, <code>iterationCount</code>
and <code>iterationLimitReached</code>. The iteration count includes the final no-change sweep. A valid
budget exit publishes the current arrival labels and predecessors and sets
<code>iterationLimitReached = 1</code>; unreachable labels are conclusive only after convergence. Invalid
inputs or nonfinite arrival arithmetic set <code>invalidCount = 1</code> and clear both stopping flags.
The existing four-word <code>summary</code> layout is unchanged. Outputs and ports need distinct buffers.
The GPU and CPU operation forms accept the same named graph values and honor the same
<code>maximumRelaxations</code> sweep budget (default: state count). Zero sweeps retain only the initial
label and report a budget exit. CPU labels use float64 internally, so rounding can affect ties
and the sweep in which convergence is certified.</p>
<h2 id="congestion-allocation-example">Congestion allocation example</h2>
<p>{/* lens:GPUNetworkRationalAccess */}</p>
<p>::lens{scenario=gpu-network.network-rational-access rung=graph}</p>
<p>A demand of ten initially chooses the first of two tied facilities, then balances to five each.
A second unreachable origin retains three units of explicitly unserved demand and a NaN score.</p>
<p>The packed gap/cycle result is <code>output.summary</code> (formerly <code>output.status</code>):
<code>[valid, equilibriumGapWithinTolerance, completedCycles, maximumGapBits]</code>. Optional named
<code>output.status</code> ports expose <code>invalidCount</code>, <code>converged</code>, <code>iterationCount</code> and
<code>iterationLimitReached</code>, each one <code>uint32</code> row. The iteration count is completed simultaneous
allocation shifts; the initial gap evaluation is cycle zero. Convergence requires the final gap
within tolerance, so an integer rounding stall still exhausts the cycle budget. A valid budget
exit publishes the allocation, final scores and unserved demand. Invalid inputs or nonfinite
arithmetic set <code>invalidCount = 1</code> and clear both stopping flags. Completed shifts remain counted
when the next gap evaluation fails. Ports reset on every encoding
and require separate output buffers. Graph and CPU operations expose the same port names; the
pure CPU result now has <code>summary</code> and named <code>status</code> arrays, and retains the cycle budget.</p>
<p>Rational-access cycles use one 64-lane workgroup within the existing 64-origin and 32-facility
limits. Each origin scores and shifts independently; facility lanes sum loads and simultaneous
inflow in ascending origin order, preserving float32 addition order and earliest-facility ties.
Uniform workgroup exits stop at equilibrium or invalid arithmetic without unrolling command nodes
per cycle. Ordered initialization and score publication preserve partial outputs on rejection.
The graph uses two kernels plus optional named-status publication. This remains a bounded reference
solver; it does not extend to larger allocations.</p>`};export{e as default};