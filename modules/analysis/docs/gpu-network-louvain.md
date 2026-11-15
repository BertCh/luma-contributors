# GPUNetworkLouvain

`GPUNetworkLouvain` (`@luma-contributors/analysis/gpu-network`) optimizes weighted
Newman modularity with multilevel Louvain. Supply `topology` for sparse parallel
moving. The existing `weights` dense reference input remains available to 32 nodes;
its seeded sequential visitation and partition choices remain unchanged. Prefer
`CPUNetworkLouvainOperation` for small dense workloads.

## Sparse input and outputs

`topology` is the analysis `GPUNetworkTopology`: packed `uint32` offsets of length
`nodeCount + 1`, packed `uint32` neighbors, and aligned packed `float32` weights.
Directed arcs occur once. Undirected edges occur in both directions and self-loop
weights are doubled, as in the dense adjacency. Parallel arcs are summed; neighbor
order is unrestricted. Undirected reciprocal weight sums must agree within relative
`1e-6` (the precision of their independent float32 reductions).

Offsets start at zero, are nondecreasing, and end at or below the slot capacity.
Slots beyond the final offset and targets `>= nodeCount` are ignored, including their
weights. Accepted weights must be finite and nonnegative. Zero weights do not create
move candidates. Bad CSR, asymmetric undirected weights, invalid parameters or
nonfinite total mass fail closed: identity labels, NaN trace, zero validity. A graph
with zero mass returns identity labels, initial objective zero and convergence.

`output.communities` has one `uint32` label per original vertex; isolates retain
their IDs. `objectiveTrace` has exactly
`1 + maximumLevels * maximumIterations` float32 rows: initial modularity followed by
one recomputed objective per completed parallel round; unused rows are NaN.
`summary` retains `[valid, converged, completedLevels, completedRounds]`.

Optional one-row `uint32` ports in `output.status` are `iterationCount`, `converged`,
`iterationLimitReached`, `invalidCount`, and `approximation`. `invalidCount` counts
invalid input graphs (zero or one). `approximation` is one: modularity optimization
is heuristic, never a global optimum certificate. Budget exhaustion publishes the
current partition and trace, with `converged = 0`, `iterationLimitReached = 1`.
A final stable moving level still needs another aggregation level to certify that
no supervertex can improve the partition.

All output views, including the summary and named status ports, must use distinct
buffers. Separate graph imports of the same default allocation also alias and are
rejected at construction. This keeps later summary/status passes from overwriting
community labels.

## Parameters and limits

Per-frame: input contents, resolution and minimum positive accepted gain. Use
`getGPUNetworkLouvainParameterValues({resolution: 1, minimumGain: 1e-5})` for the
two-row float32 `parameters` view, hosted by `GPUParameterBuffer`. Zero tolerance
is supported; a positive tolerance avoids moves driven by float32 cancellation.

Compile-time: node/slot capacities, directed mode, optional uint32 seed, output
ports, `maximumLevels` (default 8, 1–32) and `maximumIterations` per level (default
16, 1–64). Sparse node count is 1–65,535: pair keys must fit a uint32 while reserving
`0xffffffff`. CSR capacity is 0–1,048,576 slots; no degree or community-size cap.
The constructor validates the estimated command-node count against 32,768, so not
every combination of level/round maxima is accepted. The conservative bound is
`256 + maximumLevels * (224 * maximumIterations + 112) + 1`. Device storage-binding
limits are checked before creating nodes. Scratch is linear in vertices and edges.

## Algorithm and upstream evaluation

Each round uses published `GPUSort`, `GPUScan`, `GPUGather`, `GPUReduction`, and the
analysis fixed-tree segmented reducer. Sorted `(vertex, neighborCommunity)` records
provide connection weights. Each vertex proposes its best strictly improving gain;
equal gains choose the lowest community label. Proposals reserve both affected
communities. Only a proposal winning both reservations commits, so concurrent moves
have disjoint community totals and their modularity gains do not conflict. The
lowest vertex ID wins conflicts, or the lowest `vertex ^ seed` when seeded. This
seed is a deterministic priority, not a random permutation or the dense Philox order.

When local moving stabilizes after changes, a sort and segmented reduction creates
the weighted quotient graph, including self-loops. Original labels are composed
through levels. Each round/aggregation uses GPU dispatch gates. Installed alpha.14
sort/reduction nodes omit public workload geometry; a domain adapter observes their
public `ComputePass.dispatch` calls and GPU-gates the exact dimensions, without
private metadata, submission, readback, or replacement upstream kernels. Its resources
belong to the compiled command and are released on destruction.

Evaluation of the installed **published `@luma.gl/graph` 10.0.0-alpha.14**:
`GPUGraphAlgorithmsModularityOptimization`, exported from
`@luma.gl/graph/gpu-graph-algorithms`, accepts topology and GPUVector outputs,
resolution/minimumGain as construction-time numbers, up to 1,024 individual accepted
moves, and optional valid/converged scalars. Its API documentation explicitly says
it performs **single-level** local moving, accepts **one globally best move** per
round, uses unordered float atomic sums, and performs **no graph coarsening**.
It therefore cannot replace this multilevel, parallel, per-frame-parameter contract.
No upstream source or private API is imported here. The installed package version
is alpha.14 although AGENTS.md still mentions alpha.13; dependencies were not edited.

## CPU twin and verification

`GPUNetworkLouvainOperation` and `CPUNetworkLouvainOperation` accept the same graph
value names, including `topology: {offsets, neighbors, weights}` and optional status
ports. `computeNetworkLouvainOnCPU({topology, parameters}, options)` is the pure
float64 twin. It uses the same proposal/reservation/coarsening schedule but drops
GPU pair-key, scratch and command-budget caps. Near ties, overflow, subnormals and
threshold decisions can differ from GPU float32. Partitions need not match dense
sequential Louvain or NetworkX on general graphs.

`test/oracles/network-louvain-sparse.py` generates NetworkX Louvain partitions and
scores, independently cross-checked with NumPy's modularity matrix. Fixtures cover
96 weighted vertices, 48 directed vertices, parallel arcs, self-loops, padding,
isolates, and a graph that needs coarsened moves. Node specs also check exact ties,
round/level exhaustion, the largest supported shape, command estimates and generated
kernel audits. The headless spec compares every CPU/GPU output and the NetworkX
quality, plus zero/invalid/empty-slot cases. GPU execution must be verified by the
landing environment; Chromium is unavailable in the implementation sandbox.
