# K-shortest simple paths

## GPUKShortestPaths

Import from `@luma-contributors/analysis/gpu-network`. This scale-tier contributor uses
[Yen's algorithm (1971)](https://doi.org/10.1287/mnsc.17.11.712) to enumerate the first `routeCount`
simple edge paths of a directed CSR network, ordered by cost. Every round runs the possible
spur searches concurrently through `GPUNetworkReachability`, removing the root vertices and
next edges of already accepted routes. Candidates persist across rounds. Parallel CSR edges
are distinct paths; vertices never repeat, including across zero-cost cycles.

Unlike the reference-tier `GPUAlternativeRoutes`, the network has no 256-vertex limit.
Vertex count is constrained by device storage bindings and the number of concurrent spur lanes.
No coordinates are read, so the coordinate model is topological. Inputs must be packed views;
CSR offsets must be nondecreasing and in range. Unused trailing edge capacity is allowed.
Invalid neighbors, negative weights, NaNs and infinities are impassable. Invalid query endpoints
produce empty converged results. An identical source and target produces one zero-edge path.

Compile-time properties:

| Property | Default | Accepted range | Purpose |
| --- | --- | --- | --- |
| `routeCount` | Required | 1–32 | Number of requested routes per query |
| `maximumPathLength` | 64 | 1–1,024 | Vertex slots per route, root comparisons and spur lanes |
| `candidateCapacity` | 256 | 1–4,096 | Persistent pending routes per query |
| `maximumIterations` | 64 | 1–1,024 | Rounds for each shortest-path and zero-cost tie phase |

The shared command-node budget of 32,768 also applies. `estimatedCommandNodeCount` is a
conservative upper bound, `8 + routeCount * (16 + 4 * maximumIterations)`. Each shortest-path
round chains up to 64 local hops. Relaxation and tie rounds use GPU-written indirect dispatch.
Finished Yen batches gate every composed shortest-path pass as well as edge filtering and
reconstruction on the GPU. Batch predicates are combined with the live frontier dispatches, so
skipped initialization cannot reuse stale dispatch arguments from an earlier encoding. No status
is read back inside the contributor. All input view contents are per-frame, including topology weights and
query endpoints; capacities and view lengths are compile-time.

`output.vertices` and `output.edges` have `queryCount * routeCount * maximumPathLength` rows.
They are padded with `0xffffffff`; the last vertex of a route has no edge.
`output.lengths` and `output.costs` have `queryCount * routeCount` rows: empty slots have length
zero and cost positive infinity. `output.counts` has `queryCount` rows and counts the retained
routes of each query. The fixed requested top-k layout does not promise to count all simple paths.
Every output uses its own buffer, separate from inputs.

`output.status` contains four required one-row uint32 views:

| Port | Meaning |
| --- | --- |
| `converged` | Every query accepted k routes or exhausted all remaining paths |
| `iterationLimitReached` | A shortest-path/tie search did not converge, or a shortest spur exceeded `maximumPathLength` |
| `candidateOverflow` | A query's persistent candidate pool could not retain a new distinct candidate |
| `iterationCount` | Maximum Yen rounds attempted by any query, at most `routeCount` |

On either budget stop, previously certified routes remain published. A path-length stop is a
budget interruption, **not** enumeration restricted to paths of at most that length: simply dropping
an overlong spur could skip a cheaper path. A shortest-search interruption conservatively stops
all queries still active in that batch. Candidate overflow affects only its own query. Status
flags aggregate across queries, so both interruption flags can be set by different queries.

For Q queries, P spur slots, V vertices, E edge slots, C candidates and path length L, scratch is
`O(Q * P * (V + E) + Q * (C + k + P) * L)`. P is `min(L - 1, V - 1)`, at least one, and is one
when k is one. The CSR is copied per spur lane because every lane removes different edges.
Storage sizes are checked before nodes are built, including the two-slot shortest-path frontier.
Reduce query batch size or L when a binding is too large. The number of paths can still grow
exponentially; k and the candidate budget bound the enumeration, without enumerating all prefixes.

Costs follow float32 shortest-path relaxation, so very close costs can change order compared with
float64 and overflowing sums are impassable. Equal-cost membership follows reachability's acyclic
predecessor policy; the persistent pool breaks ties by CSR edge IDs. Global lexicographic tie order
is not guaranteed. Use exact binary fractions when testing exact agreement between CPU and GPU.

{/* lens:GPUKShortestPaths */}

::lens{scenario=gpu-network.k-shortest-paths rung=graph}

The directed diamond has two simple routes from vertex 0 to 3: CSR edges `[0, 2]` cost 2,
and `[1, 3]` cost 3. The lens retains padded route slots, per-query `counts` and all four
batch status ports. This synthetic run requests two routes with four vertex slots per route.

## GPUKShortestPathsOperation

The vector graph operation takes `KShortestPathsOperationProps`: graph value names replace input
and output views. `queryCount` declares the batch size; vertex count is inferred from CSR offsets. Status values remain
nested under `output.status`. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`.

## CPUKShortestPathsOperation and computeKShortestPathsOnCPU

The CPU twin accepts the same operation props and publishes every output port with the same layout.
For direct calls, `computeKShortestPathsOnCPU({topology, sources, targets}, options)` takes typed arrays
and returns typed arrays; the four status fields are flattened on its result. It uses Yen's algorithm
with the shared float64 Dijkstra and acyclic predecessor helpers. Prefer it for small networks,
where GPU submission costs dominate.

The CPU version drops GPU-only node, route, device, command and shortest-search round limits;
`maximumIterations` is ignored. Path length and candidate budgets retain the GPU's meaning.
It computes costs in float64 and rounds once at output, so near ties, overflow and predecessor
tightness may differ. Packed array lengths and CSR offset values are checked on the CPU.
The existing `CPUAlternativeRoutesOperation` and its reference GPU contributor remain available.

Independent oracle fixtures come from NetworkX `shortest_simple_paths`, using edge subdivision
to preserve distinct parallel edges. Regenerate with
`/usr/bin/python3 modules/analysis/test/oracles/k-shortest-paths.py`. Headless specs cover these
fixtures, 700-vertex networks, a 300-vertex path, graph reuse, input padding, ties and budget stops.
