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
| `maximumIterations` | 64 | 1–1,024 | Rounds per shortest-path phase and, separately, per equal-cost predecessor phase in each Yen round |

The shared command-node budget of 32,768 also applies. `estimatedCommandNodeCount` is a
conservative upper bound, `8 + routeCount * (16 + 4 * maximumIterations)`. Each shortest-path
round chains up to 64 local hops per workgroup; a round is not a guaranteed 64-hop advance.
The same per-phase round bound applies independently to each spur lane, with no shared total
across Yen rounds. Relaxation and tie rounds use GPU-written indirect dispatch.
Finished Yen batches gate every composed shortest-path pass as well as edge filtering and
reconstruction on the GPU. Batch predicates are combined with the live frontier dispatches, so
skipped initialization cannot reuse stale dispatch arguments from an earlier encoding. No status
is read back inside the contributor. All input view contents are per-frame, including topology weights and
query endpoints; capacities and view lengths are compile-time.

Use `getGPUKShortestPathsOutputLayout(options)` from the same entry point to obtain the scalar
format and required row count of every output before allocating caller-owned buffers. The
constructor, operation form and CPU twin use this function too. `queryCount` and `routeCount`
are required; `maximumPathLength` defaults to 64. Named dimensions are resolved explicitly,
including inherited getters, so validation and output slicing use the same values.
Dimensions and their products must be positive
integers fitting uint32 row counts. GPU execution limits in the table above and device binding
limits still apply to the contributor; the layout function also describes larger CPU budgets.

```typescript
import {getGPUKShortestPathsOutputLayout} from '@luma-contributors/analysis/gpu-network';

const options = {queryCount: 2, routeCount: 3, maximumPathLength: 4};
const layout = getGPUKShortestPathsOutputLayout(options);
// Allocate each output with its descriptor's format and length.
// layout.vertices = {format: 'uint32', length: 24}
// layout.costs = {format: 'float32', length: 6}
```

| Output | Descriptor | Order and meaning |
| --- | --- | --- |
| `vertices` | `layout.vertices` | Query, route, vertex slot; `uint32` |
| `edges` | `layout.edges` | Query, route, edge slot, same capacity as vertices; `uint32` |
| `lengths` | `layout.lengths` | Query, route; vertex count, `uint32` |
| `costs` | `layout.costs` | Query, route; `float32` |
| `counts` | `layout.counts` | Accepted route count per query, `uint32` |
| `status.<name>` | `layout.status.<name>` | One `uint32` row per batch for each status below |

Vertices and edges have query × route × maximumPathLength slots, padded with `0xffffffff`;
the last vertex has no edge. Lengths and costs have query × route slots: empty slots have length
zero and cost positive infinity. The fixed requested top-k layout does not promise to count all simple paths.
Every output uses its own buffer, separate from inputs.

`output.status` contains four required one-row uint32 views:

| Port | Meaning |
| --- | --- |
| `converged` | Every query accepted k routes or exhausted all remaining paths |
| `iterationLimitReached` | A shortest-path/tie search did not converge, or a shortest spur exceeded `maximumPathLength` |
| `candidateOverflow` | A query's persistent candidate pool could not retain a new distinct candidate |
| `iterationCount` | Maximum Yen rounds attempted by any query, at most `routeCount` |

On either budget stop, only the first `counts[query]` routes of each query are usable;
these previously certified routes remain published in nondecreasing cost order. Empty padded
slots do not establish that no further path exists. `iterationCount` counts Yen rounds attempted,
not frontier rounds, examined edges or path hops. Candidate capacity bounds simultaneously
pending distinct full routes per query; it does not bound the number of CSR edges examined. A path-length stop is a
budget interruption, **not** enumeration restricted to paths of at most that length: simply dropping
an overlong spur could skip a cheaper path. A shortest-search interruption conservatively stops
all queries still active in that batch. Candidate overflow affects only its own query. Status
flags aggregate across queries, so both interruption flags can be set by different queries.

`decodeGPUKShortestPaths(readback, options)` converts typed-array readbacks into
`{queries, status}`. Each query has `{count, routes}`; each retained route has
`{vertices, edges, length, cost}` with padding removed. Query and route array order matches the
output layout. A zero-edge route has one vertex and an empty edge array; a query without routes
has `count: 0` and `routes: []`. The decoder checks array formats and minimum required lengths,
accepted counts and path lengths, nonnegative retained costs, and batch status scalars. Positive
infinity is allowed for a retained CPU route whose finite float64 cost overflowed on publication
as float32; counts and lengths distinguish it from padding. NaN and negative costs are rejected.
Typed arrays from other JavaScript realms are accepted when their scalar format matches.
Typed-array byte offsets are respected and spare capacity beyond the layout is ignored. Returned
route arrays are copies. The existing output has no version header; supply the dimensions used
for the computation.

Pass the five data arrays plus the four status arrays nested under `readback.status`. For the
direct CPU result, whose status fields are flat, use
`decodeGPUKShortestPaths({...result, status: result}, options)`. The decoder returns all three
flags as booleans and `iterationCount` as a number. It retains certified routes from incomplete
batches: inspect `status.converged`, `status.iterationLimitReached` and `status.candidateOverflow`
before treating decoded routes as a complete enumeration.

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
with the shared float64 routing and acyclic predecessor helpers. Prefer it for small networks,
where GPU submission costs dominate.

The CPU version drops GPU-only node, route, device and command limits. An explicit
`maximumIterations` binds every shortest-search and zero-cost tie phase to at most 64 hops per
round. A synchronous CPU frontier stops before exceeding that allowance and publishes only
previously certified routes, with `converged = 0` and `iterationLimitReached = 1` on interruption.
Queries advance together: if any shortest-search lane interrupts, the entire still-active batch
stops before accepting that round's proposals. Routes from earlier completed rounds remain published.
Every active spur contributes its search certificate before a query-local path-length or
candidate-capacity stop is applied; a locally overlong path cannot hide another spur's interruption.
GPU workgroup scheduling can make progress through fewer hops per round. Without an explicit
CPU round budget, searches run to completion. Path length and candidate budgets retain the GPU's meaning.
It computes costs in float64 and rounds once at output, so near ties, overflow and predecessor
tightness may differ. Packed array lengths and CSR offset values are checked on the CPU.
The existing `CPUAlternativeRoutesOperation` and its reference GPU contributor remain available.

Independent oracle fixtures come from NetworkX `shortest_simple_paths`, using edge subdivision
to preserve distinct parallel edges. Regenerate with
`/usr/bin/python3 modules/analysis/test/oracles/k-shortest-paths.py`. Headless specs cover these
fixtures, 700-vertex networks, a 300-vertex path, graph reuse, input padding, ties and budget stops.

## Empty queries and byte readback

`decodeGPUKShortestPaths` also accepts `Uint8Array` and `ArrayBuffer` readbacks, including subviews
and values from another JavaScript realm. Numeric vertex/edge/count arrays must be uint32;
cost arrays must be float32. Partial elements and missing status words are rejected.
A query with no accepted routes decodes to `{count: 0, routes: []}`. Completion remains on the
returned batch `status`, independently of that query's count; inspect convergence and interruption
flags before treating empty output as proof of unreachability. Alternative routes instead uses
`nodeCount` as its path stride and returns `complete` on each query.

Zero-query batches over a valid nonempty topology have empty data columns and scalar status
`{converged: true, iterationLimitReached: false, candidateOverflow: false, iterationCount: 0}`.
An explicit CPU search budget still binds; omitted CPU budgets retain the unbounded reference route.

```ts
import {computeKShortestPathsOnCPU, decodeGPUKShortestPaths}
  from '@luma-contributors/analysis/gpu-network';
const options = {queryCount: 2, routeCount: 1, maximumPathLength: 2};
const result = computeKShortestPathsOnCPU({
  topology: {offsets: Uint32Array.of(0, 1, 1), neighbors: Uint32Array.of(1), weights: Float32Array.of(2)},
  sources: Uint32Array.of(0, 1), targets: Uint32Array.of(1, 0)
}, options);
const decoded = decodeGPUKShortestPaths({...result, status: result}, options);
console.log(decoded.queries[1]); // {count: 0, routes: []}
console.log(decoded.status.converged); // true: the whole batch completed
```
