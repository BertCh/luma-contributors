# Visibility graph and network centrality

## `addVisibilityGraphRecipe`

CSR visibility supports budgeted exact all-source metrics or deterministic sampled estimates via
`metricsMode: 'auto' | 'exact'`. Auto uses all sources when lane divisibility, binding bytes and the
32,768-command recipe budget fit. Otherwise it chooses the largest feasible sample up to 32 pivots
(or `laneCount` when larger), respecting whole batches. Explicit pivot counts and exact mode never
silently fall back. Sampled maximum depth
is a lower bound. `status.approximation`, overflow and convergence are separate statuses; explicit exact
rejects an insufficient budget.

Before adding work, call the side-effect-free preflight through the public network entry:

```typescript
import {addVisibilityGraphRecipe, type GPUNetworkVisibilityGraphCSRProps}
  from '@luma-contributors/analysis/gpu-network';

declare const csrProps: GPUNetworkVisibilityGraphCSRProps;

const plan = addVisibilityGraphRecipe.getExecutionPlan(graph, csrProps);
// plan.route: 'exact' | 'sampled'; plan.pivotCount and plan.laneCount
// plan.bindingBytes: adjacency and network metric scratch bytes per binding
// plan.estimatedCommandNodeCount: conservative total adjacency + metric command budget
const result = addVisibilityGraphRecipe(graph, csrProps);
// result.provenance.executionPlan records the same resolved plan.
```

Preflight validates adjacency storage as well as network scratch against the supplied device,
without adding nodes, declaring buffers or allocating GPU resources. The CSR-only preflight rejects
other modes and validates every supplied scalar status port before graph mutation. It returns
`feasible: true` on success and throws an explanatory validation error on refusal.

For a catalog method picker, `addVisibilityGraphRecipe.executionPlanner.getExecutionPlan(csrProps, graph)`
implements `GPUAnalysisExecutionPlanner` with a boolean `feasible`, route constraints and an array of
accuracy implications. This additive adapter uses the same selection logic; the ordinary preflight
retains its descriptive `accuracyImplications` string. `bindingBytes` includes the adjacency grid,
query partitions and padded obstacle BVH as well as centrality and integration scratch.

`pivotSelection: 'systematic'` selects viewpoint row
`floor(i * positions.length / pivotCount)` for pivot row `i`; `'caller'` uses the resident
`pivotNodes` view. Explicit `laneCount` must divide `pivotCount`. If no automatic sample fits,
the error identifies the smallest candidate and its failed constraint. Decreasing
`maximumIterations` may fit more work but can leave the runtime convergence status unset.

Network-specific contracts are `GPUNetworkVisibilityGraphCSRProps`,
`GPUNetworkVisibilityGraphCSROutput` and `GPUNetworkVisibilityGraphCSRStatus`.

Use `output` for caller-owned CSR destinations, including both adjacency fields such as `degree`
and network fields such as `metrics`. The returned fields retain the exact caller view identities.

{/* lens:addVisibilityGraphRecipe */}

The `@luma-contributors/analysis/gpu-network` recipe composes indexed visibility with
batched pivot centrality in `mode: 'csr'`. Three collinear observers at unit spacing
and a visibility radius of 1.1 form a path: degree is `[1, 2, 1]`, betweenness is
`[0, 1, 0]`, and closeness is `[2/3, 1, 2/3]`. The wall lies outside their sight lines.
All three observers are pivots, so approximation is zero. Four directed CSR slots
store the two undirected links without overflow.

The spatial [urban-morphology recipe](../gpu-spatial-analysis/urban-ecology-lenses.md)
provides adjacency and degree only in CSR mode; its dense reference mode retains
all-source metrics. Import the network recipe for CSR betweenness, pivot closeness
and reached counts. With fewer pivots, betweenness is scaled by node count divided
by pivot count, closeness follows pivot order, and approximation is one. Capacity,
candidate overflow, invalid input and convergence remain separate status fields.
An invalid viewpoint gets an empty CSR row and the rest of the adjacency is published, while
every centrality and integration metric is NaN as long as `status.invalidCount` is not zero.
Invalid pivots (out of range or repeated) withhold the whole adjacency.

::lens{scenario=gpu-network.visibility-graph-recipe rung=result caption="Inspect CSR degree, pivot centrality and completeness status."}

::lens{scenario=gpu-network.visibility-graph-recipe rung=graph caption="Follow indexed visibility, symmetric CSR construction and batched network searches."}
