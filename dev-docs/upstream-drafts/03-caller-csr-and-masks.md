# Let graph algorithms consume CSR views and live selection masks

Status: draft, not filed. Target: `@luma.gl/graph` 10.0.0-alpha.14.

## Problem

An application can produce compressed sparse row adjacency inside a `GPUCommandGraph`, with
`GraphDataView<'uint32'>` offsets and neighbors and optional aligned weights. Published graph
algorithms cannot directly consume that representation or apply per-node/per-slot input selection.
This prevents composition with GPU-produced adjacency and closures that change between submissions.

Verified against installed `dist/gpu-graph-algorithms`:

- `GPUGraphAlgorithmsTopologyProps` requires `GPUGraphAlgorithmsData` (physical source/target
  `GPUVector` columns) and caller-allocated physical adjacency vectors. `addToGraph` declares a
  topology rebuild from those source edges. Each output must have a distinct physical allocation.
- `GPUGraphAlgorithmsDegreeProps`, `GPUGraphAlgorithmsConnectedComponentsProps`,
  `GPUGraphAlgorithmsBreadthFirstSearchProps` and `GPUGraphAlgorithmsSingleSourceShortestPathProps`
  require that topology class, with physical `GPUVector` outputs. They expose no input node/slot mask.
- BFS's existing `mask` is an **output reachability mask**. It does not select input nodes or edges.
- `gpu-graph-degree-internals.js` calculates `offsets[index + 1] - offsets[index]`; it never reads
  neighbors or a closure mask. This is correct for its documented stored-adjacency degree but cannot
  represent live degree over masked slots without rebuilding offsets.

The algorithms already have `addToGraph(commandGraph)`; missing graph attachment is not the issue.
The gap is accepting existing logical views with explicit selection semantics. These are verified
API restrictions, not a claim that an existing supported invocation returns an incorrect result.

## Minimal compile-time reproduction

This uses only published luma.gl types. Given views in a caller's graph, there is no accepted degree
invocation for them. Remove the two `@ts-expect-error` lines to see alpha.14 reject both ports.
The annotations let the example itself typecheck while asserting the restrictions remain present.

```typescript
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUGraphAlgorithmsDegreeProps} from '@luma.gl/graph/gpu-graph-algorithms';

export function reproduceMissingCSRPorts(
  offsets: GraphDataView<'uint32'>,
  neighbors: GraphDataView<'uint32'>,
  nodeMask: GraphDataView<'uint32'>,
  slotMask: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): GPUGraphAlgorithmsDegreeProps {
  const topology = {
    vertexCount: 3, directed: true, nodeMask,
    forward: {
      offsets, neighbors, slotMask,
      completeness: {kind: 'guaranteed-complete' as const}
    }
  };
  return {
    // @ts-expect-error alpha.14 requires GPUGraphAlgorithmsTopology, not CSR views.
    topology,
    // @ts-expect-error alpha.14 requires a physical GPUVector output.
    output
  };
}
```

The smallest useful data fixture is a directed triangle with three slots:

```text
vertexCount = 3
offsets     = [0, 2, 3, 3]
neighbors   = [1, 2, 2]       // 0→1, 0→2, 1→2
nodeMask    = [1, 1, 1]
slotMask    = [1, 0, 1]       // close only 0→2
```

Use `seeds = [0]` and `maxDepth: 2` for **both submissions**. Alpha.14 defaults to one hop;
that default would reach only `[0, 1, 0xffffffff]` through the first mask.
Stored outgoing degree is `[2, 1, 0]`; requested live outgoing degree is **`[1, 1, 0]`**. BFS from
0 with this two-hop budget has live distances **`[0, 1, 2]`**, versus `[0, 1, 1]` without the slot mask. On the same compiled
graph, change `slotMask` to `[0, 1, 0]`: live degree becomes **`[1, 0, 0]`** and distances become
**`[0, 0xffffffff, 1]`**. These expected values are hand-worked, not GPU measurements.

## Proposed API

Add a public view representation accepted beside the existing topology by algorithms. A proposed
shape, using the existing “vertex” terminology:

```typescript
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

export type GPUGraphAlgorithmsAdjacencyView = {
  offsets: GraphDataView<'uint32'>;       // vertexCount + 1
  neighbors: GraphDataView<'uint32'>;     // slot capacity
  edgeIds?: GraphDataView<'uint32'>;      // stable edge identities, aligned with slots
  edgeWeights?: GraphDataView<'float32'>; // aligned with slots
  slotMask?: GraphDataView<'uint32'>;     // nonzero selects a stored slot
  completeness:
    | {kind: 'guaranteed-complete'}
    | {kind: 'tracked'; overflow: GraphDataView<'uint32'>}; // one GPU-resident word
};

export type GPUGraphAlgorithmsTopologyView = {
  vertexCount: number;
  directed: boolean;
  forward: GPUGraphAlgorithmsAdjacencyView;
  reverse?: GPUGraphAlgorithmsAdjacencyView;
  nodeMask?: GraphDataView<'uint32'>;     // nonzero selects a vertex
};

// Proposed view branch of degree props, for typechecking the reproduction above.
export type GPUGraphAlgorithmsViewDegreeProps = {
  topology: GPUGraphAlgorithmsTopologyView;
  output: GraphDataView<'uint32'>;
};
```

Degree/components/BFS/SSSP should accept the existing `GPUGraphAlgorithmsTopology` or this view
topology, with equivalent `GraphDataView` output ports alongside their existing physical ports.
Each direction must declare completeness: `guaranteed-complete` is a caller guarantee, while
`tracked` supplies a graph-owned, GPU-resident one-row `uint32` overflow view (zero complete,
nonzero incomplete). An omitted declaration is a type/construction error. A mutable producer that
may truncate must use `tracked`; the guarantee must hold for every submission. Reverse adjacency
has its own declaration and overflow word, independent of forward adjacency. The GPU reads that
word every submission, with no host readback.

Use the same algorithm implementation after resolving physical topology or logical views to a
common internal view contract. Keep caller-owned graph declaration and allocation; no hidden
submission, readback or CSR rebuild. Supporting packed views first is sufficient; specify that
limit rather than implying arbitrary striding or chunks. Consider `getCommandNodes(graph)` as an
additional composition method, but attachment renaming is not required for this request.

Semantics to settle in the implementation:

- Shapes and formats are construction-time; buffer contents, including masks, are per submission.
  All views must belong to the target graph. Offsets are exclusive, monotone and bounded by slot
  capacity; the last offset is the logical slot count, not `neighbors.length`. Spare capacity is
  ignored. Do not read GPU contents on the CPU to validate them.
- A slot is live only when its source and destination nodes are selected and its slot mask is
  nonzero. Missing masks select all. Excluded degree rows are zero, excluded/unreachable BFS and
  component rows use documented sentinels. An excluded BFS source produces no reachable rows.
- Forward and reverse slot orders may differ. Each direction needs its own slot mask or an explicit
  edge-ID mapping; do not reuse the forward slot mask by reverse row position. Document whether
  callers must provide reverse CSR for directed incoming traversals and weak components.
- Neighbors `>= vertexCount`, including `0xffffffff`, are absent slots: skip them in live degree,
  components and traversal before applying destination masks. Empty rows, duplicates and reciprocal
  undirected slots retain their documented meanings. Sentinel slots are neither real edges nor
  evidence of truncation; completeness is a separate condition.
- If a selected direction's tracked overflow is nonzero, refuse to publish exact traversal or
  component results. Components publish `0xffffffff` labels and `converged = 0`, matching the
  existing physical-topology implementation; BFS publishes unreachable distances/predecessors and
  clears reachability. Keep required counts distinct from stored counts. A degree operation that
  scans live slots must also refuse incomplete adjacency (construction-time rejection of tracked
  inputs is acceptable until a degree status output is added); offset-difference degree's existing
  behavior on physical topology does not establish live masked degree completeness.
- Existing degree counts an undirected self-loop once, while `GPUGraphAlgorithmsModularity` documents
  twice in its volume convention. Preserve/document each existing convention or add an explicit
  `selfLoops: 'once' | 'twice'` option for degree. This request must not silently change old results.

Start with degree and connected components, then BFS/SSSP; extend PageRank and community algorithms
under the same representation once their selection and status semantics are covered. Weighted
modularity's deterministic accumulation is a separate numerical requirement, not delivered merely
by accepting CSR views.

## Alternatives and acceptance

Rebuilding a filtered COO list and fresh topology is possible with public primitives, but adds
compaction, mappings, allocations and work each time a closure changes. Wrapping physical buffers
as `GPUVector`s does not express a graph-owned transient view or add masks to the algorithms.
Changing offsets alone cannot remove an interior slot while preserving the remaining adjacency.
Keeping a second implementation of graph algorithms downstream is the maintenance cost this
proposal aims to remove.

Test the triangle above on one compiled graph across both submissions. Also keep offsets constant
and set `nodeMask = [1, 0, 1]` with the first slot mask: live degrees are `[0, 0, 0]`, and BFS from
0 reaches only 0. Test a disabled source, isolated nodes, spare capacity with hostile trailing
neighbors, nonzero view offsets, duplicate edges, reciprocal undirected edges, one self-loop and
incomplete topology status. Assert exact degree, labels/distances and statuses, not only totals.
These distinguish implementations that ignore either mask or cache stale mask-derived state.

Two additional exact acceptance fixtures:

- `vertexCount = 1`, `offsets = [0,2]`, `neighbors = [0xffffffff,0xffffffff]`, guaranteed complete:
  live degree `[0]`, component label `[0]`, `converged = 1`. These are the right/down boundary slots
  of a one-pixel terrain raster, not two self-loops.
- `vertexCount = 2`, `offsets = [0,1,1]`, `neighbors = [1]`, tracked forward overflow:
  on one compiled component graph with enough iterations to converge, overflow `0` gives labels
  `[0,0]`, `converged = 1`; change only overflow to `1` and expect labels
  `[0xffffffff,0xffffffff]`, `converged = 0`. Change back to `0` to check stale invalidation clears.
  Identical stored adjacency must not erase this independent completeness input.

The downstream headless regression file is
`modules/analysis/test/analysis-core/upstream-drafts.spec.ts`. Until the proposed view API exists,
it probes BFS and overflow using the published physical-topology route, and sentinel degree/components
using the existing downstream consumer. The BFS baseline materializes each slot mask as sentinel
neighbors before submission; it does not test GPU-resident mask consumption by a nonexistent API.
Port these fixtures unchanged in meaning to the upstream view route when implementing it. No GPU
result from that regression file has been observed while preparing this draft.

## Downstream deletion

Deletion requires sentinel support as well as masks, completeness and parity for every caller.
Terrain spike repair emits two slots per pixel, using `0xffffffff` for absent right/down neighbors;
its one-pixel case must pass the fixture above before deleting the helper. If an initial upstream
version accepts only valid neighbors, these callers must retain the helper or a GPU compaction
adapter that rebuilds offsets, removes sentinel slots and carries status. Such a version alone
does not justify helper deletion.

Once those conditions hold, delete
`createMaskedWeakComponentNodes` in
`modules/analysis/src/analysis-core/kernels/masked-weak-components.ts` and delegate its callers.
Replace live-degree/component passes in
`modules/analysis/src/gpu-network/network-analysis/gpu-network-analytics-columns.ts` and
`modules/analysis/src/gpu-network/network-statistics/gpu-network-statistics.ts`; retain their domain
summaries, histograms and normalization. Degree delegation for statistics additionally needs its
documented twice-per-undirected-self-loop convention. Its modularity pass can only be deleted after
the published weighted/unweighted algorithm meets selection and deterministic-sum requirements.

Caller CSR support alone does not retire road noding, route reconstruction, turn costs or other
domain operations. No downstream source is changed by this draft.
