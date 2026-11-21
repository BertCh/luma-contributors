
# GPU Network


:::caution Experimental
`@luma-contributors/analysis/gpu-network` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

`@luma-contributors/analysis/gpu-network` provides analysis contributors for networks stored as CSR
adjacency, passed as one `topology: GPUNetworkTopology` (`offsets`, `neighbors`, `weights`, optional
`nodePositions`): weighted reachability and isochrones (isoband or cell-outline polygons), path
extraction, service areas, neighborhoods, snapping, cost matrices, accessibility scores, subgraph
filters, coarsening, network statistics, node-aligned analytics columns, origin-destination flow
aggregation, edge bundling, adjacency-matrix images, and network construction and analysis from line
data (noding, turn-restricted line graphs, map matching, network K function). Each contributor declares resources and
command nodes into a caller-owned [`GPUCommandGraph`](https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-command-graph) through
`getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never compiles, submits, encodes, or reads
back.

See [network structure, routes and network topology](./expansion.md)
for the bounded expansion APIs, their result contracts and reference checks.

## Choose a network task

### Start and choose

- [GPU Network concepts](./concepts.md)
- [GPU Network operators](./operators.md)

### Route and measure access

- [Alternative routes](./alternative-routes.md)
- [K-shortest simple paths](./k-shortest-paths.md)
- [Network time geography](./network-time-geography.md)

### Analyze topology and spatial patterns

- [GPUNetworkLouvain](./louvain.md)
- [Visibility graph and network centrality](./visibility-graph.md)
- [Network structure, routes and spatial topology extensions](./expansion.md)
- [Network point processes](./point-process.md)

### Analyze movement and flows

- [Bounded network optimization and trajectory analysis](./network-movement-coverage.md)
- [Network and movement measurements](./network-movement-benchmarks.md)
- [Flow models on origin-destination aggregation](./origin-destination-flow-models.md)
- [Edge volumes of map-matched tracks](./map-matched-edge-volumes.md)

### Inspect recorded outputs and status

- [Network status and count checks](./status-oracles.md)

### Compatibility URLs

These links preserve older routes; the task chooser above uses the canonical guides.

- [Flow models on origin-destination aggregation (legacy URL)](./flow-models-mob10.md)
- [Edge volumes of map-matched tracks (legacy URL)](./map-matched-edge-volumes-mob05.md)
- [Network time geography (legacy URL)](./network-time-geography-mob09.md)

See also the [sidebar-backed network chooser](../analysis-catalog.md#network-chooser).

## When to use it

Use GPU Network for map and road-network tasks that need bounded, renderable results: isochrone
bands that recolor as a cost limit moves, a path that follows a dragged endpoint, a statistics panel
or analytics columns for deck.gl styling, or aggregated flows and bundled edges for drawing.

The generic graph algorithms live in [GPU Graph](https://luma.gl/docs/api-reference/experimental/gpu-graph), and the network contributors build
on them rather than replace them:

- `GPUNetworkAnalyticsColumns` publishes degree, in-degree and weak component columns on the caller's
  CSR views with analysis-owned kernels, then normalizes the columns for styling. Published
  luma.gl graph algorithms (`GPUGraphAlgorithmsPageRank`, `GPUGraphAlgorithmsCoreNumber`,
  `GPUGraphAlgorithmsLabelPropagation`, `GPUGraphAlgorithmsModularityOptimization`) accept only
  physical `GPUGraphAlgorithmsTopology`, so call them yourself when you need ranking, core numbers
  or communities, or their raw outputs and convergence status.
- `GPUNetworkStatistics` reduces a CSR to one summary row (counts, components, degree histograms,
  optional modularity) for a stats panel.
- `GPUNetworkReachability` is a multi-source, cost-limited shortest-path search with isochrone
  bands and cycle-safe predecessors, all written to caller-owned views. Use
  `GPUGraphAlgorithmsSingleSourceShortestPath` when you need plain bounded shortest paths from one source and
  the GPU Graph topology and status contracts.
- `GPUGraphAlgorithmsForceLayout` and `GPUGraphAlgorithmsSpatialForceLayout` position nodes; `GPUEdgeBundling` then
  turns the straight edges into renderable polylines.

## Quick start

Use the [complete GPU walkthrough](../operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for device creation, buffer imports, submission, typed readback and cleanup. For reachability,
replace its point-density inputs with this complete three-node directed CSR fixture.
The only edge is 1 → 2 with cost 2; node 0 is isolated.

```ts
export const reachabilityInputs = {
  topology: {
    offsets: Uint32Array.of(0, 0, 1, 1),
    neighbors: Uint32Array.of(2),
    weights: Float32Array.of(2)
  },
  startNodes: Uint32Array.of(1),
  costLimit: Float32Array.of(10),
  costBreaks: Float32Array.of(0, 2, 5, 10)
};
```

Import every array once, then create scalar `uint32` views for offsets, neighbors and start nodes,
and scalar `float32` views for weights, cost limit and cost breaks, with lengths equal to their
array lengths. Pass the three topology views together as `topology` to `GPUNetworkReachability`,
and the remaining views as `startNodes`, `costLimit` and `costBreaks`. Set
`maximumIterations: 4` for this fixture.

Allocate the following outputs in separate buffers and pass their views as `output`:

| Port | Format and rows | Expected readback |
| --- | --- | --- |
| `costs` | `float32`, 3 | `[Infinity, 0, 2]` |
| `bands` | `uint32`, 3 | `[0xffffffff, 0, 1]` |
| `bandCounts` | `uint32`, 4 | `[1, 1, 0, 0]` |
| `status.converged` | `uint32`, 1 | `[1]` |
| `status.iterationLimitReached` | `uint32`, 1 | `[0]` |

There is exactly **one start slot**. Every slot is active when `startCount` is omitted; if reserving
spare slots, supply a one-row `uint32` `startCount` view containing 1. Spare zero-filled slots
would seed node 0 too. Cost breaks must be initialized in ascending order in the edge-weight units;
a node belongs to the first break greater than or equal to its cost, or `0xffffffff` beyond
the final break. Node 0 stays unreachable and contributes to no band.

Compile, encode, submit and read these buffers with the walkthrough's lifecycle. Check
`iterationLimitReached` with the costs: when it is 1, propagate incompleteness, suppress dependent
results, or continue/fall back rather than treating unreached nodes as proof of no route.

## CPU first run

These array functions need no GPU device. Edge weights are travel costs in the caller's chosen
unit (seconds in this example); rows are zero-based vertex indices.

```ts
import {
  computeNetworkReachabilityOnCPU,
  computeNetworkCostMatrixOnCPU
} from '@luma-contributors/analysis/gpu-network';
const topology = {
  offsets: Uint32Array.of(0, 1, 2, 2),
  neighbors: Uint32Array.of(1, 2),
  weights: Float32Array.of(2, 3)
};
const reachable = computeNetworkReachabilityOnCPU({
  topology, startNodes: Uint32Array.of(0)
});
console.log(Array.from(reachable.costs)); // [0, 2, 5]
const matrix = computeNetworkCostMatrixOnCPU({
  topology,
  startNodes: Uint32Array.of(0, 2),
  targetNodes: Uint32Array.of(1, 2)
}, {rowCount: 2});
console.log(Array.from(matrix.costs)); // [2, 5, Infinity, 0]
```

The OD matrix has two origin rows (vertices 0 and 2) and two target columns (vertices 1 and 2).
Origin 2 cannot reach target 1, so that entry is positive infinity. Without an explicit iteration
budget these CPU searches run to completion; supplying a budget retains its bounded-search contract.

| Task | Entry and reading contract |
| --- | --- |
| Weak component labels | `GPUNetworkAnalyticsColumns` components output, one row per vertex |
| Strong component labels | `GPUNetworkStructure` or `GPUNetworkStronglyConnectedComponents`; inspect SCC status |
| Component summaries | `GPUNetworkStatistics` packed summary and its decoder |
| Route geometry | `GPUNetworkPathExtraction`; forward predecessor completeness when composing |
| OD costs | `GPUNetworkCostMatrix` / `computeNetworkCostMatrixOnCPU`; `targetNodes` sets column order |
| Lexicographic simple alternatives | `GPUAlternativeRoutes`: CSR edge-slot tie order, `nodeCount` path stride, per-query completion |
| Bounded Yen alternatives | `GPUKShortestPaths`: `maximumPathLength` path stride, batch interruption status |

`computeNetworkStructureOnCPU` accepts either flat `{offsets, neighbors, reverseEdgeSlots?}`
or named `{topology: {offsets, neighbors}, reverseEdgeSlots?}` input. Reverse slots are needed
for undirected resilience outputs; SCC-only use does not require them.

## Concepts and operators

- [GPU Network concepts](./concepts.md): conventions, units and the GPU execution model.
- [GPU Network operators](./operators.md): every contributor, its inputs, outputs, parameters and how it
  computes.

## Catalog

[Browse network operations in the catalog](/explore/?data=network) to inspect contracts and
available recordings. A recording's own provenance identifies its dataset, device and execution;
listing an operation does not establish a GPU qualification run.

## Limits and compatibility

- GPU Network is experimental and WebGPU-only.
- CSR graphs use packed views only. Edge weights must be non-negative where a contributor takes costs.
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (`dev-docs/roadmap.md`) tracks what is still open.

## Related modules

- [GPU Graph](https://luma.gl/docs/api-reference/experimental/gpu-graph) supplies the generic algorithms, topology views, and layouts.
- [GPU Raster](../gpu-raster/README.md) has the raster counterpart of cost distance and isolines.
- [GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core/concepts) defines contributors, composition levels, and graph ownership.

- [Bounded alternative routes](./alternative-routes.md)
