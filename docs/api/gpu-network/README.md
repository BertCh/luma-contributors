
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

See [network structure, alternative routes, edge centrality and spatial topology extensions](./expansion.md)
for the bounded expansion APIs, their result contracts and reference checks.

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

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkReachability,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const graph = new GPUCommandGraph(device, {id: 'network'});
const startNodes = new GPUParameterBuffer(device, {id: 'starts', format: 'uint32', length: 4});
const costLimit = new GPUParameterBuffer(device, {id: 'cost-limit', format: 'float32', length: 1});
const costBreaks = new GPUParameterBuffer(device, {id: 'cost-breaks', format: 'float32', length: 4});

// `offsets`, `neighbors`, `weights`, `costs`, `bands`, `bandCounts`, and `status.converged` are graph
// views created on `graph`. Build the CSR on the GPU with `GPUCOOToCSR`. One topology object
// feeds every network contributor.
const topology: GPUNetworkTopology = {offsets, neighbors, weights};
graph.add(new GPUNetworkReachability({
  topology,
  startNodes: startNodes.importToGraph(graph),
  costLimit: costLimit.importToGraph(graph),
  costBreaks: costBreaks.importToGraph(graph),
  maximumIterations: 48,
  output: {costs, bands, bandCounts, status: {converged}}
}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite the start node and cost limit, encode the same compiled graph
startNodes.write(new Uint32Array([startNode, 0, 0, 0]));
costLimit.write(new Float32Array([1800]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

`encode()` only records; submit with `device.submit()`. [Running contributors](../gpu-spatial-analysis/concepts.md#running-contributors)
covers `compileAsync()`, rebuilding without a stall, submission and why `parameters` is `undefined`.

## Concepts and operators

- [GPU Network concepts](./concepts.md): conventions, units and the GPU execution model.
- [GPU Network operators](./operators.md): every contributor, its inputs, outputs, parameters and how it
  computes.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-network}

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
