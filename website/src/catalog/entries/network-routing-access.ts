// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: network routing and access. Street and flow networks: routing, centrality,
// isochrones, service areas, accessibility, allocation and network point processes.

import type {CatalogEntry} from '../schema';

export const entries: CatalogEntry[] = [
  {
    id: 'network-reachability',
    symbol: 'GPUNetworkReachability',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network reachability',
    summary:
      'Shortest-path travel costs, bands and predecessors from one or many start nodes over a road network.',
    explanation:
      'Best for drive-time and walk-time maps. It relaxes a directed CSR network with a GPU frontier queue until costs stop improving. Optional cost breaks give isochrone bands; predecessors give the paths.',
    dataTypes: {input: ['network', 'points'], output: ['network']},
    useCases: ['route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'CSR network: offsets, neighbors, per-edge cost.'
      },
      {
        name: 'startNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Origin node ids, rewritten every frame without recompiling.'
      },
      {
        name: 'costLimit',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Stop expanding beyond this cost (one-row view).'
      },
      {
        name: 'costBreaks',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Band edges, e.g. 5, 10, 15 minutes.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        default: '64',
        description: 'Round budget; one round chains up to localIterations hops.'
      },
      {
        name: 'localIterations',
        type: 'number | undefined',
        default: '32',
        description: 'Hops chained per round inside a workgroup.'
      },
      {
        name: 'laneCount',
        type: 'number | undefined',
        default: '1',
        description: 'Independent searches run side by side.'
      },
      {
        name: 'output',
        type: 'GPUNetworkReachabilityOutput',
        required: true,
        description: 'Views to fill: costs, predecessors, bands, bandCounts, status.'
      }
    ],
    outputs: [
      {
        name: 'costs',
        type: "GraphDataView<'float32'>",
        description: 'Per-node minimum cost; Infinity when unreached.'
      },
      {
        name: 'predecessors',
        type: "GraphDataView<'uint32'>",
        description: 'Per-node predecessor on a shortest path (optional).'
      },
      {
        name: 'bands',
        type: "GraphDataView<'uint32'>",
        description: 'Per-node isochrone band id (optional).'
      },
      {
        name: 'bandCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Nodes per band (optional).'
      },
      {
        name: 'status.converged',
        type: "GraphDataView<'uint32'>",
        description: '1 when relaxation reached a fixpoint.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNetworkReachability} from '@luma-contributors/analysis/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'reach'});
// topology {offsets, neighbors, weights}, starts, costBreaks and output views live on \`graph\`
graph.add(new GPUNetworkReachability({
  topology,
  startNodes: starts.importToGraph(graph),
  costBreaks: costBreaks.importToGraph(graph),
  maximumIterations: 48,
  output: {costs, bands, status: {converged}}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'isochrones',
      caption: 'Travel cost spreads from the start; breaks cut it into bands.',
      seed: 4301
    },
    related: [
      'network-service-areas',
      'network-isochrones',
      'network-path-extraction',
      'network-cost-matrix'
    ],
    lensScenario: 'gpu-network.network-reachability',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-service-areas',
    symbol: 'GPUNetworkServiceAreas',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network service areas',
    summary:
      'Assigns every network node to its nearest facility by travel cost: a Voronoi diagram on the network.',
    explanation:
      'Shows which facility serves which part of a street network. It runs multi-source relaxation with nearest-label propagation. Per-facility node counts and cost sums come along.',
    dataTypes: {input: ['network'], output: ['network']},
    useCases: ['route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'CSR network: offsets, neighbors, per-edge cost.'
      },
      {
        name: 'facilities',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Facility node ids.'
      },
      {
        name: 'facilityCosts',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional head-start cost per facility.'
      },
      {
        name: 'costLimit',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Nodes costlier than this stay unassigned.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        description: 'Relaxation round budget.'
      },
      {
        name: 'localIterations',
        type: 'number | undefined',
        default: '16',
        description: 'Hops chained per round.'
      },
      {
        name: 'output',
        type: 'GPUNetworkServiceAreasOutput',
        required: true,
        description: 'Views to fill: assignments, costs, predecessors, per-facility totals.'
      }
    ],
    outputs: [
      {
        name: 'assignments',
        type: "GraphDataView<'uint32'>",
        description: 'Nearest facility index per node.'
      },
      {
        name: 'costs',
        type: "GraphDataView<'float32'>",
        description: 'Cost to that facility (optional).'
      },
      {
        name: 'facilityNodeCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Nodes served per facility (optional).'
      },
      {
        name: 'facilityCostSums',
        type: "GraphDataView<'float32'>",
        description: 'Summed cost per facility (optional).'
      },
      {
        name: 'status.converged',
        type: "GraphDataView<'uint32'>",
        description: '1 when labels settled.'
      }
    ],
    example: `import {GPUNetworkServiceAreas} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkServiceAreas({
  topology,
  facilities: facilityNodes.importToGraph(graph),
  costLimit: costLimit.importToGraph(graph),
  maximumIterations: 48,
  output: {assignments, costs, facilityNodeCounts}
}));`,
    preview: {
      kind: 'isochrones',
      caption: 'Each color is the set of nodes closest to one facility.',
      seed: 4302
    },
    related: ['network-reachability', 'location-allocation'],
    lensScenario: 'gpu-network.network-service-areas',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-isochrones',
    symbol: 'GPUNetworkIsochrones',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network isochrones',
    summary:
      'Turns network travel costs into contoured isochrone polygons, or the outline of reached cells.',
    explanation:
      'Best when you need drawable travel-time areas, not per-node costs. Edge-interpolated costs are splatted to a raster and contoured into bands. Feed it costs from reachability or service areas.',
    dataTypes: {input: ['network'], output: ['polygons', 'raster']},
    useCases: ['route', 'transform'],
    domains: ['transport', 'urban', 'cartography'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {nodePositions}',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-node costs from reachability.'
      },
      {
        name: 'costBreaks',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Band edges for the contours.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame extent, break count and buffer radius.'
      },
      {
        name: 'raster',
        type: 'GPUNetworkIsochronesRaster | undefined',
        description: 'Splat raster size and its triangle outputs.'
      },
      {
        name: 'cellOutline',
        type: 'GPUNetworkIsochronesCellOutline | undefined',
        description: 'Outline rings of reached cells instead of a raster.'
      },
      {
        name: 'assignments',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional facility labels to isolate per-facility areas.'
      },
      {
        name: 'output',
        type: 'GPUNetworkIsochronesOutput | undefined',
        description: 'Convergence status.'
      }
    ],
    outputs: [
      {
        name: 'raster.output.triangles',
        type: 'GraphDataView<"float32x2">',
        description: 'Isoband triangle vertices, three per triangle.'
      },
      {
        name: 'raster.output.triangleBands',
        type: 'GraphDataView<"uint32">',
        description: 'Cost band of each triangle.'
      },
      {
        name: 'cellOutline',
        type: 'GPUNetworkIsochronesCellOutline',
        description: 'Outline of the reached cells, when requested instead of a raster.'
      },
      {
        name: 'status',
        type: '{converged?}',
        description: 'Optional flag: 1 when label propagation finished.'
      }
    ],
    example: `import {
  GPUNetworkIsochrones, getGPUNetworkIsochronesParameterValues
} from '@luma-contributors/analysis/gpu-network';

// topology, costs (from GPUNetworkReachability) and output views live on \`graph\`
graph.add(new GPUNetworkIsochrones({
  topology, costs,
  costBreaks: costBreaks.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  raster: {width: 256, height: 256, output: {triangles, triangleBands, status: {count, overflow}}}
}));
// parameters: getGPUNetworkIsochronesParameterValues({breakCount: 4, extent, bufferRadius: 50})`,
    preview: {
      kind: 'isochrones',
      caption: 'Nested travel-time polygons around the origin.',
      seed: 4303
    },
    related: ['network-reachability', 'network-service-areas', 'drive-time-catchment'],
    lensScenario: 'gpu-network.network-isochrones',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-cost-matrix',
    symbol: 'GPUNetworkCostMatrix',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network cost matrix',
    summary:
      'Bounded many-to-all shortest-path cost matrix: one search per row over a directed network.',
    explanation:
      'Builds origin-destination costs for accessibility, allocation or disruption studies. Each row runs one search from its start nodes, optionally cut off by cost. Keep the matrix on the GPU and feed accessibility directly.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'CSR network: offsets, neighbors, per-edge cost.'
      },
      {
        name: 'startNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Start nodes, grouped into rows.'
      },
      {
        name: 'startsPerRow',
        type: 'number | undefined',
        default: '1',
        description: 'Start nodes per matrix row.'
      },
      {
        name: 'startCosts',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional head-start cost per start node.'
      },
      {
        name: 'targetNodes',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Restrict columns to these nodes.'
      },
      {
        name: 'costLimit',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Stop expanding beyond this cost.'
      },
      {name: 'laneCount', type: 'number | undefined', description: 'Rows searched concurrently.'},
      {
        name: 'output',
        type: 'GPUNetworkCostMatrixOutput',
        required: true,
        description: 'costs plus optional converged status.'
      }
    ],
    outputs: [
      {
        name: 'costs',
        type: "GraphDataView<'float32'>",
        description: 'Row-major rows x columns costs; Infinity when unreached.'
      },
      {
        name: 'status.converged',
        type: "GraphDataView<'uint32'>",
        description: '1 when every search finished.'
      }
    ],
    example: `import {GPUNetworkCostMatrix} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkCostMatrix({
  topology,
  startNodes: origins.importToGraph(graph),
  costLimit: costLimit.importToGraph(graph),
  targetNodes: destinations.importToGraph(graph),
  maximumIterations: 64,
  output: {costs, status: {converged}}
}));`,
    preview: {
      kind: 'network',
      caption: 'Many origins, one search each, costs to every node.',
      seed: 4304
    },
    related: [
      'network-accessibility',
      'network-cost-weights',
      'network-reachability',
      'network-cost-table'
    ],
    lensScenario: 'gpu-network.network-cost-matrix',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-cost-table',
    symbol: 'GPUNetworkCostTable',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network cost table',
    summary:
      'Precomputes a bounded all-pairs cost table of a static network, as a sparse CSR of reachable pairs.',
    explanation:
      'A fast lookup for map matching and repeated routing on a fixed network. Every node searches out to the cost limit. Reachable targets are stored sparsely, with overflow reported on the GPU.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['route', 'index'],
    domains: ['transport'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'CSR network: offsets, neighbors, per-edge cost.'
      },
      {
        name: 'costLimit',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Radius of every search (one-row view).'
      },
      {name: 'laneCount', type: 'number | undefined', description: 'Searches run concurrently.'},
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        description: 'Round budget per batch.'
      },
      {
        name: 'localIterations',
        type: 'number | undefined',
        default: '16',
        description: 'Hops chained per round.'
      },
      {
        name: 'output',
        type: 'GPUNetworkCostTableOutput',
        required: true,
        description: 'offsets, targets, costs and overflow status.'
      }
    ],
    outputs: [
      {
        name: 'offsets',
        type: "GraphDataView<'uint32'>",
        description: 'Row offsets into targets and costs.'
      },
      {
        name: 'targets',
        type: "GraphDataView<'uint32'>",
        description: 'Reachable target node per entry.'
      },
      {name: 'costs', type: "GraphDataView<'float32'>", description: 'Cost per entry.'},
      {
        name: 'status.overflow',
        type: "GraphDataView<'uint32'>",
        description: '1 when the table capacity was exceeded.'
      }
    ],
    example: `import {GPUNetworkCostTable} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkCostTable({
  topology,
  costLimit: radius.importToGraph(graph),
  maximumIterations: 48,
  output: {offsets, targets, costs, status: {overflow, converged}}
}));`,
    preview: {
      kind: 'network',
      caption: 'Every node keeps the costs to its nearby nodes.',
      seed: 4305
    },
    related: ['network-cost-matrix', 'map-matching'],
    lensScenario: 'gpu-network.network-cost-table',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-cost-weights',
    symbol: 'GPUNetworkCostWeights',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network cost weights',
    summary:
      'Converts network costs into spatial weights with distance decay, so any weights-based tool runs on a network.',
    explanation:
      'Runs Moran, Gi* or smoothing with network travel time instead of straight-line distance. It reads a cost matrix or reachability result and writes CSR weights. Decay shape and cutoff come from the per-frame parameters.',
    dataTypes: {input: ['network'], output: ['spatial-weights']},
    useCases: ['relate', 'find-patterns'],
    domains: ['spatial-statistics', 'transport'],
    params: [
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Network costs from a cost matrix.'
      },
      {name: 'rowCount', type: 'number | undefined', description: 'Number of origin rows.'},
      {
        name: 'columns',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional column node ids.'
      },
      {
        name: 'orientation',
        type: '"rows" | "columns" | undefined',
        description: 'Whether costs are laid out by rows or columns.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame decay kind, scale and cutoff.'
      },
      {
        name: 'output',
        type: 'GPUNetworkCostWeightsOutput',
        required: true,
        description: 'CSR weights: offsets, neighbors, weights.'
      }
    ],
    outputs: [
      {name: 'offsets', type: "GraphDataView<'uint32'>", description: 'CSR row offsets.'},
      {
        name: 'neighbors',
        type: "GraphDataView<'uint32'>",
        description: 'Neighbor index per entry.'
      },
      {name: 'weights', type: "GraphDataView<'float32'>", description: 'Decayed weight per entry.'},
      {
        name: 'distances',
        type: "GraphDataView<'float32'>",
        description: 'Network cost per entry (optional).'
      },
      {
        name: 'status.overflow',
        type: "GraphDataView<'uint32'>",
        description: '1 when capacity was exceeded.'
      }
    ],
    example: `import {GPUNetworkCostWeights} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkCostWeights({
  costs: matrixCosts,
  rowCount: originCount,
  parameters: decayParameters.importToGraph(graph),
  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}
}));
// then pass {offsets, neighbors, weights} wherever GPUSpatialWeights are read`,
    preview: {
      kind: 'spatial-weights',
      caption: 'Neighbors by travel time, weighted by decay.',
      seed: 4306
    },
    related: ['network-cost-matrix', 'network-accessibility'],
    lensScenario: 'gpu-network.network-cost-weights',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-accessibility',
    symbol: 'GPUNetworkAccessibility',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network accessibility',
    summary:
      'Scores per-origin accessibility from a cost matrix: cumulative opportunities, gravity and catchment ratios.',
    explanation:
      'Counts how many jobs, clinics or parks each origin can reach. It reads a retained cost matrix and a per-frame threshold. Cumulative, gravity and floating-catchment scores can be written in one pass.',
    dataTypes: {input: ['table', 'network'], output: ['table']},
    useCases: ['route', 'summarize'],
    domains: ['urban', 'demographics', 'transport'],
    params: [
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Cost matrix from GPUNetworkCostMatrix.'
      },
      {
        name: 'orientation',
        type: '"opportunity-rows" | "origin-rows" | undefined',
        description: 'Matrix layout: opportunity-rows or origin-rows.'
      },
      {
        name: 'opportunityWeights',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Opportunities (jobs, beds) per destination.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame threshold and gravity decay.'
      },
      {
        name: 'cumulative',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Output: opportunities within the threshold.'
      },
      {
        name: 'gravity',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Output: distance-decayed opportunities.'
      },
      {
        name: 'catchment',
        type: 'GPUNetworkAccessibilityCatchment | undefined',
        description: 'Optional floating-catchment ratios with demand.'
      }
    ],
    outputs: [
      {
        name: 'cumulative',
        type: "GraphDataView<'float32'>",
        description: 'Opportunities reachable within the threshold.'
      },
      {
        name: 'gravity',
        type: "GraphDataView<'float32'>",
        description: 'Gravity-weighted accessibility score.'
      },
      {
        name: 'catchment.output',
        type: "GraphDataView<'float32'>",
        description: 'Catchment-ratio accessibility (optional).'
      }
    ],
    example: `import {GPUNetworkAccessibility} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkAccessibility({
  costs: matrix.costs,
  opportunityWeights: jobs.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  cumulative,
  gravity
}));
// parameters: getGPUNetworkAccessibilityParameterValues({...}) holds the threshold and decay`,
    preview: {kind: 'hotspots', caption: 'Origins that reach many opportunities glow.', seed: 4307},
    related: [
      'network-cost-matrix',
      'catchment-accessibility',
      'facility-scenario',
      'network-disruption'
    ],
    lensScenario: 'gpu-network.network-accessibility',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'catchment-accessibility',
    symbol: 'GPUCatchmentAccessibility',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-spatial-analysis',
    title: 'Catchment accessibility (2SFCA)',
    summary:
      'Two-step and three-step floating catchment accessibility on any spatial weights, Euclidean or network.',
    explanation:
      'Measures healthcare or service access while accounting for competing demand. Step one computes supply-to-demand ratios per facility; step two sums the ratios each demand location can reach. Weights decide what counts as reachable.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['route', 'summarize'],
    domains: ['demographics', 'urban'],
    params: [
      {
        name: 'method',
        type: 'GPUCatchmentMethod',
        required: true,
        description: 'Which catchment method to run.'
      },
      {
        name: 'supply',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Capacity per facility (beds, doctors).'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand per location (population).'
      },
      {
        name: 'facilityWeights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Spatial weights from each facility to demand.'
      },
      {
        name: 'demandWeights',
        type: 'GPUSpatialWeights | undefined',
        description: 'Weights from each demand location to facilities.'
      },
      {
        name: 'output',
        type: 'GPUCatchmentAccessibilityOutput',
        required: true,
        description: 'accessibility plus optional ratios and facility counts.'
      }
    ],
    outputs: [
      {
        name: 'accessibility',
        type: "GraphDataView<'float32'>",
        description: 'Accessibility score per demand location.'
      },
      {
        name: 'ratios',
        type: "GraphDataView<'float32'>",
        description: 'Supply-to-demand ratio per facility (optional).'
      },
      {
        name: 'reachableFacilities',
        type: "GraphDataView<'uint32'>",
        description: 'Facilities within reach (optional).'
      }
    ],
    example: `import {GPUCatchmentAccessibility} from '@luma-contributors/analysis/gpu-spatial-analysis';

graph.add(new GPUCatchmentAccessibility({
  method: '2sfca',
  supply: beds.importToGraph(graph),
  demand: population.importToGraph(graph),
  facilityWeights,
  output: {accessibility, ratios}
}));
// facilityWeights: GPUSpatialWeights, e.g. from GPUNetworkCostWeights`,
    preview: {
      kind: 'hotspots',
      caption: 'Access drops where many people share few facilities.',
      seed: 4308
    },
    related: ['network-accessibility', 'huff-trade-areas', 'network-cost-weights'],
    lensScenario: 'gpu-spatial-analysis.catchment-accessibility',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'huff-trade-areas',
    symbol: 'GPUHuffTradeAreas',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-spatial-analysis',
    title: 'Huff trade areas',
    summary: 'Huff gravity-model probabilities that each demand location patronises each facility.',
    explanation:
      'Best for retail and service market areas where attractive, nearby facilities win more customers. The probability is attractiveness over distance decay, normalized across competing facilities. The most likely facility becomes the trade area.',
    dataTypes: {input: ['points', 'spatial-weights'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['urban', 'demographics'],
    params: [
      {
        name: 'attractiveness',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Attractiveness per facility (size, rating).'
      },
      {
        name: 'demandWeights',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Weights from demand locations to facilities.'
      },
      {
        name: 'facilityWeights',
        type: 'GPUSpatialWeights | undefined',
        description: 'Optional facility-side weights for expected demand.'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Demand per location, for expected demand totals.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Per-frame exponents.'
      },
      {
        name: 'output',
        type: 'GPUHuffTradeAreasOutput',
        required: true,
        description: 'tradeArea plus optional probabilities.'
      }
    ],
    outputs: [
      {
        name: 'tradeArea',
        type: "GraphDataView<'uint32'>",
        description: 'Most likely facility per demand location.'
      },
      {
        name: 'probabilities',
        type: "GraphDataView<'float32'>",
        description: 'Patronage probability per entry (optional).'
      },
      {
        name: 'tradeAreaProbability',
        type: "GraphDataView<'float32'>",
        description: 'Probability of the winning facility (optional).'
      },
      {
        name: 'expectedDemand',
        type: "GraphDataView<'float32'>",
        description: 'Expected demand per facility (optional).'
      }
    ],
    example: `import {GPUHuffTradeAreas} from '@luma-contributors/analysis/gpu-spatial-analysis';

graph.add(new GPUHuffTradeAreas({
  attractiveness: storeSize.importToGraph(graph),
  demandWeights,
  demand: households.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  output: {tradeArea, probabilities, expectedDemand}
}));`,
    preview: {
      kind: 'polygon-overlay',
      caption: 'Overlapping trade areas split by attractiveness and distance.',
      seed: 4309
    },
    related: ['catchment-accessibility', 'network-service-areas'],
    lensScenario: 'gpu-spatial-analysis.huff-trade-areas',
    reference: 'api/gpu-spatial-analysis/operators-accessibility',
    maturity: 'stable'
  },
  {
    id: 'location-allocation',
    symbol: 'GPULocationAllocation',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-spatial-analysis',
    title: 'Location allocation',
    summary:
      'Deterministic p-median, p-center and coverage facility selection over a caller-owned cost matrix.',
    explanation:
      'Chooses where to put facilities so demand is served well. It starts from a greedy pick and improves it with swaps, all on the GPU. Choose the objective with `operation`.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['urban', 'transport'],
    params: [
      {
        name: 'operation',
        type: 'GPULocationAllocationOperation',
        required: true,
        description: 'Objective: p-median, p-center, coverage and so on.'
      },
      {
        name: 'costs',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Dense demand x facility cost matrix.'
      },
      {
        name: 'sparseCosts',
        type: 'GPULocationAllocationSparseCosts | undefined',
        description: 'Sparse alternative to the dense matrix.'
      },
      {
        name: 'facilityCount',
        type: 'number',
        required: true,
        description: 'Number of candidate facilities.'
      },
      {
        name: 'maximumFacilityCount',
        type: 'number',
        required: true,
        description: 'Number of facilities to select.'
      },
      {
        name: 'demandWeights',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Weight per demand point.'
      },
      {
        name: 'maximumSwapIterations',
        type: 'number | undefined',
        description: 'Swap improvement rounds.'
      },
      {
        name: 'coverageDistance',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Coverage radius for coverage objectives.'
      }
    ],
    outputs: [
      {
        name: 'selectedFacilities',
        type: "GraphDataView<'uint32'>",
        description: 'Chosen facility indices.'
      },
      {
        name: 'assignments',
        type: "GraphDataView<'uint32'>",
        description: 'Facility assigned to each demand point.'
      },
      {
        name: 'assignedCosts',
        type: "GraphDataView<'float32'>",
        description: 'Cost to the assigned facility (optional).'
      },
      {name: 'objective', type: "GraphDataView<'float32'>", description: 'Final objective value.'},
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Result status code.'}
    ],
    example: `import {GPULocationAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';

graph.add(new GPULocationAllocation({
  operation: 'p-median',
  costs: matrix.costs,
  facilityCount: 40,
  maximumFacilityCount: 5,
  demandWeights: population.importToGraph(graph),
  maximumSwapIterations: 20,
  output: {selectedFacilities, assignments, objective, status}
}));`,
    preview: {
      kind: 'point-clusters',
      caption: 'Five sites chosen; demand joins its nearest one.',
      seed: 4310
    },
    related: [
      'bounded-facility-allocation',
      'network-facility-optimization',
      'network-service-areas',
      'network-cost-matrix'
    ],
    lensScenario: 'gpu-spatial-analysis.location-allocation',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'bounded-facility-allocation',
    symbol: 'GPUBoundedFacilityAllocation',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-spatial-analysis',
    title: 'Bounded facility allocation',
    summary:
      'Exhaustive p-center or capacitated p-median: scores every facility subset (up to 65536) in parallel.',
    explanation:
      'Best when the candidate set is small and you want the exact optimum, not a heuristic. Each GPU thread evaluates one subset against the cost matrix. Capacities are enforced for the capacitated variant.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['urban', 'transport'],
    params: [
      {
        name: 'operation',
        type: '"p-center" | "capacitated-p-median"',
        required: true,
        description: "'p-center' or 'capacitated-p-median'."
      },
      {
        name: 'facilityCount',
        type: 'number',
        required: true,
        description: 'Number of candidate facilities.'
      },
      {name: 'selectCount', type: 'number', required: true, description: 'How many to select.'},
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand x facility cost matrix.'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand per point.'
      },
      {
        name: 'capacities',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Capacity per facility.'
      },
      {
        name: 'output',
        type: '{selectedFacilities, assignments}',
        required: true,
        description: 'selectedFacilities, assignments, objective, status.'
      }
    ],
    outputs: [
      {name: 'selectedFacilities', type: "GraphDataView<'uint32'>", description: 'Best subset.'},
      {
        name: 'assignments',
        type: "GraphDataView<'uint32'>",
        description: 'Facility per demand point.'
      },
      {
        name: 'objective',
        type: "GraphDataView<'float32'>",
        description: 'Optimal objective value.'
      },
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Result status code.'}
    ],
    example: `import {GPUBoundedFacilityAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';

graph.add(new GPUBoundedFacilityAllocation({
  operation: 'capacitated-p-median',
  facilityCount: 12,
  selectCount: 4,
  costs: costs.importToGraph(graph),
  demand: demand.importToGraph(graph),
  capacities: capacities.importToGraph(graph),
  output: {selectedFacilities, assignments, objective, status}
}));`,
    preview: {kind: 'point-clusters', caption: 'All subsets tried; the best four win.', seed: 4311},
    related: ['location-allocation', 'network-facility-optimization'],
    lensScenario: 'gpu-spatial-analysis.bounded-facility-allocation',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'network-facility-optimization',
    symbol: 'GPUNetworkFacilityOptimization',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network facility optimization',
    summary:
      'Exhaustive facility candidates evaluated in fixed-size GPU batches: capacitated p-median and backup coverage.',
    explanation:
      'Best for exact siting problems with capacities or backup coverage thresholds. Candidates are enumerated in batches so larger sets stay bounded. Cap the work with `maximumCandidates`.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['urban', 'transport'],
    params: [
      {
        name: 'operation',
        type: '"capacitated-p-median" | "backup-coverage" | "p-dispersion"',
        required: true,
        description: "'capacitated-p-median', 'backup-coverage' and similar."
      },
      {
        name: 'facilityCount',
        type: 'number',
        required: true,
        description: 'Number of candidate facilities.'
      },
      {name: 'selectCount', type: 'number | undefined', description: 'How many to select.'},
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand x facility cost matrix.'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand per point.'
      },
      {
        name: 'capacities',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Capacity per facility.'
      },
      {
        name: 'threshold',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Coverage cost threshold.'
      },
      {
        name: 'maximumCandidates',
        type: 'number | undefined',
        description: 'Upper bound on subsets examined.'
      }
    ],
    outputs: [
      {name: 'selectedFacilities', type: "GraphDataView<'uint32'>", description: 'Best subset.'},
      {
        name: 'assignments',
        type: "GraphDataView<'uint32'>",
        description: 'Facility per demand point.'
      },
      {name: 'objective', type: "GraphDataView<'float32'>", description: 'Objective value.'},
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Result status code.'}
    ],
    example: `import {GPUNetworkFacilityOptimization} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkFacilityOptimization({
  operation: 'backup-coverage',
  facilityCount: 20,
  selectCount: 5,
  costs: costs.importToGraph(graph),
  demand: demand.importToGraph(graph),
  capacities: capacities.importToGraph(graph),
  threshold: threshold.importToGraph(graph),
  output: {selectedFacilities, assignments, objective, status}
}));`,
    preview: {kind: 'point-clusters', caption: 'Pick sites that cover demand twice.', seed: 4312},
    related: ['bounded-facility-allocation', 'location-allocation'],
    lensScenario: 'gpu-network.network-facility-optimization',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'stable'
  },
  {
    id: 'facility-scenario',
    symbol: 'addFacilityScenarioRecipe',
    kind: 'recipe',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Facility scenario',
    summary:
      'Evaluates one live facility scenario against a reusable cost matrix: access, nearest facility and coverage.',
    explanation:
      'Best for what-if siting: toggle facilities in a mask and re-score without rebuilding the matrix. Everything is read per frame, so sliders and clicks need no recompile. Results include accessibility, nearest cost and covered demand.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['urban', 'demographics'],
    params: [
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand x facility cost matrix, reused across scenarios.'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand per point.'
      },
      {
        name: 'supply',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Capacity per facility.'
      },
      {
        name: 'facilityMask',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Nonzero entries mark facilities that are open.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame coverage and decay settings.'
      },
      {
        name: 'outputs',
        type: 'Partial<GPUFacilityScenarioOutput> | undefined',
        description: 'Any of the scenario output views.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: 'GPUFacilityScenarioOutput',
        description:
          'accessibility, nearestCosts, nearestFacilities, coveredDemand, uncoveredDemand.'
      },
      {
        name: 'contributors',
        type: 'GPUCommandNodeProducer[]',
        description: 'Graph nodes the recipe added, in execution order.'
      },
      {
        name: 'intermediates',
        type: 'Record<string, unknown>',
        description: 'Named ports between stages.'
      },
      {
        name: 'status',
        type: 'GPURecipeStatusPort',
        description: 'Completeness and convergence views per stage.'
      }
    ],
    example: `import {addFacilityScenarioRecipe} from '@luma-contributors/analysis/gpu-network';

const recipe = addFacilityScenarioRecipe(graph, {
  costs: matrix.costs,
  demand: demand.importToGraph(graph),
  supply: supply.importToGraph(graph),
  facilityMask: openMask.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  outputs: {accessibility, nearestCosts, nearestFacilities}
});
// rewrite openMask each frame to try another scenario`,
    preview: {
      kind: 'point-clusters',
      caption: 'Switch facilities on and off; access updates live.',
      seed: 4313
    },
    related: ['network-accessibility', 'network-disruption', 'location-allocation'],
    lensScenario: 'gpu-network.facility-scenario-recipe',
    reference: 'api/gpu-network/expansion',
    maturity: 'stable'
  },
  {
    id: 'network-disruption',
    symbol: 'addNetworkDisruptionRecipe',
    kind: 'recipe',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network disruption',
    summary:
      'Compares baseline and disrupted routing: cost change, newly unreachable nodes and lost accessibility.',
    explanation:
      'Shows what a closed bridge or flooded road does to travel and access. It runs the cost matrix twice, with and without the edge mask, and differences them without Infinity arithmetic. Opportunity weights give the accessibility change.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['route', 'detect-change'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'search',
        type: 'Omit<GPUNetworkCostMatrixProps, "id" | "output">',
        required: true,
        description: 'Cost-matrix settings: topology, starts, limits.'
      },
      {name: 'rowCount', type: 'number', required: true, description: 'Number of origin rows.'},
      {
        name: 'edgeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Zero marks closed edge slots.'
      },
      {
        name: 'replacementCosts',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Costs to use instead of closing edges.'
      },
      {
        name: 'opportunityWeights',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Opportunities per destination.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame threshold and decay.'
      },
      {
        name: 'outputs',
        type: 'Partial<GPUNetworkDisruptionOutput> | undefined',
        description: 'Any of the disruption output views.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: 'GPUNetworkDisruptionOutput',
        description: 'Baseline and scenario costs, differences, newlyUnreachable, accessibility.'
      },
      {
        name: 'contributors',
        type: 'GPUCommandNodeProducer[]',
        description: 'Graph nodes the recipe added, in execution order.'
      },
      {
        name: 'intermediates',
        type: 'Record<string, unknown>',
        description: 'Named ports between stages.'
      },
      {
        name: 'status',
        type: 'GPURecipeStatusPort',
        description: 'Completeness and convergence views per stage.'
      }
    ],
    example: `import {addNetworkDisruptionRecipe} from '@luma-contributors/analysis/gpu-network';

const recipe = addNetworkDisruptionRecipe(graph, {
  search: {topology, startNodes: origins, costLimit, maximumIterations: 64},
  rowCount: originCount,
  edgeMask: openEdges.importToGraph(graph),
  opportunityWeights: jobs.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  outputs: {costDifferences, newlyUnreachable, accessibilityDifferences}
});`,
    preview: {
      kind: 'network',
      caption: 'Close an edge and watch the reachable area shrink.',
      seed: 4314
    },
    related: ['network-accessibility', 'network-cost-matrix', 'facility-scenario'],
    lensScenario: 'gpu-network.network-disruption-recipe',
    reference: 'api/gpu-network/expansion',
    maturity: 'stable'
  },
  {
    id: 'drive-time-catchment',
    symbol: 'addDriveTimeCatchmentRecipe',
    kind: 'recipe',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Drive-time catchment',
    summary:
      'Snaps facilities and demand to the network, then bands demand by drive time to the nearest facility.',
    explanation:
      'Best for catchments such as demand within 5, 10 and 15 minutes of a clinic. It snaps points, runs service areas, bands the times and can build isochrone polygons. Band counts and value sums are written on the GPU.',
    dataTypes: {input: ['points', 'network'], output: ['table', 'polygons']},
    useCases: ['route', 'summarize'],
    domains: ['urban', 'transport', 'demographics'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {nodePositions}',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'facilities',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Facility coordinates (float32x2).'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Demand coordinates (float32x2).'
      },
      {
        name: 'demandValues',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional value per demand point to sum by band.'
      },
      {
        name: 'costBreaks',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Band edges in network cost units.'
      },
      {
        name: 'bandCapacity',
        type: 'number | undefined',
        description: 'Maximum number of bands reported.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        description: 'Relaxation round budget.'
      },
      {
        name: 'isochrones',
        type: 'GPUDriveTimeIsochroneOptions | undefined',
        description: 'Optional isochrone polygon settings.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: 'Record<string, unknown>',
        description: 'assignments, nodeCosts, demandTimes, demandBands and band statistics.'
      },
      {
        name: 'contributors',
        type: 'GPUCommandNodeProducer[]',
        description: 'Graph nodes the recipe added, in execution order.'
      },
      {
        name: 'intermediates',
        type: 'Record<string, unknown>',
        description: 'Named ports between stages.'
      },
      {
        name: 'status',
        type: 'GPURecipeStatusPort & {serviceAreaConverged}',
        description: 'Completeness and convergence views per stage.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Resolved coordinate context of the positions.'
      },
      {
        name: 'startNodes',
        type: "GraphDataView<'uint32'>",
        description: 'Facility start nodes, two edge endpoints each.'
      },
      {
        name: 'startCosts',
        type: "GraphDataView<'float32'>",
        description: 'Cost from each facility to its start nodes.'
      },
      {
        name: 'assignments, nodeCosts, demandTimes, demandBands, bands',
        type: 'mixed views',
        description: 'Further result fields; see the reference.'
      }
    ],
    example: `import {addDriveTimeCatchmentRecipe} from '@luma-contributors/analysis/gpu-network';

const recipe = addDriveTimeCatchmentRecipe(graph, {
  topology,                       // CSR with nodePositions
  facilities: facilityXY.importToGraph(graph),
  demand: demandXY.importToGraph(graph),
  demandValues: population.importToGraph(graph),
  costBreaks: breaks.importToGraph(graph),   // 5, 10, 15 minutes
  bandCapacity: 8,
  maximumIterations: 64,
  outputs: {demandBands, bands: {keys, counts, sumValues}}
});`,
    preview: {
      kind: 'isochrones',
      caption: 'Demand colored by drive-time band to the nearest facility.',
      seed: 4315
    },
    related: ['network-service-areas', 'network-isochrones', 'network-snapping'],
    lensScenario: 'gpu-network.drive-time-catchment-recipe',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-snapping',
    symbol: 'GPUNetworkSnapping',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network snapping',
    summary:
      'Snaps points to the nearest road edge and returns the snap position, distance and entry costs.',
    explanation:
      'Puts addresses, GPS fixes or facilities onto a street network before routing. Each point finds its nearest edge; the cost to either end lets it act as a virtual node. Distances honor the spatial context, planar or great-circle.',
    dataTypes: {input: ['points', 'network'], output: ['points', 'table']},
    useCases: ['relate', 'route'],
    domains: ['transport', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Point coordinates to snap.'
      },
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {weights?}',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'maximumSnapDistance',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Ignore points farther than this (one-row view).'
      },
      {
        name: 'candidateCapacity',
        type: 'number | undefined',
        description: 'Candidate edges kept per point in the grid search.'
      },
      {
        name: 'spatialSort',
        type: 'boolean | undefined',
        description: 'Sort points spatially for faster search.'
      },
      {
        name: 'startDirection',
        type: 'GPUNetworkSnappingStartDirection | undefined',
        description: 'Which ends of the edge a start may leave through.'
      },
      {
        name: 'output',
        type: 'GPUNetworkSnappingOutput',
        required: true,
        description: 'snappedEdges plus distances, positions and start nodes.'
      }
    ],
    outputs: [
      {
        name: 'snappedEdges',
        type: "GraphDataView<'uint32'>",
        description: 'Nearest edge slot per point.'
      },
      {
        name: 'snapFractions',
        type: "GraphDataView<'float32'>",
        description: 'Position along the edge, 0 to 1 (optional).'
      },
      {
        name: 'snapDistances',
        type: "GraphDataView<'float32'>",
        description: 'Distance from point to edge (optional).'
      },
      {
        name: 'snappedPositions',
        type: "GraphDataView<'float32x2'>",
        description: 'Snapped coordinates (optional).'
      },
      {
        name: 'startNodes',
        type: "GraphDataView<'uint32'>",
        description: 'Edge end nodes, ready for reachability (optional).'
      }
    ],
    example: `import {GPUNetworkSnapping} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkSnapping({
  positions: points.importToGraph(graph),
  topology,
  maximumSnapDistance: maximumDistance.importToGraph(graph),
  output: {snappedEdges, snapDistances, snappedPositions}
}));`,
    preview: {kind: 'network', caption: 'Points slide onto the closest street.', seed: 4316},
    related: ['network-reachability', 'map-matching', 'drive-time-catchment'],
    lensScenario: 'gpu-network.network-snapping',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'map-matching',
    symbol: 'GPUMapMatching',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Map matching',
    summary:
      'Matches noisy GPS tracks to the road network with a hidden Markov model (Newson and Krumm).',
    explanation:
      'Turns raw GPS traces into the roads that were driven. Each fix keeps candidate edges; the HMM balances GPS distance against route plausibility. Optional route tables speed up the transition step.',
    dataTypes: {input: ['trajectories', 'network'], output: ['lines', 'table']},
    useCases: ['relate', 'route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'GPS fixes, concatenated across tracks.'
      },
      {
        name: 'trackOffsets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Row offsets of each track in positions.'
      },
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {weights?}',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame noise and route-slack settings.'
      },
      {
        name: 'candidateCount',
        type: 'number | undefined',
        default: '8',
        description: 'Candidate edges kept per fix.'
      },
      {
        name: 'cellSize',
        type: 'number',
        required: true,
        description: 'Spatial grid cell size for candidate search.'
      },
      {
        name: 'bounds',
        type: '{minimum, maximum}',
        required: true,
        description: 'Extent of the network.'
      },
      {
        name: 'output',
        type: 'GPUMapMatchingOutput',
        required: true,
        description: 'matchedEdges plus fractions, snapped positions and breaks.'
      }
    ],
    outputs: [
      {name: 'matchedEdges', type: "GraphDataView<'uint32'>", description: 'Matched edge per fix.'},
      {
        name: 'matchedFractions',
        type: "GraphDataView<'float32'>",
        description: 'Position along the matched edge (optional).'
      },
      {
        name: 'snappedPositions',
        type: "GraphDataView<'float32x2'>",
        description: 'Matched coordinates (optional).'
      },
      {
        name: 'trackLogLikelihoods',
        type: "GraphDataView<'float32'>",
        description: 'Log-likelihood per track (optional).'
      },
      {
        name: 'breaks',
        type: "GraphDataView<'uint32'>",
        description: 'Fixes where the match broke (optional).'
      }
    ],
    example: `import {GPUMapMatching} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUMapMatching({
  positions: fixes.importToGraph(graph),
  trackOffsets: trackOffsets.importToGraph(graph),
  topology,
  parameters: parameters.importToGraph(graph),
  cellSize: 100,
  bounds: {minimum: [0, 0], maximum: [10000, 10000]},
  output: {matchedEdges, matchedFractions, snappedPositions}
}));`,
    preview: {
      kind: 'trajectories',
      caption: 'Scattered GPS fixes lock onto the roads they followed.',
      seed: 4317
    },
    related: ['network-snapping', 'network-cost-table', 'track-quality'],
    lensScenario: 'gpu-network.map-matching',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'track-quality',
    symbol: 'addTrackQualityRecipe',
    kind: 'recipe',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Track quality',
    summary:
      'Flags bad GPS fixes by speed and gap, and optionally adds track similarity and map-match completeness.',
    explanation:
      'Screens tracks before analysis. It computes step speeds, lengths and durations, then flags jumps and gaps against two thresholds. Add `similarity` or `mapMatching` for route distances and completeness.',
    dataTypes: {input: ['trajectories'], output: ['table']},
    useCases: ['measure', 'transform'],
    domains: ['transport', 'data-engineering'],
    params: [
      {
        name: 'trajectory',
        type: 'Omit<GPUTrajectoryMetricsProps, "id" | "output">',
        required: true,
        description: 'Positions, track offsets and timestamps.'
      },
      {
        name: 'thresholds',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Two rows: maximum speed and maximum time gap.'
      },
      {
        name: 'similarity',
        type: 'GPUTrackSimilarityProps | "output"',
        description: 'Optional track-similarity settings.'
      },
      {
        name: 'mapMatching',
        type: 'GPUMapMatchingProps | "output"',
        description: 'Optional map-matching settings for completeness.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: '{metrics, flags, flaggedCounts}',
        description: 'Per-fix flag bits, flagged counts and speed and length metrics.'
      },
      {
        name: 'contributors',
        type: 'GPUCommandNodeProducer[]',
        description: 'Graph nodes the recipe added, in execution order.'
      },
      {
        name: 'intermediates',
        type: '{positions, trackOffsets}',
        description: 'Projected positions and track offsets.'
      },
      {name: 'status', type: '{stages}', description: 'Per-stage completeness views.'}
    ],
    example: `import {addTrackQualityRecipe} from '@luma-contributors/analysis/gpu-network';

const recipe = addTrackQualityRecipe(graph, {
  trajectory: {
    positions: fixes.importToGraph(graph),
    trackOffsets: trackOffsets.importToGraph(graph),
    timestamps: times.importToGraph(graph)
  },
  thresholds: thresholds.importToGraph(graph)   // [maximum speed, maximum gap]
});
// the recipe's return value exposes its flags and metrics views`,
    preview: {kind: 'trajectories', caption: 'Tracks with jumps and gaps marked.', seed: 4318},
    related: ['map-matching'],
    lensScenario: 'gpu-network.track-quality-recipe',
    reference: 'api/gpu-network/expansion',
    maturity: 'stable'
  },
  {
    id: 'network-path-extraction',
    symbol: 'GPUNetworkPathExtraction',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network path extraction',
    summary:
      'Turns a predecessor array into ordered node and edge lists per target, ready for a path layer.',
    explanation:
      'Best after reachability to draw the actual route to each target. It walks predecessors back to the start on the GPU and emits compact, ordered lists. Path costs and found flags come with them.',
    dataTypes: {input: ['network'], output: ['lines']},
    useCases: ['route'],
    domains: ['transport'],
    params: [
      {
        name: 'predecessors',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Predecessor array from reachability.'
      },
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-node costs from reachability.'
      },
      {
        name: 'targets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Target node ids.'
      },
      {
        name: 'targetCount',
        type: 'number | GraphDataView<"uint32"> | undefined',
        description: 'Number of targets, compiled in or read per frame.'
      },
      {
        name: 'predecessorsComplete',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional flag that the predecessors are complete.'
      },
      {name: 'maximumPathLength', type: 'number | undefined', description: 'Longest path emitted.'},
      {
        name: 'nodeIds',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional node id remapping.'
      },
      {
        name: 'output',
        type: 'GPUNetworkPathExtractionOutput',
        required: true,
        description: 'Compact node list with line offsets and path costs.'
      }
    ],
    outputs: [
      {
        name: 'ids',
        type: "GraphDataView<'uint32'>",
        description: 'Ordered node ids, all paths concatenated.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Start of each path in ids (optional).'
      },
      {
        name: 'pathCosts',
        type: "GraphDataView<'float32'>",
        description: 'Total cost per path (optional).'
      },
      {
        name: 'pathFound',
        type: "GraphDataView<'uint32'>",
        description: '1 when the target was reached (optional).'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Written path vertex count and overflow flag.'
      }
    ],
    example: `import {GPUNetworkPathExtraction} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkPathExtraction({
  predecessors: reach.predecessors,
  costs: reach.costs,
  targets: targets.importToGraph(graph),
  maximumPathLength: 512,
  output: {ids, status: {count, overflow}, lineOffsets, pathCosts, pathFound}
}));`,
    preview: {
      kind: 'shortest-path',
      caption: 'The route back from a target to the start lights up.',
      seed: 4319
    },
    related: ['network-reachability', 'alternative-routes'],
    lensScenario: 'gpu-network.network-path-extraction',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'alternative-routes',
    symbol: 'GPUAlternativeRoutes',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Alternative routes',
    summary:
      'Bounded uniform-cost search for several simple edge paths per query, queries run independently on the GPU.',
    explanation:
      'Shows route options between origin and destination pairs. Each query expands paths in cost order up to your limits. Results include vertices, edges, lengths and costs per route.',
    dataTypes: {input: ['network'], output: ['lines']},
    useCases: ['route'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'CSR network: offsets, neighbors, per-edge cost.'
      },
      {
        name: 'sources',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Source node per query.'
      },
      {
        name: 'targets',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Target node per query.'
      },
      {
        name: 'routeCount',
        type: 'number',
        required: true,
        description: 'Routes returned per query.'
      },
      {
        name: 'maximumExpansions',
        type: 'number | undefined',
        default: '100000',
        description: 'Search expansions per query.'
      },
      {
        name: 'maximumFrontierSize',
        type: 'number | undefined',
        default: '1024',
        description: 'Frontier capacity per query.'
      },
      {
        name: 'output',
        type: '{vertices, edges}',
        required: true,
        description: 'vertices, edges, lengths, costs, counts, complete.'
      }
    ],
    outputs: [
      {
        name: 'vertices',
        type: "GraphDataView<'uint32'>",
        description: 'Node ids along each route.'
      },
      {name: 'edges', type: "GraphDataView<'uint32'>", description: 'Edge ids along each route.'},
      {name: 'lengths', type: "GraphDataView<'uint32'>", description: 'Hop count per route.'},
      {name: 'costs', type: "GraphDataView<'float32'>", description: 'Total cost per route.'},
      {name: 'counts', type: "GraphDataView<'uint32'>", description: 'Routes found per query.'},
      {
        name: 'complete',
        type: "GraphDataView<'uint32'>",
        description: '1 when the top-k routes are certified or exhausted.'
      }
    ],
    example: `import {GPUAlternativeRoutes} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUAlternativeRoutes({
  topology,
  sources: sources.importToGraph(graph),
  targets: targets.importToGraph(graph),
  routeCount: 3,
  maximumExpansions: 50000,
  output: {vertices, edges, lengths, costs, counts, complete}
}));`,
    preview: {
      kind: 'shortest-path',
      caption: 'Three routes between the same pair, cheapest first.',
      seed: 4320
    },
    related: ['network-path-extraction', 'network-point-routing'],
    lensScenario: 'gpu-network.alternative-routes',
    reference: 'api/gpu-network/expansion',
    maturity: 'stable'
  },
  {
    id: 'network-point-routing',
    symbol: 'GPUNetworkPointRouting',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Point routing with turns',
    summary:
      'Batched static routing between positions on directed edges, with explicit incoming turn state.',
    explanation:
      'Best for many origin-destination queries where turn penalties or restrictions matter. Queries are edge positions plus fractions along them. Pair it with the line graph for turn-aware networks.',
    dataTypes: {input: ['network', 'points'], output: ['table']},
    useCases: ['route'],
    domains: ['transport'],
    params: [
      {
        name: 'turns',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Allowed [incomingEdge, outgoingEdge] turns.'
      },
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Cost per turn transition.'
      },
      {
        name: 'queries',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: '[sourceEdge, targetEdge] per query.'
      },
      {
        name: 'fractions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Position along the source and target edges.'
      },
      {name: 'maximumRelaxations', type: 'number | undefined', description: 'Relaxation rounds.'},
      {
        name: 'output',
        type: '{costs, status}',
        required: true,
        description: 'costs, status and predecessor edges.'
      }
    ],
    outputs: [
      {name: 'costs', type: "GraphDataView<'float32'>", description: 'Route cost per query.'},
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Query status codes.'},
      {
        name: 'predecessorEdges',
        type: "GraphDataView<'uint32'>",
        description: 'Predecessor edge per state, for path recovery.'
      }
    ],
    example: `import {GPUNetworkPointRouting} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkPointRouting({
  turns: turns.importToGraph(graph),
  costs: turnCosts.importToGraph(graph),
  queries: queries.importToGraph(graph),
  fractions: fractions.importToGraph(graph),
  output: {costs, status, predecessorEdges}
}));`,
    preview: {
      kind: 'shortest-path',
      caption: 'Routes that respect turn costs between street positions.',
      seed: 4321
    },
    related: ['network-line-graph', 'network-time-routing', 'network-snapping'],
    lensScenario: 'gpu-network.network-point-routing',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'stable'
  },
  {
    id: 'network-time-routing',
    symbol: 'GPUNetworkTimeRouting',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Time-dependent routing',
    summary:
      'FIFO earliest-arrival routing on time-varying travel times over prepared restriction states.',
    explanation:
      'Best for rush-hour routing and schedules where edge cost depends on departure time. Costs are affine durations or piecewise arrival samples. Results are earliest arrival times per state.',
    dataTypes: {input: ['network', 'time-series'], output: ['table']},
    useCases: ['route'],
    domains: ['transport'],
    params: [
      {
        name: 'stateVertices',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Vertex of each routing state.'
      },
      {
        name: 'transitions',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'State transitions as pairs.'
      },
      {
        name: 'costModel',
        type: '"affine-duration" | "piecewise-arrival" | undefined',
        description: "'affine-duration' or 'piecewise-arrival'."
      },
      {
        name: 'costs',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: '[base duration, slope] per transition, or piecewise samples.'
      },
      {
        name: 'endpoints',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: '[initial state, destination vertex] per query.'
      },
      {
        name: 'departure',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Departure time per query.'
      },
      {name: 'maximumRelaxations', type: 'number | undefined', description: 'Relaxation rounds.'},
      {
        name: 'output',
        type: '{arrivals}',
        required: true,
        description: 'arrivals, predecessor transitions and summary.'
      }
    ],
    outputs: [
      {
        name: 'arrivals',
        type: "GraphDataView<'float32'>",
        description: 'Earliest arrival time per state.'
      },
      {
        name: 'predecessorTransitions',
        type: "GraphDataView<'uint32'>",
        description: 'Transition used to reach each state.'
      },
      {name: 'summary', type: "GraphDataView<'uint32'>", description: 'Convergence summary words.'}
    ],
    example: `import {GPUNetworkTimeRouting} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkTimeRouting({
  stateVertices: vertices.importToGraph(graph),
  transitions: transitions.importToGraph(graph),
  costModel: 'affine-duration',
  costs: durations.importToGraph(graph),
  endpoints: endpoints.importToGraph(graph),
  departure: departureTimes.importToGraph(graph),
  output: {arrivals, predecessorTransitions, summary}
}));`,
    preview: {
      kind: 'time-series',
      caption: 'Earliest arrival grows as departure moves into rush hour.',
      seed: 4322
    },
    related: ['network-point-routing', 'network-reachability'],
    lensScenario: 'gpu-network.network-time-routing',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'stable'
  },
  {
    id: 'network-line-graph',
    symbol: 'GPUNetworkLineGraph',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Turn-aware line graph',
    summary:
      'Builds the edge-based line graph of a road network so turn costs and bans become ordinary edge weights.',
    explanation:
      'Best before routing when turns matter: left-turn penalties, U-turn bans, signals. Each road edge becomes a node; each legal turn becomes an arc with a cost. The output is a topology you feed straight to reachability.',
    dataTypes: {input: ['network'], output: ['network']},
    useCases: ['transform', 'route'],
    domains: ['transport'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {nodePositions}',
        required: true,
        description: 'CSR road network with node positions.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame turn penalty settings.'
      },
      {
        name: 'bannedTurns',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional turns to remove.'
      },
      {
        name: 'output',
        type: 'GPUNetworkLineGraphOutput',
        required: true,
        description: 'Line-graph topology, arc count and overflow.'
      }
    ],
    outputs: [
      {
        name: 'topology.offsets',
        type: "GraphDataView<'uint32'>",
        description: 'CSR offsets of the line graph.'
      },
      {
        name: 'topology.neighbors',
        type: "GraphDataView<'uint32'>",
        description: 'Next edge per arc.'
      },
      {
        name: 'topology.weights',
        type: "GraphDataView<'float32'>",
        description: 'Turn-aware cost per arc.'
      },
      {
        name: 'status.overflow',
        type: "GraphDataView<'uint32'>",
        description: '1 when arc capacity was exceeded.'
      },
      {
        name: 'arcCount',
        type: "GraphDataView<'uint32'> | undefined",
        description: 'Optional one-row arc count.'
      }
    ],
    example: `import {GPUNetworkLineGraph} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkLineGraph({
  topology,
  parameters: turnParameters.importToGraph(graph),
  bannedTurns: bans.importToGraph(graph),
  output: {topology: {offsets, neighbors, weights}, status: {overflow}}
}));
// feed the output topology to GPUNetworkReachability`,
    preview: {
      kind: 'network',
      caption: 'Roads become nodes; turns become the edges between them.',
      seed: 4323
    },
    related: ['network-point-routing', 'network-reachability'],
    lensScenario: 'gpu-network.network-line-graph',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-noding',
    symbol: 'GPUNetworkNoding',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network noding',
    summary:
      'Builds a routable planar network from raw linestrings by splitting them at intersections.',
    explanation:
      'Prepares messy road centerlines for routing. It finds crossings with a spatial sort, snaps within a tolerance and emits nodes, edges and CSR topology. Uncertain intersections are counted for review.',
    dataTypes: {input: ['lines'], output: ['network']},
    useCases: ['transform', 'relate'],
    domains: ['transport', 'cartography', 'data-engineering'],
    params: [
      {
        name: 'lines',
        type: 'GPULineGeometryPort | "sourceIds">',
        required: true,
        description: 'Line geometry: vertex positions and line offsets.'
      },
      {
        name: 'intersectionCapacity',
        type: 'number',
        required: true,
        description: 'Maximum intersections found.'
      },
      {
        name: 'tolerance',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Snap tolerance in length units (one-row view).'
      },
      {
        name: 'spatialSort',
        type: 'boolean | undefined',
        description: 'Sort segments spatially for faster search.'
      },
      {name: 'leafCapacity', type: 'number | undefined', description: 'Segments per spatial leaf.'},
      {
        name: 'output',
        type: 'GPUNetworkNodingOutput',
        required: true,
        description: 'pieces, nodes, edges, topology and status.'
      }
    ],
    outputs: [
      {name: 'pieces', type: 'GPULineSplitPieces', description: 'Line pieces after splitting.'},
      {
        name: 'nodes',
        type: 'GPUNetworkNodingNodes',
        description: 'Unique intersection and end nodes.'
      },
      {name: 'edges', type: 'GPUNetworkNodingEdges', description: 'Edges between nodes.'},
      {
        name: 'topology',
        type: 'GPUNetworkNodingTopology',
        description: 'CSR topology ready for routing.'
      },
      {
        name: 'status.overflow',
        type: "GraphDataView<'uint32'>",
        description: '1 when capacity was exceeded.'
      }
    ],
    example: `import {GPUNetworkNoding} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkNoding({
  lines: {kind: 'lines', positions, lineOffsets},
  intersectionCapacity: 100000,
  tolerance: tolerance.importToGraph(graph),
  spatialSort: true,
  output: nodingOutput
}));
// nodingOutput.topology feeds GPUNetworkReachability`,
    preview: {kind: 'lines', caption: 'Crossing lines split at every intersection.', seed: 4324},
    related: ['network-snapping', 'network-structure'],
    lensScenario: 'gpu-network.network-noding',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-centrality',
    symbol: 'GPUNetworkCentrality',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network centrality',
    summary:
      'Radius-bounded betweenness and closeness of a weighted road network, exact or estimated from pivots.',
    explanation:
      'Finds the streets and junctions that carry the most shortest paths. Searches from pivot nodes feed Brandes accumulation; sample pivots to scale to large networks. It needs the network and its transpose.',
    dataTypes: {input: ['network'], output: ['network']},
    useCases: ['find-patterns', 'measure'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'Forward CSR network.'
      },
      {
        name: 'reverseTopology',
        type: 'GPUNetworkTopology',
        required: true,
        description: 'Same edges reversed, with identical weights.'
      },
      {
        name: 'pivotNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Nodes to measure from; all nodes is exact.'
      },
      {
        name: 'costLimit',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Radius: paths longer than this are ignored.'
      },
      {
        name: 'betweennessScale',
        type: 'number | undefined',
        description: 'Scale; use nodeCount / pivotCount when sampling.'
      },
      {name: 'laneCount', type: 'number | undefined', description: 'Pivots processed per batch.'},
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        default: '64',
        description: 'Round budget per phase.'
      },
      {
        name: 'output',
        type: 'GPUNetworkCentralityOutput',
        required: true,
        description: 'betweenness, closeness, harmonicCloseness, reachedCounts.'
      }
    ],
    outputs: [
      {
        name: 'betweenness',
        type: "GraphDataView<'float32'>",
        description: 'Shortest paths through each node (optional).'
      },
      {
        name: 'edgeBetweenness',
        type: "GraphDataView<'float32'>",
        description: 'Shortest paths through each edge (optional).'
      },
      {
        name: 'closeness',
        type: "GraphDataView<'float32'>",
        description: 'Closeness per pivot (optional).'
      },
      {
        name: 'harmonicCloseness',
        type: "GraphDataView<'float32'>",
        description: 'Harmonic closeness per pivot (optional).'
      },
      {
        name: 'status.converged',
        type: "GraphDataView<'uint32'>",
        description: '1 when every phase finished.'
      }
    ],
    example: `import {GPUNetworkCentrality} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkCentrality({
  topology,
  reverseTopology,
  pivotNodes: pivots.importToGraph(graph),
  costLimit: radius.importToGraph(graph),
  betweennessScale: nodeCount / pivotCount,
  output: {betweenness, closeness, status: {converged}}
}));`,
    preview: {
      kind: 'network',
      caption: 'The busiest streets stand out as thick, bright edges.',
      seed: 4325
    },
    related: ['network-structure', 'network-statistics', 'network-reachability'],
    lensScenario: 'gpu-network.network-centrality',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-structure',
    symbol: 'GPUNetworkStructure',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network structure',
    summary:
      'Exact strong components, bridges, articulation points and biconnected components on GPU-resident CSR.',
    explanation:
      'Finds the weak spots of a network: roads whose loss disconnects it, and islands that cannot reach the rest. Results are exact within the bounds of the algorithm. Choose which outputs to compute by supplying them.',
    dataTypes: {input: ['network'], output: ['network', 'table']},
    useCases: ['find-patterns', 'measure'],
    domains: ['transport', 'ecology'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network, connectivity only.'
      },
      {
        name: 'reverseEdgeSlots',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Reverse slot of every CSR edge.'
      },
      {
        name: 'output',
        type: '{strongComponents?}',
        required: true,
        description: 'strongComponents, bridges, articulationPoints, biconnectedComponents.'
      }
    ],
    outputs: [
      {
        name: 'strongComponents',
        type: "GraphDataView<'uint32'>",
        description: 'Strongly connected component label per node.'
      },
      {
        name: 'bridges',
        type: "GraphDataView<'uint32'>",
        description: 'Edges whose removal disconnects the graph.'
      },
      {
        name: 'articulationPoints',
        type: "GraphDataView<'uint32'>",
        description: 'Nodes whose removal disconnects the graph.'
      },
      {
        name: 'biconnectedComponents',
        type: "GraphDataView<'uint32'>",
        description: 'Biconnected component label per edge.'
      }
    ],
    example: `import {GPUNetworkStructure} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkStructure({
  topology,
  reverseEdgeSlots: reverseSlots.importToGraph(graph),
  output: {bridges, articulationPoints, strongComponents}
}));`,
    preview: {
      kind: 'network',
      caption: 'Bridges and cut points glow where the network is fragile.',
      seed: 4326
    },
    related: ['network-centrality', 'network-statistics', 'network-disruption'],
    lensScenario: 'gpu-network.network-structure',
    reference: 'api/gpu-network/expansion',
    maturity: 'stable'
  },
  {
    id: 'network-statistics',
    symbol: 'GPUNetworkStatistics',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network statistics',
    summary:
      'Reduces a network to a compact summary: counts, components, degree histogram and community sizes.',
    explanation:
      'Best for a stats panel that updates as filters change. One GPU pass counts live nodes and edges, finds weak components and bins degrees. All results are packed in a single uint32 view.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['transport', 'data-engineering'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network, connectivity only.'
      },
      {name: 'directed', type: 'boolean | undefined', description: 'Treat edge slots as directed.'},
      {
        name: 'nodeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are live nodes.'
      },
      {
        name: 'edgeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are live edge slots.'
      },
      {
        name: 'communities',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional community labels to summarize.'
      },
      {
        name: 'degreeBinCount',
        type: 'number | undefined',
        default: '32',
        description: 'Number of degree histogram bins.'
      },
      {
        name: 'degreeBinning',
        type: '"linear" | "log2" | undefined',
        description: "'linear' or 'log2' degree bins."
      },
      {
        name: 'output',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Packed summary words.'
      }
    ],
    outputs: [
      {
        name: 'output',
        type: "GraphDataView<'uint32'>",
        description: 'Counts, component stats and degree histogram, packed.'
      }
    ],
    example: `import {GPUNetworkStatistics} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkStatistics({
  topology,
  nodeMask: liveNodes.importToGraph(graph),
  degreeBinCount: 16,
  degreeBinning: 'log2',
  output: summary
}));
// decode \`summary\` with the statistics layout helpers on the CPU`,
    preview: {kind: 'histogram', caption: 'Degree distribution of the live network.', seed: 4327},
    related: ['network-analytics-columns', 'network-structure', 'network-subgraph-filter'],
    lensScenario: 'gpu-network.network-statistics',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-analytics-columns',
    symbol: 'GPUNetworkAnalyticsColumns',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network analytics columns',
    summary:
      'Publishes node-aligned degree, in-degree and component columns, with normalized variants for styling.',
    explanation:
      'Colors or size nodes by structure in a deck.gl layer. Each column is a raw value, with an optional zero-to-one normalized copy from a GPU min/max reduction. Components use label propagation.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['measure', 'classify'],
    domains: ['transport', 'data-engineering'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network, connectivity only.'
      },
      {
        name: 'reverseTopology',
        type: 'GPUNetworkAdjacency | "offsets">',
        description: 'Reverse CSR, needed for in-degree.'
      },
      {
        name: 'degree',
        type: 'GPUNetworkAnalyticsColumn<"uint32"> | undefined',
        description: 'Outgoing degree column and optional normalized copy.'
      },
      {
        name: 'inDegree',
        type: 'GPUNetworkAnalyticsColumn<"uint32"> | undefined',
        description: 'Incoming degree column and optional normalized copy.'
      },
      {
        name: 'components',
        type: '{output, maximumIterations?}',
        description: 'Weak component labels with iteration cap.'
      }
    ],
    outputs: [
      {
        name: 'degree.output',
        type: "GraphDataView<'uint32'>",
        description: 'Outgoing degree per node.'
      },
      {
        name: 'degree.normalized',
        type: "GraphDataView<'float32'>",
        description: 'Degree scaled to 0 to 1 (optional).'
      },
      {
        name: 'inDegree.output',
        type: "GraphDataView<'uint32'>",
        description: 'Incoming degree per node.'
      },
      {
        name: 'components.output',
        type: "GraphDataView<'uint32'>",
        description: 'Component label per node.'
      }
    ],
    example: `import {GPUNetworkAnalyticsColumns} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkAnalyticsColumns({
  topology,
  reverseTopology: {offsets: reverseOffsets, neighbors: reverseNeighbors},
  degree: {output: degree, normalized: degreeNormalized},
  inDegree: {output: inDegree},
  components: {output: components, maximumIterations: 32}
}));`,
    preview: {kind: 'network', caption: 'Nodes sized and colored by degree.', seed: 4328},
    related: ['network-statistics', 'network-centrality'],
    lensScenario: 'gpu-network.network-analytics-columns',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-neighborhood',
    symbol: 'GPUNetworkNeighborhood',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network neighborhood',
    summary:
      'The k-hop ego network around start nodes: hop distances, node and edge masks, compact id lists.',
    explanation:
      'Best for click-to-explore in a graph view: show what is within a few hops of a node. Hop counts are computed breadth-first from per-frame starts. Masks drive highlighting; compact lists feed draw calls.',
    dataTypes: {input: ['network'], output: ['network']},
    useCases: ['relate', 'find-patterns'],
    domains: ['data-engineering', 'transport'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network, connectivity only.'
      },
      {
        name: 'startNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Ego nodes, rewritten every frame.'
      },
      {
        name: 'startCount',
        type: 'number | GraphDataView<"uint32"> | undefined',
        description: 'Number of starts, compiled in or per frame.'
      },
      {
        name: 'hops',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Hop limit as a one-row view.'
      },
      {name: 'maximumHops', type: 'number | undefined', description: 'Compile-time cap on hops.'},
      {
        name: 'output',
        type: 'GPUNetworkNeighborhoodOutput',
        required: true,
        description: 'hopDistances, nodeMask, edgeMask, compact node and edge lists.'
      }
    ],
    outputs: [
      {
        name: 'hopDistances',
        type: "GraphDataView<'uint32'>",
        description: 'Hops from the nearest start per node.'
      },
      {
        name: 'nodeMask',
        type: "GraphDataView<'uint32'>",
        description: '1 for nodes inside the neighborhood (optional).'
      },
      {
        name: 'edgeMask',
        type: "GraphDataView<'uint32'>",
        description: '1 for induced edge slots (optional).'
      },
      {
        name: 'nodes',
        type: 'GPUCompactOutput',
        description: 'Compact list of neighborhood nodes (optional).'
      }
    ],
    example: `import {GPUNetworkNeighborhood} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkNeighborhood({
  topology,
  startNodes: picked.importToGraph(graph),
  hops: hops.importToGraph(graph),
  maximumHops: 6,
  output: {hopDistances, nodeMask, edgeMask}
}));`,
    preview: {
      kind: 'network',
      caption: 'Click a node; rings of neighbors light up hop by hop.',
      seed: 4329
    },
    related: ['network-subgraph-filter', 'network-reachability'],
    lensScenario: 'gpu-network.network-neighborhood',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-subgraph-filter',
    symbol: 'GPUNetworkSubgraphFilter',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network subgraph filter',
    summary:
      'Turns attribute and time predicates on nodes and edges into a consistent induced subgraph.',
    explanation:
      'Best for brushing and time sliders on a network: keep nodes and edges that pass, drop edges whose ends fail. Masks and compact lists come out on the GPU. Optionally drop isolated nodes and emit the induced topology.',
    dataTypes: {input: ['network', 'table'], output: ['network']},
    useCases: ['transform', 'relate'],
    domains: ['data-engineering', 'transport'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network, connectivity only.'
      },
      {
        name: 'nodeColumns',
        type: 'readonly GraphDataView<"float32">[] | undefined',
        description: 'Float columns with node attributes.'
      },
      {
        name: 'edgeColumns',
        type: 'readonly GraphDataView<"float32">[] | undefined',
        description: 'Float columns with edge attributes.'
      },
      {
        name: 'edgeTimes',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Per-edge time stamps.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Per-frame ranges for the predicates.'
      },
      {
        name: 'dropIsolated',
        type: 'boolean | undefined',
        description: 'Drop nodes left without edges.'
      },
      {name: 'directed', type: 'boolean | undefined', description: 'Treat edge slots as directed.'},
      {
        name: 'output',
        type: 'GPUNetworkSubgraphFilterOutput',
        required: true,
        description: 'nodeMask, edgeMask, counts, compact lists, induced topology.'
      }
    ],
    outputs: [
      {name: 'nodeMask', type: "GraphDataView<'uint32'>", description: '1 for surviving nodes.'},
      {
        name: 'edgeMask',
        type: "GraphDataView<'uint32'>",
        description: '1 for surviving edge slots.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Live node and edge counts (optional).'
      },
      {
        name: 'inducedTopology',
        type: 'GPUNetworkSubgraphFilterInducedTopology',
        description: 'Compacted CSR of the subgraph (optional).'
      }
    ],
    example: `import {GPUNetworkSubgraphFilter} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkSubgraphFilter({
  topology,
  edgeColumns: [speed, volume],
  edgeTimes: times.importToGraph(graph),
  parameters: ranges.importToGraph(graph),
  dropIsolated: true,
  output: {nodeMask, edgeMask, counts}
}));`,
    preview: {kind: 'network', caption: 'Drag a range: edges outside it fade out.', seed: 4330},
    related: ['network-neighborhood', 'network-statistics'],
    lensScenario: 'gpu-network.network-subgraph-filter',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-coarsening',
    symbol: 'GPUNetworkCoarsening',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network coarsening',
    summary:
      'Collapses a network by group label into a summary graph of compound nodes, centroids and weighted edges.',
    explanation:
      'Best for zoomed-out views of huge networks: one node per community or tile. Group nodes counts, centroids, bounds and value sums are computed on the GPU. Edges between groups are merged with counts and weights.',
    dataTypes: {input: ['network'], output: ['network']},
    useCases: ['summarize', 'transform'],
    domains: ['data-engineering', 'transport'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'labels',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Group label per node.'
      },
      {
        name: 'groupCapacity',
        type: 'number',
        required: true,
        description: 'Maximum number of groups.'
      },
      {
        name: 'nodeValues',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional node value to sum per group.'
      },
      {
        name: 'nodeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are live nodes.'
      },
      {name: 'directed', type: 'boolean | undefined', description: 'Treat edge slots as directed.'},
      {
        name: 'fixedPointScale',
        type: 'number | undefined',
        default: '65536',
        description: 'Fixed-point scale for weight sums.'
      },
      {
        name: 'output',
        type: 'GPUNetworkCoarseningOutput',
        required: true,
        description: 'Per-group stats and merged group edges.'
      }
    ],
    outputs: [
      {name: 'groupNodeCount', type: "GraphDataView<'uint32'>", description: 'Nodes per group.'},
      {
        name: 'groupCentroid',
        type: "GraphDataView<'float32x2'>",
        description: 'Group centroid (optional).'
      },
      {
        name: 'groupBounds',
        type: "GraphDataView<'float32x4'>",
        description: 'Group bounding box (optional).'
      },
      {
        name: 'edgeTargetGroups',
        type: "GraphDataView<'uint32'>",
        description: 'Target group of each merged edge.'
      },
      {
        name: 'edgeCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Original edges merged into each group edge.'
      },
      {
        name: 'edges',
        type: 'GPUCompactOutput',
        description: 'Superedge list; `ids` is the source group.'
      }
    ],
    example: `import {GPUNetworkCoarsening} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkCoarsening({
  topology,
  labels: communities.importToGraph(graph),
  groupCapacity: 256,
  output: {groupNodeCount, groupCentroid, edges, edgeTargetGroups, edgeCounts}
}));`,
    preview: {
      kind: 'network',
      caption: 'Hundreds of nodes fold into a handful of compound nodes.',
      seed: 4331
    },
    related: ['network-louvain', 'network-statistics'],
    lensScenario: 'gpu-network.network-coarsening',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-louvain',
    symbol: 'GPUNetworkLouvain',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Louvain communities',
    summary:
      'Reproducible weighted Louvain community detection for small graphs of up to 32 vertices.',
    explanation:
      'Best for communities in small dense graphs, such as flows between a few zones. Modularity moves repeat over levels with a seeded order. The objective trace shows convergence.',
    dataTypes: {input: ['network'], output: ['table']},
    useCases: ['find-patterns'],
    domains: ['urban', 'spatial-statistics'],
    params: [
      {
        name: 'nodeCount',
        type: 'number',
        required: true,
        description: 'Number of vertices (at most 32).'
      },
      {
        name: 'weights',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Symmetric nonnegative row-major adjacency matrix.'
      },
      {name: 'directed', type: 'boolean | undefined', description: 'Treat the matrix as directed.'},
      {
        name: 'seed',
        type: 'number | undefined',
        description: 'Seed for the vertex visiting order.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Resolution gamma and minimum modularity improvement.'
      },
      {
        name: 'maximumLevels',
        type: 'number | undefined',
        default: '8',
        description: 'Aggregation levels.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        default: '16',
        description: 'Local move rounds per level.'
      },
      {
        name: 'output',
        type: '{communities, objectiveTrace}',
        required: true,
        description: 'communities, objectiveTrace, summary.'
      }
    ],
    outputs: [
      {
        name: 'communities',
        type: "GraphDataView<'uint32'>",
        description: 'Community label per vertex.'
      },
      {
        name: 'objectiveTrace',
        type: "GraphDataView<'float32'>",
        description: 'Modularity after each round.'
      },
      {name: 'summary', type: "GraphDataView<'uint32'>", description: 'Level and round counts.'}
    ],
    example: `import {GPUNetworkLouvain} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkLouvain({
  nodeCount: 24,
  weights: flowMatrix.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  seed: 4332,
  output: {communities, objectiveTrace, summary}
}));`,
    preview: {
      kind: 'point-clusters',
      caption: 'Zones with heavy exchange join one community.',
      seed: 4333
    },
    related: ['network-coarsening', 'flow-aggregation'],
    lensScenario: 'gpu-network.network-louvain',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'partial'
  },
  {
    id: 'adjacency-matrix',
    symbol: 'GPUAdjacencyMatrix',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Adjacency matrix',
    summary:
      'Bins a graph into an R x R adjacency-matrix image on the GPU, for a matrix view linked to a node-link view.',
    explanation:
      'Shows dense networks as a matrix next to the node-link picture. Edge counts or weight sums are binned at the resolution you choose, honoring masks and a node order. A texture output is ready to sample.',
    dataTypes: {input: ['network'], output: ['raster']},
    useCases: ['summarize', 'classify'],
    domains: ['data-engineering', 'urban'],
    params: [
      {
        name: 'topology',
        type: 'GPUNetworkAdjacency',
        required: true,
        description: 'Forward CSR network.'
      },
      {name: 'resolution', type: 'number', required: true, description: 'Matrix side R in cells.'},
      {
        name: 'order',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Node-to-row permutation (see Adjacency matrix order).'
      },
      {
        name: 'directed',
        type: 'boolean | undefined',
        description: 'Row is source, column is target.'
      },
      {
        name: 'mirrorSlots',
        type: 'boolean | undefined',
        description: 'Undirected only: mirror counts across the diagonal.'
      },
      {
        name: 'nodeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are live nodes.'
      },
      {
        name: 'edgeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are live edge slots.'
      },
      {
        name: 'output',
        type: '{counts, weightSums?}',
        required: true,
        description: 'counts, weightSums, maxima and an optional texture.'
      }
    ],
    outputs: [
      {name: 'counts', type: "GraphDataView<'uint32'>", description: 'Edges per matrix cell.'},
      {
        name: 'weightSums',
        type: "GraphDataView<'uint32'>",
        description: 'Fixed-point weight sum per cell (optional).'
      },
      {
        name: 'maximumCount',
        type: "GraphDataView<'uint32'>",
        description: 'Largest count, for color scaling (optional).'
      },
      {
        name: 'texture',
        type: "GraphTextureView<'r32float'>",
        description: 'Matrix image ready to sample (optional).'
      }
    ],
    example: `import {GPUAdjacencyMatrix} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUAdjacencyMatrix({
  topology,
  resolution: 256,
  order: order.importToGraph(graph),
  directed: true,
  output: {counts, maximumCount, texture}
}));`,
    preview: {
      kind: 'cells',
      caption: 'Edges collapse into a matrix whose blocks reveal communities.',
      seed: 4334
    },
    related: ['adjacency-matrix-order', 'network-louvain'],
    lensScenario: 'gpu-network.adjacency-matrix',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'adjacency-matrix-order',
    symbol: 'GPUAdjacencyMatrixOrder',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Adjacency matrix order',
    summary:
      'Computes a node ordering from group labels so communities form blocks in the adjacency matrix.',
    explanation:
      'Best before drawing an adjacency matrix. Nodes are sorted by group, with optional tie keys. The permutation feeds the matrix pass.',
    dataTypes: {input: ['network', 'table'], output: ['table']},
    useCases: ['transform', 'classify'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'groups',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Group label per node, such as a community.'
      },
      {
        name: 'tieKeys',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional secondary sort key per node.'
      },
      {
        name: 'output',
        type: 'GPUAdjacencyMatrixOrderOutput',
        required: true,
        description: 'order: node to matrix position.'
      }
    ],
    outputs: [
      {name: 'order', type: "GraphDataView<'uint32'>", description: 'Matrix position per node.'}
    ],
    example: `import {GPUAdjacencyMatrixOrder} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUAdjacencyMatrixOrder({
  groups: communities.importToGraph(graph),
  tieKeys: degree.importToGraph(graph),
  output: {order}
}));
// pass \`order\` to GPUAdjacencyMatrix`,
    preview: {
      kind: 'cells',
      caption: 'Sorted by community, the matrix shows clear blocks.',
      seed: 4335
    },
    related: ['adjacency-matrix', 'network-louvain'],
    lensScenario: 'gpu-network.adjacency-matrix-order',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'edge-bundling',
    symbol: 'GPUEdgeBundling',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Edge bundling',
    summary:
      'Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines.',
    explanation:
      'Untangles hairball flow maps: similar edges are pulled together along density gradients. Output polylines go straight to a path layer. Increase iterations for tighter bundles.',
    dataTypes: {input: ['network'], output: ['lines']},
    useCases: ['transform', 'classify'],
    domains: ['cartography', 'urban'],
    params: [
      {
        name: 'nodePositions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Node coordinates.'
      },
      {
        name: 'sourceNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Source node per edge.'
      },
      {
        name: 'targetNodes',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Target node per edge.'
      },
      {
        name: 'edgeMask',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Nonzero entries are edges to bundle.'
      },
      {
        name: 'pointsPerEdge',
        type: 'number | undefined',
        default: '16',
        description: 'Polyline points per edge.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        default: '15',
        description: 'Bundling iterations.'
      },
      {
        name: 'densityResolution',
        type: 'number | undefined',
        default: '256',
        description: 'Density grid size.'
      },
      {
        name: 'output',
        type: 'GPUEdgeBundlingOutput',
        required: true,
        description: 'paths, lineOffsets, drawRecord.'
      }
    ],
    outputs: [
      {
        name: 'paths',
        type: "GraphDataView<'float32x2'>",
        description: 'Polyline vertices of all edges.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Start of each polyline (optional).'
      },
      {
        name: 'drawRecord',
        type: "GraphDataView<'uint32'>",
        description: 'Indirect draw record (optional).'
      }
    ],
    example: `import {GPUEdgeBundling} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUEdgeBundling({
  nodePositions: positions.importToGraph(graph),
  sourceNodes: sources.importToGraph(graph),
  targetNodes: targets.importToGraph(graph),
  pointsPerEdge: 16,
  maximumIterations: 15,
  output: {paths, lineOffsets}
}));`,
    preview: {kind: 'flows', caption: 'Straight edges relax into smooth bundles.', seed: 4336},
    related: ['flow-aggregation', 'adjacency-matrix'],
    lensScenario: 'gpu-network.edge-bundling',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'flow-aggregation',
    symbol: 'GPUFlowAggregation',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Flow aggregation',
    summary:
      'Aggregates origin-destination movements into a weight-sorted top-K flow list with per-zone totals.',
    explanation:
      'Draws the biggest flows between zones from millions of trips. Origins and destinations map to zones, optionally filtered by a time window. The strongest pairs and zone in and out totals stay on the GPU.',
    dataTypes: {input: ['points', 'table'], output: ['network', 'table']},
    useCases: ['summarize', 'relate'],
    domains: ['urban', 'transport'],
    params: [
      {
        name: 'zones',
        type: 'GPUFlowAggregationZones',
        required: true,
        description: 'Zone lookup used to assign points.'
      },
      {
        name: 'origins',
        type: 'GPUFloat32Positions | undefined',
        description: 'Origin coordinates.'
      },
      {
        name: 'destinations',
        type: 'GPUFloat32Positions | undefined',
        description: 'Destination coordinates.'
      },
      {
        name: 'weights',
        type: 'GraphDataView<"float32"> | GraphVectorView<"float32">',
        description: 'Weight per movement.'
      },
      {
        name: 'timeWindow',
        type: 'GPUFlowAggregationTimeWindow | undefined',
        description: 'Keep movements inside a time range.'
      },
      {
        name: 'excludeSelfFlows',
        type: 'boolean | undefined',
        description: 'Drop flows whose ends share a zone.'
      },
      {
        name: 'pairCapacity',
        type: 'number',
        required: true,
        description: 'Maximum distinct zone pairs kept.'
      },
      {
        name: 'output',
        type: 'GPUFlowAggregationOutput',
        required: true,
        description: 'Top-K flow list and per-zone totals.'
      }
    ],
    outputs: [
      {
        name: 'flowOriginZoneIds',
        type: "GraphDataView<'uint32'>",
        description: 'Origin zone of each flow (optional).'
      },
      {
        name: 'flowDestinationZoneIds',
        type: "GraphDataView<'uint32'>",
        description: 'Destination zone of each flow (optional).'
      },
      {
        name: 'flowWeights',
        type: "GraphDataView<'float32'>",
        description: 'Summed weight per flow (optional).'
      },
      {
        name: 'zoneOutWeights',
        type: "GraphDataView<'float32'>",
        description: 'Total outflow per zone (optional).'
      },
      {
        name: 'zoneInWeights',
        type: "GraphDataView<'float32'>",
        description: 'Total inflow per zone (optional).'
      },
      {
        name: 'ids',
        type: "GraphDataView<'uint32'>",
        description: 'Compact pair ids, one row per reported flow.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {pairOverflow?}',
        description: 'Flow count and overflow flags.'
      }
    ],
    example: `import {GPUFlowAggregation} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUFlowAggregation({
  zones,
  origins: origins.importToGraph(graph),
  destinations: destinations.importToGraph(graph),
  weights: passengers.importToGraph(graph),
  excludeSelfFlows: true,
  pairCapacity: 4096,
  output: {ids, status: {count, overflow}, flowWeights, zoneOutWeights, zoneInWeights}
}));`,
    preview: {
      kind: 'flows',
      caption: 'Millions of trips reduce to the strongest zone-to-zone flows.',
      seed: 4337
    },
    related: ['edge-bundling', 'network-flow'],
    lensScenario: 'gpu-network.flow-aggregation',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-flow',
    symbol: 'GPUNetworkFlow',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network flow',
    summary:
      'Min-cost max-flow by successive shortest augmenting paths with reverse arcs, plus the minimum cut.',
    explanation:
      'Best for capacity questions: how much can move from A to B, and where is the bottleneck. Augmentation repeats until no path remains. The cut marks the saturated frontier.',
    dataTypes: {input: ['network'], output: ['network', 'table']},
    useCases: ['simulate', 'route'],
    domains: ['transport'],
    params: [
      {name: 'nodeCount', type: 'number', required: true, description: 'Number of nodes.'},
      {
        name: 'edges',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Arcs as source and target node pairs.'
      },
      {
        name: 'capacities',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Capacity per arc.'
      },
      {
        name: 'costs',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Optional cost per arc for min-cost flow.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Source, sink and required flow (max flow when 0xffffffff).'
      },
      {
        name: 'maximumAugmentations',
        type: 'number | undefined',
        description: 'Augmenting path budget.'
      },
      {
        name: 'output',
        type: '{flow, cut}',
        required: true,
        description: 'flow, cut, summary and exact cost.'
      }
    ],
    outputs: [
      {name: 'flow', type: "GraphDataView<'uint32'>", description: 'Flow on each arc.'},
      {
        name: 'cut',
        type: "GraphDataView<'uint32'>",
        description: 'Source-side flag of each node in the minimum cut.'
      },
      {
        name: 'summary',
        type: "GraphDataView<'uint32'>",
        description: 'Total flow and augmentation count.'
      },
      {
        name: 'exactCost',
        type: "GraphDataView<'uint32x2'>",
        description: 'Exact total cost as two words (optional).'
      }
    ],
    example: `import {GPUNetworkFlow} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkFlow({
  nodeCount: 64,
  edges: arcs.importToGraph(graph),
  capacities: capacities.importToGraph(graph),
  costs: arcCosts.importToGraph(graph),
  parameters: sourceAndSink.importToGraph(graph),
  output: {flow, cut, summary}
}));`,
    preview: {
      kind: 'flows',
      caption: 'Flow saturates the bottleneck arcs; the cut splits the network.',
      seed: 4338
    },
    related: ['flow-aggregation', 'network-rational-access'],
    lensScenario: 'gpu-network.network-flow',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'partial'
  },
  {
    id: 'network-rational-access',
    symbol: 'GPUNetworkRationalAccess',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Rational access',
    summary:
      'Rational-agent access model: demand is allocated across facilities with congestion equalization.',
    explanation:
      'Shows where people would actually go when facilities fill up. Travel and supply are normalized, then demand shifts between facilities until congestion equalizes. Run integral or fractional demand.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['route', 'model'],
    domains: ['demographics', 'urban'],
    params: [
      {
        name: 'costs',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand x facility cost matrix.'
      },
      {
        name: 'demand',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Demand per point.'
      },
      {
        name: 'supply',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Capacity per facility.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Per-frame decay and step settings.'
      },
      {
        name: 'maximumCycles',
        type: 'number | undefined',
        default: '150',
        description: 'Equalization cycles.'
      },
      {
        name: 'demandMode',
        type: '"integral" | "fractional" | undefined',
        description: "'integral' or 'fractional' demand."
      },
      {
        name: 'stepMode',
        type: '"fraction" | "absolute" | undefined',
        description: "'fraction' or 'absolute' step."
      },
      {
        name: 'output',
        type: '{assignments, scores}',
        required: true,
        description: 'assignments, scores, unservedDemand, status.'
      }
    ],
    outputs: [
      {
        name: 'assignments',
        type: "GraphDataView<'float32'>",
        description: 'Demand allocated to each facility.'
      },
      {
        name: 'scores',
        type: "GraphDataView<'float32'>",
        description: 'Access score per demand point.'
      },
      {
        name: 'unservedDemand',
        type: "GraphDataView<'float32'>",
        description: 'Demand left unserved.'
      },
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Cycle count and convergence.'}
    ],
    example: `import {GPUNetworkRationalAccess} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkRationalAccess({
  costs: matrix.costs,
  demand: demand.importToGraph(graph),
  supply: supply.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  demandMode: 'fractional',
  output: {assignments, scores, unservedDemand, status}
}));`,
    preview: {
      kind: 'flows',
      caption: 'Demand spills to the next facility as the nearest one fills.',
      seed: 4339
    },
    related: ['network-accessibility', 'catchment-accessibility', 'huff-trade-areas'],
    lensScenario: 'gpu-network.network-rational-access',
    reference: 'api/gpu-network/network-movement-coverage',
    maturity: 'partial'
  },
  {
    id: 'network-k-function',
    symbol: 'GPUNetworkKFunction',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network K function',
    summary:
      'Network-constrained Ripley K of events on a road network, with a Monte Carlo envelope.',
    explanation:
      'Tests whether crashes or crimes cluster along roads, not just in the plane. Events snap to the network; pairs are counted by shortest-path distance per band. Simulated events give the envelope.',
    dataTypes: {input: ['points', 'network'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'transport'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Event coordinates.'
      },
      {
        name: 'topology',
        type: 'GPUNetworkTopology & {nodePositions}',
        required: true,
        description: 'CSR network with node positions.'
      },
      {
        name: 'maximumDistance',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Largest distance band (one-row view).'
      },
      {
        name: 'networkLength',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Total network length (one-row view).'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Per-frame seed and simulation settings.'
      },
      {
        name: 'bandCount',
        type: 'number | undefined',
        default: '32',
        description: 'Number of distance bands.'
      },
      {
        name: 'simulationCount',
        type: 'number | undefined',
        default: '0',
        description: 'Monte Carlo simulations for the envelope.'
      },
      {
        name: 'output',
        type: 'GPUNetworkKFunctionOutput',
        required: true,
        description: 'kValues, pairCounts, envelope, status.'
      }
    ],
    outputs: [
      {name: 'kValues', type: "GraphDataView<'float32'>", description: 'K per distance band.'},
      {
        name: 'pairCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Event pairs per band (optional).'
      },
      {
        name: 'envelope',
        type: "GraphDataView<'float32'>",
        description: 'Simulation envelope per band (optional).'
      },
      {
        name: 'status.converged',
        type: "GraphDataView<'uint32'>",
        description: '1 when every search finished.'
      }
    ],
    example: `import {GPUNetworkKFunction} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkKFunction({
  positions: events.importToGraph(graph),
  topology,
  maximumDistance: maximumDistance.importToGraph(graph),
  networkLength: networkLength.importToGraph(graph),
  parameters: parameters.importToGraph(graph),
  bandCount: 32,
  simulationCount: 99,
  output: {kValues, envelope}
}));`,
    preview: {
      kind: 'histogram',
      caption: 'K rises above the envelope where events cluster on the network.',
      seed: 4340
    },
    related: ['network-inhomogeneous-k', 'network-kernel-density'],
    lensScenario: 'gpu-network.network-k-function',
    reference: 'api/gpu-network/operators',
    maturity: 'stable'
  },
  {
    id: 'network-inhomogeneous-k',
    symbol: 'GPUNetworkInhomogeneousK',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Inhomogeneous network K',
    summary:
      'Inhomogeneous network K: clustering of events beyond what a varying intensity already explains.',
    explanation:
      'Best when event density varies (more crashes on busy roads) and you want clustering beyond that. Each pair is weighted by the inverse intensities at both ends. Choose Ang edge correction and a normalization.',
    dataTypes: {input: ['points', 'network'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'transport'],
    params: [
      {
        name: 'eventNodeIds',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Network node of each event.'
      },
      {
        name: 'intensity',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Estimated intensity per event.'
      },
      {
        name: 'nodeDistances',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Event-major shortest-path distances to every node.'
      },
      {
        name: 'edgeNodes',
        type: 'GraphDataView<"uint32x2">',
        required: true,
        description: 'Edge endpoints.'
      },
      {
        name: 'edgeLengths',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Edge lengths.'
      },
      {
        name: 'radii',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Distances at which K is evaluated.'
      },
      {
        name: 'correction',
        type: '"none" | "ang" | undefined',
        default: "'ang'",
        description: "Edge correction: 'none' or 'ang'."
      },
      {
        name: 'normalization',
        type: '"network-length" | "inverse-intensity"',
        default: "'network-length'",
        description: "Denominator: 'network-length' or 'inverse-intensity'."
      }
    ],
    outputs: [
      {name: 'k', type: "GraphDataView<'float32'>", description: 'Inhomogeneous K per radius.'},
      {
        name: 'numerator',
        type: "GraphDataView<'float32'>",
        description: 'Weighted pair sum per radius.'
      },
      {
        name: 'normalization',
        type: "GraphDataView<'float32'>",
        description: 'Denominator per radius.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Overflow and pair-count status.'
      }
    ],
    example: `import {GPUNetworkInhomogeneousK} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkInhomogeneousK({
  eventNodeIds: eventNodes.importToGraph(graph),
  intensity: intensity.importToGraph(graph),
  nodeDistances: distances.importToGraph(graph),
  nodeCount,
  edgeNodes: edgeNodes.importToGraph(graph),
  edgeLengths: edgeLengths.importToGraph(graph),
  radii: radii.importToGraph(graph),
  output: {k, numerator, normalization, status}
}));`,
    preview: {
      kind: 'histogram',
      caption: 'K corrected for uneven intensity along the network.',
      seed: 4341
    },
    related: ['network-k-function', 'network-kernel-density'],
    lensScenario: 'gpu-network.network-inhomogeneous-k',
    reference: 'api/gpu-spatial-analysis/point-process',
    maturity: 'stable'
  },
  {
    id: 'network-kernel-density',
    symbol: 'GPUNetworkKernelDensity',
    kind: 'class',
    importPath: '@luma-contributors/analysis/gpu-network',
    title: 'Network kernel density',
    summary:
      "Network KDE with a triangular shortest-path kernel, corrected by each event's total kernel mass.",
    explanation:
      'Maps event intensity along streets without smearing across blocks. Kernels follow shortest paths and are mass-corrected at junctions. Output is an intensity per sample point.',
    dataTypes: {input: ['points', 'network'], output: ['network']},
    useCases: ['interpolate', 'find-patterns'],
    domains: ['spatial-statistics', 'transport', 'urban'],
    params: [
      {
        name: 'distances',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Query-major shortest-path distances, queries x events.'
      },
      {name: 'eventCount', type: 'number', required: true, description: 'Number of events.'},
      {
        name: 'quadratureLengths',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Length represented by each sample.'
      },
      {
        name: 'bandwidth',
        type: 'number',
        required: true,
        description: 'Kernel bandwidth in network units.'
      },
      {
        name: 'excludedEventIds',
        type: 'GraphDataView<"uint32"> | undefined',
        description: 'Events to leave out, e.g. for cross-validation.'
      },
      {
        name: 'output',
        type: '{intensity, kernelMass}',
        required: true,
        description: 'intensity, kernelMass, status.'
      }
    ],
    outputs: [
      {
        name: 'intensity',
        type: "GraphDataView<'float32'>",
        description: 'Density per network sample.'
      },
      {
        name: 'kernelMass',
        type: "GraphDataView<'float32'>",
        description: 'Total kernel mass per event.'
      },
      {name: 'status', type: "GraphDataView<'uint32'>", description: 'Status words.'}
    ],
    example: `import {GPUNetworkKernelDensity} from '@luma-contributors/analysis/gpu-network';

graph.add(new GPUNetworkKernelDensity({
  distances: distances.importToGraph(graph),
  eventCount,
  quadratureLengths: lengths.importToGraph(graph),
  bandwidth: 400,
  output: {intensity, kernelMass, status}
}));`,
    preview: {
      kind: 'density-surface',
      caption: 'Event density follows the street, not the block.',
      seed: 4342
    },
    related: ['network-k-function', 'network-inhomogeneous-k'],
    lensScenario: 'gpu-network.network-kernel-density',
    reference: 'api/gpu-network/point-process',
    maturity: 'stable'
  }
];
