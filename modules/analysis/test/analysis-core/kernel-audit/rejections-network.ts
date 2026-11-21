// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the network cases. */
export const NETWORK_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUFlowAggregation',
    name: 'maximum pairCapacity 2^23 with weights and sorted sums, rows at the binding limit',
    kind: 'dispatch',
    limit: 4294967295,
    rejects: /^Error: flow-aggregation rows x maximumProbeCount must be at most 4294967295$/
  },
  {
    exportName: 'GPUMapMatching',
    name: 'entryCapacity above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: map-matching entryCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPUMapMatching',
    name: 'edges above the maximum',
    kind: 'binding',
    limit: 11184810,
    rejects: /^Error: map-matching topology edges must be at most 11184810$/
  },
  {
    exportName: 'GPUNetworkSnapping',
    name: 'candidate search, edges above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: network-snapping topology edges with candidateCapacity must be at most 8388608$/
  },
  {
    exportName: 'GPUNetworkSnapping',
    name: 'exact scan, points times edges above the maximum',
    kind: 'dispatch',
    limit: 134217728,
    rejects:
      /^Error: network-snapping positions times topology edges without candidateCapacity must be at most 134217728$/
  },
  {
    exportName: 'GPUNetworkCentrality',
    name: 'local iterations 1, maximum rounds, four lanes, over the command node cap',
    kind: 'dispatch',
    limit: 32768,
    rejects: /^Error: network-centrality estimated command node count must be at most 32768$/
  },
  {
    exportName: 'GPUNetworkReachability',
    name: 'nodes above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: network-reachability output.costs rows must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'events at the pair-count limit, row blocks past the invocation limit',
    kind: 'dispatch',
    limit: 4294967295,
    rejects: /^Error: network-k-function rowsPerBlock x events must be at most 4294967295$/
  },
  {
    // n (n - 1) / 2 unordered pairs must fit the uint32 histogram and pairCounts (NET-110).
    exportName: 'GPUNetworkKFunction',
    name: 'events above the pair-count limit',
    kind: 'dispatch',
    limit: 92682,
    rejects: /^Error: network-k-function eventCount must be at most 92682$/
  },
  {
    exportName: 'GPUNetworkCoarsening',
    name: 'weighted superedges above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects:
      /^Error: network-coarsening output.edges.ids rows with output.edgeWeights must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkLineGraph',
    name: 'planar, base edges above the maximum, arcs and banned turns at the binding limit',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: network-line-graph topology edges must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkLineGraph',
    name: 'great-circle, base edges above the maximum, arcs and banned turns at the binding limit',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: network-line-graph topology edges with a great-circle spatialContext must be at most 8388608$/
  },
  {
    exportName: 'GPUNetworkNoding',
    name: 'intersection capacity above the maximum',
    kind: 'binding',
    limit: 1048576,
    rejects: /^Error: network-noding intersectionCapacity must be an integer in \[1, 1048576\]$/
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'events at the pair-count limit, one-row blocks, over the command node budget',
    kind: 'dispatch',
    limit: 32768,
    rejects: /^Error: network-k-function estimated command node count must be at most 32768$/
  },
  {
    exportName: 'GPUNetworkCostMatrix',
    name: 'lane count 1, 40 rows, maximum rounds, over the command node budget',
    kind: 'dispatch',
    limit: 32768,
    rejects: /^Error: network-cost-matrix estimated command node count must be at most 32768$/
  },
  {
    exportName: 'GPUNetworkCostTable',
    name: 'lane count 1, 600 nodes, default rounds, over the command node budget',
    kind: 'dispatch',
    limit: 32768,
    rejects: /^Error: network-cost-table estimated command node count must be at most 32768$/
  },
  {
    exportName: 'GPULocationAllocation',
    name: 'p-median, 11,000 greedy steps, over the command node budget',
    kind: 'dispatch',
    limit: 32768,
    rejects: /^Error: location-allocation estimated command node count must be at most 32768$/
  }
];
