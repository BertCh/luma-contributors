// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network structure-family lens scenarios (one entry per scenario).

import {registerLensScenarios} from '../../define-lens-scenarios';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};

registerLensScenarios(
  {
    entry: 'gpu-network',
    audience: 'analyst',
    defaultRung: 'result',
    dataset
  },
  [
    {
      id: 'gpu-network.network-strongly-connected-components',
      element: 'GPUNetworkStronglyConnectedComponents',
      title: 'Synthetic directed components with an isolated node',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      dataset: {
        id: 'synthetic-four-node-directed-components-v1',
        source: 'Deterministic example constructed in expansion.lens.ts',
        licence: 'MIT'
      },
      load: async () => {
        const module = await import('./expansion.lens');
        return {
          run: device => module.runExpansion(device, 'network-strongly-connected-components'),
          describeStructure: device =>
            module.describeExpansion(device, 'network-strongly-connected-components')
        };
      }
    },
    {
      id: 'gpu-network.network-analytics-columns',
      element: 'GPUNetworkAnalyticsColumns',
      title: 'Chicago Loop intersection degrees',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-analytics-columns.lens')
    },
    {
      id: 'gpu-network.network-centrality',
      element: 'GPUNetworkCentrality',
      title: 'Chicago Loop closeness of eight sampled pivots',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-centrality.lens')
    },
    {
      id: 'gpu-network.network-statistics',
      element: 'GPUNetworkStatistics',
      title: 'Chicago Loop network summary statistics',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-statistics.lens')
    },
    {
      id: 'gpu-network.network-subgraph-filter',
      element: 'GPUNetworkSubgraphFilter',
      title: 'Chicago Loop short-block subgraph',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-subgraph-filter.lens')
    },
    {
      id: 'gpu-network.network-coarsening',
      element: 'GPUNetworkCoarsening',
      title: 'Chicago Loop coarsened to four zones',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      load: () => import('./network-coarsening.lens')
    },
    {
      id: 'gpu-network.adjacency-matrix',
      element: 'GPUAdjacencyMatrix',
      title: 'Chicago Loop adjacency matrix in zone order',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./adjacency-matrix.lens')
    },
    {
      id: 'gpu-network.adjacency-matrix-order',
      element: 'GPUAdjacencyMatrixOrder',
      title: 'Chicago Loop intersections ordered by zone',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      load: () => import('./adjacency-matrix-order.lens')
    },
    {
      id: 'gpu-network.network-line-graph',
      element: 'GPUNetworkLineGraph',
      title: 'Chicago Loop turn-cost line graph',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      load: () => import('./network-line-graph.lens')
    }
  ]
);
