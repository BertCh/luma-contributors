// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network construction-family lens scenarios (one entry per scenario).

import {registerLensScenarios} from '../../define-lens-scenarios';

const roadsDataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};
const polylinesDataset = {
  id: 'n-chicago-loop-polylines',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract (raw centrelines)'
};
const tracesDataset = {
  id: 'n-chicago-loop-traces',
  licence: 'Synthetic, derived from OpenStreetMap data (ODbL 1.0)',
  source: 'SIMULATED GPS traces on OpenStreetMap roads (not real vehicles)'
};
const crashesDataset = {
  id: 'n-chicago-loop-crashes',
  licence: 'City of Chicago Data Portal Terms of Use',
  source: 'City of Chicago traffic crashes 2023 (sampled) on OpenStreetMap roads'
};
const flowsDataset = {
  id: 'n-chicago-od-flows',
  licence: 'City of Chicago Data Portal Terms of Use',
  source: 'City of Chicago taxi trips 2023 between community areas'
};

registerLensScenarios(
  {
    entry: 'gpu-network',
    audience: 'analyst',
    defaultRung: 'result'
  },
  [
    {
      id: 'gpu-network.network-cost-table',
      element: 'GPUNetworkCostTable',
      title: 'Chicago Loop 30-second travel-cost table',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      dataset: roadsDataset,
      load: () => import('./network-cost-table.lens')
    },
    {
      id: 'gpu-network.network-cost-weights',
      element: 'GPUNetworkCostWeights',
      title: 'Chicago Loop facility catchment weights',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      dataset: roadsDataset,
      load: () => import('./network-cost-weights.lens')
    },
    {
      id: 'gpu-network.network-noding',
      element: 'GPUNetworkNoding',
      title: 'Chicago Loop centrelines noded into a network',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      dataset: polylinesDataset,
      load: () => import('./network-noding.lens')
    },
    {
      id: 'gpu-network.map-matching',
      element: 'GPUMapMatching',
      title: 'Simulated GPS tracks matched to Loop roads',
      rungs: ['result', 'datum', 'kernel', 'graph'],
      dataset: tracesDataset,
      load: () => import('./map-matching.lens')
    },
    {
      id: 'gpu-network.network-k-function',
      element: 'GPUNetworkKFunction',
      title: 'Network K function of Chicago Loop crashes',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      dataset: crashesDataset,
      load: () => import('./network-k-function.lens')
    },
    {
      id: 'gpu-network.edge-bundling',
      element: 'GPUEdgeBundling',
      title: 'Chicago taxi flows bundled by density',
      rungs: ['result', 'datum', 'iterations', 'kernel', 'graph'],
      dataset: flowsDataset,
      load: () => import('./edge-bundling.lens')
    },
    {
      id: 'gpu-network.flow-aggregation',
      element: 'GPUFlowAggregation',
      title: 'Chicago taxi flows ranked by zone pair',
      rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
      dataset: flowsDataset,
      load: () => import('./flow-aggregation.lens')
    }
  ]
);
