// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';
import type {NetworkOptimizationKind} from './optimization-coverage.lens';

const scenarios: readonly [NetworkOptimizationKind, string, string][] = [
  [
    'point-routing',
    'GPUNetworkPointRouting',
    'Fractional directed-edge routes and a forbidden reverse journey'
  ],
  ['flow', 'GPUNetworkFlow', 'Integral residual flow and a minimum-cut certificate'],
  ['louvain', 'GPUNetworkLouvain', 'Disconnected weighted pairs and a modularity trace'],
  [
    'facility-optimization',
    'GPUNetworkFacilityOptimization',
    'An exact capacity assignment that greedy selection misses'
  ],
  ['time-routing', 'GPUNetworkTimeRouting', 'FIFO travel costs evaluated at each arrival'],
  [
    'rational-access',
    'GPUNetworkRationalAccess',
    'Congestion balancing with explicitly unserved demand'
  ]
];
for (const [kind, element, title] of scenarios) {
  defineLensScenario({
    id: `gpu-network.network-${kind}`,
    entry: 'gpu-network',
    element,
    title,
    audience: 'analyst',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: `synthetic-network-${kind}-v1`,
      source: 'Deterministic bounded scenario in optimization-coverage.lens.ts',
      licence: 'MIT'
    },
    load: async () => {
      const module = await import('./optimization-coverage.lens');
      return {
        run: device => module.runNetworkOptimizationCoverage(device, kind),
        describeStructure: device => module.describeNetworkOptimizationCoverage(device, kind)
      };
    }
  });
}
