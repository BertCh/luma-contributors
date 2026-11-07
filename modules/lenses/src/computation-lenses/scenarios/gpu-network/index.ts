// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network lens scenarios (side-effect imports): the network pack first, then one
// import line per scenario registration file.

import '../../../gpu-network/computation-pack/index';
import {defineLensScenario} from '../../define-lens-scenarios';
import './register-search';
import './register-structure';
import './register-construction';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};
defineLensScenario({
  id: 'gpu-network.network-reachability',
  element: 'GPUNetworkReachability',
  title: 'Chicago Loop reachability wavefront',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph'],
  dataset,
  load: () => import('./network-reachability.lens')
});
defineLensScenario({
  id: 'gpu-network.network-reachability.chained',
  element: 'GPUNetworkReachability',
  title: 'Chicago Loop chained frontier rounds',
  rungs: ['result', 'iterations', 'kernel', 'graph'],
  dataset,
  example: {path: 'scenarios/gpu-network/network-reachability.example.ts'},
  load: () => import('./network-reachability.chained.lens')
});

import './register-expansion';

import './register-optimization-coverage';

import './point-process.lens';
