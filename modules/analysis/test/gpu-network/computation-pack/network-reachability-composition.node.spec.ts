// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

test('GPUNetworkReachability unrolls relaxation and tie levels into generated kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'reachability-composition'});
  const view = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
  try {
    const nodes = new GPUNetworkReachability({
      id: 'reach',
      maximumIterations: 4,
      maximumTieIterations: 3,
      localIterations: 1,
      topology: {
        offsets: view('offsets', 5),
        neighbors: view('neighbors', 4),
        weights: createTransientView(graph, 'weights', 'float32', 4)
      },
      startNodes: view('starts', 1),
      costBreaks: createTransientView(graph, 'thresholds', 'float32', 5),
      output: {
        costs: createTransientView(graph, 'costs', 'float32', 4),
        bands: view('bands', 4),
        bandCounts: view('counts', 5),
        predecessors: view('predecessors', 4),
        status: {converged: view('converged', 1), iterationCount: view('iterationCount', 1)}
      }
    }).getCommandNodes(graph);
    expect(getGPUCommandNodeKernelSources(nodes).length).toBeGreaterThan(0);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
    const ids = nodes.map(node => node.id);
    expect(ids.filter(id => /^reach-relax-\d+$/.test(id))).toHaveLength(4);
    expect(ids.filter(id => /^reach-tie-level-\d+$/.test(id))).toHaveLength(3);
  } finally {
    device.destroy();
  }
});
