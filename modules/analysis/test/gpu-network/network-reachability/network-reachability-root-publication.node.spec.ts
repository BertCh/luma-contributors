// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/gpu-network-reachability';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it.each([
  false,
  true
])('root publication has no concurrent non-atomic level writes (labels: %s)', labels => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const topology = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 5),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
    weights: createTransientView(graph, 'weights', 'float32', 3)
  };
  const seeds = createTransientView(graph, 'seeds', 'uint32', 3);
  const output = {
    costs: createTransientView(graph, 'costs', 'float32', 4),
    predecessors: createTransientView(graph, 'predecessors', 'uint32', 4)
  };
  const contributor = labels
    ? new GPUNetworkServiceAreas({
        ...{topology, facilities: seeds},
        output: {
          ...output,
          assignments: createTransientView(graph, 'assignments', 'uint32', 4)
        }
      })
    : new GPUNetworkReachability({topology, startNodes: seeds, output});
  try {
    const nodes = contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-tie-roots'));
    expect(nodes).toHaveLength(1);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels).toHaveLength(1);
    const kernel = kernels[0];
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    // Duplicate seed rows share a level address. Equal values do not make plain stores race-free.
    expect(kernel.source).toMatch(/levels:\s*array<atomic<u32>>/);
    expect(kernel.source).toContain('atomicStore(&levels[levelsOffset + node], 0u)');
  } finally {
    device.destroy();
  }
});
