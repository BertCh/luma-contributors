// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUNetworkPotentialPathArea} from '../../../src/gpu-network/network-time-geography/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('potential path area resets every edge slot before writing live CSR rows', () => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {input, output, graph} = context;
  try {
    const nodes = new GPUNetworkPotentialPathArea({
      topology: {
        offsets: input('offsets', 'uint32', [0, 2, 2]),
        neighbors: input('neighbors', 'uint32', [1, 1]),
        weights: input('weights', 'float32', [1, 1])
      },
      forwardCosts: input('forward', 'float32', [0, 1]),
      reverseCosts: input('reverse', 'float32', [1, 0]),
      forwardComplete: input('forward-complete', 'uint32', [1]),
      reverseComplete: input('reverse-complete', 'uint32', [1]),
      budget: input('budget', 'float32', [2]),
      output: {
        edgeSlack: output('slack', 'float32', 2),
        edgeInPrism: output('edges', 'uint32', 2),
        nodeInPrism: output('nodes', 'uint32', 2)
      }
    }).getCommandNodes(graph);
    const reset = nodes.find(node => node.id === 'network-potential-path-area-clear-edge-outputs');
    expect(reset, 'inactive capacity slots need an unconditional writer').toBeDefined();
    const source = getParsedKernelSources([reset!]).get(reset!.id)!;
    expect(source).toContain('edgeSlack[edgeSlackOffset + index] = bitcast<f32>(infinityBits);');
    expect(source).toContain('edgeInPrism[edgeInPrismOffset + index] = 0u;');
    expect(nodes.indexOf(reset!)).toBeLessThan(nodes.findIndex(node => node.id.endsWith('-edges')));
  } finally {
    context.destroy();
    device.destroy();
  }
});
