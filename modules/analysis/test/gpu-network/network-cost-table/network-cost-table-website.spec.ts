// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createStreetNetwork} from '../../../../../website/src/gpu/demos/street-network';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';

for (const capacity of [3000, 7834]) {
  it(`publishes the square-world demo row and completeness at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const network = createStreetNetwork(1, 144);
    const weights = new Float32Array(network.slotCount);
    network.fillWeights(weights, 0);
    const harness = createOptimizationHarness(device);
    const output = {
      offsets: harness.output('uint32', network.nodeCount + 1),
      targets: harness.output('uint32', capacity),
      costs: harness.output('float32', capacity),
      status: {
        complete: harness.output('uint32', 1),
        requiredCount: harness.output('uint32', 1),
        converged: harness.output('uint32', 1),
        overflow: harness.output('uint32', 1)
      }
    };
    const read = await harness.run(
      new GPUNetworkCostTable({
        topology: {
          offsets: harness.input('uint32', network.offsets),
          neighbors: harness.input('uint32', network.neighbors),
          weights: harness.input('float32', weights)
        },
        costLimit: harness.input('float32', Float32Array.of(0.4)),
        maximumIterations: 48,
        output
      })
    );
    expect(read(output.status.converged)).toEqual([1]);
    expect(read(output.status.requiredCount)).toEqual([7834]);
    expect(read(output.status.overflow)).toEqual([Number(capacity === 3000)]);
    expect(read(output.status.complete)).toEqual([Number(capacity === 7834)]);
    expect(read(output.offsets)[144] - read(output.offsets)[143]).toBe(capacity === 3000 ? 0 : 32);
  });
}
