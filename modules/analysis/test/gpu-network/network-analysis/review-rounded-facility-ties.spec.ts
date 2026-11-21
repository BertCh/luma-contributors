// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('B1 keeps the facility with a tight path when rounded downstream totals coincide', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  const assignments = output('assignments', 'uint32', 5);
  const counts = output('facility-counts', 'uint32', 2);
  const costs = output('costs', 'float32', 5);
  const predecessors = output('predecessors', 'uint32', 5);
  try {
    graph.add(
      new GPUNetworkServiceAreas({
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 2, 3, 4, 4]),
          neighbors: input('neighbors', 'uint32', [2, 3, 3, 4]),
          weights: input('weights', 'float32', [0.3, 0.7, 0.4, 1])
        },
        facilities: input('facilities', 'uint32', [0, 1]),
        maximumIterations: 8,
        maximumLabelIterations: 8,
        localIterations: 1,
        output: {assignments, costs, predecessors, facilityNodeCounts: counts}
      })
    );
    context.run();
    // f32(f32(0.3) + f32(0.4)) > f32(0.7). Only facility 1 has a tight path
    // to node 3, so the sole tight predecessor of node 4 carries facility 1's label.
    expect(await read(costs)).toEqual([
      0, 0, 0.30000001192092896, 0.699999988079071, 1.7000000476837158
    ]);
    expect(await read(assignments)).toEqual([0, 1, 0, 1, 1]);
    expect(await read(counts)).toEqual([2, 3]);
    expect(await read(predecessors)).toEqual([0xffffffff, 0xffffffff, 0, 1, 3]);
  } finally {
    context.destroy();
  }
});
