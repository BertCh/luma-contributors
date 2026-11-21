// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {CPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const maximumIterations of [1, 16]) {
  it(`derives both operation stopping words from an eight-edge chain (${maximumIterations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // One local hop in one round leaves node 8 unresolved. Sixteen rounds
    // exceed the eight edges plus the unchanged sweep, including dependency phases.
    for (const Operation of [GPUNetworkCentralityOperation, CPUNetworkCentralityOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 8)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8)},
          weights: {format: 'float32', values: new Float32Array(8).fill(1)},
          reverseOffsets: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 2, 3, 4, 5, 6, 7, 8)},
          reverseNeighbors: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7)},
          pivots: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [
          new Operation({
            nodeCount: 9,
            edgeCount: 8,
            reverseEdgeCount: 8,
            pivotCount: 1,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            reverseTopology: {
              offsets: 'reverseOffsets',
              neighbors: 'reverseNeighbors',
              weights: 'weights'
            },
            pivotNodes: 'pivots',
            maximumIterations,
            localIterations: 1,
            output: {
              closeness: 'closeness',
              status: {converged: 'converged', iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: ['converged', 'limit']
      });
      expect(Array.from(rows['limit'])).toEqual([maximumIterations === 1 ? 1 : 0]);
      expect(Array.from(rows['converged'])).toEqual([maximumIterations === 1 ? 0 : 1]);
    }
  });
}
