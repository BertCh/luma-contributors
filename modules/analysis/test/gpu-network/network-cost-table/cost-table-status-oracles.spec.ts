// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {CPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const maximumIterations of [1, 4]) {
  it(`pins every operation stopping word from three-node pair enumeration (${maximumIterations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // One local hop finds three diagonal pairs and two edges; (0,2) remains
    // unresolved. Four rounds settle all six downstream pairs, fitting capacity 9.
    for (const Operation of [GPUNetworkCostTableOperation, CPUNetworkCostTableOperation]) {
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          cutoff: {format: 'float32', values: Float32Array.of(Infinity)}
        },
        operations: [
          new Operation({
            nodeCount: 3,
            capacity: 9,
            maximumIterations,
            localIterations: 1,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            costLimit: 'cutoff',
            output: {
              offsets: 'outputOffsets',
              targets: 'targets',
              costs: 'costs',
              status: {
                complete: 'complete',
                converged: 'converged',
                iterationLimitReached: 'limit',
                overflow: 'overflow',
                requiredCount: 'required'
              }
            }
          })
        ],
        outputs: ['complete', 'converged', 'limit', 'overflow', 'required']
      });
      expect(
        ['complete', 'converged', 'limit', 'overflow', 'required'].map(name => rows[name][0])
      ).toEqual(maximumIterations === 1 ? [0, 0, 1, 0, 5] : [1, 1, 0, 0, 6]);
    }
  });
}
