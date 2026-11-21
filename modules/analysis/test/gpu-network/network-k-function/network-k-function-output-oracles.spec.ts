// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues
} from '../../../src/gpu-network/network-k-function/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('network K reports an independently stopped search with the limit flag alone or absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumIterations of [1, 8])
    for (const withFlag of [true, false]) {
      const context = createOracleContext(device);
      const {graph, input, output, read} = context;
      const kValues = output('k', 'float32', 2);
      const iterationLimitReached = withFlag ? output('limit', 'uint32') : undefined;
      // Six-node undirected unit chain. Events at opposite ends seed only their incident
      // endpoints; one one-hop round cannot reach the far endpoint and cannot certify a fixpoint.
      // With eight rounds the distance is 5 < 6: one unordered pair, K=2*1*5/2²=2.5.
      graph.add(
        new GPUNetworkKFunction({
          topology: {
            offsets: input('offsets', 'uint32', [0, 1, 3, 5, 7, 9, 10]),
            neighbors: input('neighbors', 'uint32', [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]),
            weights: input('weights', 'float32', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            nodePositions: input('nodes', 'float32x2', [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0])
          },
          positions: input('events', 'float32x2', [0, 0, 5, 0]),
          maximumDistance: input('distance', 'float32', [6]),
          networkLength: input('length', 'float32', [5]),
          parameters: input(
            'parameters',
            'uint32',
            Array.from(getGPUNetworkKFunctionParameterValues({seed: 1, activeSimulations: 0}))
          ),
          bandCount: 2,
          maximumIterations,
          localIterations: 1,
          rowsPerBlock: 2,
          laneCount: 1,
          output: {kValues, status: iterationLimitReached ? {iterationLimitReached} : undefined}
        })
      );
      try {
        context.run();
        if (iterationLimitReached)
          expect(await read(iterationLimitReached)).toEqual([maximumIterations === 1 ? 1 : 0]);
        expect(await read(kValues)).toEqual(maximumIterations === 1 ? [0, 0] : [0, 2.5]);
      } finally {
        context.destroy();
      }
    }
});
