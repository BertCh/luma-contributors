// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkNeighborhood} from '../../../src/gpu-network/network-analysis/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';
import type {GPUCompactOutput} from '../../../src/analysis-core/kernels/gpu-contributor-types';

it('neighborhood compact diagnostics distinguish final truncation from candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Directed 0->1->2, hop radius 2: three nodes and two induced arcs, each compact capacity 1.
  for (const role of ['nodes', 'edges'] as const)
    for (const diagnostic of [undefined, 'requiredCount', 'candidateOverflow'] as const) {
      const context = createOracleContext(device);
      const {graph, input, output, read} = context;
      const hopDistances = output('hops-out', 'uint32', 3);
      const compact: GPUCompactOutput = {
        ids: output('ids', 'uint32'),
        status: {
          count: output('count', 'uint32'),
          overflow: output('overflow', 'uint32'),
          ...(diagnostic ? {[diagnostic]: output('diagnostic', 'uint32')} : {})
        }
      };
      graph.add(
        new GPUNetworkNeighborhood({
          topology: {
            offsets: input('offsets', 'uint32', [0, 1, 2, 2]),
            neighbors: input('neighbors', 'uint32', [1, 2])
          },
          startNodes: input('start', 'uint32', [0]),
          hops: input('hops', 'uint32', [2]),
          maximumHops: 2,
          nodeIds: input('node-ids', 'uint32', [41, 17, 90]),
          edgeIds: input('edge-ids', 'uint32', [501, 92]),
          output: {hopDistances, [role]: compact}
        })
      );
      try {
        context.run();
        expect(await read(hopDistances)).toEqual([0, 1, 2]);
        expect(await read(compact.ids)).toEqual(role === 'nodes' ? [41] : [501]);
        expect(await read(compact.status.count)).toEqual([1]);
        expect(await read(compact.status.overflow)).toEqual([1]);
        if (diagnostic)
          expect(await read(compact.status[diagnostic]!)).toEqual([
            diagnostic === 'requiredCount' ? (role === 'nodes' ? 3 : 2) : 0
          ]);
      } finally {
        context.destroy();
      }
    }
});
