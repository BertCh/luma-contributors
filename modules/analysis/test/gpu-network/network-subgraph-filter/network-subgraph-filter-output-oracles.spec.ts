// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkSubgraphFilter} from '../../../src/gpu-network/network-subgraph-filter/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('subgraph compaction writes zero candidate loss independently of final output truncation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 0->1->2->3 with node 2 masked: nodes [0,1,3] survive, only slot 0 survives.
  for (const role of ['liveNodes', 'liveEdgeSlots'] as const)
    for (const withDiagnostic of [true, false]) {
      const context = createOracleContext(device);
      const {graph, input, output, read} = context;
      const nodeMask = output('node-mask', 'uint32', 4);
      const edgeMask = output('edge-mask', 'uint32', 3);
      const compact = {
        ids: output('ids', 'uint32', role === 'liveNodes' ? 2 : 1),
        status: {
          count: output('count', 'uint32'),
          overflow: output('overflow', 'uint32'),
          candidateOverflow: withDiagnostic ? output('candidate', 'uint32') : undefined
        }
      };
      graph.add(
        new GPUNetworkSubgraphFilter({
          topology: {
            offsets: input('offsets', 'uint32', [0, 1, 2, 3, 3]),
            neighbors: input('neighbors', 'uint32', [1, 2, 3])
          },
          directed: true,
          nodeMask: input('selection', 'uint32', [1, 1, 0, 1]),
          output: {nodeMask, edgeMask, [role]: compact}
        })
      );
      try {
        context.run();
        expect(await read(nodeMask)).toEqual([1, 1, 0, 1]);
        expect(await read(edgeMask)).toEqual([1, 0, 0]);
        expect(await read(compact.ids)).toEqual(role === 'liveNodes' ? [0, 1] : [0]);
        expect(await read(compact.status.count)).toEqual(role === 'liveNodes' ? [2] : [1]);
        expect(await read(compact.status.overflow)).toEqual(role === 'liveNodes' ? [1] : [0]);
        if (compact.status.candidateOverflow)
          expect(await read(compact.status.candidateOverflow)).toEqual([0]);
      } finally {
        context.destroy();
      }
    }
});
