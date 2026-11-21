// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUNetworkSubgraphFilter} from '../../../src/gpu-network/network-subgraph-filter/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('subgraph filter refuses either candidate-loss output when it aliases the input edge mask', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const role of ['liveNodes', 'liveEdgeSlots'] as const) {
      const context = createOracleContext(device);
      const {input, output} = context;
      const edgeMask = input('selection', 'uint32', [1]);
      const props = {
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 1]),
          neighbors: input('neighbors', 'uint32', [1])
        },
        directed: true,
        edgeMask,
        output: {
          nodeMask: output('node-mask', 'uint32', 2),
          edgeMask: output('edge-mask', 'uint32', 1),
          [role]: {
            ids: output('ids', 'uint32', 1),
            status: {
              count: output('count', 'uint32'),
              overflow: output('overflow', 'uint32'),
              candidateOverflow: edgeMask
            }
          }
        }
      };
      try {
        expect(() => new GPUNetworkSubgraphFilter(props), role).toThrow(
          /network-subgraph-filter.*(outputs|alias|overlap)/
        );
      } finally {
        context.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('GPUNetworkSubgraphFilter accepts aligned disjoint compact status slices (D077)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const role of ['liveNodes', 'liveEdgeSlots'] as const) {
      const context = createOracleContext(device);
      const {graph, input, output} = context;
      try {
        // 256-byte spacing keeps physical storage bindings disjoint in the same publish dispatch.
        const storage = output('status-storage', 'uint32', 193);
        const scalar = (byteOffset: number) =>
          graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset});
        const compact = {
          ids: output('ids', 'uint32', 1),
          status: {
            count: scalar(0),
            overflow: scalar(256),
            candidateOverflow: scalar(512),
            requiredCount: scalar(768)
          }
        };
        const contributor = new GPUNetworkSubgraphFilter({
          topology: {
            offsets: input('offsets', 'uint32', [0, 1, 1]),
            neighbors: input('neighbors', 'uint32', [1])
          },
          directed: true,
          output: {
            nodeMask: output('node-mask', 'uint32', 2),
            edgeMask: output('edge-mask', 'uint32', 1),
            [role]: compact
          }
        });
        expect(() => contributor.getCommandNodes(graph), role).not.toThrow();
      } finally {
        context.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});
