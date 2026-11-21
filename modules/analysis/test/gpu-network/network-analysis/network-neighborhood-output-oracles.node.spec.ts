// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUNetworkNeighborhood} from '../../../src/gpu-network/network-analysis/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('neighborhood refuses either candidate-loss output when it aliases the hops input', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const role of ['nodes', 'edges'] as const) {
      const context = createOracleContext(device);
      const {input, output} = context;
      const hops = input('hops', 'uint32', [2]);
      const props = {
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 2, 2]),
          neighbors: input('neighbors', 'uint32', [1, 2])
        },
        startNodes: input('start', 'uint32', [0]),
        hops,
        output: {
          hopDistances: output('distances', 'uint32', 3),
          [role]: {
            ids: output('ids', 'uint32', 1),
            status: {
              count: output('count', 'uint32'),
              overflow: output('overflow', 'uint32'),
              candidateOverflow: hops
            }
          }
        }
      };
      try {
        expect(() => new GPUNetworkNeighborhood(props), role).toThrow(
          /network-neighborhood.*(outputs|alias|overlap)/
        );
      } finally {
        context.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('GPUNetworkNeighborhood accepts aligned disjoint compact status slices (D077)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const role of ['nodes', 'edges'] as const) {
      const context = createOracleContext(device);
      const {graph, input, output} = context;
      try {
        // Neighborhood requires distinct physical output buffers even for disjoint ranges.
        // Exercise nonzero logical slices without violating that dispatch-binding contract.
        const scalar = (byteOffset: number) =>
          graph.createDataView(
            output(`status-storage-${byteOffset}`, 'uint32', byteOffset / 4 + 1).buffer,
            {format: 'uint32', length: 1, byteOffset}
          );
        const compact = {
          ids: output('ids', 'uint32', 1),
          status: {
            count: scalar(0),
            overflow: scalar(256),
            candidateOverflow: scalar(512),
            requiredCount: scalar(768)
          }
        };
        const contributor = new GPUNetworkNeighborhood({
          topology: {
            offsets: input('offsets', 'uint32', [0, 1, 2, 2]),
            neighbors: input('neighbors', 'uint32', [1, 2])
          },
          startNodes: input('start', 'uint32', [0]),
          hops: input('hops', 'uint32', [2]),
          output: {hopDistances: output('distances', 'uint32', 3), [role]: compact}
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
