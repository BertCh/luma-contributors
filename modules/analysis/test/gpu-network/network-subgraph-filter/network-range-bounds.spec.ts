// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkSubgraphFilter} from '../../../src/gpu-network/network-subgraph-filter/gpu-network-subgraph-filter';
import {getGPUNetworkSubgraphFilterParameterValues} from '../../../src/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('decimal attribute and time bounds retain exact float32 membership (PCK-04)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  try {
    const {input, output, graph} = context;
    const nodeMask = output('node-mask', 'uint32', 2);
    const edgeMask = output('edge-mask', 'uint32');
    const parameters = getGPUNetworkSubgraphFilterParameterValues(
      {nodeColumnCount: 1, edgeColumnCount: 0, hasEdgeTimes: true},
      {nodeAttributeRanges: [[0, 0.7]], closedEdgeTimeWindow: [0, 0.1]}
    );
    graph.add(
      new GPUNetworkSubgraphFilter({
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 1]),
          neighbors: input('neighbors', 'uint32', [1])
        },
        directed: true,
        // 0x3f333333 < 0.7; 0x3dcccccd > 0.1. Data are already representable.
        nodeColumns: [input('attributes', 'float32', [0.699999988079071, 0.699999988079071])],
        edgeTimes: input('times', 'float32', [0.10000000149011612]),
        parameters: input('parameters', 'float32', Array.from(parameters)),
        output: {nodeMask, edgeMask}
      })
    );
    context.run();
    expect(Array.from(await context.read(nodeMask))).toEqual([1, 1]);
    expect(Array.from(await context.read(edgeMask))).toEqual([0]);
  } finally {
    context.destroy();
  }
});

it('zero rows retain membership at subnormal attribute and closed time bounds (S3)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  try {
    const {input, output, graph} = context;
    const nodeMask = output('zero-node-mask', 'uint32', 2);
    const edgeMask = output('zero-edge-mask', 'uint32');
    const parameters = getGPUNetworkSubgraphFilterParameterValues(
      {nodeColumnCount: 1, edgeColumnCount: 0, hasEdgeTimes: true},
      {nodeAttributeRanges: [[0, 2 ** -151]], closedEdgeTimeWindow: [-1, -(2 ** -151)]}
    );
    graph.add(
      new GPUNetworkSubgraphFilter({
        topology: {
          offsets: input('zero-offsets', 'uint32', [0, 1, 1]),
          neighbors: input('zero-neighbors', 'uint32', [1])
        },
        directed: true,
        nodeColumns: [input('zero-attributes', 'float32', [0, -0])],
        edgeTimes: input('zero-times', 'float32', [0]),
        parameters: input('zero-parameters', 'float32', Array.from(parameters)),
        output: {nodeMask, edgeMask}
      })
    );
    context.run();
    // Both signed zeros lie in [0, 2^-151); zero is above the closed time end -2^-151.
    expect(Array.from(await context.read(nodeMask))).toEqual([1, 1]);
    expect(Array.from(await context.read(edgeMask))).toEqual([0]);
  } finally {
    context.destroy();
  }
});
