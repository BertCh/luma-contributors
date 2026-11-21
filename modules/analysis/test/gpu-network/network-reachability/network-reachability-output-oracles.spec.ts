// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkReachability,
  GPUNetworkReachabilityOperation
} from '../../../src/gpu-network/network-reachability/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createOracleContext,
  poisonOperationOutputs
} from '../adjacency-matrix/network-output-oracle-test-utils';

it('reachability counts exactly one edge visit and resets both counter words on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withCounter of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read, write} = context;
    const startNodes = input('start', 'uint32', [0]);
    const costs = output('costs', 'float32', 2);
    const relaxationCount = withCounter ? output('relaxations', 'uint32', 2) : undefined;
    // Only node 0 has an outgoing slot. It is visited once; visiting node 1 scans zero edges.
    graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: input('offsets', 'uint32', [0, 1, 1]),
          neighbors: input('neighbors', 'uint32', [1]),
          weights: input('weights', 'float32', [3])
        },
        startNodes,
        maximumIterations: 4,
        localIterations: 1,
        output: {costs, status: relaxationCount ? {relaxationCount} : undefined}
      })
    );
    try {
      context.run();
      expect(await read(costs)).toEqual([0, 3]);
      if (relaxationCount) expect(await read(relaxationCount)).toEqual([1, 0]);
      write(startNodes, [99]);
      context.run();
      expect(await read(costs)).toEqual([Infinity, Infinity]);
      if (relaxationCount) expect(await read(relaxationCount)).toEqual([0, 0]);
    } finally {
      context.destroy();
    }
  }
});

it('reachability operation has hand-counted bands and edge counters with optional ports alone', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Unit chain 0->1->2: costs [0,1,2], <= thresholds [1,2] give bands [0,0,1],
  // Add the dead-target 0->99 slot: three slots are scanned, two edges are reached.
  // This distinguishes relaxation work from reached edges without scheduling-dependent revisits.
  for (const selected of [
    undefined,
    'relaxationCount',
    'bands',
    'bandCounts',
    'reachedEdgeCount'
  ] as const) {
    const withBreaks = selected === 'bands' || selected === 'bandCounts';
    const operation = poisonOperationOutputs(
      new GPUNetworkReachabilityOperation({
        nodeCount: 3,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        maximumIterations: 4,
        localIterations: 1,
        costBreaks: withBreaks ? 'breaks' : undefined,
        costBreakCount: withBreaks ? 2 : undefined,
        output: {
          costs: 'costs',
          bands: selected === 'bands' ? 'bands' : undefined,
          bandCounts: selected === 'bandCounts' ? 'bandCounts' : undefined,
          status:
            selected === 'relaxationCount' || selected === 'reachedEdgeCount'
              ? {[selected]: selected}
              : undefined
        }
      })
    );
    const values = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 99, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
        starts: {format: 'uint32', values: Uint32Array.of(0)},
        breaks: {format: 'float32', values: Float32Array.of(1, 2)}
      },
      operations: [operation],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    expect(Array.from(values['costs'])).toEqual([0, 1, 2]);
    if (selected)
      expect(Array.from(values[selected])).toEqual(
        selected === 'bands'
          ? [0, 0, 1]
          : selected === 'bandCounts'
            ? [2, 1]
            : selected === 'relaxationCount'
              ? [3, 0]
              : [2, 0]
      );
  }
});
