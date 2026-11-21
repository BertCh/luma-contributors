// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkReachability} from '../../src/gpu-network/network-reachability/gpu-network-reachability';
import {GPUNetworkTimeRoutingOperation} from '../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createOracleContext} from './adjacency-matrix/network-output-oracle-test-utils';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';

it('F7 overlapping CSR rows retain the source-specific tie parent and edge slot', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  try {
    const costs = output('costs', 'float32', 4);
    const parents = output('parents', 'uint32', 4);
    const slots = output('slots', 'uint32', 4);
    const converged = output('converged', 'uint32');
    graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: input('offsets', 'uint32', [0, 2, 2, 0, 2]),
          neighbors: input('neighbors', 'uint32', [1, 1]),
          weights: input('weights', 'float32', [0, 1])
        },
        startNodes: input('starts', 'uint32', [0, 3]),
        startCosts: input('start-costs', 'float32', [19, 2]),
        maximumIterations: 4,
        maximumTieIterations: 4,
        output: {costs, predecessors: parents, predecessorEdges: slots, status: {converged}}
      })
    );
    context.run();
    // Source 3 reaches node 1 at cost 2 through slot 0; source 0 starts at cost 19.
    const values = await read(costs);
    expect(Number.isFinite(values[1])).toBe(true);
    expect(values[1]).toBe(2);
    expect((await read(parents))[1]).toBe(3);
    expect((await read(slots))[1]).toBe(0);
    expect(await read(converged)).toEqual([1]);
  } finally {
    context.destroy();
  }
});

it.each([
  [1, 0],
  [2 ** 25, 1.5]
])('elapsed operation publishes duration separately at clock %s', async (departure, duration) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      vertices: {format: 'uint32', values: Uint32Array.of(0, 1)},
      transitions: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
      costs: {format: 'float32x2', values: Float32Array.of(duration, 0)},
      endpoints: {format: 'uint32', values: Uint32Array.of(0, 1)},
      departure: {format: 'float32', values: Float32Array.of(departure)}
    },
    operations: [
      new GPUNetworkTimeRoutingOperation({
        stateCount: 2,
        transitionCount: 1,
        stateVertices: 'vertices',
        transitions: 'transitions',
        costs: 'costs',
        endpoints: 'endpoints',
        departure: 'departure',
        output: {
          arrivals: 'arrivals',
          elapsedTimes: 'elapsed',
          predecessorTransitions: 'parents',
          summary: 'summary'
        }
      })
    ],
    outputs: ['arrivals', 'elapsed']
  });
  expect(Array.from(rows['elapsed'])).toEqual([0, duration]);
  expect(Array.from(rows['arrivals'])).toEqual(
    Array.from(Float32Array.of(departure, departure + duration))
  );
});
