// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkTravelTimePercentilesOperation} from '../../../src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation';
import {
  GPUNetworkTravelTimePercentilesOperation,
  type NetworkTravelTimePercentilesOperationProps
} from '../../../src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it.each([
  {
    departureCount: 2,
    percentiles: [49.99999999, 50, 50.00000001, 100],
    costs: [5, 85],
    maximumRelaxations: undefined,
    expected: [5, 5, 90, 90],
    reached: [2, 2],
    converged: 1
  },
  {
    departureCount: 25,
    percentiles: [28, 56 - 2 ** -47, 56, 56 + 2 ** -47, 100],
    costs: [1, 1],
    maximumRelaxations: undefined,
    expected: [7, 14, 14, 15, 25],
    reached: [25, 25],
    converged: 1
  },
  {
    departureCount: 2,
    percentiles: [50.00000001],
    costs: [5, 85],
    maximumRelaxations: 0,
    expected: [Infinity],
    reached: [2, 0],
    converged: 0
  }
])('selects hand-calculated nearest ranks for $departureCount departures on both backends', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: NetworkTravelTimePercentilesOperationProps = {
    stateCount: 2,
    transitionCount: 1,
    stateVertices: 'vertices',
    transitions: 'transitions',
    costs: 'costs',
    initialState: 'initial',
    window: 'window',
    vertexCount: 2,
    departureCount: fixture.departureCount,
    percentiles: fixture.percentiles,
    maximumRelaxations: fixture.maximumRelaxations,
    output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
  };
  for (const Operation of [
    GPUNetworkTravelTimePercentilesOperation,
    CPUNetworkTravelTimePercentilesOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        vertices: {format: 'uint32', values: Uint32Array.of(0, 1)},
        transitions: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
        costs: {format: 'float32x2', values: Float32Array.from(fixture.costs)},
        initial: {format: 'uint32', values: Uint32Array.of(0)},
        window: {format: 'float32', values: Float32Array.of(0, 1)}
      },
      operations: [new Operation(props)],
      outputs: ['times', 'reached', 'status']
    });
    expect(Array.from(result['times'])).toEqual([
      ...fixture.percentiles.map(() => 0),
      ...fixture.expected
    ]);
    expect(Array.from(result['reached'])).toEqual(fixture.reached);
    expect(Array.from(result['status'])).toEqual(
      Array.from({length: fixture.departureCount}, () => [1, fixture.converged]).flat()
    );
  }
});

it.each([
  50.00000001,
  NaN
])('keeps the constructed median and output shape after caller mutation to %s on both backends', async mutation => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUNetworkTravelTimePercentilesOperation,
    CPUNetworkTravelTimePercentilesOperation
  ]) {
    const percentiles = [50];
    const operation = new Operation({
      stateCount: 2,
      transitionCount: 1,
      stateVertices: 'vertices',
      transitions: 'transitions',
      costs: 'costs',
      initialState: 'initial',
      window: 'window',
      vertexCount: 2,
      departureCount: 2,
      percentiles,
      output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
    });
    percentiles[0] = mutation;
    percentiles.push(100);
    const result = await runOperationGraph(device, {
      inputs: {
        vertices: {format: 'uint32', values: Uint32Array.of(0, 1)},
        transitions: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
        costs: {format: 'float32x2', values: Float32Array.of(5, 85)},
        initial: {format: 'uint32', values: Uint32Array.of(0)},
        window: {format: 'float32', values: Float32Array.of(0, 1)}
      },
      operations: [operation],
      outputs: ['times', 'reached', 'status']
    });
    expect(Array.from(result['times'])).toEqual([0, 5]);
    expect(Array.from(result['reached'])).toEqual([2, 2]);
    expect(Array.from(result['status'])).toEqual([1, 1, 1, 1]);
  }
});

it.each([
  false,
  true
])('honors a zero budget from an inherited getter on both backends: %s', async inherited => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPUNetworkTravelTimePercentilesOperation,
    CPUNetworkTravelTimePercentilesOperation
  ]) {
    const props = Object.assign(
      inherited
        ? Object.create({
            get maximumRelaxations() {
              return 0;
            }
          })
        : {maximumRelaxations: 0},
      {
        stateCount: 2,
        transitionCount: 1,
        stateVertices: 'vertices',
        transitions: 'transitions',
        costs: 'costs',
        initialState: 'initial',
        window: 'window',
        vertexCount: 2,
        departureCount: 2,
        percentiles: [50],
        output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
      }
    );
    const result = await runOperationGraph(device, {
      inputs: {
        vertices: {format: 'uint32', values: Uint32Array.of(0, 1)},
        transitions: {format: 'uint32x2', values: Uint32Array.of(0, 1)},
        costs: {format: 'float32x2', values: Float32Array.of(5, 0)},
        initial: {format: 'uint32', values: Uint32Array.of(0)},
        window: {format: 'float32', values: Float32Array.of(0, 1)}
      },
      operations: [new Operation(props)],
      outputs: ['times', 'reached', 'status']
    });
    expect(Array.from(result['times'])).toEqual([0, Infinity]);
    expect(Array.from(result['reached'])).toEqual([2, 0]);
    expect(Array.from(result['status'])).toEqual([1, 0, 1, 0]);
  }
});

for (const fixture of ['affine', 'piecewise-wait', 'invalid-initial', 'no-transitions'] as const) {
  it(`travel-time percentile twins compare every output: ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const piecewise = fixture === 'piecewise-wait';
    const props: NetworkTravelTimePercentilesOperationProps = {
      stateCount: 4,
      transitionCount: fixture === 'no-transitions' ? 0 : 2,
      stateVertices: 'vertices',
      transitions: 'transitions',
      costs: 'costs',
      initialState: 'initial',
      window: 'window',
      vertexCount: 3,
      departureCount: 3,
      percentiles: [1, 50, 100],
      costModel: piecewise ? 'piecewise-arrival' : 'affine-duration',
      samplesPerTransition: piecewise ? 2 : undefined,
      output: {travelTimes: 'times', reachedCounts: 'reached', status: 'status'}
    };
    const inputs = {
      vertices: {format: 'uint32' as const, values: new Uint32Array([0, 1, 1, 2])},
      transitions: {
        format: 'uint32x2' as const,
        values: new Uint32Array(fixture === 'no-transitions' ? [] : [0, 1, 0, 2])
      },
      costs: {
        format: 'float32x2' as const,
        values: new Float32Array(
          fixture === 'no-transitions'
            ? []
            : piecewise
              ? [4, 8, 8, 12, 4, 24, 8, 28]
              : [2, 0.5, 20, 0]
        )
      },
      initial: {
        format: 'uint32' as const,
        values: new Uint32Array([fixture === 'invalid-initial' ? 99 : 0])
      },
      window: {format: 'float32' as const, values: new Float32Array([0, 2])}
    };
    const outputs = ['times', 'reached', 'status'];
    const results = [];
    for (const Operation of [
      GPUNetworkTravelTimePercentilesOperation,
      CPUNetworkTravelTimePercentilesOperation
    ])
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    expect(results[0]['reached']).toEqual(results[1]['reached']);
    expect(results[0]['status']).toEqual(results[1]['status']);
    if (fixture === 'affine') {
      // t = 0, 2, 4 gives durations 2, 3, 4 through the first state at vertex one.
      expect(Array.from(results[0]['times'])).toEqual([
        0,
        0,
        0,
        2,
        3,
        4,
        Infinity,
        Infinity,
        Infinity
      ]);
      expect(Array.from(results[0]['reached'])).toEqual([3, 3, 0]);
      expect(Array.from(results[0]['status'])).toEqual([1, 1, 1, 1, 1, 1]);
    }
    // Relative/absolute float tolerance 1e-5; integer flags/counts and infinities are exact.
    for (let index = 0; index < 9; index++) {
      const expected = results[1]['times'][index];
      if (!Number.isFinite(expected)) expect(results[0]['times'][index]).toBe(expected);
      else
        expect(Math.abs(results[0]['times'][index] - expected)).toBeLessThanOrEqual(
          1e-5 * Math.max(1, Math.abs(expected))
        );
    }
  });
}
