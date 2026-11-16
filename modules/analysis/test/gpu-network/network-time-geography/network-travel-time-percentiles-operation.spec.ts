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
