// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {
  GPUNetworkTimeRoutingOperation,
  type NetworkTimeRoutingOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of ['affine', 'piecewise', 'invalid', 'no-transitions'] as const) {
  it(`routing twins compare all arrivals, predecessors and summary words: ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const piecewise = fixture === 'piecewise';
    const props: NetworkTimeRoutingOperationProps = {
      stateCount: 4,
      transitionCount: fixture === 'no-transitions' ? 0 : 3,
      stateVertices: 'vertices',
      transitions: 'transitions',
      costs: 'costs',
      endpoints: 'endpoints',
      departure: 'departure',
      costModel: piecewise ? 'piecewise-arrival' : 'affine-duration',
      samplesPerTransition: piecewise ? 2 : undefined,
      output: {
        arrivals: 'arrivals',
        predecessorTransitions: 'predecessors',
        summary: 'summary',
        status: {
          invalidCount: 'invalidCount',
          converged: 'converged',
          iterationCount: 'iterationCount',
          iterationLimitReached: 'iterationLimitReached'
        }
      }
    };
    const inputs = {
      vertices: {format: 'uint32' as const, values: new Uint32Array([0, 1, 2, 99])},
      transitions: {
        format: 'uint32x2' as const,
        values: new Uint32Array(fixture === 'no-transitions' ? [] : [1, 2, 0, 1, 0, 2])
      },
      costs: {
        format: 'float32x2' as const,
        values: new Float32Array(
          fixture === 'no-transitions'
            ? []
            : piecewise
              ? [0, 3, 16, 19, 0, 2, 16, 18, 0, 20, 16, 36]
              : [3, 0.25, 2, 0.5, 20, 0]
        )
      },
      endpoints: {format: 'uint32' as const, values: new Uint32Array([0, 2])},
      departure: {
        format: 'float32' as const,
        values: new Float32Array([fixture === 'invalid' ? -1 : 4])
      }
    };
    const outputs = [
      'arrivals',
      'predecessors',
      'summary',
      'invalidCount',
      'converged',
      'iterationCount',
      'iterationLimitReached'
    ];
    const results = [];
    for (const Operation of [GPUNetworkTimeRoutingOperation, CPUNetworkTimeRoutingOperation])
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    // These dyadic fixtures have exact labels and target-arrival bits. Integer tolerance is zero.
    expect(results[0]['predecessors']).toEqual(results[1]['predecessors']);
    expect(results[0]['summary']).toEqual(results[1]['summary']);
    for (const field of ['invalidCount', 'converged', 'iterationCount', 'iterationLimitReached'])
      expect(results[0][field]).toEqual(results[1][field]);
    for (let state = 0; state < 4; state++) {
      const expected = results[1]['arrivals'][state];
      if (!Number.isFinite(expected)) expect(results[0]['arrivals'][state]).toBe(expected);
      else
        expect(Math.abs(results[0]['arrivals'][state] - expected)).toBeLessThanOrEqual(
          1e-5 * Math.max(1, Math.abs(expected))
        );
    }
  });
}
