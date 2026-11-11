// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation';
import {
  GPUNetworkRationalAccessOperation,
  type NetworkRationalAccessOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of [
  'integral',
  'fractional',
  'simultaneous-inflow',
  'unreachable',
  'invalid'
] as const) {
  it(`rational-access twins compare all values for ${scene} input`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fractional = scene === 'fractional';
    const simultaneousInflow = scene === 'simultaneous-inflow';
    const props: NetworkRationalAccessOperationProps = {
      id: 'rational',
      originCount: fractional ? 1 : 2,
      facilityCount: 2,
      costs: 'costs',
      demand: 'demand',
      supply: 'supply',
      parameters: 'parameters',
      maximumCycles: simultaneousInflow ? 1 : 3,
      initialLimitCycles: simultaneousInflow ? 1 : 0,
      demandMode: fractional ? 'fractional' : 'integral',
      stepMode: fractional ? 'absolute' : 'fraction',
      output: {
        assignments: 'assignments',
        scores: 'scores',
        unservedDemand: 'unserved',
        status: 'status'
      }
    };
    const inputs = {
      costs: {
        format: 'float32' as const,
        values: simultaneousInflow
          ? new Float32Array(4)
          : fractional
            ? Float32Array.of(0, 2)
            : scene === 'unreachable'
              ? Float32Array.of(0, 2, Infinity, Infinity)
              : Float32Array.of(0, 2, 10, 0)
      },
      demand: {
        format: 'float32' as const,
        values: simultaneousInflow
          ? Float32Array.of(3, 3)
          : fractional
            ? Float32Array.of(10)
            : Float32Array.of(10, 2)
      },
      supply: {
        format: 'float32' as const,
        values: simultaneousInflow
          ? Float32Array.of(1, 3)
          : scene === 'invalid'
            ? Float32Array.of(0, 4)
            : Float32Array.of(2, 4)
      },
      parameters: {
        format: 'float32' as const,
        values: Float32Array.of(
          fractional ? 2 : 1,
          fractional ? 2 : 1,
          fractional ? 1.5 : 1,
          fractional ? 1.5 : 1,
          50,
          0
        )
      }
    };
    const outputs = ['assignments', 'scores', 'unserved', 'status'];
    const results: Record<string, Record<string, OperationGraphRows>> = {};
    for (const location of ['gpu', 'cpu'])
      results[location] = await runOperationGraph(device, {
        inputs,
        operations: [
          location === 'gpu'
            ? new GPUNetworkRationalAccessOperation(props)
            : new CPUNetworkRationalAccessOperation(props)
        ],
        outputs
      });
    for (const port of outputs) {
      const expected = results['cpu'][port];
      const actual = results['gpu'][port];
      expect(actual.length).toBe(expected.length);
      for (let row = 0; row < expected.length; row++) {
        if (port === 'status' && row === 3) {
          // Packed float32 gap bits are compared as floats, relative 2e-5.
          const expectedGap = new Float32Array(Uint32Array.of(expected[row]).buffer)[0];
          const actualGap = new Float32Array(Uint32Array.of(actual[row]).buffer)[0];
          expect(Math.abs(actualGap - expectedGap)).toBeLessThanOrEqual(
            2e-5 * Math.abs(expectedGap)
          );
        } else if (Number.isNaN(expected[row])) expect(Number.isNaN(actual[row])).toBe(true);
        else if (
          port === 'status' ||
          (port === 'assignments' && !fractional) ||
          port === 'unserved'
        )
          expect(actual[row] === expected[row]).toBe(true);
        else
          expect(Math.abs(actual[row] - expected[row])).toBeLessThanOrEqual(
            2e-5 * Math.abs(expected[row])
          );
      }
    }
    if (scene === 'integral') {
      expect(Array.from(results['cpu']['assignments'])).toEqual([7, 3, 0, 2]);
      expect(Array.from(results['cpu']['status'].subarray(0, 3))).toEqual([1, 0, 3]);
    }
    if (simultaneousInflow) {
      // Each origin requests 3. Incoming 6 / capacity 3 limits each shift to 1.5;
      // ties-to-even rounds it to 2. Final loads are (2,4), costs (2,4/3).
      for (const location of ['gpu', 'cpu']) {
        expect(Array.from(results[location]['assignments'])).toEqual([1, 2, 1, 2]);
        expect(Array.from(results[location]['unserved'])).toEqual([0, 0]);
        expect(Array.from(results[location]['status'].subarray(0, 3))).toEqual([1, 0, 1]);
        for (const score of results[location]['scores'])
          expect(Math.abs(score - 14 / 9)).toBeLessThanOrEqual(2e-5 * (14 / 9));
        const gap = new Float32Array(Uint32Array.of(results[location]['status'][3]).buffer)[0];
        expect(Math.abs(gap - 2 / 3)).toBeLessThanOrEqual(2e-5 * (2 / 3));
      }
    }
  });
}
