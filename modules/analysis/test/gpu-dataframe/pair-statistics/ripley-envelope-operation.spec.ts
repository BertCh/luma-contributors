// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURipleyEnvelopeOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {
  GPURipleyEnvelopeOperation,
  type RipleyEnvelopeOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeRipleyOnCPU} from './ripley-oracle';

it('both Ripley operations ignore observedK capacity past radiusCount', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RipleyEnvelopeOperationProps = {
    positions: 'positions',
    parameters: 'parameters',
    observedK: 'observed',
    rowCount: 2,
    gridSize: [4, 4],
    radiusCount: 1,
    simulationCount: 2,
    statistic: 'k',
    output: {
      lower: 'lower',
      upper: 'upper',
      globalLower: 'globalLower',
      globalUpper: 'globalUpper',
      theoretical: 'theoretical',
      statistics: 'statistics',
      unplacedCount: 'unplaced'
    }
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 1)},
    observed: {format: 'float32' as const, values: Float32Array.of(1, 99)},
    parameters: {
      format: 'float32' as const,
      values: getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 2,
        edgeCorrection: 'none'
      })
    }
  };
  const outputs = [
    'lower',
    'upper',
    'globalLower',
    'globalUpper',
    'theoretical',
    'statistics',
    'unplaced'
  ];
  // Radius 2 exceeds the unit-square diameter: simulated K and observed K are all 1.
  // Thus T = 2 - sqrt(1/pi), with two exceedances and p = 1. The trailing 99 is unused.
  const deviation = 2 - Math.sqrt(1 / Math.PI);
  const expected: Record<string, readonly number[]> = {
    lower: [1],
    upper: [1],
    globalLower: [1],
    globalUpper: [Math.PI * (2 + deviation) ** 2],
    theoretical: [4 * Math.PI],
    statistics: [2, 2, deviation, deviation, 1, 1],
    unplaced: [0]
  };
  const gpu = await runOperationGraph(device, {
    inputs,
    outputs,
    operations: [new GPURipleyEnvelopeOperation(props)]
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    outputs,
    operations: [new CPURipleyEnvelopeOperation(props)]
  });
  for (const port of outputs) {
    expect(gpu[port].length).toBe(expected[port].length);
    expect(cpu[port].length).toBe(expected[port].length);
    for (let index = 0; index < expected[port].length; index++) {
      const truth = expected[port][index];
      const exact =
        ['lower', 'upper', 'unplaced'].includes(port) ||
        (port === 'statistics' && [0, 1, 4, 5].includes(index));
      for (const result of [gpu, cpu]) {
        if (exact) expect(result[port][index] === truth).toBe(true);
        else
          expect(Math.abs(result[port][index] - truth)).toBeLessThanOrEqual(2e-5 * Math.abs(truth));
      }
      // Derived curves allow 2e-5 relative error; copies and integer-valued metadata are exact.
      if (exact) expect(gpu[port][index] === cpu[port][index]).toBe(true);
      else
        expect(Math.abs(gpu[port][index] - cpu[port][index])).toBeLessThanOrEqual(
          2e-5 * Math.abs(cpu[port][index])
        );
    }
  }
});

for (const edgeCorrection of ['none', 'border', 'isotropic'] as const)
  for (const statistic of ['k', 'l', 'l-minus-r'] as const)
    for (const excluded of [false, true]) {
      it(`Ripley ${edgeCorrection}/${statistic}, excluded=${excluded} compares every rectangle output`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const props: RipleyEnvelopeOperationProps = {
          positions: 'positions',
          parameters: 'parameters',
          rowCount: 6,
          mask: 'mask',
          observedK: 'observed',
          gridSize: [4, 4],
          radiusCount: 4,
          simulationCount: 6,
          statistic,
          output: {
            lower: 'lower',
            upper: 'upper',
            globalLower: 'globalLower',
            globalUpper: 'globalUpper',
            theoretical: 'theoretical',
            statistics: 'statistics',
            unplacedCount: 'unplaced'
          }
        };
        const inputs = {
          positions: {
            format: 'float32x2' as const,
            values: Float32Array.of(0.3, 0.2, 0.8, 0.3, 0.4, 0.6, 0.7, 0.9, 0.2, 0.8, 0.6, 0.1)
          },
          mask: {
            format: 'uint32' as const,
            values: excluded ? new Uint32Array(6) : Uint32Array.of(1, 1, 0, 1, 1, 1)
          },
          observed: {format: 'float32' as const, values: new Float32Array(4)},
          parameters: {
            format: 'float32' as const,
            values: getGPURipleyParameterValues({
              bounds: [0, 0, 1, 1],
              maximumDistance: 0.5,
              edgeCorrection,
              seed: 31
            })
          }
        };
        // Supply K of this same masked pattern. In the excluded case every ordinate is NaN.
        inputs.observed.values.set(
          computeRipleyOnCPU(
            {positions: inputs.positions.values, mask: inputs.mask.values},
            {bounds: [0, 0, 1, 1], maximumDistance: 0.5, edgeCorrection},
            props.radiusCount
          ).k
        );
        const outputs = [
          'lower',
          'upper',
          'globalLower',
          'globalUpper',
          'theoretical',
          'statistics',
          'unplaced'
        ];
        const gpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new GPURipleyEnvelopeOperation(props)]
        });
        const cpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new CPURipleyEnvelopeOperation(props)]
        });
        for (const port of outputs)
          for (let index = 0; index < cpu[port].length; index++) {
            const expected = cpu[port][index];
            const actual = gpu[port][index];
            if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
            else if (
              port === 'unplaced' ||
              (port === 'statistics' && [0, 1, 5].includes(index)) ||
              expected === 0
            )
              expect(actual === expected).toBe(true);
            // Positive estimates use relative tolerance; only signed L-minus-r curves allow cancellation error.
            else
              expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
                2e-4 * Math.abs(expected) +
                  (statistic === 'l-minus-r' && port !== 'statistics' ? 2e-5 : 0)
              );
          }
      });
    }
