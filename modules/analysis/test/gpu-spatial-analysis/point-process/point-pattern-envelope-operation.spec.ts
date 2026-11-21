// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {
  GPUPointPatternEnvelopeOperation,
  type PointPatternEnvelopeOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const statistic of [
  'maximum-deviation',
  'integrated-squared-deviation',
  'extreme-rank'
] as const)
  for (const invalid of [false, true]) {
    it(`point-pattern ${statistic}, invalid=${invalid} compares every output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: PointPatternEnvelopeOperationProps = {
        observed: 'observed',
        simulations: 'simulations',
        reference: 'reference',
        integrationWeights: 'weights',
        curveRowCount: 3,
        simulationCount: 4,
        nullModel: 'fixed externally simulated null',
        fittedParameters: 'fixed-external',
        statistic,
        rank: 2,
        significanceLevel: 0.4,
        output: {
          lower: 'lower',
          upper: 'upper',
          statistics: 'statistics',
          status: 'status',
          ...(statistic === 'extreme-rank' ? {pValueInterval: 'interval'} : {})
        }
      };
      const inputs = {
        observed: {format: 'float32' as const, values: Float32Array.of(2, 1, -1)},
        simulations: {
          format: 'float32' as const,
          values: Float32Array.of(invalid ? NaN : 0, 0, 1, 1, 2, -2, 3, 1, 0, 2, 3, -1)
        },
        reference: {format: 'float32' as const, values: Float32Array.of(0.25, 0.5, -0.25)},
        weights: {format: 'float32' as const, values: Float32Array.of(1, 2, 0.5)}
      };
      const outputs = [
        'lower',
        'upper',
        'statistics',
        'status',
        ...(statistic === 'extreme-rank' ? ['interval'] : [])
      ];
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPUPointPatternEnvelopeOperation(props)]
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPUPointPatternEnvelopeOperation(props)]
      });
      for (const port of outputs) {
        const expectedLength =
          port === 'statistics' ? 4 : port === 'status' ? 1 : port === 'interval' ? 2 : 3;
        expect(cpu[port].length).toBe(expectedLength);
        expect(gpu[port].length).toBe(expectedLength);
      }
      for (const port of outputs)
        for (let index = 0; index < cpu[port].length; index++) {
          const expected = cpu[port][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(gpu[port][index])).toBe(true);
          // Copies, integer counts and half-integer ranks are exact; quotients allow 2e-6 relative.
          else if (
            ['lower', 'upper', 'status'].includes(port) ||
            (port === 'statistics' &&
              (index === 3 || (statistic === 'extreme-rank' && index === 0))) ||
            expected === 0
          )
            expect(gpu[port][index] === expected).toBe(true);
          else
            expect(Math.abs(gpu[port][index] - expected)).toBeLessThanOrEqual(
              2e-6 * Math.abs(expected)
            );
        }
    });
  }

it('TWN-41 retains the residual through the independent exceedance decision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const props: PointPatternEnvelopeOperationProps = {
      observed: 'observed',
      simulations: 'simulations',
      reference: 'reference',
      integrationWeights: 'weights',
      curveRowCount: 2,
      simulationCount: 19,
      nullModel: 'specified null',
      fittedParameters: 'none',
      statistic: 'integrated-squared-deviation',
      output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
    };
    for (const Operation of [GPUPointPatternEnvelopeOperation, CPUPointPatternEnvelopeOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          observed: {format: 'float32', values: Float32Array.of(1, 2 ** -13)},
          simulations: {
            format: 'float32',
            values: Float32Array.from({length: 38}, (_, index) => (index % 2 ? 0 : 1))
          },
          reference: {format: 'float32', values: Float32Array.of(0, 0)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)}
        },
        outputs: ['statistics', 'status'],
        operations: [new Operation(props)]
      });
      // 1+2^-26 exceeds all nineteen statistics of 1: zero exceedances, p=1/20.
      expect([...result['statistics']]).toEqual([1, Math.fround(1 / 20), Math.fround(1 / 20), 19]);
      expect([...result['status']]).toEqual([0]);
    }
  } finally {
    device.destroy();
  }
});

it('DEF-12 maximum deviation ignores absent, zero and unit integration weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const weight of [undefined, 0, 1]) {
      const result = await runOperationGraph(device, {
        inputs: {
          observed: {format: 'float32', values: Float32Array.of(3)},
          simulations: {format: 'float32', values: Float32Array.of(1, 2)},
          reference: {format: 'float32', values: Float32Array.of(0)},
          ...(weight === undefined
            ? {}
            : {weights: {format: 'float32' as const, values: Float32Array.of(weight)}})
        },
        outputs: ['statistics', 'status'],
        operations: [
          new GPUPointPatternEnvelopeOperation({
            observed: 'observed',
            simulations: 'simulations',
            reference: 'reference',
            ...(weight === undefined ? {} : {integrationWeights: 'weights'}),
            curveRowCount: 1,
            nullModel: 'specified null',
            fittedParameters: 'none',
            output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
          })
        ]
      });
      expect([...result['statistics']]).toEqual([3, Math.fround(1 / 3), Math.fround(1 / 3), 2]);
      expect([...result['status']]).toEqual([0]);
    }
  } finally {
    device.destroy();
  }
});

it('B2 compares exact products and tiny normal addends before Monte Carlo decisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const [observed, simulated] of [
      [
        [1 + 2 ** -13, 1 - 2 ** -13],
        [1, 1]
      ],
      [
        [2 ** -40, 2 ** -53],
        [2 ** -40, 0]
      ]
    ]) {
      // Exact excesses are respectively 2^-25 and 2^-106, both strictly positive.
      // Nineteen identical simulated curves therefore give zero exceedances and p=1/20.
      for (const Operation of [
        GPUPointPatternEnvelopeOperation,
        CPUPointPatternEnvelopeOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            observed: {format: 'float32', values: Float32Array.from(observed)},
            simulations: {
              format: 'float32',
              values: Float32Array.from(Array.from({length: 19}, () => simulated).flat())
            },
            reference: {format: 'float32', values: Float32Array.of(0, 0)},
            weights: {format: 'float32', values: Float32Array.of(1, 1)}
          },
          outputs: ['statistics', 'status'],
          operations: [
            new Operation({
              observed: 'observed',
              simulations: 'simulations',
              reference: 'reference',
              integrationWeights: 'weights',
              curveRowCount: 2,
              simulationCount: 19,
              statistic: 'integrated-squared-deviation',
              nullModel: 'specified',
              fittedParameters: 'none',
              output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
            })
          ]
        });
        expect(result['statistics'].length).toBe(4);
        expect(Number.isFinite(result['statistics'][0])).toBe(true);
        expect(result['statistics'][1]).toBe(Math.fround(1 / 20));
        expect(result['statistics'][2]).toBe(Math.fround(1 / 20));
        expect(result['statistics'][3]).toBe(19);
        expect([...result['status']]).toEqual([0]);
      }
    }
  } finally {
    device.destroy();
  }
});
