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
import {requireTestDevice} from '../../utils/require-test-device';

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
