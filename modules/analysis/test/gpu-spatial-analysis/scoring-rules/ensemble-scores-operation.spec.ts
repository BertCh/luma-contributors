// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUEnsembleScoresOperation} from '../../../src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation';
import {
  GPUEnsembleScoresOperation,
  type EnsembleScoresOperationProps
} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fair of [false, true]) {
  for (const invalid of [false, true]) {
    it(`ensemble operations agree on all scores, fair=${fair}, invalid pair=${invalid}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: EnsembleScoresOperationProps = {
        ensemble: 'ensemble',
        observations: 'observations',
        memberCount: 4,
        dimension: 3,
        estimator: fair ? 'fair' : 'empirical',
        variogramOrder: 0.5,
        variogramPairs: 'pairs',
        variogramWeights: 'weights',
        output: {crps: 'crps', energyScore: 'energy', variogramScore: 'variogram'}
      };
      const inputs = {
        ensemble: {
          format: 'float32' as const,
          values: new Float32Array([-2, 1, 4, 3, -1, 2, 0, 4, -3, 5, 2, 1])
        },
        observations: {format: 'float32' as const, values: new Float32Array([1, 0, 2])},
        pairs: {
          format: 'uint32x2' as const,
          values: new Uint32Array([0, 1, 1, 2, 2, invalid ? 3 : 0, 1, 1])
        },
        weights: {format: 'float32' as const, values: new Float32Array([0.25, 2, 1.5, 3])}
      };
      const outputs = ['crps', 'energy', 'variogram'];
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPUEnsembleScoresOperation(props)]
      });
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPUEnsembleScoresOperation(props)]
      });
      for (const output of outputs) {
        expect(gpu[output].length).toBe(cpu[output].length);
        for (let index = 0; index < cpu[output].length; index++) {
          const expected = cpu[output][index];
          if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
          else
            expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-5 * Math.abs(expected));
        }
      }
    });
  }
}

it('single-member operations agree with the absolute error and Euclidean distance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUEnsembleScoresOperation, CPUEnsembleScoresOperation]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        ensemble: {format: 'float32', values: new Float32Array([3, 4])},
        observations: {format: 'float32', values: new Float32Array([0, 0])}
      },
      operations: [
        new Operation({
          ensemble: 'ensemble',
          observations: 'observations',
          memberCount: 1,
          dimension: 2,
          output: {crps: 'crps', energyScore: 'energy'}
        })
      ],
      outputs: ['crps', 'energy']
    });
    expect(Array.from(rows['crps'])).toEqual([3, 4]);
    expect(rows['energy'][0]).toBe(5);
  }
});
