// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSinkhornDivergenceOperation} from '../../../src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation';
import {
  GPUSinkhornDivergenceOperation,
  type SinkhornDivergenceOperationProps
} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const frame of ['ordinary', 'limited', 'empty', 'bad-parameters'] as const) {
  it(`Sinkhorn operations agree on every value and diagnostic: ${frame}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SinkhornDivergenceOperationProps = {
      grid: {width: 2, height: 2, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      source: 'source',
      target: 'target',
      parameters: 'parameters',
      maximumIterations: frame === 'limited' ? 1 : 50,
      output: {
        divergence: 'divergence',
        components: 'components',
        marginalError: 'error',
        status: {
          converged: 'converged',
          iterationLimitReached: 'limited',
          iterationCount: 'iterations',
          invalidCount: 'invalid',
          approximation: 'approximation'
        }
      }
    };
    const inputs = {
      source: {format: 'float32' as const, values: new Float32Array([1, 2, 3, 4])},
      target: {
        format: 'float32' as const,
        values: frame === 'empty' ? new Float32Array(4) : new Float32Array([4, 1, 3, 2])
      },
      parameters: {
        format: 'float32' as const,
        values:
          frame === 'bad-parameters'
            ? new Float32Array(6)
            : getGPUSinkhornDivergenceParameterValues({
                cellWidth: 1.5,
                cellHeight: 0.75,
                epsilon: 2,
                tolerance: 0.002,
                epsilonStart: 4,
                scalingFactor: 0.5
              })
      }
    };
    const outputs = [
      'divergence',
      'components',
      'error',
      'converged',
      'limited',
      'iterations',
      'invalid',
      'approximation'
    ];
    const cpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new CPUSinkhornDivergenceOperation(props)]
    });
    const gpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new GPUSinkhornDivergenceOperation(props)]
    });
    // Positive values: relative 3e-4. The convergence residual approaches zero and has absolute
    // tolerance 3e-6. All integer diagnostics are exact for these threshold-separated scenes.
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let index = 0; index < cpu[output].length; index++) {
        const expected = cpu[output][index];
        if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
        else if (output === 'error')
          expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-6);
        else if (output === 'divergence' || output === 'components')
          expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-4 * Math.abs(expected));
        else expect(gpu[output][index]).toBe(expected);
      }
    }
    if (frame === 'ordinary') expect(cpu['converged'][0]).toBe(1);
    if (frame === 'limited') expect(cpu['limited'][0]).toBe(1);
  });
}
