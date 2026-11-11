// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {GPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation';
import {GPU_TRANSITION_HOMOGENEITY} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const [name, classCount, conditionCount, counts] of [
  ['ordinary', 2, 3, Uint32Array.of(9, 1, 1, 9, 5, 5, 5, 5, 1, 9, 9, 1)],
  ['moderate-df4-tail', 2, 3, Uint32Array.of(3, 1, 1, 3, 2, 2, 2, 2, 1, 3, 3, 1)],
  ['empty', 2, 2, new Uint32Array(8)],
  ['failed-chain', 2, 2, Uint32Array.of(0, 1, 0, 0, 2, 2, 1, 3)],
  ['single-chain', 2, 1, Uint32Array.of(3, 7, 2, 8)],
  ['reducible-escape', 2, 1, Uint32Array.of(1, 0, 1, 0)]
] as const) {
  it(`transition diagnostic graph twins agree: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const output = {
      homogeneity: conditionCount > 1 ? 'homogeneity' : undefined,
      steadyStates: 'steadyStates',
      stateSparsity: 'stateSparsity',
      status: {converged: 'converged'}
    };
    const outputs = [
      'steadyStates',
      'stateSparsity',
      'converged',
      ...(conditionCount > 1 ? ['homogeneity'] : [])
    ];
    const results = [];
    for (const Operation of [
      GPUTransitionDiagnosticsOperation,
      CPUTransitionDiagnosticsOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {counts: {format: 'uint32', values: counts}},
          operations: [new Operation({counts: 'counts', classCount, conditionCount, output})],
          outputs
        })
      );
    }
    for (const port of outputs) {
      const expected = results[1][port];
      const actual = results[0][port];
      expect(actual.length).toBe(expected.length);
      for (let index = 0; index < expected.length; index++) {
        const value = expected[index];
        if (Number.isNaN(value)) expect(actual[index], `${port}[${index}]`).toBeNaN();
        else if (port === 'converged' || port === 'stateSparsity')
          expect(actual[index]).toBe(value);
        else {
          // All output rows: stationary 2e-6 absolute; homogeneity statistics 1e-5 relative,
          // p-values 2e-4 absolute, including the GPU gamma approximation documented upstream.
          const tolerance =
            port === 'steadyStates'
              ? 2e-6
              : index === 1 || index === 3
                ? 2e-4
                : 1e-5 * Math.max(1, Math.abs(value));
          expect(Math.abs(actual[index] - value), `${port}[${index}]`).toBeLessThanOrEqual(
            tolerance
          );
        }
      }
    }
    if (name === 'moderate-df4-tail') {
      const likelihoodRatio = 8 * (3 * Math.log(1.5) + Math.log(0.5));
      for (const rows of results) {
        expect(rows['homogeneity'][GPU_TRANSITION_HOMOGENEITY.Q]).toBeCloseTo(4, 4);
        expect(rows['homogeneity'][GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM]).toBe(4);
        expect(
          Math.abs(rows['homogeneity'][GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE] - 3 * Math.exp(-2))
        ).toBeLessThanOrEqual(2e-4);
        expect(
          Math.abs(
            rows['homogeneity'][GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE] -
              Math.exp(-likelihoodRatio / 2) * (1 + likelihoodRatio / 2)
          )
        ).toBeLessThanOrEqual(2e-4);
      }
    }
  });
}
