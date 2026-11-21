// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOrdinaryLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {AUDIT_OLS_FIXTURES} from './audit-ols-fixtures';

for (const fixture of AUDIT_OLS_FIXTURES) {
  it(`GPU OLS every port matches independent SciPy QR/SVD: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
        response: {format: 'float32', values: Float32Array.from(fixture.response)},
        mask: {format: 'uint32', values: Uint32Array.from(fixture.mask)}
      },
      operations: [
        new GPUOrdinaryLeastSquaresOperation({
          rowCount: fixture.response.length,
          predictorCount: 2,
          tileRowCount: 4,
          predictors: 'predictors',
          response: 'response',
          mask: 'mask',
          output: {
            status: {code: 'status'},
            coefficients: 'coefficients',
            standardErrors: 'standardErrors',
            tStatistics: 'tStatistics',
            summary: 'summary',
            residuals: 'residuals',
            fitted: 'fitted',
            predictions: 'predictions'
          }
        })
      ],
      outputs: Object.keys(fixture.expected)
    });
    for (const name of Object.keys(fixture.expected) as (keyof typeof fixture.expected)[]) {
      expect(result[name].length).toBe(fixture.expected[name].length);
      fixture.expected[name].forEach((expected, index) => {
        const actual = result[name][index];
        if (Number.isNaN(expected)) expect(actual, `${name}[${index}]`).toBeNaN();
        else {
          expect(Number.isFinite(actual), `${name}[${index}]`).toBe(true);
          // Small, well-conditioned fixture: f32 moments, Cholesky and residual fourth powers.
          // The floor keeps a strict absolute bound on the small KB/BP statistics and p-values.
          const tolerance = (name === 'summary' ? 5e-4 : 1e-4) * Math.max(1, Math.abs(expected));
          expect(Math.abs(actual - expected), `${name}[${index}]`).toBeLessThanOrEqual(tolerance);
        }
      });
    }
  });
}
