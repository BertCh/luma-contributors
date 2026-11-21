// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-regression-diagnostics-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const fixture of fixtures) {
  it(`GPU diagnostics independent annihilator oracle: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
        neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
        weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
        predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
        response: {format: 'float32', values: Float32Array.from(fixture.response)},
        residuals: {format: 'float32', values: Float32Array.from(fixture.residuals)},
        mask: {format: 'uint32', values: Uint32Array.from(fixture.mask)}
      },
      operations: [
        new GPUSpatialRegressionDiagnosticsOperation({
          rowCount: fixture.rowCount,
          predictorCount: fixture.predictorCount,
          restandardizeRows: fixture.restandardizeRows,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          predictors: 'predictors',
          response: 'response',
          residuals: 'residuals',
          mask: 'mask',
          output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
        })
      ],
      outputs: ['tests', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    // Float32 Gram/trace solves and cancellation in the information J: 0.5% on these
    // moderate-condition fixtures; integer counts and degrees must be exact.
    expectRegressionRowsClose(result['tests'], fixture.expected.tests, 5e-3);
    expectRegressionRowsClose(result['summary'], fixture.expected.summary, 5e-3);
    for (const slot of [0, 12])
      expect(result['summary'][slot]).toBe(fixture.expected.summary[slot]);
    for (let row = 0; row < 6; row++)
      expect(result['tests'][row * 3 + 1]).toBe(fixture.expected.tests[row * 3 + 1]);
  });
}
