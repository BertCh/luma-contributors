// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-error-gm-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const fixture of fixtures) {
  it(`GPU GM matches every independent oracle output: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
        neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
        weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
        predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
        response: {format: 'float32', values: Float32Array.from(fixture.response)},
        traceParameters: {format: 'uint32', values: Uint32Array.of(fixture.seed)}
      },
      operations: [
        new GPUSpatialErrorGMOperation({
          rowCount: fixture.response.length,
          predictorCount: 2,
          estimator: fixture.estimator as 'homoskedastic' | 'kelejian-prucha',
          traceMode: fixture.mode as 'exact' | 'stochastic',
          traceProbeCount: fixture.probeCount,
          traceParameters: 'traceParameters',
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          predictors: 'predictors',
          response: 'response',
          output: {
            table: 'table',
            summary: 'summary',
            residuals: 'residuals',
            status: {code: 'code'}
          }
        })
      ],
      outputs: ['table', 'summary', 'residuals', 'code']
    });
    expect(Array.from(result['code'])).toEqual(fixture.expected.status);
    // Exact integer/boolean slots cannot pass as unwritten zero buffers.
    for (const slot of [0, 9, 11, 12])
      expect(result['summary'][slot]).toBe(fixture.expected.summary[slot]);
    // 0.5% allows float32 reductions, two 2x2 covariance inversions and parameter search.
    // These fixtures have moderate condition numbers, including after the exact dyadic shift.
    for (const port of ['table', 'summary', 'residuals'] as const)
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        5e-3
      );
  });
}
