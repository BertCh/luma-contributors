// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {SPREG_ERROR_GM_REFERENCE} from './spatial-error-gm-reference';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

it('Kelejian-Prucha accepts traceProbeCount=0 and recovers the hand-derived four-row fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = ['code', 'table', 'summary', 'residuals'];
  const results = [];
  for (const Operation of [GPUSpatialErrorGMOperation, CPUSpatialErrorGMOperation]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 3, 6, 9, 12)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2)},
          weights: {format: 'float32', values: new Float32Array(12).fill(1)},
          predictors: {format: 'float32', values: Float32Array.of(-1, 0, 1, 2)},
          response: {format: 'float32', values: Float32Array.of(0, 1, 4, 9)}
        },
        operations: [
          new Operation({
            rowCount: 4,
            predictorCount: 1,
            estimator: 'kelejian-prucha',
            traceProbeCount: 0,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            predictors: 'predictors',
            response: 'response',
            output: {
              status: {code: 'code'},
              table: 'table',
              summary: 'summary',
              residuals: 'residuals'
            }
          })
        ],
        outputs: names
      })
    );
  }
  for (const result of results) {
    expect(result['code'][0]).toBe(0);
    expectRegressionRowsClose(
      [result['table'][0], result['table'][4], result['table'][8]],
      [2, 3, -0.33],
      5e-3
    );
    expectRegressionRowsClose(result['residuals'], [1, -1, -1, 1], 5e-3);
  }
  for (const name of names)
    expectRegressionRowsClose(results[0][name], results[1][name], name === 'code' ? 0 : 5e-3);
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const) {
  for (const scene of ['ordinary', 'stochastic', 'singular', 'zero-links'] as const) {
    it(`GM operations agree on ${estimator}/${scene}: 5e-3 relative, status and flags exact`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Rook lattice has unequal row weights (corners, edges and interior), two nontrivial
      // predictors and a nonzero lambda. Stochastic mode uses a fixed nonzero registered seed.
      const fixture = SPREG_ERROR_GM_REFERENCE.lattice;
      const names = ['code', 'table', 'summary', 'residuals'];
      const results = [];
      for (const Operation of [GPUSpatialErrorGMOperation, CPUSpatialErrorGMOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              offsets: {
                format: 'uint32',
                values:
                  scene === 'zero-links'
                    ? new Uint32Array(fixture.offsets.length)
                    : Uint32Array.from(fixture.offsets)
              },
              neighbors: {format: 'uint32', values: Uint32Array.from(fixture.neighbors)},
              weights: {format: 'float32', values: Float32Array.from(fixture.weights)},
              predictors: {
                format: 'float32',
                values:
                  scene === 'singular'
                    ? new Float32Array(fixture.predictors.length)
                    : Float32Array.from(fixture.predictors)
              },
              response: {format: 'float32', values: Float32Array.from(fixture.response)},
              traceParameters: {
                format: 'uint32',
                values: getGPUSpatialWeightsTracesParameterValues({seed: 42})
              }
            },
            operations: [
              new Operation({
                rowCount: fixture.response.length,
                predictorCount: 2,
                estimator,
                traceMode: scene === 'stochastic' ? 'stochastic' : 'exact',
                traceProbeCount: 64,
                traceParameters: 'traceParameters',
                weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
                predictors: 'predictors',
                response: 'response',
                output: {
                  status: {code: 'code'},
                  table: 'table',
                  summary: 'summary',
                  residuals: 'residuals'
                }
              })
            ],
            outputs: names
          })
        );
      }
      const expectedStatus = scene === 'singular' ? 1 : scene === 'zero-links' ? 3 : 0;
      for (const result of results) expect(Array.from(result['code'])).toEqual([expectedStatus]);
      for (const name of names)
        expectRegressionRowsClose(results[0][name], results[1][name], name === 'code' ? 0 : 5e-3);
      if (expectedStatus === 0) {
        for (const slot of [0, 9, 11, 12])
          expect(results[0]['summary'][slot] === results[1]['summary'][slot]).toBe(true);
        expect(Math.abs(results[1]['summary'][1])).toBeGreaterThan(0.1);
        expect(results[1]['table'][3]).toBeGreaterThan(1e-8);
        expect(results[1]['table'][3]).toBeLessThan(0.99);
      }
    });
  }
}
