// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation';
import {GPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {fixture, getLikelihoodAuditInput} from './likelihood-mathematical-audit-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const transformed of [false, true]) {
  for (const kind of ['lag', 'error'] as const) {
    it(`audits every GPU ${kind} ML port against dense NumPy: transformed=${transformed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getLikelihoodAuditInput(transformed);
      const expected = fixture.expected[transformed ? (`${kind}Transformed` as const) : kind];
      const Operation =
        kind === 'lag'
          ? GPUSpatialLagMaximumLikelihoodOperation
          : GPUSpatialErrorMaximumLikelihoodOperation;
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: input.weights.offsets},
          neighbors: {format: 'uint32', values: input.weights.neighbors},
          weights: {format: 'float32', values: input.weights.weights},
          predictors: {format: 'float32', values: input.predictors},
          response: {format: 'float32', values: input.response}
        },
        operations: [
          new Operation({
            rowCount: 12,
            predictorCount: 2,
            maximumPower: 64,
            traceMode: 'exact',
            tileRowCount: 5,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            predictors: 'predictors',
            response: 'response',
            output: {
              status: {code: 'status', converged: 'converged'},
              table: 'table',
              summary: 'summary',
              residuals: 'residuals',
              powerTraceStandardErrors: 'powerTraceStandardErrors',
              ...(kind === 'lag'
                ? {powerTraces: 'powerTraces', powerSums: 'powerSums', infinityNorm: 'infinityNorm'}
                : {})
            }
          })
        ],
        outputs: Object.keys(expected)
      });
      for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
        // Float32 OLS, trace chains and information inversion: 0.5% relative allowance.
        // Exact zero compares as zero; the comparator rejects NaN for every finite expectation.
        expectRegressionRowsClose(
          result[port],
          Float32Array.from(expected[port], value => value ?? NaN),
          port === 'status' || port === 'converged' ? 0 : 5e-3
        );
      }
    });
  }
}

for (const scale of [2 ** -34, 2 ** 34]) {
  it(`error ML pseudo-R squared retains its dense value at response scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getLikelihoodAuditInput(false);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: input.weights.offsets},
        neighbors: {format: 'uint32', values: input.weights.neighbors},
        weights: {format: 'float32', values: input.weights.weights},
        predictors: {format: 'float32', values: input.predictors},
        response: {
          format: 'float32',
          values: Float32Array.from(input.response, value => scale * value)
        }
      },
      operations: [
        new GPUSpatialErrorMaximumLikelihoodOperation({
          rowCount: 12,
          predictorCount: 2,
          maximumPower: 64,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          predictors: 'predictors',
          response: 'response',
          output: {status: {code: 'status'}, table: 'table', summary: 'summary'}
        })
      ],
      outputs: ['status', 'summary']
    });
    expect(result['status'][0]).toBe(0);
    // Correlation is invariant to positive response rescaling; the literal comes from NumPy.
    expectRegressionRowsClose([result['summary'][6]], [fixture.expected.error.summary[6]!], 5e-3);
  });
}
