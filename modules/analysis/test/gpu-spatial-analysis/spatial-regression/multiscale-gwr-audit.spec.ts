// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createMultiscaleAuditFixture,
  createMultiscaleOverflowFixture
} from './multiscale-gwr-audit-fixtures';

it('R6 CPU and GPU MGWR operations suppress inference when published variance overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createMultiscaleOverflowFixture();
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: input.positions},
        predictors: {format: 'float32', values: input.predictors},
        response: {format: 'float32', values: input.response},
        parameters: {format: 'float32', values: input.parameters}
      },
      operations: [
        new Operation({
          rowCount: 8,
          predictorCount: 1,
          maximumIterations: 4,
          positions: 'positions',
          predictors: 'predictors',
          response: 'response',
          parameters: 'parameters',
          inference: {
            output: {
              standardErrors: 'errors',
              effectiveParameters: 'effective',
              summary: 'summary',
              status: 'inferenceStatus'
            }
          },
          output: {
            coefficients: 'coefficients',
            fitted: 'fitted',
            status: {codes: 'codes', iterationCount: 'iterations'}
          }
        })
      ],
      outputs: [
        'coefficients',
        'fitted',
        'codes',
        'errors',
        'effective',
        'summary',
        'inferenceStatus'
      ]
    });
    expect(Array.from(result['coefficients'])).toEqual(Array(16).fill(0));
    expect(Array.from(result['fitted'])).toEqual(Array(8).fill(0));
    expect(Array.from(result['codes'])).toEqual(Array(8).fill(0));
    expect(Array.from(result['summary'])).toEqual([2, 2, Infinity, 6]);
    expect(Array.from(result['inferenceStatus'])).toEqual([2]);
    expect(Array.from(result['errors']).every(Number.isNaN)).toBe(true);
    expect(Array.from(result['effective']).every(Number.isNaN)).toBe(true);
  }
});

for (const kernel of ['gaussian', 'bisquare'] as const) {
  for (const varianceConvention of [undefined, 'residual-df', 'smoother-df'] as const) {
    it(`MGWR ${kernel}/${varianceConvention ?? 'conditional'} CPU/GPU operations each match every independent block-solve output`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const rescaled of [false, true]) {
        const {input, expected} = createMultiscaleAuditFixture(
          kernel,
          rescaled,
          varianceConvention
        );
        for (const Operation of [
          GPUMultiscaleGeographicallyWeightedRegressionOperation,
          CPUMultiscaleGeographicallyWeightedRegressionOperation
        ]) {
          const result = await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: input.positions},
              predictors: {format: 'float32', values: input.predictors},
              response: {format: 'float32', values: input.response},
              parameters: {format: 'float32', values: input.parameters},
              mask: {format: 'uint32', values: input.mask},
              noiseVariance: {format: 'float32', values: input.noiseVariance}
            },
            operations: [
              new Operation({
                rowCount: 14,
                predictorCount: 1,
                kernel,
                maximumIterations: 4,
                indexGridSize: rescaled ? [8, 4] : false,
                positions: 'positions',
                predictors: 'predictors',
                response: 'response',
                parameters: 'parameters',
                mask: 'mask',
                noiseVariance: 'noiseVariance',
                inference: varianceConvention
                  ? {
                      varianceConvention,
                      output: {
                        standardErrors: 'inferenceStandardErrors',
                        effectiveParameters: 'inferenceEffectiveParameters',
                        summary: 'inferenceSummary',
                        status: 'inferenceStatus'
                      }
                    }
                  : undefined,
                output: {
                  coefficients: 'coefficients',
                  fitted: 'fitted',
                  selectedBandwidths: 'selectedBandwidths',
                  conditionalStandardErrors: 'conditionalStandardErrors',
                  status: {
                    codes: 'codes',
                    iterationCount: 'iterationCount',
                    componentSearchIterationLimitReached: 'componentSearchIterationLimitReached'
                  }
                }
              })
            ],
            outputs: Object.keys(expected)
          });
          for (const [name, values] of Object.entries(expected)) {
            expect(result[name].length, name).toBe(values.length);
            values.forEach((value, index) => {
              if (Number.isNaN(value)) expect(result[name][index], `${name}[${index}]`).toBeNaN();
              else {
                expect(Number.isFinite(result[name][index]), `${name}[${index}]`).toBe(true);
                // Short float32 local sums, two sweeps and 12x12 smoother propagation.
                const tolerance =
                  name === 'codes' ||
                  name === 'iterationCount' ||
                  name === 'inferenceStatus' ||
                  name === 'componentSearchIterationLimitReached' ||
                  name === 'selectedBandwidths'
                    ? 0
                    : 1e-4;
                expect(
                  Math.abs(result[name][index] - value),
                  `${name}[${index}]`
                ).toBeLessThanOrEqual(tolerance);
              }
            });
          }
        }
      }
    });
  }
}
