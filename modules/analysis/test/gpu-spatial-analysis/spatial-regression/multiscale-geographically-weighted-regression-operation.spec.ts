// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixtures from './multiscale-geographically-weighted-regression-operation-fixtures.json';

for (const scene of ['rank', 'gaussian-tie', 'bisquare-tie'] as const) {
  it(`MGWR ranks distinct usable bandwidths and observes exact ties: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rank = scene === 'rank';
    const names = [
      'coefficients',
      'fitted',
      'codes',
      'iterationCount',
      'selectedBandwidths',
      'conditionalStandardErrors'
    ];
    const results = [];
    for (const Operation of [
      GPUMultiscaleGeographicallyWeightedRegressionOperation,
      CPUMultiscaleGeographicallyWeightedRegressionOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {
              format: 'float32x2',
              values: rank ? Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0) : new Float32Array(8)
            },
            predictors: {format: 'float32', values: Float32Array.of(1, -1, 1, -1)},
            response: {format: 'float32', values: Float32Array.of(1, 2, 3, 4)},
            parameters: {
              format: 'float32',
              values: rank ? Float32Array.of(2, 10, 10) : Float32Array.of(1e-6, 10, 2)
            },
            bandwidthCandidates: {
              format: 'float32',
              values: rank ? Float32Array.of(10, 1, 10, 1) : Float32Array.of(10, 1, 2, 20)
            },
            noiseVariance: {format: 'float32', values: Float32Array.of(1)}
          },
          operations: [
            new Operation({
              rowCount: 4,
              predictorCount: 1,
              kernel: scene === 'bisquare-tie' ? 'bisquare' : 'gaussian',
              positions: 'positions',
              predictors: 'predictors',
              response: 'response',
              parameters: 'parameters',
              bandwidthCandidates: 'bandwidthCandidates',
              bandwidthCandidateCount: 2,
              noiseVariance: 'noiseVariance',
              output: {
                coefficients: 'coefficients',
                fitted: 'fitted',
                status: {codes: 'codes', iterationCount: 'iterationCount'},
                selectedBandwidths: 'selectedBandwidths',
                conditionalStandardErrors: 'conditionalStandardErrors'
              }
            })
          ],
          outputs: names
        })
      );
    }
    // The node spec derives the finite unequal rank scores and the exact tie score 64/9 by hand.
    for (const result of results) {
      expect(Array.from(result['selectedBandwidths'])).toEqual(rank ? [1, 10] : [10, 2]);
      expect(Array.from(result['codes'])).toEqual([0, 0, 0, 0]);
      expect(result['iterationCount'][0]).toBe(1);
      if (!rank) expectRegressionRowsClose(result['fitted'], [2, 3, 2, 3], 5e-3);
    }
    for (const name of names)
      expectRegressionRowsClose(
        results[0][name],
        results[1][name],
        ['codes', 'iterationCount', 'selectedBandwidths'].includes(name) ? 0 : 5e-3
      );
  });
}

for (const kernel of ['gaussian', 'bisquare'] as const) {
  for (const scene of [
    'fixed',
    'selection',
    'excluded',
    'singular',
    'full-residual',
    'full-smoother',
    'full-excluded'
  ] as const) {
    it(`MGWR operations agree on ${kernel}/${scene}: 5e-3 relative, integer outputs exact`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[0];
      const fullInference = scene.startsWith('full-');
      const excluded = scene === 'excluded' || scene === 'full-excluded';
      const names = [
        'coefficients',
        'fitted',
        'codes',
        'iterationCount',
        'selectedBandwidths',
        'conditionalStandardErrors'
      ];
      if (fullInference)
        names.push(
          'inferenceStandardErrors',
          'inferenceEffectiveParameters',
          'inferenceSummary',
          'inferenceStatus'
        );
      const results = [];
      for (const Operation of [
        GPUMultiscaleGeographicallyWeightedRegressionOperation,
        CPUMultiscaleGeographicallyWeightedRegressionOperation
      ])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: Float32Array.from(fixture.positions)},
              predictors: {
                format: 'float32',
                values:
                  scene === 'singular'
                    ? new Float32Array(24)
                    : Float32Array.from(fixture.predictors)
              },
              response: {format: 'float32', values: Float32Array.from(fixture.response)},
              parameters: {format: 'float32', values: Float32Array.of(1e-3, 3.5, 2, 2.75)},
              mask: {
                format: 'uint32',
                values: excluded ? new Uint32Array(12) : Uint32Array.from(fixture.mask)
              },
              noiseVariance: {format: 'float32', values: Float32Array.of(0.25)},
              ...(scene === 'selection'
                ? {
                    bandwidthCandidates: {
                      format: 'float32' as const,
                      values: Float32Array.of(0.01, 3.5, 0.01, 2, 0.01, 2.75)
                    }
                  }
                : {})
            },
            operations: [
              new Operation({
                rowCount: 12,
                predictorCount: 2,
                kernel,
                maximumIterations: 128,
                positions: 'positions',
                predictors: 'predictors',
                response: 'response',
                parameters: 'parameters',
                mask: 'mask',
                noiseVariance: 'noiseVariance',
                ...(fullInference
                  ? {
                      inference: {
                        varianceConvention:
                          scene === 'full-smoother'
                            ? ('smoother-df' as const)
                            : ('residual-df' as const),
                        output: {
                          standardErrors: 'inferenceStandardErrors',
                          effectiveParameters: 'inferenceEffectiveParameters',
                          summary: 'inferenceSummary',
                          status: 'inferenceStatus'
                        }
                      }
                    }
                  : {}),
                ...(scene === 'selection'
                  ? {bandwidthCandidates: 'bandwidthCandidates', bandwidthCandidateCount: 2}
                  : {}),
                output: {
                  coefficients: 'coefficients',
                  fitted: 'fitted',
                  status: {codes: 'codes', iterationCount: 'iterationCount'},
                  selectedBandwidths: 'selectedBandwidths',
                  conditionalStandardErrors: 'conditionalStandardErrors'
                }
              })
            ],
            outputs: names
          })
        );
      for (const name of names)
        expectRegressionRowsClose(
          results[0][name],
          results[1][name],
          ['codes', 'iterationCount', 'selectedBandwidths', 'inferenceStatus'].includes(name)
            ? 0
            : 5e-3
        );
      for (const result of results)
        expect(Array.from(result['codes'])).toEqual(
          fixture.mask.map(keep => (excluded || !keep ? 1 : scene === 'singular' ? 2 : 0))
        );
      if (scene === 'fixed' || scene === 'selection') {
        expect(results[1]['iterationCount'][0]).toBeGreaterThan(1);
        expect(results[1]['conditionalStandardErrors'][0]).toBeGreaterThan(0);
      }
      if (fullInference)
        for (const result of results) expect(result['inferenceStatus'][0]).toBe(excluded ? 2 : 0);
    });
  }
}
