// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './multiscale-bandwidth-selection-fixtures.json';

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number | null>,
  tolerance: number
) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index++) {
    const value = expected[index] ?? NaN;
    if (Number.isNaN(value)) expect(actual[index], `entry ${index}`).toBeNaN();
    else
      expect(Math.abs(actual[index] - value), `entry ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
  }
}

for (const fixture of fixtures) {
  it(`GPU and CPU match NumPy ${fixture.kernel}/${fixture.criterion}/${fixture.searchMethod}/inference=${fixture.inference}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const golden = fixture.searchMethod === 'golden-section';
    const outputs = ['coefficients', 'fitted', 'codes', 'iterationCount', 'selectedBandwidths'];
    if (fixture.inference)
      outputs.push(
        'inferenceSummary',
        'inferenceEffectiveParameters',
        'inferenceStandardErrors',
        'inferenceStatus'
      );
    for (const Operation of [
      GPUMultiscaleGeographicallyWeightedRegressionOperation,
      CPUMultiscaleGeographicallyWeightedRegressionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: Float32Array.from(fixture.positions)},
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          response: {format: 'float32', values: Float32Array.from(fixture.response)},
          parameters: {format: 'float32', values: Float32Array.from(fixture.parameters)},
          mask: {format: 'uint32', values: Uint32Array.from(fixture.mask)},
          bandwidthCandidates: {
            format: 'float32',
            values: Float32Array.from(fixture.bandwidthCandidates)
          }
        },
        operations: [
          new Operation({
            rowCount: fixture.response.length,
            predictorCount: 2,
            kernel: fixture.kernel as 'gaussian' | 'bisquare',
            maximumIterations: 64,
            positions: 'positions',
            predictors: 'predictors',
            response: 'response',
            parameters: 'parameters',
            mask: 'mask',
            bandwidthCandidates: 'bandwidthCandidates',
            bandwidthCandidateCount: fixture.bandwidthCandidateCount,
            bandwidthSelection: {
              criterion: fixture.criterion as 'AICc' | 'leave-one-out',
              searchMethod: fixture.searchMethod as 'finite-grid' | 'golden-section',
              maximumSearchIterations: golden ? 12 : undefined
            },
            inference: fixture.inference
              ? {
                  output: {
                    summary: 'inferenceSummary',
                    effectiveParameters: 'inferenceEffectiveParameters',
                    standardErrors: 'inferenceStandardErrors',
                    status: 'inferenceStatus'
                  }
                }
              : undefined,
            output: {
              coefficients: 'coefficients',
              fitted: 'fitted',
              selectedBandwidths: 'selectedBandwidths',
              status: {codes: 'codes', iterationCount: 'iterationCount'}
            }
          })
        ],
        outputs
      });
      expect(Array.from(result['codes'])).toEqual(fixture.mask.map(keep => (keep ? 0 : 1)));
      for (const port of ['coefficients', 'fitted', 'selectedBandwidths'] as const)
        expectClose(result[port], fixture.expected[port], golden ? 2e-2 : 5e-3);
      if (fixture.inference) {
        expect(result['inferenceStatus'][0]).toBe(0);
        for (const port of [
          'inferenceSummary',
          'inferenceEffectiveParameters',
          'inferenceStandardErrors'
        ] as const)
          expectClose(result[port], fixture.expected[port]!, 2e-2);
        expect(result['inferenceSummary'][0]).toBeCloseTo(
          Array.from(result['inferenceEffectiveParameters']).reduce((sum, value) => sum + value, 0),
          4
        );
      }
    }
  });
}

for (const scene of [
  'excluded',
  'invalid-bounds',
  'aicc-saturated',
  'finite-tie',
  'golden-tie'
] as const) {
  it(`MGWR AICc selection handles ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const golden = scene === 'invalid-bounds' || scene === 'golden-tie';
    const count = 8;
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
              values:
                scene === 'aicc-saturated'
                  ? Float32Array.from({length: count * 2}, (_, index) =>
                      index % 2 ? 0 : index * 100
                    )
                  : new Float32Array(count * 2)
            },
            predictors: {
              format: 'float32',
              values: Float32Array.from({length: count}, (_, index) => (index % 2 ? -1 : 1))
            },
            response: {format: 'float32', values: Float32Array.of(1, 2, 4, 3, 2, 1, 5, 3)},
            parameters: {format: 'float32', values: Float32Array.of(1e-3, 2, 2)},
            mask: {
              format: 'uint32',
              values: new Uint32Array(count).fill(scene === 'excluded' ? 0 : 1)
            },
            bandwidthCandidates: {
              format: 'float32',
              values:
                scene === 'invalid-bounds'
                  ? Float32Array.of(-1, 10, 20, 2)
                  : Float32Array.of(2, 10, 2, 20)
            }
          },
          operations: [
            new Operation({
              rowCount: count,
              predictorCount: 1,
              kernel: 'bisquare',
              maximumIterations: 4,
              positions: 'positions',
              predictors: 'predictors',
              response: 'response',
              parameters: 'parameters',
              mask: 'mask',
              bandwidthCandidates: 'bandwidthCandidates',
              bandwidthCandidateCount: 2,
              bandwidthSelection: {
                criterion: 'AICc',
                searchMethod: golden ? 'golden-section' : 'finite-grid',
                maximumSearchIterations: golden ? 4 : undefined
              },
              output: {
                coefficients: 'coefficients',
                fitted: 'fitted',
                selectedBandwidths: 'selectedBandwidths',
                status: {codes: 'codes', iterationCount: 'iterationCount'}
              }
            })
          ],
          outputs: ['coefficients', 'fitted', 'selectedBandwidths', 'codes', 'iterationCount']
        })
      );
    }
    for (const result of results) {
      expect(Array.from(result['codes'])).toEqual(
        new Array(count).fill(scene === 'excluded' ? 1 : scene.endsWith('tie') ? 0 : 2)
      );
      if (scene.endsWith('tie'))
        expectClose(
          result['selectedBandwidths'],
          golden ? [2 + 0.38197 * 8, 2 + 0.38197 * 18] : [2, 2],
          1e-6
        );
    }
    for (const port of ['coefficients', 'fitted', 'selectedBandwidths', 'codes', 'iterationCount'])
      expectClose(results[0][port], results[1][port], 5e-3);
  });
}

it('MGWR AICc golden search respects leading/trailing view padding on indexed and dense paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = fixtures.find(
    scene =>
      scene.kernel === 'bisquare' &&
      scene.criterion === 'AICc' &&
      scene.searchMethod === 'golden-section' &&
      !scene.inference
  )!;
  for (const indexGridSize of [false, [3, 3]] as const) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const padding = 64; // Aligned storage offsets; hostile finite values outside every view.
    const sentinel = -12345;
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const packed =
        values instanceof Uint32Array
          ? new Uint32Array(padding + values.length + padding).fill(0xffffffff)
          : new Float32Array(padding + values.length + padding).fill(sentinel);
      packed.set(values, padding);
      const buffer = device.createBuffer({
        data: packed,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const whole = importGraphBuffer(graph, name, buffer, format);
      return {
        buffer,
        view: graph.createDataView(whole.buffer, {
          format,
          byteOffset: padding * 4,
          length: values.length / (format === 'float32x2' ? 2 : 1)
        })
      };
    };
    const coefficients = input('coefficients', 'float32', new Float32Array(75).fill(sentinel));
    const fitted = input('fitted', 'float32', new Float32Array(25).fill(sentinel));
    const selected = input('selected', 'float32', new Float32Array(3).fill(sentinel));
    const codes = input('codes', 'uint32', new Uint32Array(25));
    const iterations = input('iterations', 'uint32', new Uint32Array(1));
    graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        positions: input('positions', 'float32x2', Float32Array.from(fixture.positions)).view,
        predictors: input('predictors', 'float32', Float32Array.from(fixture.predictors)).view,
        response: input('response', 'float32', Float32Array.from(fixture.response)).view,
        parameters: input('parameters', 'float32', Float32Array.from(fixture.parameters)).view,
        mask: input('mask', 'uint32', Uint32Array.from(fixture.mask)).view,
        bandwidthCandidates: input(
          'candidates',
          'float32',
          Float32Array.from(fixture.bandwidthCandidates)
        ).view,
        bandwidthCandidateCount: 2,
        predictorCount: 2,
        kernel: 'bisquare',
        maximumIterations: 32,
        indexGridSize,
        bandwidthSelection: {
          criterion: 'AICc',
          searchMethod: 'golden-section',
          maximumSearchIterations: 12
        },
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          selectedBandwidths: selected.view,
          status: {codes: codes.view, iterationCount: iterations.view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const [output, expected] of [
        [coefficients, fixture.expected.coefficients],
        [fitted, fixture.expected.fitted],
        [selected, fixture.expected.selectedBandwidths]
      ] as const) {
        const values = await readFloat32(output.buffer);
        expect(values.slice(0, padding)).toEqual(new Array(padding).fill(sentinel));
        expect(values.slice(padding + expected.length)).toEqual(new Array(padding).fill(sentinel));
        expectClose(values.slice(padding, padding + expected.length), expected, 2e-2);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
