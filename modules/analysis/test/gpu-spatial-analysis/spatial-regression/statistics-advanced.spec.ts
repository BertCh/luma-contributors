// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from '../../oracles/fixtures/statistics-advanced.json';
async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-4) {
  expected.forEach((value, index) =>
    Number.isNaN(value)
      ? expect(actual[index]).toBeNaN()
      : expect(
          Math.abs(actual[index] - value),
          `element ${index}: ${actual[index]} vs ${value}`
        ).toBeLessThan(tolerance)
  );
}

function uploadWeights(
  input: Awaited<ReturnType<typeof setup>>['input'],
  weights: {offsets: number[]; neighbors: number[]; weights: number[]}
) {
  return {
    offsets: input('uint32', weights.offsets),
    neighbors: input('uint32', weights.neighbors),
    weights: input('float32', weights.weights)
  };
}

for (const data of reference.panels)
  for (const effects of ['random', 'unit'] as const)
    it(`direct spreg panel ${effects} ML and conditional covariance seed${data.seed}`, async () => {
      const context = await setup();
      const {graph, input, output} = context;
      try {
        const coefficients = output('float32', 5),
          covariance = output('float32', 16),
          summary = output('float32', 6),
          status = output('uint32', 1),
          fitted = output('float32', 48),
          residuals = output('float32', 48),
          hausman = output('float32', 3),
          hausmanStatus = output('uint32', 1);
        graph.add(
          new GPUSpatialPanelLagMaximumLikelihood({
            weights: uploadWeights(input, data.weights),
            periodCount: data.periods,
            predictorCount: 2,
            predictors: input('float32', data.predictors),
            response: input('float32', data.response),
            effects,
            maximumIterations: 128,
            tolerance: 2e-5,
            output: {
              coefficients: coefficients.view,
              covariance: covariance.view,
              summary: summary.view,
              status: status.view,
              fitted: fitted.view,
              residuals: residuals.view
            },
            ...(effects === 'random'
              ? {hausman: {tests: hausman.view, status: hausmanStatus.view}}
              : {})
          })
        );
        context.run();
        expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
        const estimates = await readFloat32(coefficients.buffer);
        const values = await readFloat32(covariance.buffer);
        close(
          effects === 'random' ? estimates : estimates.slice(1, 4),
          data[effects].coefficients,
          0.003
        );
        close(
          effects === 'random'
            ? values
            : [...values.slice(5, 8), ...values.slice(9, 12), ...values.slice(13, 16)],
          data[effects].covariance,
          0.003
        );
        close(await readFloat32(residuals.buffer), data[effects].residuals, 0.003);
        close(await readFloat32(fitted.buffer), data[effects].fitted, 0.003);
        expect((await readFloat32(summary.buffer))[0]).toBeCloseTo(data[effects].variance, 3);
        expect((await readFloat32(summary.buffer))[1]).toBeCloseTo(data[effects].logLikelihood, 2);
        if (effects === 'random') {
          const test = await readFloat32(hausman.buffer);
          if (data.hausman[0] >= 0) {
            expect(Array.from(await readUint32(hausmanStatus.buffer))).toEqual([0]);
            close(test, data.hausman, 0.025);
          } else {
            expect(Array.from(await readUint32(hausmanStatus.buffer))).toEqual([2]);
            expect(test[0]).toBeNaN();
            expect(test[2]).toBeNaN();
          }
        }
      } finally {
        context.destroy();
      }
    });

for (const convention of ['residual-df', 'smoother-df'] as const)
  it(`direct mgwr full backfit covariance, effective parameters and ${convention} variance`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.multiscale;
    try {
      const coefficients = output('float32', 72),
        fitted = output('float32', 24),
        codes = output('uint32', 24),
        iterationCount = output('uint32', 1),
        standardErrors = output('float32', 72),
        effective = output('float32', 3),
        summary = output('float32', 4),
        status = output('uint32', 1);
      graph.add(
        new GPUMultiscaleGeographicallyWeightedRegression({
          positions: input('float32x2', data.positions),
          predictors: input('float32', data.predictors),
          predictorCount: 2,
          response: input('float32', data.response),
          parameters: input('float32', [1e-6, ...data.bandwidths]),
          kernel: 'gaussian',
          maximumIterations: 128,
          inference: {
            varianceConvention: convention,
            output: {
              standardErrors: standardErrors.view,
              effectiveParameters: effective.view,
              summary: summary.view,
              status: status.view
            }
          },
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: iterationCount.view}
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(codes.buffer))).toEqual(Array(24).fill(0));
      expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
      close(await readFloat32(coefficients.buffer), data.coefficients, 0.003);
      close(await readFloat32(effective.buffer), data.effectiveParameters, 0.003);
      const expected = data.summary.slice();
      if (convention === 'smoother-df') {
        expected[3] = 24 - 2 * expected[0] + expected[1];
        expected[2] *= data.summary[3] / expected[3];
      }
      close(await readFloat32(summary.buffer), expected, 0.003);
      close(
        await readFloat32(standardErrors.buffer),
        data.standardErrors.map(value => value * Math.sqrt(expected[2] / data.summary[2])),
        0.003
      );
    } finally {
      context.destroy();
    }
  });

for (const failure of [
  'missing',
  'singular',
  'outer-exhaustion',
  'search-exhaustion',
  'invalid-weights',
  'within-singular',
  'zero-variance'
] as const)
  it(`random panel exposes ${failure}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.panels[0];
    try {
      const predictors = data.predictors.slice();
      if (failure === 'singular')
        for (let row = 0; row < 48; row++) predictors[row * 2 + 1] = predictors[row * 2];
      if (failure === 'within-singular')
        for (let row = 0; row < 48; row++) predictors[row * 2] = row % 8;
      const response = data.response.map(value => (failure === 'zero-variance' ? 0 : value));
      if (failure === 'missing') response[3] = NaN;
      const coefficients = output('float32', 5),
        covariance = output('float32', 16),
        summary = output('float32', 6),
        status = output('uint32', 1),
        fitted = output('float32', 48),
        residuals = output('float32', 48);
      graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          weights: uploadWeights(input, {
            ...data.weights,
            weights: data.weights.weights.map(value =>
              failure === 'invalid-weights' ? value * 2 : value
            )
          }),
          effects: failure === 'within-singular' ? 'unit' : 'random',
          periodCount: 6,
          predictorCount: 2,
          predictors: input('float32', predictors),
          response: input('float32', response),
          maximumIterations: failure === 'outer-exhaustion' ? 1 : 64,
          maximumSearchIterations: failure === 'search-exhaustion' ? 1 : 32,
          output: {
            coefficients: coefficients.view,
            covariance: covariance.view,
            summary: summary.view,
            status: status.view,
            fitted: fitted.view,
            residuals: residuals.view
          }
        })
      );
      context.run();
      expect((await readUint32(status.buffer))[0]).toEqual(
        failure === 'missing' || failure === 'invalid-weights'
          ? 1
          : failure === 'singular' || failure === 'within-singular' || failure === 'zero-variance'
            ? 2
            : 3
      );
      expect((await readFloat32(covariance.buffer)).every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });

for (const failure of ['singular', 'exhaustion'] as const)
  it(`full MGWR inference rejects ${failure} instead of publishing conditional errors`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    const data = reference.multiscale;
    try {
      const predictors = data.predictors.map((value, index) =>
        failure === 'singular' && index % 2 === 1 ? 0 : value
      );
      const coefficients = output('float32', 72),
        fitted = output('float32', 24),
        codes = output('uint32', 24),
        iterationCount = output('uint32', 1),
        standardErrors = output('float32', 72),
        effective = output('float32', 3),
        summary = output('float32', 4),
        status = output('uint32', 1);
      graph.add(
        new GPUMultiscaleGeographicallyWeightedRegression({
          positions: input('float32x2', data.positions),
          predictors: input('float32', predictors),
          predictorCount: 2,
          response: input('float32', data.response),
          parameters: input('float32', [1e-6, ...data.bandwidths]),
          maximumIterations: failure === 'exhaustion' ? 1 : 16,
          inference: {
            output: {
              standardErrors: standardErrors.view,
              effectiveParameters: effective.view,
              summary: summary.view,
              status: status.view
            }
          },
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: iterationCount.view}
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(codes.buffer))).toEqual(
        Array(24).fill(failure === 'singular' ? 2 : 3)
      );
      expect(Array.from(await readUint32(status.buffer))).toEqual([1]);
      expect((await readFloat32(standardErrors.buffer)).every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });

it('finite-grid partial-response LOO selection propagates full MGWR uncertainty against actual mgwr', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const data = reference.multiscale;
  const expected = reference.candidateMultiscale;
  try {
    const coefficients = output('float32', 72),
      fitted = output('float32', 24),
      codes = output('uint32', 24),
      iterationCount = output('uint32', 1),
      standardErrors = output('float32', 72),
      effective = output('float32', 3),
      summary = output('float32', 4),
      status = output('uint32', 1),
      bandwidths = output('float32', 3);
    graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        positions: input('float32x2', data.positions),
        predictors: input('float32', data.predictors),
        predictorCount: 2,
        response: input('float32', data.response),
        parameters: input('float32', [1e-6, ...data.bandwidths]),
        kernel: 'gaussian',
        maximumIterations: 128,
        bandwidthCandidates: input('float32', expected.candidates),
        bandwidthCandidateCount: 3,
        inference: {
          output: {
            standardErrors: standardErrors.view,
            effectiveParameters: effective.view,
            summary: summary.view,
            status: status.view
          }
        },
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          selectedBandwidths: bandwidths.view,
          status: {codes: codes.view, iterationCount: iterationCount.view}
        }
      })
    );
    context.run();
    expect(Array.from(await readUint32(codes.buffer))).toEqual(Array(24).fill(0));
    expect(Array.from(await readUint32(status.buffer))).toEqual([0]);
    close(await readFloat32(bandwidths.buffer), expected.bandwidths);
    close(await readFloat32(coefficients.buffer), expected.coefficients, 0.003);
    close(await readFloat32(effective.buffer), expected.effectiveParameters, 0.003);
    close(await readFloat32(standardErrors.buffer), expected.standardErrors, 0.003);
  } finally {
    context.destroy();
  }
});
