// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import reference from '../../oracles/fixtures/adaptive-generalized-gwr.json';
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

it('fixed and adaptive local GLMs agree with independent statsmodels compositions, including distance ties', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const results = reference.models.map((model, index) => {
    const coefficients = output('float32', 48);
    const fitted = output('float32', 24);
    const codes = output('uint32', 24);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        id: `adaptive-oracle-${index}`,
        family: model.family === 'poisson' ? 'poisson' : 'logistic',
        kernel: model.kernel === 'gaussian' ? 'gaussian' : 'bisquare',
        bandwidthMode: model.mode === 'adaptive' ? 'adaptive' : undefined,
        positions: input('float32x2', reference.positions),
        predictors: input('float32', reference.predictors),
        response: input('float32', model.response),
        offset: input('float32', model.offset),
        mask: input('uint32', reference.mask),
        predictorCount: 1,
        parameters: input('float32', [model.mode === 'adaptive' ? 16 : 3, 1e-5]),
        maximumIterations: 64,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: {codes: codes.view, iterationCount: output('uint32', 24).view}
        }
      })
    );
    return {model, coefficients, fitted, codes};
  });
  try {
    context.run();
    const surfaces: number[][] = [];
    for (const {model, coefficients, fitted, codes} of results) {
      expect(Array.from(await readUint32(codes.buffer, 24))).toEqual(
        reference.mask.map(value => (value ? 0 : 1))
      );
      const actual = await readFloat32(coefficients.buffer, 48);
      surfaces.push(actual);
      model.coefficients.forEach((value, index) =>
        value === null
          ? expect(actual[index]).toBeNaN()
          : expect(Math.abs(actual[index] - value)).toBeLessThan(reference.provenance.tolerance)
      );
      const predictions = await readFloat32(fitted.buffer, 24);
      model.fitted.forEach((value, index) =>
        value === null
          ? expect(predictions[index]).toBeNaN()
          : expect(Math.abs(predictions[index] - value)).toBeLessThan(
              reference.provenance.tolerance
            )
      );
    }
    expect(Math.abs(surfaces[0][0] - surfaces[1][0])).toBeGreaterThan(0.001);
  } finally {
    context.destroy();
  }
});

it('adaptive GLMs report masked/nonfinite exclusions, invalid k, insufficient and zero-distance neighborhoods, boundary responses and iteration limits', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const cases = [
    {name: 'too-many', bandwidth: 5, expected: [2, 2, 2, 2]},
    {name: 'self-only', bandwidth: 1, expected: [2, 2, 2, 2]},
    {name: 'invalid-response', bandwidth: 4, response: [2, 1, 0, 1], expected: [1, 2, 2, 2]},
    {
      name: 'poisson-boundary',
      family: 'poisson',
      bandwidth: 4,
      response: [0, 0, 0, 0],
      expected: [5, 5, 5, 5]
    },
    {name: 'fractional', bandwidth: 2.5, expected: [4, 4, 4, 4]},
    {
      name: 'zero-distance',
      bandwidth: 4,
      positions: [0, 0, 0, 0, 0, 0, 0, 0],
      expected: [2, 2, 2, 2]
    },
    {name: 'boundary', bandwidth: 4, response: [0, 0, 0, 0], expected: [5, 5, 5, 5]},
    {name: 'one-class', bandwidth: 4, response: [1, 1, 1, 1], expected: [5, 5, 5, 5]},
    {name: 'excluded', bandwidth: 3, predictors: [NaN, 1, 2, 3], expected: [1, 0, 0, 0]},
    {name: 'masked', bandwidth: 4, mask: [0, 1, 1, 1], expected: [1, 2, 2, 2]},
    {name: 'limit', bandwidth: 4, maximumIterations: 1, expected: [3, 3, 3, 3]}
  ];
  const results = cases.map(testCase => {
    const codes = output('uint32', 4);
    const coefficients = output('float32', 8);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        id: testCase.name,
        family: testCase.family === 'poisson' ? 'poisson' : 'logistic',
        bandwidthMode: 'adaptive',
        positions: input('float32x2', testCase.positions ?? [0, 0, 1, 0, 2, 0, 3, 0]),
        predictors: input('float32', testCase.predictors ?? [0, 1, 2, 3]),
        predictorCount: 1,
        response: input('float32', testCase.response ?? [0, 1, 0, 1]),
        mask: input('uint32', testCase.mask ?? [1, 1, 1, 1]),
        parameters: input('float32', [testCase.bandwidth, 1e-5]),
        maximumIterations: testCase.maximumIterations ?? 64,
        output: {
          coefficients: coefficients.view,
          fitted: output('float32', 4).view,
          status: {codes: codes.view, iterationCount: output('uint32', 4).view}
        }
      })
    );
    return {testCase, codes, coefficients};
  });
  try {
    context.run();
    for (const {testCase, codes, coefficients} of results) {
      expect(Array.from(await readUint32(codes.buffer, 4)), testCase.name).toEqual(
        testCase.expected
      );
      const actual = await readFloat32(coefficients.buffer, 8);
      testCase.expected.forEach((code, index) => {
        if (code !== 0) {
          expect(actual[index * 2]).toBeNaN();
          expect(actual[index * 2 + 1]).toBeNaN();
        }
      });
    }
  } finally {
    context.destroy();
  }
});
