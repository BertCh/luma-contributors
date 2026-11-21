// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {applyFinalGeneralizedCase, type FinalGeneralizedCase} from './local-regression-w5-fixtures';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  GENERALIZED_AUDIT_MODELS,
  createGeneralizedAuditFixture,
  applyGeneralizedAuditReviewCase,
  type GeneralizedAuditReviewCase,
  type GeneralizedAuditModel
} from './generalized-gwr-audit-fixtures';

async function runFit(
  model: GeneralizedAuditModel,
  options: {
    rescaled?: boolean;
    indexed?: boolean;
    predictExcluded?: boolean;
    queryOffset?: number;
    criterion?: 'aicc' | 'loo-deviance';
    reviewCase?: GeneralizedAuditReviewCase;
    finalCase?: FinalGeneralizedCase;
    leaveOneOut?: boolean;
  } = {}
) {
  const fixture = createGeneralizedAuditFixture(model, options.rescaled);
  applyGeneralizedAuditReviewCase(fixture, options.reviewCase);
  applyFinalGeneralizedCase(fixture, options.finalCase);
  if (options.queryOffset !== undefined) fixture.offset[12] = options.queryOffset;
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(
      format === 'uint32' ? new Uint32Array(length).fill(999) : new Float32Array(length).fill(-999)
    );
    buffers.push(buffer);
    return {
      buffer,
      format,
      view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)
    };
  };
  const outputs = {
    coefficients: output('float32', 28),
    fitted: output('float32', 14),
    codes: output('uint32', 14),
    iterationCount: output('uint32', 14),
    standardErrors: output('float32', 28),
    influence: output('float32', 14)
  };
  const selection = {
    scores: output('float32', 2),
    evaluatedBandwidths: output('float32', 2),
    bandwidth: output('float32', 1),
    status: output('uint32', 1),
    iterationCount: output('uint32', 1),
    converged: output('uint32', 1),
    iterationLimitReached: output('uint32', 1)
  };
  graph.add(
    new GPUGeneralizedGeographicallyWeightedRegression({
      family: model.family,
      kernel: model.kernel,
      bandwidthMode: model.bandwidthMode,
      positions: input('float32x2', fixture.positions),
      predictors: input('float32', fixture.predictors),
      predictorCount: 1,
      response: input('float32', fixture.response),
      offset: input('float32', fixture.offset),
      mask: input('uint32', fixture.mask),
      parameters: input('float32', fixture.parameters),
      predictExcluded: options.predictExcluded ?? true,
      maximumIterations: 64,
      leaveOneOut: options.leaveOneOut,
      // Empty cells and tied distances exercise the shared training index as well as dense fits.
      indexGridSize: options.indexed ? [8, 4] : false,
      gaussianCutoff: options.indexed ? 4 : undefined,
      inference: {standardErrors: outputs.standardErrors.view, influence: outputs.influence.view},
      bandwidthSelection: options.criterion
        ? {
            criterion: options.criterion,
            bandwidths: options.reviewCase === 'large-counts' ? [10, 0.5] : [2.5, 3],
            output: {
              scores: selection.scores.view,
              evaluatedBandwidths: selection.evaluatedBandwidths.view,
              bandwidth: selection.bandwidth.view,
              status: selection.status.view
            },
            status: {
              iterationCount: selection.iterationCount.view,
              converged: selection.converged.view,
              iterationLimitReached: selection.iterationLimitReached.view
            }
          }
        : undefined,
      output: {
        coefficients: outputs.coefficients.view,
        fitted: outputs.fitted.view,
        status: {codes: outputs.codes.view, iterationCount: outputs.iterationCount.view}
      }
    })
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  const read = async (port: ReturnType<typeof output>) =>
    Array.from(
      port.format === 'uint32'
        ? await readUint32(port.buffer, port.view.length)
        : await readFloat32(port.buffer, port.view.length)
    );
  try {
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    return {
      fit: Object.fromEntries(
        await Promise.all(
          Object.entries(outputs).map(async ([name, port]) => [name, await read(port)])
        )
      ) as typeof fixture.expected,
      selection: Object.fromEntries(
        await Promise.all(
          Object.entries(selection).map(async ([name, port]) => [name, await read(port)])
        )
      ) as Record<keyof typeof selection, number[]>
    };
  } finally {
    compiled?.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

it('W5-51 heterogeneous Poisson exposures converge in both units on WebGPU', async () => {
  for (const indexed of [false, true]) {
    for (const [finalCase, intercept] of [
      ['feasible-exposure', -5],
      ['shifted-feasible-exposure', -1]
    ] as const) {
      const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {finalCase, indexed});
      expect(fit.codes).toEqual(Array(14).fill(0));
      for (let row = 0; row < 14; row++) {
        expect(Number.isFinite(fit.coefficients[2 * row])).toBe(true);
        expect(Number.isFinite(fit.coefficients[2 * row + 1])).toBe(true);
        expect(fit.coefficients[2 * row]).toBeCloseTo(intercept, 4);
        expect(fit.coefficients[2 * row + 1]).toBeCloseTo(-5, 4);
        expect(Number.isFinite(fit.fitted[row])).toBe(true);
        if (row === 0) expect(fit.fitted[row] / Math.exp(-23)).toBeCloseTo(1, 4);
        else expect(fit.fitted[row]).toBeCloseTo(1, 5);
      }
    }
  }
});

it('W5-52 eligible LOO extrapolation and bandwidth selection use training anchors on WebGPU', async () => {
  for (const indexed of [false, true]) {
    const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {
      finalCase: 'eligible-extrapolation',
      leaveOneOut: true,
      indexed
    });
    expect(fit.codes[12]).toBe(0);
    expect(Number.isFinite(fit.fitted[12])).toBe(true);
    expect(fit.fitted[12]).toBeCloseTo(2.90666e-6, 9);
    const {selection} = await runFit(GENERALIZED_AUDIT_MODELS[0], {
      finalCase: 'eligible-extrapolation',
      criterion: 'loo-deviance',
      indexed
    });
    for (const [index, expected] of [3.2169481, 3.20898537].entries()) {
      expect(Number.isFinite(selection.scores[index])).toBe(true);
      expect(selection.scores[index]).toBeCloseTo(expected, 4);
    }
    expect(selection.bandwidth).toEqual([3]);
  }
});

it('B1 fractional exposure-unit shifts retain feasible rounded links on WebGPU', async () => {
  for (const indexed of [false, true]) {
    for (const [finalCase, shift] of [
      ['fractional-exposure', 0],
      ['shifted-fractional-exposure', 2.004]
    ] as const) {
      const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {finalCase, indexed});
      expect(fit.codes.slice(0, 6)).toEqual(Array(6).fill(0));
      expect(Number.isFinite(fit.coefficients[0])).toBe(true);
      expect(fit.coefficients[0]).toBeCloseTo(-23 - shift, 4);
      expect(fit.coefficients[1]).toBeCloseTo(-10, 4);
      expect(Number.isFinite(fit.fitted[0])).toBe(true);
      expect(fit.fitted[0] / Math.exp(-23)).toBeCloseTo(1, 4);
    }
  }
});

it('R1 Poisson LOO distinguishes small deviances of large counts on WebGPU', async () => {
  const {selection} = await runFit(GENERALIZED_AUDIT_MODELS[0], {
    criterion: 'loo-deviance',
    reviewCase: 'large-counts'
  });
  // Float32 fitted means near 10000 resolve about .001; 1e-5 score error leaves the winner clear.
  expect(selection.scores[0]).toBeCloseTo(0.000754351, 5);
  expect(selection.scores[1]).toBeCloseTo(0.000374729, 5);
  expect(selection.bandwidth).toEqual([0.5]);
});

it('R3 changing Poisson exposure units shifts only the intercept on WebGPU', async () => {
  for (const [reviewCase, shift] of [
    ['offset-shift', -14],
    ['exposure-shift-up', 100],
    ['exposure-shift-down', -100]
  ] as const) {
    const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {reviewCase});
    expect(fit.codes).toEqual(Array(14).fill(0));
    const {expected} = createGeneralizedAuditFixture(GENERALIZED_AUDIT_MODELS[0]);
    for (let row = 0; row < 14; row++) {
      expect(fit.coefficients[2 * row]).toBeCloseTo(expected.coefficients[2 * row] - shift, 4);
      expect(fit.coefficients[2 * row + 1]).toBeCloseTo(expected.coefficients[2 * row + 1], 4);
      expect(fit.fitted[row]).toBeCloseTo(expected.fitted[row], 4);
    }
  }
});

it('R4 held-out extrapolation preserves the full-rank local training fit on WebGPU', async () => {
  for (const indexed of [false, true]) {
    const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {reviewCase: 'extrapolation', indexed});
    expect(fit.codes.slice(12)).toEqual([0, 0]);
    for (const row of [12, 13]) {
      expect(fit.coefficients[2 * row]).toBeCloseTo(0.5547189562170504, 5);
      expect(fit.coefficients[2 * row + 1]).toBeCloseTo(-0.013428224342895145, 5);
      expect(fit.influence[row]).toBe(0);
    }
    expect(fit.fitted[12]).toBeCloseTo(0.00000290666, 9);
    expect(fit.fitted[13]).toBeCloseTo(2.5, 5);
  }
});

it('R5 a singular first information matrix publishes zero completed steps on WebGPU', async () => {
  const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {reviewCase: 'singular'});
  expect(fit.codes).toEqual(Array(14).fill(2));
  expect(fit.iterationCount).toEqual(Array(14).fill(0));
});

it('R9 logistic query offset 100 publishes a valid probability while Poisson overflow is refused', async () => {
  const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[4], {queryOffset: 100});
  expect(fit.codes[12]).toBe(0);
  expect(fit.fitted[12]).toBe(1);
  expect(fit.influence[12]).toBe(0);
  expect(fit.coefficients[24]).toBeCloseTo(0, 5);
  expect(fit.coefficients[25]).toBeCloseTo(-0.25, 5);
});

it.each(
  GENERALIZED_AUDIT_MODELS
)('$family/$kernel/$bandwidthMode agrees with independent MLE, covariance and hat diagonal literals on every output', async model => {
  for (const options of [{}, {rescaled: true, indexed: true}, {predictExcluded: false}]) {
    const {fit: actual} = await runFit(model, options);
    const {expected} = createGeneralizedAuditFixture(
      model,
      'rescaled' in options && options.rescaled
    );
    if ('predictExcluded' in options) {
      expected.codes.splice(12, 2, 1, 1);
      expected.iterationCount.splice(12, 2, 0, 0);
      for (const name of ['coefficients', 'standardErrors'] as const) expected[name].fill(NaN, 24);
      for (const name of ['fitted', 'influence'] as const) expected[name].fill(NaN, 12);
    }
    expect(actual.codes).toEqual(expected.codes);
    expect(actual.iterationCount).toEqual(expected.iterationCount);
    for (const name of ['coefficients', 'fitted', 'standardErrors', 'influence'] as const) {
      expect(actual[name]).toHaveLength(expected[name].length);
      actual[name].forEach((value, index) => {
        if (Number.isNaN(expected[name][index])) expect(value).toBeNaN();
        else {
          expect(Number.isFinite(value), `${name}[${index}]`).toBe(true);
          // Short float32 reductions and Cholesky/IRLS: 1e-4, versus <3e-7 in Node interpretation.
          expect(Math.abs(value - expected[name][index]), `${name}[${index}]`).toBeLessThan(1e-4);
        }
      });
    }
  }
});

it('Poisson prediction overflow preserves completed steps and suppresses every dependent output', async () => {
  for (const indexed of [false, true]) {
    const {fit} = await runFit(GENERALIZED_AUDIT_MODELS[0], {queryOffset: 100, indexed});
    expect(fit.codes).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0]);
    expect(fit.iterationCount).toEqual(Array(14).fill(5));
    expect(fit.coefficients.slice(24, 26).every(Number.isNaN)).toBe(true);
    expect(fit.standardErrors.slice(24, 26).every(Number.isNaN)).toBe(true);
    expect(fit.fitted[12]).toBeNaN();
    expect(fit.influence[12]).toBeNaN();
  }
});

it.each(
  GENERALIZED_AUDIT_MODELS.filter(model => model.bandwidthMode === 'fixed')
)('$family/$kernel AICc and LOO score outputs agree with independent weighted category fits', async model => {
  for (const criterion of ['aicc', 'loo-deviance'] as const) {
    const {selection} = await runFit(model, {criterion, indexed: true});
    const expected = criterion === 'aicc' ? model.aicc : model.loo;
    selection.scores.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[index])).toBeLessThan(1e-4);
    });
    expect(selection.bandwidth).toEqual([expected[0] < expected[1] ? 2.5 : 3]);
    expect(selection.evaluatedBandwidths).toEqual([2.5, 3]);
    expect(selection.status).toEqual([0]);
    expect(selection.iterationCount).toEqual([1]);
    expect(selection.converged).toEqual([1]);
    expect(selection.iterationLimitReached).toEqual([0]);
  }
});
