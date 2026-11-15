// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  GPU_ITERATIVE_REGRESSION_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from '../../oracles/fixtures/generalized-gwr-aicc.json';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const width = format === 'float32x2' ? 2 : 1;
    const padding = 3 * width;
    const data = [...new Array(padding).fill(12345), ...values, ...new Array(padding).fill(12345)];
    const buffer = createOutputBuffer(device, data.length);
    buffer.write(format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    buffers.push(buffer);
    const whole = importGraphBuffer(graph, `port-${serial++}`, buffer, format, data.length / width);
    const view = graph.createDataView(whole.buffer, {
      format,
      byteOffset: padding * 4,
      length: values.length / width
    });
    return {buffer, view, padding, length: values.length};
  };
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => port(format, values).view;
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    port(format, new Array(length).fill(0));
  const read = async (
    result: {buffer: Buffer; padding: number; length: number},
    integer = false
  ) => {
    const values = integer ? await readUint32(result.buffer) : await readFloat32(result.buffer);
    expect(Array.from(values).slice(0, result.padding)).toEqual(
      new Array(result.padding).fill(12345)
    );
    expect(Array.from(values).slice(result.padding + result.length)).toEqual(
      new Array(result.padding).fill(12345)
    );
    return Array.from(values).slice(result.padding, result.padding + result.length);
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    device,
    graph,
    input,
    output,
    port,
    read,
    run: () => {
      compiled ??= graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

it('fixed/adaptive Poisson/logistic AICc golden selection agrees with independent NumPy IRLS, with ties, masks, offsets and padded views', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const results = reference.models.map((model, modelIndex) => {
      const scores = output('float32', 24);
      const history = output('float32', 24);
      const bandwidth = output('float32', 1);
      const status = output('uint32', 1);
      const converged = output('uint32', 1);
      const iterationCount = output('uint32', 1);
      const iterationLimitReached = output('uint32', 1);
      const coefficients = output('float32', 96);
      const fitted = output('float32', 48);
      const codes = output('uint32', 48);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          id: `aicc-${modelIndex}`,
          family: model.family === 'poisson' ? 'poisson' : 'logistic',
          kernel: model.kernel === 'gaussian' ? 'gaussian' : 'bisquare',
          bandwidthMode: model.mode === 'fixed' ? 'fixed' : 'adaptive',
          // Exercise the shared grid, including empty cells; dense variants cover the other half.
          indexGridSize: modelIndex % 2 ? [16, 12] : undefined,
          positions: input('float32x2', reference.positions),
          predictors: input('float32', reference.predictors),
          predictorCount: 1,
          response: input('float32', model.response),
          offset: input('float32', model.offset),
          mask: input('uint32', reference.mask),
          parameters: input('float32', [1, 1e-6]),
          maximumIterations: 64,
          predictExcluded: true,
          bandwidthSelection: {
            method: 'golden-section',
            maximumIterations: 12,
            parameters: input('float32', model.bounds),
            output: {
              scores: scores.view,
              evaluatedBandwidths: history.view,
              bandwidth: bandwidth.view,
              status: status.view
            },
            status: {
              converged: converged.view,
              iterationCount: iterationCount.view,
              iterationLimitReached: iterationLimitReached.view
            }
          },
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: output('uint32', 48).view}
          }
        })
      );
      return {
        model,
        scores,
        history,
        bandwidth,
        status,
        converged,
        iterationCount,
        iterationLimitReached,
        coefficients,
        fitted,
        codes
      };
    });
    context.run();
    for (const result of results) {
      const {model} = result;
      expect(await read(result.status, true)).toEqual([0]);
      const iterations = (await read(result.iterationCount, true))[0];
      expect(iterations).toBeLessThanOrEqual(12);
      expect(iterations).toBeGreaterThan(0);
      const chosen = (await read(result.bandwidth))[0];
      if (model.mode === 'fixed') expect(Math.abs(chosen - model.bandwidth)).toBeLessThan(0.03);
      const scores = await read(result.scores);
      const history = await read(result.history);
      const selectedReference =
        model.mode === 'adaptive'
          ? model.candidates.find(candidate => candidate.bandwidth === chosen)
          : model;
      expect(selectedReference).toBeDefined();
      if (model.mode === 'adaptive') {
        // Integer scores/fits exhaust the full supplied interval, independently of GPU stopping.
        // Local golden search is not a global optimizer of this possibly non-unimodal curve.
        const evaluated = model.candidates.filter(candidate =>
          history.slice(0, 2 * iterations).includes(candidate.bandwidth)
        );
        const best = Math.min(...evaluated.map(candidate => candidate.score));
        expect(selectedReference!.score - best).toBeLessThan(0.006);
        for (let slot = 0; slot < 2 * iterations; slot++) {
          const expected = model.candidates.find(
            candidate => candidate.bandwidth === history[slot]
          );
          expect(expected).toBeDefined();
          expect(Math.abs(scores[slot] - expected!.score)).toBeLessThan(0.006);
        }
      } else {
        for (let slot = 0; slot < Math.min(2 * iterations, model.history.length); slot++) {
          expect(Math.abs(history[slot] - model.history[slot].bandwidth)).toBeLessThan(0.002);
          expect(Math.abs(scores[slot] - model.history[slot].score)).toBeLessThan(0.006);
        }
      }
      for (let slot = 2 * iterations; slot < 24; slot++) {
        expect(history[slot]).toBeNaN();
        expect(scores[slot]).toBeNaN();
      }
      expect(await read(result.converged, true)).toEqual([1]);
      expect(await read(result.iterationLimitReached, true)).toEqual([0]);
      expect(await read(result.codes, true)).toEqual(new Array(48).fill(0));
      const coefficients = await read(result.coefficients);
      const predictions = await read(result.fitted);
      coefficients.forEach((value, index) =>
        expect(Math.abs(value - selectedReference!.coefficients[index])).toBeLessThan(0.004)
      );
      predictions.forEach((value, index) =>
        expect(Math.abs(value - selectedReference!.fitted[index])).toBeLessThan(0.005)
      );
    }
  } finally {
    context.destroy();
  }
});

it('golden selection publishes the best candidate at its budget, rejects invalid bounds/empty training, and preserves exact grid ties', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const model = reference.models[0];
    const cases = [
      {
        name: 'budget',
        bounds: model.bounds,
        expectedStatus: 0,
        expectedIterations: 1,
        expectedLimit: 1
      },
      {
        name: 'reversed',
        bounds: [8, 2, 1e-4],
        expectedStatus: 1,
        expectedIterations: 0,
        expectedLimit: 0
      },
      {
        name: 'nonfinite',
        bounds: [2, NaN, 1e-4],
        expectedStatus: 1,
        expectedIterations: 0,
        expectedLimit: 0
      },
      {
        name: 'bad-tolerance',
        bounds: [2, 8, 0],
        expectedStatus: 1,
        expectedIterations: 0,
        expectedLimit: 0
      },
      {
        name: 'empty',
        bounds: model.bounds,
        mask: new Array(48).fill(0),
        expectedStatus: 1,
        expectedIterations: 1,
        expectedLimit: 1
      }
    ];
    const results = cases.map(testCase => {
      const scores = output('float32', 2);
      const bandwidth = output('float32', 1);
      const status = output('uint32', 1);
      const converged = output('uint32', 1);
      const iterationCount = output('uint32', 1);
      const iterationLimitReached = output('uint32', 1);
      const coefficients = output('float32', 96);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          id: testCase.name,
          family: 'poisson',
          positions: input('float32x2', reference.positions),
          predictors: input('float32', reference.predictors),
          predictorCount: 1,
          response: input('float32', model.response),
          offset: input('float32', model.offset),
          mask: input('uint32', testCase.mask ?? reference.mask),
          parameters: input('float32', [1, 1e-6]),
          maximumIterations: 64,
          bandwidthSelection: {
            method: 'golden-section',
            maximumIterations: 1,
            parameters: input('float32', testCase.bounds),
            output: {scores: scores.view, bandwidth: bandwidth.view, status: status.view},
            status: {
              converged: converged.view,
              iterationCount: iterationCount.view,
              iterationLimitReached: iterationLimitReached.view
            }
          },
          output: {
            coefficients: coefficients.view,
            fitted: output('float32', 48).view,
            status: {codes: output('uint32', 48).view, iterationCount: output('uint32', 48).view}
          }
        })
      );
      return {
        testCase,
        scores,
        bandwidth,
        status,
        converged,
        iterationCount,
        iterationLimitReached,
        coefficients
      };
    });
    const tied = output('float32', 1);
    const tiedScores = output('float32', 3);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        id: 'grid-ties',
        family: 'poisson',
        positions: input('float32x2', reference.positions),
        predictors: input('float32', reference.predictors),
        predictorCount: 1,
        response: input('float32', model.response),
        offset: input('float32', model.offset),
        mask: input('uint32', reference.mask),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 64,
        bandwidthSelection: {
          criterion: 'aicc',
          bandwidths: [model.bandwidth, model.bandwidth, model.bandwidth],
          output: {scores: tiedScores.view, bandwidth: tied.view, status: output('uint32', 1).view}
        },
        output: {
          coefficients: output('float32', 96).view,
          fitted: output('float32', 48).view,
          status: {codes: output('uint32', 48).view, iterationCount: output('uint32', 48).view}
        }
      })
    );
    context.run();
    for (const result of results) {
      expect(await read(result.status, true), result.testCase.name).toEqual([
        result.testCase.expectedStatus
      ]);
      expect(await read(result.iterationCount, true)).toEqual([result.testCase.expectedIterations]);
      expect(await read(result.converged, true)).toEqual([0]);
      expect(await read(result.iterationLimitReached, true)).toEqual([
        result.testCase.expectedLimit
      ]);
      if (result.testCase.expectedStatus) {
        expect((await read(result.bandwidth))[0]).toBeNaN();
        expect((await read(result.coefficients)).every(Number.isNaN)).toBe(true);
      } else {
        const scores = await read(result.scores);
        expect((await read(result.bandwidth))[0]).toBeCloseTo(
          model.history[scores[0] <= scores[1] ? 0 : 1].bandwidth,
          5
        );
        expect((await read(result.coefficients)).filter(Number.isFinite).length).toBeGreaterThan(0);
      }
    }
    expect((await read(tied))[0]).toBeCloseTo(model.bandwidth, 5);
    (await read(tiedScores)).forEach(value =>
      expect(Math.abs(value - model.score)).toBeLessThan(0.006)
    );
  } finally {
    context.destroy();
  }
});

it('local GLMs name complete/quasi separation, preserve boundary status, and publish the last finite nonseparated iterate', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const cases = [
      // A strict separating direction is (0,1); quasi cases have both classes on x=0.
      {
        name: 'complete',
        family: 'logistic',
        predictor: [-2, -1, 1, 2],
        response: [0, 0, 1, 1],
        expected: 6
      },
      {
        name: 'quasi',
        family: 'logistic',
        predictor: [-1, 0, 0, 1],
        response: [0, 0, 1, 1],
        expected: 6
      },
      // Poisson recession direction (0,-1) vanishes on positive-count rows and decreases zero rows.
      {
        name: 'poisson-quasi',
        family: 'poisson',
        predictor: [0, 0, 1, 2],
        response: [1, 2, 0, 0],
        expected: 6
      },
      {
        name: 'poisson-boundary',
        family: 'poisson',
        predictor: [-1, 0, 0, 1],
        response: [0, 0, 0, 0],
        expected: 5
      },
      {
        name: 'logistic-boundary',
        family: 'logistic',
        predictor: [-1, 0, 0, 1],
        response: [1, 1, 1, 1],
        expected: 5
      },
      // Each x has positive and zero observations: finite MLE, even at large slope/tight tolerance.
      {
        name: 'overlap',
        family: 'logistic',
        predictor: [-1, -1, 1, 1],
        response: [0, 1, 0, 1],
        expected: 0
      },
      {
        name: 'limited',
        family: 'poisson',
        predictor: [-1, 0, 0, 1],
        response: [1, 2, 1, 3],
        maximumIterations: 1,
        expected: 3
      }
    ];
    const results = cases.map(testCase => {
      const codes = output('uint32', 4);
      const coefficients = output('float32', 8);
      const fitted = output('float32', 4);
      graph.add(
        new GPUGeneralizedGeographicallyWeightedRegression({
          id: testCase.name,
          family: testCase.family === 'poisson' ? 'poisson' : 'logistic',
          positions: input('float32x2', new Array(8).fill(0)),
          predictors: input('float32', testCase.predictor),
          predictorCount: 1,
          response: input('float32', testCase.response),
          parameters: input('float32', [1, 1e-6]),
          maximumIterations: testCase.maximumIterations ?? 128,
          output: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            status: {codes: codes.view, iterationCount: output('uint32', 4).view}
          }
        })
      );
      return {testCase, codes, coefficients, fitted};
    });
    context.run();
    expect(GPU_ITERATIVE_REGRESSION_STATUS.SEPARATION).toBe(6);
    for (const result of results) {
      expect(await read(result.codes, true), result.testCase.name).toEqual(
        new Array(4).fill(result.testCase.expected)
      );
      const values = [...(await read(result.coefficients)), ...(await read(result.fitted))];
      expect(
        values.every(
          result.testCase.expected === 0 || result.testCase.expected === 3
            ? Number.isFinite
            : Number.isNaN
        )
      ).toBe(true);
    }
  } finally {
    context.destroy();
  }
});

it('AICc selection excludes held-out responses and resets histories/gates when per-frame bounds change', async () => {
  const context = await setup();
  const {graph, input, output, port, read} = context;
  try {
    const model = reference.models[0];
    const bounds = port('float32', [2, 8, 1e6]);
    const response = port('float32', model.response);
    const scores = output('float32', 8);
    const history = output('float32', 8);
    const bandwidth = output('float32', 1);
    const status = output('uint32', 1);
    const iterationCount = output('uint32', 1);
    const iterationLimitReached = output('uint32', 1);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'poisson',
        positions: input('float32x2', reference.positions),
        predictors: input('float32', reference.predictors),
        predictorCount: 1,
        response: response.view,
        offset: input('float32', model.offset),
        mask: input('uint32', reference.mask),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 64,
        indexGridSize: [16, 12],
        bandwidthSelection: {
          method: 'golden-section',
          maximumIterations: 4,
          parameters: bounds.view,
          output: {
            scores: scores.view,
            evaluatedBandwidths: history.view,
            bandwidth: bandwidth.view,
            status: status.view
          },
          status: {
            iterationCount: iterationCount.view,
            iterationLimitReached: iterationLimitReached.view
          }
        },
        output: {
          coefficients: output('float32', 96).view,
          fitted: output('float32', 48).view,
          status: {codes: output('uint32', 48).view, iterationCount: output('uint32', 48).view}
        }
      })
    );
    context.run();
    const firstScores = await read(scores);
    const firstBandwidth = await read(bandwidth);
    expect(await read(iterationCount, true)).toEqual([1]);
    response.buffer.write(
      Float32Array.from(model.response.map((value, row) => (reference.mask[row] ? value : 100000))),
      response.padding * 4
    );
    context.run();
    expect(await read(scores)).toEqual(firstScores);
    expect(await read(bandwidth)).toEqual(firstBandwidth);
    bounds.buffer.write(Float32Array.from([8, 2, 1e-4]), bounds.padding * 4);
    context.run();
    expect(await read(status, true)).toEqual([1]);
    expect(await read(iterationCount, true)).toEqual([0]);
    expect((await read(scores)).every(Number.isNaN)).toBe(true);
    expect((await read(history)).every(Number.isNaN)).toBe(true);
    bounds.buffer.write(Float32Array.from([2, 8, 1e-9]), bounds.padding * 4);
    context.run();
    expect(await read(status, true)).toEqual([0]);
    expect(await read(iterationCount, true)).toEqual([4]);
    expect(await read(iterationLimitReached, true)).toEqual([1]);
    expect((await read(scores)).every(Number.isFinite)).toBe(true);
  } finally {
    context.destroy();
  }
});

it('AICc rejects a saturated two-row model with a nonpositive correction denominator', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const scores = output('float32', 2);
    const status = output('uint32', 1);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'poisson',
        positions: input('float32x2', [0, 0, 1, 0]),
        predictors: input('float32', [0, 1]),
        predictorCount: 1,
        response: input('float32', [1, 2]),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 64,
        bandwidthSelection: {
          criterion: 'aicc',
          bandwidths: [1, 2],
          output: {scores: scores.view, bandwidth: output('float32', 1).view, status: status.view}
        },
        output: {
          coefficients: output('float32', 4).view,
          fitted: output('float32', 2).view,
          status: {codes: output('uint32', 2).view, iterationCount: output('uint32', 2).view}
        }
      })
    );
    context.run();
    expect(await read(status, true)).toEqual([1]);
    expect((await read(scores)).every(Number.isNaN)).toBe(true);
  } finally {
    context.destroy();
  }
});

it('does not certify the finite near-separated logistic MLE when tiny observations oppose the direction (review 1)', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const coefficients = output('float32', 8);
    const fitted = output('float32', 4);
    const codes = output('uint32', 4);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'logistic',
        positions: input('float32x2', new Array(8).fill(0)),
        predictors: input('float32', [-1, -1e-6, 1e-6, 1]),
        predictorCount: 1,
        response: input('float32', [0, 1, 0, 1]),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 128,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: {codes: codes.view, iterationCount: output('uint32', 4).view}
        }
      })
    );
    context.run();
    // By symmetry the intercept is zero. The strictly decreasing slope score is
    // 2 sigmoid(-slope) - 2 epsilon sigmoid(epsilon * slope); its unique root is the finite MLE.
    const epsilon = Math.fround(1e-6);
    let lower = 0;
    let upper = 30;
    for (let iteration = 0; iteration < 80; iteration++) {
      const slope = (lower + upper) / 2;
      const score = 1 / (1 + Math.exp(slope)) - epsilon / (1 + Math.exp(-epsilon * slope));
      if (score > 0) lower = slope;
      else upper = slope;
    }
    const expectedSlope = (lower + upper) / 2;
    expect(expectedSlope).toBeCloseTo(14.50865, 4);
    const statuses = await read(codes, true);
    const values = await read(coefficients);
    statuses.forEach((status, focus) => {
      // Tight tolerance can exhaust float32 IRLS; a finite iterate must still be published.
      expect([0, 3], `focus ${focus}`).toContain(status);
      expect(Math.abs(values[2 * focus])).toBeLessThan(5e-5);
      expect(Math.abs(values[2 * focus + 1] - expectedSlope)).toBeLessThan(0.2);
    });
    expect((await read(fitted)).every(Number.isFinite)).toBe(true);
  } finally {
    context.destroy();
  }
});

it('rejects overlapping search/final status and preserves adjacent status slices through a converged search (review 2)', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const model = reference.models[0];
    const statusStorage = output('uint32', 49);
    const codes = graph.createDataView(statusStorage.view.buffer, {
      format: 'uint32',
      byteOffset: statusStorage.view.byteOffset,
      length: 48
    });
    const overlapped = graph.createDataView(codes.buffer, {
      format: 'uint32',
      byteOffset: codes.byteOffset,
      length: 1
    });
    const converged = graph.createDataView(codes.buffer, {
      format: 'uint32',
      byteOffset: codes.byteOffset + 48 * 4,
      length: 1
    });
    const props = {
      family: 'poisson' as const,
      positions: input('float32x2', reference.positions),
      predictors: input('float32', reference.predictors),
      predictorCount: 1,
      response: input('float32', model.response),
      offset: input('float32', model.offset),
      mask: input('uint32', reference.mask),
      parameters: input('float32', [1, 1e-6]),
      maximumIterations: 64,
      predictExcluded: true,
      bandwidthSelection: {
        method: 'golden-section' as const,
        maximumIterations: 2,
        parameters: input('float32', [2, 8, 1e6]),
        output: {
          scores: output('float32', 4).view,
          bandwidth: output('float32', 1).view,
          status: output('uint32', 1).view
        }
      },
      output: {
        coefficients: output('float32', 96).view,
        fitted: output('float32', 48).view,
        status: {codes, iterationCount: output('uint32', 48).view}
      }
    };
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          bandwidthSelection: {...props.bandwidthSelection, status: {converged: overlapped}}
        })
    ).toThrow(/overlap/);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        ...props,
        bandwidthSelection: {...props.bandwidthSelection, status: {converged}}
      })
    );
    context.run();
    const statuses = await read(statusStorage, true);
    expect(statuses.slice(0, 48)).toEqual(new Array(48).fill(0));
    expect(statuses[48]).toBe(1);
  } finally {
    context.destroy();
  }
});

it('finishes the narrow adaptive bracket [40,44] against exhaustive NumPy scores before reporting convergence (search review 1)', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const model = reference.models.find(
      candidate =>
        candidate.family === 'logistic' &&
        candidate.kernel === 'bisquare' &&
        candidate.mode === 'adaptive'
    )!;
    const scores = output('float32', 16);
    const history = output('float32', 16);
    const bandwidth = output('float32', 1);
    const status = output('uint32', 1);
    const converged = output('uint32', 1);
    const iterationCount = output('uint32', 1);
    const iterationLimitReached = output('uint32', 1);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'logistic',
        kernel: 'bisquare',
        bandwidthMode: 'adaptive',
        positions: input('float32x2', reference.positions),
        predictors: input('float32', reference.predictors),
        predictorCount: 1,
        response: input('float32', model.response),
        mask: input('uint32', reference.mask),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 64,
        bandwidthSelection: {
          method: 'golden-section',
          maximumIterations: 8,
          parameters: input('float32', [40, 44, 0.0001]),
          output: {
            scores: scores.view,
            evaluatedBandwidths: history.view,
            bandwidth: bandwidth.view,
            status: status.view
          },
          status: {
            converged: converged.view,
            iterationCount: iterationCount.view,
            iterationLimitReached: iterationLimitReached.view
          }
        },
        output: {
          coefficients: output('float32', 96).view,
          fitted: output('float32', 48).view,
          status: {codes: output('uint32', 48).view, iterationCount: output('uint32', 48).view}
        }
      })
    );
    context.run();
    expect(await read(status, true)).toEqual([0]);
    expect(await read(bandwidth)).toEqual([44]);
    expect(await read(converged, true)).toEqual([1]);
    expect(await read(iterationCount, true)).toEqual([3]);
    expect(await read(iterationLimitReached, true)).toEqual([0]);
    const evaluated = await read(history);
    const values = await read(scores);
    for (const candidate of model.candidates.filter(value => value.bandwidth >= 40)) {
      const slot = evaluated.indexOf(candidate.bandwidth);
      expect(slot).toBeGreaterThanOrEqual(0);
      expect(Math.abs(values[slot] - candidate.score)).toBeLessThan(0.006);
    }
    expect(evaluated.slice(6).every(Number.isNaN)).toBe(true);
    expect(values.slice(6).every(Number.isNaN)).toBe(true);
  } finally {
    context.destroy();
  }
});

it('finds a finite fixed bisquare model above two invalid interiors without discarding that side (search review 2)', async () => {
  const context = await setup();
  const {graph, input, output, read} = context;
  try {
    const scores = output('float32', 16);
    const history = output('float32', 16);
    const bandwidth = output('float32', 1);
    const status = output('uint32', 1);
    const converged = output('uint32', 1);
    const iterationLimitReached = output('uint32', 1);
    const coefficients = output('float32', 16);
    const fitted = output('float32', 8);
    const codes = output('uint32', 8);
    graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: 'logistic',
        kernel: 'bisquare',
        positions: input('float32x2', [0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, 100, 0, 100, 0]),
        predictors: input('float32', [-1, 1, -1, 1, -1, 1, -1, 1]),
        predictorCount: 1,
        response: input('float32', [0, 0, 0, 0, 1, 1, 1, 1]),
        parameters: input('float32', [1, 1e-6]),
        maximumIterations: 64,
        bandwidthSelection: {
          method: 'golden-section',
          maximumIterations: 8,
          parameters: input('float32', [1, 150, 0.0001]),
          output: {
            scores: scores.view,
            evaluatedBandwidths: history.view,
            bandwidth: bandwidth.view,
            status: status.view
          },
          status: {converged: converged.view, iterationLimitReached: iterationLimitReached.view}
        },
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: {codes: codes.view, iterationCount: output('uint32', 8).view}
        }
      })
    );
    context.run();
    expect(await read(status, true)).toEqual([0]);
    const selected = (await read(bandwidth))[0];
    expect(selected).toBeGreaterThan(100);
    expect(selected).toBeLessThanOrEqual(150);
    expect(await read(converged, true)).toEqual([0]);
    expect(await read(iterationLimitReached, true)).toEqual([1]);
    const evaluated = await read(history);
    const values = await read(scores);
    expect(values[0]).toBeNaN();
    expect(values[1]).toBeNaN();
    expect(evaluated).toContain(150);
    // Balanced +/-1 predictors give slope zero, p = w/(1+w) at the all-zero site and
    // p = 1/(1+w) at the all-one site. Thus D = 16 log(1+w), K = 4/(1+w), independently of IRLS.
    const getScore = (candidate: number) => {
      const weight = (1 - (100 / candidate) ** 2) ** 2;
      const trace = 4 / (1 + weight);
      return 16 * Math.log1p(weight) + 2 * trace + (2 * trace * (trace + 1)) / (7 - trace);
    };
    expect(getScore(120)).toBeCloseTo(18.945362, 5);
    for (let slot = 0; slot < evaluated.length; slot++) {
      if (evaluated[slot] <= 100) expect(values[slot]).toBeNaN();
      else expect(Math.abs(values[slot] - getScore(evaluated[slot]))).toBeLessThan(0.005);
    }
    expect(await read(codes, true)).toEqual(new Array(8).fill(0));
    const estimates = await read(coefficients);
    const means = await read(fitted);
    const weight = (1 - (100 / selected) ** 2) ** 2;
    for (let row = 0; row < 8; row++) {
      expect(
        Math.abs(estimates[2 * row] - (row < 4 ? Math.log(weight) : -Math.log(weight)))
      ).toBeLessThan(0.003);
      expect(Math.abs(estimates[2 * row + 1])).toBeLessThan(0.003);
      expect(
        Math.abs(means[row] - (row < 4 ? weight / (1 + weight) : 1 / (1 + weight)))
      ).toBeLessThan(0.003);
    }
  } finally {
    context.destroy();
  }
});
