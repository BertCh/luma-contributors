// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {GPUSpatialRegressionDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {REVIEW_REGRESSION_FIXTURE} from './rr-weights-regression-review-fixture';

function createFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `review2-input-${serial++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length + 2);
    buffers.push(buffer);
    const values = [...new Array(length).fill(77), 98, 99];
    buffer.write(format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values));
    const backing = importGraphBuffer(
      graph,
      `review2-output-${serial++}`,
      buffer,
      format,
      length + 2
    );
    return {buffer, view: graph.createDataView(backing.buffer, {format, length, byteOffset: 0})};
  };
  const data = REVIEW_REGRESSION_FIXTURE;
  const predictors = input('float32', data.predictors).view;
  const response = input('float32', data.response).view;
  const weights = {
    offsets: input('uint32', data.weights.offsets).view,
    neighbors: input('uint32', data.weights.neighbors).view,
    weights: input('float32', data.weights.weights).view
  };
  return {
    graph,
    buffers,
    input,
    output,
    predictors,
    response,
    weights,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

it('R7 complete recipe propagates retained-count loss to diagnostics, specification and residual Moran', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const required = fixture.input('uint32', Uint32Array.of(8));
  const code = fixture.output('uint32', 1),
    decision = fixture.output('uint32', 1);
  const localI = fixture.output('float32', 8),
    probabilities = fixture.output('float32', 8);
  const parameters = new GPUParameterBuffer(device, {
    id: 'count-recipe-parameters',
    format: 'float32',
    length: 8
  });
  try {
    addSpatialRegressionRecipe(fixture.graph, {
      predictors: fixture.predictors,
      response: fixture.response,
      predictorCount: 1,
      parameters,
      weights: {...fixture.weights, status: {requiredCount: required.view}},
      output: {
        diagnostics: {status: {code: code.view}},
        specification: {decision: decision.view},
        residualMoran: {localI: localI.view, pValues: probabilities.view}
      }
    });
    const compiled = fixture.graph.compile();
    try {
      // rr-weights-regression-review2.oracle.py: I_i = (n - 1) c_i (Wc)_i / sum(c^2).
      // Sample scaling is 7/M2 here; population scaling 8/M2 is wrong by a factor of 8/7.
      const expectedLocalI = [
        -0.004803921568627451, -0.014019607843137254, -0.715, -0.465, -1.3038235294117646,
        -0.4906862745098039, -0.42892156862745096, -0.6004901960784313
      ];
      for (const count of [8, 9, 8]) {
        required.buffer.write(Uint32Array.of(count));
        submitGraph(device, compiled, undefined);
        expect(Array.from(await readUint32(code.buffer, 3))).toEqual([count === 9 ? 3 : 0, 98, 99]);
        expect(Array.from(await readUint32(decision.buffer, 3))).toEqual([
          count === 9 ? 4 : 0,
          98,
          99
        ]);
        const values = await readFloat32(localI.buffer, 10),
          probabilitiesReadback = await readFloat32(probabilities.buffer, 10);
        expect(values.slice(8)).toEqual([98, 99]);
        expect(probabilitiesReadback.slice(8)).toEqual([98, 99]);
        if (count === 9) {
          expect(values.slice(0, 8).every(Number.isNaN)).toBe(true);
          expect(probabilitiesReadback.slice(0, 8).every(Number.isNaN)).toBe(true);
        } else
          for (const [index, expected] of expectedLocalI.entries()) {
            expect(Number.isFinite(values[index])).toBe(true);
            expect(Math.abs(values[index] - expected)).toBeLessThanOrEqual(2e-4); // float32 centered moments.
          }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    parameters.destroy();
    fixture.destroy();
  }
});

it('R8 complete 2SLS retains point estimates while HAC count loss suppresses covariance and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const required = fixture.input('uint32', Uint32Array.of(8));
  const covariance = fixture.output('float32', 9),
    covarianceCode = fixture.output('uint32', 1);
  const table = fixture.output('float32', 12),
    summary = fixture.output('float32', 16),
    code = fixture.output('uint32', 1);
  try {
    fixture.graph.add(
      new GPUSpatialTwoStageLeastSquares({
        weights: fixture.weights,
        predictors: fixture.predictors,
        response: fixture.response,
        predictorCount: 1,
        output: {table: table.view, summary: summary.view, status: {code: code.view}},
        robust: {
          method: 'HAC',
          kernel: {
            offsets: fixture.input('uint32', Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)).view,
            neighbors: fixture.input('uint32', Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7)).view,
            weights: fixture.input('float32', new Float32Array(8).fill(1)).view,
            status: {requiredCount: required.view}
          },
          output: {covariance: covariance.view, status: {code: covarianceCode.view}}
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      const expectedCovariance = [
        2.7335123628111035, 0.7509776511140532, -0.8997389744133288, 0.750977651114053,
        0.4840167229861271, -0.41019456878484994, -0.8997389744133288, -0.4101945687848499,
        0.4117634060642176
      ];
      for (const count of [8, 9, 8]) {
        required.buffer.write(Uint32Array.of(count));
        submitGraph(device, compiled, undefined);
        expect(Array.from(await readUint32(code.buffer, 3))).toEqual([0, 98, 99]);
        expect(Array.from(await readUint32(covarianceCode.buffer, 3))).toEqual([
          count === 9 ? 4 : 0,
          98,
          99
        ]);
        const values = await readFloat32(covariance.buffer, 11),
          coefficients = await readFloat32(table.buffer, 14);
        expect(values.slice(9)).toEqual([98, 99]);
        expect(coefficients.slice(12)).toEqual([98, 99]);
        for (const [index, expected] of REVIEW_REGRESSION_FIXTURE.twoStageCoefficients.entries()) {
          expect(Number.isFinite(coefficients[index * 4])).toBe(true);
          expect(Math.abs(coefficients[index * 4] - expected)).toBeLessThanOrEqual(2e-3);
          if (count === 9) expect(Number.isNaN(coefficients[index * 4 + 1])).toBe(true);
        }
        if (count === 9) expect(values.slice(0, 9).every(Number.isNaN)).toBe(true);
        else
          for (const [index, expected] of expectedCovariance.entries()) {
            expect(Number.isFinite(values[index])).toBe(true);
            expect(Math.abs(values[index] - expected)).toBeLessThanOrEqual(3e-3); // float32 IV sandwich.
          }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    fixture.destroy();
  }
});

it('R9 complete panel evaluates count loss on the entity graph before expansion and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const required = fixture.input('uint32', Uint32Array.of(4));
  const table = fixture.output('float32', 12),
    summary = fixture.output('float32', 22),
    code = fixture.output('uint32', 1);
  const tests = fixture.output('float32', 6),
    diagnosticCode = fixture.output('uint32', 1);
  try {
    fixture.graph.add(
      new GPUSpatialPanelErrorMaximumLikelihood({
        predictors: fixture.predictors,
        response: fixture.response,
        predictorCount: 1,
        periodCount: 2,
        maximumPower: 256,
        traceMode: 'exact',
        tailExtrapolation: true,
        weights: {
          offsets: fixture.input('uint32', Uint32Array.of(0, 1, 2, 3, 4)).view,
          neighbors: fixture.input('uint32', Uint32Array.of(1, 2, 3, 0)).view,
          weights: fixture.input('float32', Float32Array.of(1, 1, 1, 1)).view,
          status: {requiredCount: required.view}
        },
        output: {table: table.view, summary: summary.view, status: {code: code.view}},
        diagnostics: {tests: tests.view, status: diagnosticCode.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (const count of [4, 5, 4]) {
        required.buffer.write(Uint32Array.of(count));
        submitGraph(device, compiled, undefined);
        expect(Array.from(await readUint32(code.buffer, 3))).toEqual([count === 5 ? 3 : 0, 98, 99]);
        expect(Array.from(await readUint32(diagnosticCode.buffer, 3))).toEqual([
          count === 5 ? 1 : 0,
          98,
          99
        ]);
        const values = await readFloat32(tests.buffer, 8),
          coefficients = await readFloat32(table.buffer, 14);
        expect(values.slice(6)).toEqual([98, 99]);
        expect(coefficients.slice(12)).toEqual([98, 99]);
        if (count === 5) {
          expect(values.slice(0, 6).every(Number.isNaN)).toBe(true);
          expect(coefficients.slice(0, 12).every(Number.isNaN)).toBe(true);
        } else {
          expect(Number.isFinite(values[0])).toBe(true);
          expect(Math.abs(values[0] - 1.7313633782924958)).toBeLessThanOrEqual(2e-3);
          // Independent dense profile optimum; tolerance covers float32 likelihood search and its series tail.
          for (const [index, expected] of [-0.6275645006895683, 0.9317603209776758].entries()) {
            expect(Number.isFinite(coefficients[(index + 1) * 4])).toBe(true);
            expect(Math.abs(coefficients[(index + 1) * 4] - expected)).toBeLessThanOrEqual(4e-3);
          }
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    fixture.destroy();
  }
});

it('W5-27 2SLS keeps point estimates while a complete-only HAC certificate loses and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const complete = fixture.input('uint32', Uint32Array.of(1));
  const clear = fixture.input('uint32', Uint32Array.of(0));
  const covariance = fixture.output('float32', 9),
    covarianceCode = fixture.output('uint32', 1);
  const table = fixture.output('float32', 12),
    summary = fixture.output('float32', 16),
    code = fixture.output('uint32', 1);
  try {
    fixture.graph.add(
      new GPUSpatialTwoStageLeastSquares({
        weights: fixture.weights,
        predictors: fixture.predictors,
        response: fixture.response,
        predictorCount: 1,
        output: {table: table.view, summary: summary.view, status: {code: code.view}},
        robust: {
          method: 'HAC',
          kernel: {
            offsets: fixture.input('uint32', Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)).view,
            neighbors: fixture.input('uint32', Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7)).view,
            weights: fixture.input('float32', new Float32Array(8).fill(1)).view,
            status: {complete: complete.view, overflow: clear.view, candidateOverflow: clear.view}
          },
          output: {covariance: covariance.view, status: {code: covarianceCode.view}}
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (const value of [1, 0, 1]) {
        complete.buffer.write(Uint32Array.of(value));
        submitGraph(device, compiled, undefined);
        expect(Array.from(await readUint32(code.buffer, 3))).toEqual([0, 98, 99]);
        expect(Array.from(await readUint32(covarianceCode.buffer, 3))).toEqual([
          value ? 0 : 4,
          98,
          99
        ]);
        const coefficients = await readFloat32(table.buffer, 14),
          values = await readFloat32(covariance.buffer, 11);
        for (const [index, expected] of REVIEW_REGRESSION_FIXTURE.twoStageCoefficients.entries()) {
          expect(Number.isFinite(coefficients[4 * index])).toBe(true);
          expect(Math.abs(coefficients[4 * index] - expected)).toBeLessThan(2e-3);
        }
        if (!value) expect(values.slice(0, 9).every(Number.isNaN)).toBe(true);
        else {
          // Identity HAC equals HC0: independent sandwich from the existing R8 fixture.
          const expected = [
            2.7335123628111035, 0.7509776511140532, -0.8997389744133288, 0.750977651114053,
            0.4840167229861271, -0.41019456878484994, -0.8997389744133288, -0.4101945687848499,
            0.4117634060642176
          ];
          for (const [index, reference] of expected.entries()) {
            expect(Number.isFinite(values[index])).toBe(true);
            expect(Math.abs(values[index] - reference)).toBeLessThan(3e-3);
          }
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    fixture.destroy();
  }
});

for (const location of ['whole', 'root', 'later-child'] as const)
  for (const kind of ['traces', 'collinearity', 'diagnostics'] as const) {
    if (kind === 'diagnostics' && location !== 'whole') continue;
    it(`W5-27 GPU ${kind} consumes ${location} certificates with clear flags through 1/0/1`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createFixture(device);
      const complete = fixture.input('uint32', Uint32Array.of(1));
      const clear = fixture.input('uint32', Uint32Array.of(0));
      const status = {complete: complete.view, overflow: clear.view, candidateOverflow: clear.view};
      const createCSR = (start: number, count: number) => ({
        offsets: fixture.input(
          'uint32',
          Uint32Array.from({length: count + 1}, (_, row) => 3 * row)
        ).view,
        neighbors: fixture.input(
          'uint32',
          Uint32Array.from(
            Array.from({length: count}, (_, local) =>
              [0, 1, 2, 3].filter(neighbor => neighbor !== start + local)
            ).flat()
          )
        ).view,
        weights: fixture.input('float32', new Float32Array(3 * count).fill(1 / 3)).view
      });
      const whole = {...createCSR(0, 4), status};
      const weights =
        location === 'whole'
          ? whole
          : {
              rowCount: 4,
              status:
                location === 'root'
                  ? status
                  : {overflow: clear.view, candidateOverflow: clear.view},
              chunks: [0, 2].map(rowStart => ({
                rowStart,
                rowCount: 2,
                weights: {
                  ...createCSR(rowStart, 2),
                  ...(location === 'later-child' && rowStart === 2 ? {status} : {})
                }
              }))
            };
      const predictors = fixture.input('float32', Float32Array.of(0, 0, 1, 1)).view;
      const residuals = fixture.input('float32', Float32Array.of(1, -1, -1, 1)).view;
      const response = fixture.input('float32', Float32Array.of(3, 1, 2, 4)).view;
      const primary = fixture.output(
        'float32',
        kind === 'traces' ? 2 : kind === 'collinearity' ? 4 : 13
      );
      const secondary = fixture.output(
        'float32',
        kind === 'traces' ? 1 : kind === 'collinearity' ? 4 : 18
      );
      const code = fixture.output('uint32', kind === 'collinearity' ? 4 : 1);
      const powerSums = fixture.output('float32', 2);
      try {
        const contributor =
          kind === 'traces'
            ? new GPUSpatialWeightsTraces({
                weights,
                maximumPower: 2,
                mode: 'exact',
                output: {
                  powerTraces: primary.view,
                  powerSums: powerSums.view,
                  infinityNorm: secondary.view
                }
              })
            : kind === 'collinearity'
              ? new GPULocalCollinearity({
                  weights,
                  predictors,
                  predictorCount: 1,
                  output: {
                    varianceInflationFactors: primary.view,
                    conditionNumbers: secondary.view,
                    status: code.view
                  }
                })
              : new GPUSpatialRegressionDiagnostics({
                  weights: whole,
                  predictors,
                  residuals,
                  response,
                  predictorCount: 1,
                  output: {summary: primary.view, tests: secondary.view, status: {code: code.view}}
                });
        fixture.graph.add(contributor);
        const compiled = fixture.graph.compile();
        try {
          for (const value of [1, 0, 1]) {
            complete.buffer.write(Uint32Array.of(value));
            submitGraph(device, compiled, undefined);
            const values = await readFloat32(primary.buffer, primary.view.length);
            if (!value) {
              if (kind === 'diagnostics') {
                expect(values[0]).toBe(4);
                expect(values[12]).toBe(0);
                expect(values.slice(1, 12).every(Number.isNaN)).toBe(true);
              } else expect(values.every(Number.isNaN)).toBe(true);
            } else {
              expect(values.every(Number.isFinite)).toBe(kind !== 'diagnostics');
              if (kind === 'traces') expect(Math.abs(values[1] - 4 / 3)).toBeLessThan(1e-5); // W=(11'-I)/3: tr(W²)=4/3.
              if (kind === 'collinearity')
                expect(values.every(number => Math.abs(number - 1) < 1e-5)).toBe(true);
            }
            if (kind !== 'traces') {
              const codes = await readUint32(code.buffer, code.view.length);
              expect(Array.from(codes)).toEqual(
                new Array(code.view.length).fill(value ? 0 : kind === 'collinearity' ? 1 : 3)
              );
            }
          }
        } finally {
          compiled.destroy();
        }
      } finally {
        fixture.destroy();
      }
    });
  }
