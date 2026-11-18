// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {getGPUOrdinaryLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {createGPURegressionModelResult} from '../../../src/analysis-core/contracts/model-result';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPU prediction fixes the combined small correction before adding a large response mean', () => {
  // Regrouping the two -0.25 mean corrections into the large mean reproduces Metal v2-1.
  expect(Math.fround(Math.fround(5000001 - 0.25) - 0.25)).toBe(5000001);
  expect(Math.fround(5000001 + Math.fround(-0.25 - 0.25))).toBe(5000000.5);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `prediction-rounding-${serial++}`, format, length);
  try {
    const contributors = [
      new GPUOrdinaryLeastSquares({
        id: 'ols-rounding',
        predictors: view('float32', 4),
        response: view('float32', 4),
        predictorCount: 1,
        output: {
          coefficients: view('float32', 2),
          standardErrors: view('float32', 2),
          tStatistics: view('float32', 2),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)},
          fitted: view('float32', 4),
          predictions: view('float32', 4)
        }
      }),
      new GPUSpatialRegimeRegression({
        id: 'regime-rounding',
        predictors: view('float32', 4),
        response: view('float32', 4),
        predictorCount: 1,
        regimes: view('uint32', 4),
        regimeCount: 1,
        output: {
          coefficients: view('float32', 2),
          fitted: view('float32', 4),
          status: view('uint32', 1),
          counts: view('uint32', 1)
        }
      })
    ];
    for (const contributor of contributors) {
      const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(
        kernel => kernel.id === 'ols-rounding-residuals' || kernel.id === 'regime-rounding-predict'
      )!.source;
      expect(source).toContain(
        'getRoundedDifference(centeredContribution + meanCorrection, 0.0, runtimeZero)'
      );
      expect(source).toMatch(/(?:value|fittedOut\[.*?\]) = getCenteredRegressionPrediction\(/);
      if (contributor instanceof GPUOrdinaryLeastSquares) {
        expect(source).toMatch(/predictionOut\[.*?\] = getCenteredRegressionPrediction\(/);
      }
    }
  } finally {
    device.destroy();
  }
});

it('predicts excluded rows with centered slopes at projected-coordinate magnitude', () => {
  const result = computeOrdinaryLeastSquaresOnCPU(
    {
      predictors: Float32Array.of(4999998, 4999998, 5000002, 5000002, 5000000, NaN),
      response: Float32Array.of(-15, -15, 17, 17, NaN, 1),
      mask: Uint32Array.of(1, 1, 1, 1, 0, 0)
    },
    1
  );
  expect(result.status[0]).toBe(0);
  // The published raw float32 intercept loses the level of 1 at the predictor center.
  expect(Math.fround(result.coefficients[0] + Math.fround(5000000 * result.coefficients[1]))).toBe(
    0
  );
  expect(result.predictions[4]).toBe(1);
  expect(Number.isNaN(result.predictions[5])).toBe(true);
  expect(Number.isNaN(result.fitted[4])).toBe(true);
});

it('CPU centered fit preserves a rare varying predictor and the exceptional held-out row', () => {
  const trainingCount = 2048;
  const predictors = new Float32Array(trainingCount + 1).fill(5000000);
  const response = new Float32Array(trainingCount + 1);
  const mask = new Uint32Array(trainingCount + 1).fill(1);
  predictors[0] = 5000000.5;
  predictors[trainingCount] = 5000000.5;
  response[0] = 1;
  response[trainingCount] = NaN;
  mask[trainingCount] = 0;
  const result = computeOrdinaryLeastSquaresOnCPU({predictors, response, mask}, 1);
  expect(result.status[0]).toBe(0);
  expect(result.coefficients[1]).toBeCloseTo(2, 5);
  expect(result.predictions[trainingCount]).toBeCloseTo(1, 5);
});

for (const scene of ['representable-intercept', 'finite-variance'] as const) {
  it(`CPU regression oracle preserves ${scene}`, () => {
    const predictors =
      scene === 'representable-intercept'
        ? Float32Array.of(5000000, 5000000.5, 5000000, 5000000.5)
        : Float32Array.of(2 ** 86, 2 ** 86 + 2 ** 63, 2 ** 86, 2 ** 86 + 2 ** 63);
    const response =
      scene === 'representable-intercept'
        ? Float32Array.from(predictors, value => value + 0.5)
        : Float32Array.of(0, 1, 0, 1);
    const result = computeOrdinaryLeastSquaresOnCPU({predictors, response}, 1);
    expect(result.status[0]).toBe(0);
    if (scene === 'representable-intercept') {
      expect(Array.from(result.coefficients)).toEqual([0.5, 1]);
    }
    expect(Array.from(result.fitted)).toEqual(Array.from(response));
  });
}

it('OLS decoded AICc is null at the successful-fit boundary and with a distinct criterion count', () => {
  for (const response of [Float32Array.of(1, 2, 2), Float32Array.of(1, 2, 2, 4)]) {
    const fit = computeOrdinaryLeastSquaresOnCPU(
      {predictors: Float32Array.from(response, (_, row) => row), response},
      1
    );
    expect(fit.status[0]).toBe(0);
    const decoded = getGPUOrdinaryLeastSquaresModelResult(fit);
    expect(Number.isFinite(decoded.akaikeInformationCriterion)).toBe(true);
    if (response.length === 3) expect(decoded.correctedAkaikeInformationCriterion).toBeNull();
    else {
      expect(decoded.correctedAkaikeInformationCriterion).toBeGreaterThan(
        decoded.akaikeInformationCriterion!
      );
      const withVariance = createGPURegressionModelResult({
        kind: decoded.kind,
        coefficients: decoded.coefficients,
        rowCount: decoded.rowCount,
        residualSumOfSquares: decoded.residualSumOfSquares,
        standardErrorSigmaSquared: decoded.standardErrorSigmaSquared,
        statusCode: decoded.status.code,
        logLikelihood: decoded.logLikelihood,
        informationCriterionParameterCount: 3
      });
      expect(withVariance.correctedAkaikeInformationCriterion).toBeNull();
    }
  }
});
