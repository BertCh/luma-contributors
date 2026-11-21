// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {makeOLSAnalysisSummary} from '../../../../examples/deck/kepler-compute/src/analysis-ols-report';
import {getOLSInferenceLabel} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/regression/regression-inference';
import {
  computeOrdinaryLeastSquaresOnCPU,
  getGPUOrdinaryLeastSquaresModelResult,
  getGPUOrdinaryLeastSquaresParameterValues,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA
} from '@luma-contributors/analysis/gpu-spatial-analysis';

test('OLS report uses applied ridge metadata and resets inference with each readback', () => {
  const fit = (ridgeLambda: number) =>
    computeOrdinaryLeastSquaresOnCPU(
      {
        predictors: Float32Array.of(0, 1, 2, 3),
        response: Float32Array.of(1, 3, 4, 8),
        parameters: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})
      },
      1
    );
  const readback = fit(0.1);
  // The shader applies the float32 parameter, even when the caller requested binary64 0.1.
  const settings: Parameters<typeof makeOLSAnalysisSummary>[1] = {
    responseColumn: 'y',
    predictorColumns: ['x'],
    intercept: 'included',
    ridgeLambda: 0.1,
    variance: 'degrees-of-freedom',
    missing: 'complete-cases'
  };
  const ridge = makeOLSAnalysisSummary(readback, settings);
  expect(ridge.ridgeLambda).toBe(0.10000000149011612);
  expect(ridge.inferenceMethod).toBe('ridge-approximate');
  expect(ridge.coefficients).toHaveLength(2);
  expect(
    ridge.coefficients.every(
      coefficient => coefficient.standardError === null && coefficient.pValue === null
    )
  ).toBe(true);
  expect(getOLSInferenceLabel(getGPUOrdinaryLeastSquaresModelResult(readback))).toMatch(
    /approximate.*inverse-penalized-normal-matrix/
  );
  const ordinary = fit(0);
  expect(makeOLSAnalysisSummary(ordinary, settings)).toMatchObject({
    ridgeLambda: 0,
    inference: 'classical-homoskedastic',
    inferenceMethod: 'ordinary-student-t'
  });
  ordinary.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA] = NaN;
  expect(makeOLSAnalysisSummary(ordinary, settings).inference).toBe('unavailable');
  expect(getOLSInferenceLabel(getGPUOrdinaryLeastSquaresModelResult(ordinary))).toBe(
    'inference convention unavailable'
  );
});
