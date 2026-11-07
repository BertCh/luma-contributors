// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {solveBenchmarkSystem} from '../spatial-interpolation/interpolation-performance-fixtures';

/** Independent f64 block GLS for the two-equation, one-predictor benchmark fixture. */
export function calculateSURBenchmarkReference(
  predictors: ArrayLike<number>,
  responses: ArrayLike<number>
) {
  const rows = responses.length / 2;
  const initialResiduals = new Float64Array(rows * 2);
  for (let equation = 0; equation < 2; equation++) {
    let predictorMean = 0;
    let responseMean = 0;
    for (let row = 0; row < rows; row++) {
      predictorMean += predictors[equation * rows + row] / rows;
      responseMean += responses[equation * rows + row] / rows;
    }
    let squareSum = 0;
    let crossSum = 0;
    for (let row = 0; row < rows; row++) {
      const centered = predictors[equation * rows + row] - predictorMean;
      squareSum += centered * centered;
      crossSum += centered * (responses[equation * rows + row] - responseMean);
    }
    const slope = crossSum / squareSum;
    for (let row = 0; row < rows; row++) {
      initialResiduals[equation * rows + row] =
        responses[equation * rows + row] -
        responseMean -
        slope * (predictors[equation * rows + row] - predictorMean);
    }
  }
  const residualCovariance = Array.from({length: 4}, (_, index) => {
    let value = 0;
    for (let row = 0; row < rows; row++) {
      value +=
        (initialResiduals[Math.floor(index / 2) * rows + row] *
          initialResiduals[(index % 2) * rows + row]) /
        rows;
    }
    return value;
  });
  const determinant =
    residualCovariance[0] * residualCovariance[3] - residualCovariance[1] * residualCovariance[2];
  const precision = [
    residualCovariance[3],
    -residualCovariance[1],
    -residualCovariance[2],
    residualCovariance[0]
  ].map(value => value / determinant);
  const information = Array.from({length: 4}, () => new Array<number>(4).fill(0));
  const right = new Array<number>(4).fill(0);
  for (let row = 0; row < rows; row++) {
    const design = [1, predictors[row], 1, predictors[rows + row]];
    for (let first = 0; first < 4; first++) {
      const equation = Math.floor(first / 2);
      for (let other = 0; other < 2; other++) {
        right[first] +=
          design[first] * precision[equation * 2 + other] * responses[other * rows + row];
      }
      for (let second = 0; second < 4; second++) {
        information[first][second] +=
          design[first] * precision[equation * 2 + Math.floor(second / 2)] * design[second];
      }
    }
  }
  const coefficients = solveBenchmarkSystem(information, right);
  const covariance = new Array<number>(16);
  for (let column = 0; column < 4; column++) {
    const solution = solveBenchmarkSystem(
      information,
      Array.from({length: 4}, (_, row) => Number(row === column))
    );
    for (let row = 0; row < 4; row++) covariance[row * 4 + column] = solution[row];
  }
  const residuals = Array.from({length: responses.length}, (_, index) => {
    const equation = Math.floor(index / rows);
    return (
      responses[index] -
      coefficients[equation * 2] -
      predictors[index] * coefficients[equation * 2 + 1]
    );
  });
  return {coefficients, covariance, residualCovariance, residuals};
}
