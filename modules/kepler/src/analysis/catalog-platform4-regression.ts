// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeOutputSpec,
  makeStringParameter,
  readNumericProperty,
  type Values
} from './catalog-platform3-b-common';

const MAXIMUM_PREDICTORS = 15;

type RegressionSample = {
  response: Float64Array;
  predictors: Float64Array;
  predictorCount: number;
  featureIndex: Int32Array;
  coordinates: [number, number][];
  count: number;
  droppedCount: number;
  names: string[];
};

function readRegressionSample(
  collection: CatalogFeatureCollection,
  values: Values
): RegressionSample {
  const responseName = (values['responseProperty'] as string).trim();
  const names = (values['predictorProperties'] as string)
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  // The model needs a response and 1 to 15 predictor properties.
  if (!responseName || names.length < 1 || names.length > MAXIMUM_PREDICTORS)
    throw new RangeError('Name a response and 1 to 15 comma separated predictor properties');
  const response: number[] = [];
  const predictors: number[] = [];
  const featureIndex: number[] = [];
  const coordinates: [number, number][] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const position = getPointPositions(feature.geometry)[0];
    const y = readNumericProperty(feature, responseName);
    const row = names.map(name => readNumericProperty(feature, name));
    // Float32 rounding must stay finite for the GPU contributor.
    if (
      !position ||
      y === null ||
      row.some(value => value === null) ||
      ![y, ...(row as number[])].every(value => Number.isFinite(Math.fround(value)))
    ) {
      droppedCount++;
      return;
    }
    response.push(Math.fround(y));
    for (const value of row as number[]) predictors.push(Math.fround(value));
    featureIndex.push(index);
    coordinates.push([position[0], position[1]]);
  });
  // The fit needs more complete rows than coefficients.
  if (response.length <= names.length + 1)
    throw new RangeError('Too few complete rows for the number of predictors');
  return {
    response: Float64Array.from(response),
    predictors: Float64Array.from(predictors),
    predictorCount: names.length,
    featureIndex: Int32Array.from(featureIndex),
    coordinates,
    count: response.length,
    droppedCount,
    names
  };
}

type OlsFit = {
  coefficients: number[];
  standardErrors: number[];
  tStatistics: number[];
  fitted: Float64Array;
  residuals: Float64Array;
  rSquared: number;
  adjustedRSquared: number;
  residualSumOfSquares: number;
  sigmaSquared: number;
};

/** Centered normal equations in float64 with Gauss-Jordan inversion. */
export function fitOrdinaryLeastSquares(sample: RegressionSample): OlsFit {
  const {count, predictorCount: p} = sample;
  const means = new Float64Array(p);
  let responseMean = 0;
  for (let row = 0; row < count; row++) {
    responseMean += sample.response[row];
    for (let column = 0; column < p; column++) means[column] += sample.predictors[row * p + column];
  }
  responseMean /= count;
  for (let column = 0; column < p; column++) means[column] /= count;
  const gram = Array.from({length: p}, () => new Float64Array(2 * p));
  const cross = new Float64Array(p);
  let totalSumOfSquares = 0;
  for (let row = 0; row < count; row++) {
    const centeredResponse = sample.response[row] - responseMean;
    totalSumOfSquares += centeredResponse * centeredResponse;
    for (let a = 0; a < p; a++) {
      const centeredA = sample.predictors[row * p + a] - means[a];
      cross[a] += centeredA * centeredResponse;
      for (let b = 0; b < p; b++)
        gram[a][b] += centeredA * (sample.predictors[row * p + b] - means[b]);
    }
  }
  const scale = Math.max(...gram.map((row, index) => row[index]), 1e-300);
  for (let a = 0; a < p; a++) gram[a][p + a] = 1;
  for (let pivot = 0; pivot < p; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < p; row++)
      if (Math.abs(gram[row][pivot]) > Math.abs(gram[best][pivot])) best = row;
    // A vanishing pivot means the predictors are collinear.
    if (Math.abs(gram[best][pivot]) < 1e-12 * scale)
      throw new RangeError('Predictors are collinear (singular normal equations)');
    [gram[pivot], gram[best]] = [gram[best], gram[pivot]];
    const divisor = gram[pivot][pivot];
    for (let column = 0; column < 2 * p; column++) gram[pivot][column] /= divisor;
    for (let row = 0; row < p; row++) {
      if (row === pivot) continue;
      const factor = gram[row][pivot];
      if (factor !== 0)
        for (let column = 0; column < 2 * p; column++)
          gram[row][column] -= factor * gram[pivot][column];
    }
  }
  const inverse = gram.map(row => row.slice(p));
  const slopes = new Float64Array(p);
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) slopes[a] += inverse[a][b] * cross[b];
  let intercept = responseMean;
  for (let a = 0; a < p; a++) intercept -= slopes[a] * means[a];
  const fitted = new Float64Array(count);
  const residuals = new Float64Array(count);
  let residualSumOfSquares = 0;
  for (let row = 0; row < count; row++) {
    let value = intercept;
    for (let a = 0; a < p; a++) value += slopes[a] * sample.predictors[row * p + a];
    fitted[row] = value;
    residuals[row] = sample.response[row] - value;
    residualSumOfSquares += residuals[row] ** 2;
  }
  const degrees = count - p - 1;
  const sigmaSquared = residualSumOfSquares / degrees;
  const slopeVariances = inverse.map((row, index) => row[index] * sigmaSquared);
  let interceptVariance = sigmaSquared / count;
  for (let a = 0; a < p; a++)
    for (let b = 0; b < p; b++)
      interceptVariance += sigmaSquared * means[a] * inverse[a][b] * means[b];
  const coefficients = [intercept, ...slopes];
  const standardErrors = [Math.sqrt(interceptVariance), ...slopeVariances.map(Math.sqrt)];
  const rSquared =
    totalSumOfSquares > 0 ? 1 - residualSumOfSquares / totalSumOfSquares : Number.NaN;
  return {
    coefficients,
    standardErrors,
    tStatistics: coefficients.map((value, index) => value / standardErrors[index]),
    fitted,
    residuals,
    rSquared,
    adjustedRSquared: 1 - ((1 - rSquared) * (count - 1)) / degrees,
    residualSumOfSquares,
    sigmaSquared
  };
}

function buildRegressionOutput(
  sample: RegressionSample,
  fit: OlsFit,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (sample.droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position or with a missing or non-numeric response or predictor are skipped (complete cases only).',
      sample.droppedCount
    );
  const coefficients = makeCatalogTable('ols-coefficients', 'OLS coefficients', {
    term: 'string',
    estimate: 'float64',
    standard_error: 'float64',
    t_statistic: 'float64'
  });
  ['(intercept)', ...sample.names].forEach((term, index) =>
    appendCatalogRow(coefficients, {
      term,
      estimate: fit.coefficients[index],
      standard_error: fit.standardErrors[index],
      t_statistic: fit.tStatistics[index]
    })
  );
  const residuals = makeCatalogTable(
    'ols-residuals',
    'OLS fitted values and residuals',
    {feature_index: 'int32', fitted: 'float64', residual: 'float64', geometry: 'string'},
    'geometry'
  );
  for (let row = 0; row < sample.count; row++)
    appendCatalogRow(residuals, {
      feature_index: sample.featureIndex[row],
      fitted: fit.fitted[row],
      residual: fit.residuals[row],
      geometry: makeCatalogGeometryText('Point', sample.coordinates[row])
    });
  return {
    tables: [coefficients, residuals],
    diagnostics,
    summary: {
      rowCount: sample.count,
      predictorCount: sample.predictorCount,
      rSquared: fit.rSquared,
      adjustedRSquared: fit.adjustedRSquared,
      residualSumOfSquares: fit.residualSumOfSquares,
      sigmaSquared: fit.sigmaSquared
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

async function fitOrdinaryLeastSquaresGpu(
  device: Device,
  sample: RegressionSample
): Promise<OlsFit> {
  const {count, predictorCount: p} = sample;
  return withCatalogGpuSession(device, 'catalog-regression-ols', async session => {
    const coefficients = session.output('coefficients', 'float32', p + 1);
    const standardErrors = session.output('standard-errors', 'float32', p + 1);
    const tStatistics = session.output('t-statistics', 'float32', p + 1);
    const summary = session.output('summary', 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH);
    const residuals = session.output('residuals', 'float32', count);
    const fitted = session.output('fitted', 'float32', count);
    const code = session.output('code', 'uint32', 1);
    session.graph.add(
      new GPUOrdinaryLeastSquares({
        id: 'catalog-regression-ols',
        predictorCount: p,
        predictors: session.input('predictors', Float32Array.from(sample.predictors), 'float32'),
        response: session.input('response', Float32Array.from(sample.response), 'float32'),
        output: {
          status: {code: code.view},
          coefficients: coefficients.view,
          standardErrors: standardErrors.view,
          tStatistics: tStatistics.view,
          summary: summary.view,
          residuals: residuals.view,
          fitted: fitted.view
        }
      })
    );
    session.run();
    // Status 1 is a singular normal matrix and 2 too few rows; the CPU engine reports both.
    if ((await session.readScalar(code)) !== 0)
      throw new CatalogGpuUnsupportedError(
        'The GPU solver reported a singular or underdetermined fit'
      );
    const summaryValues = await session.readFloat32(summary);
    const residualSumOfSquares =
      summaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES];
    return {
      coefficients: Array.from(await session.readFloat32(coefficients)),
      standardErrors: Array.from(await session.readFloat32(standardErrors)),
      tStatistics: Array.from(await session.readFloat32(tStatistics)),
      fitted: Float64Array.from(await session.readFloat32(fitted)),
      residuals: Float64Array.from(await session.readFloat32(residuals)),
      rSquared: summaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED],
      adjustedRSquared: summaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED],
      residualSumOfSquares,
      sigmaSquared: residualSumOfSquares / (count - p - 1)
    };
  });
}

const OLS_BINDING: CatalogOperationBinding = {
  id: 'regression-ols',
  title: 'Ordinary least squares regression',
  domain: 'analysis',
  summary:
    'Fits a numeric point property on one to fifteen numeric predictor properties with an intercept and publishes coefficients, standard errors, t statistics, fit statistics, fitted values and residuals.',
  capabilityIds: ['spatial-regression'],
  contributorNames: ['GPUOrdinaryLeastSquares'],
  gpuContributorNames: ['GPUOrdinaryLeastSquares'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Observations', ['point'], 100000, 100000)],
  parameters: [
    makeStringParameter('responseProperty', 'Response property'),
    makeStringParameter('predictorProperties', 'Predictor properties (comma separated)')
  ],
  outputs: [
    makeOutputSpec('ols-coefficients', 'OLS coefficients', 'none', ['term'], 16),
    makeOutputSpec(
      'ols-residuals',
      'OLS fitted values and residuals',
      'point',
      ['feature_index'],
      100000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Complete cases only: points without a position or with a missing, non-numeric or float32-overflowing response or predictor are skipped and counted.',
    'Needs more complete rows than coefficients (n > predictors + 1); collinear predictors throw on the CPU engine, and the GPU engine hands a singular fit to the CPU engine.',
    'Standard errors use RSS / (n - p) with p counting the intercept; the GPU engine solves float32 centered normal equations, so it matches the CPU float64 fit to float32 tolerance.'
  ],
  limitation:
    'Aspatial OLS: no spatial lag or error terms, geographically weighted regression, robust covariance or diagnostics tables (the contributor computes Jarque-Bera, Koenker-Bassett and Breusch-Pagan but they are not published).',
  run: (inputs, values) => {
    const sample = readRegressionSample(inputs['points'], values);
    return buildRegressionOutput(sample, fitOrdinaryLeastSquares(sample));
  },
  runGpu: async (context, inputs, values) => {
    const sample = readRegressionSample(inputs['points'], values);
    return buildRegressionOutput(sample, await fitOrdinaryLeastSquaresGpu(context.device, sample), [
      'GPUOrdinaryLeastSquares'
    ]);
  }
};

export const PLATFORM4_REGRESSION_BINDINGS: readonly CatalogOperationBinding[] = [OLS_BINDING];
