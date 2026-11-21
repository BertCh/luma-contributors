// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import type {Device} from '@luma.gl/core';
import {
  GPUGeographicallyWeightedRegression,
  GPUProbitRegression,
  GPUSeeminglyUnrelatedRegression,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  collectAllPositions,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  readNumericProperty,
  type Values
} from './catalog-platform3-b-common';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';
import {makeRecipeEnumParameter} from './catalog-platform5-recipe-common';
import {solveDenseLinearSystem} from './dense-linear-system';

const MAXIMUM_PREDICTORS = 10;

type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;

/** Complete cases of a point regression in a local frame; positions and data are float32 rounded. */
type RegressionRows = {
  frame: LocalFrame;
  response: Float64Array;
  predictors: Float64Array;
  predictorCount: number;
  positions: Float64Array;
  featureIndex: Int32Array;
  coordinates: [number, number][];
  count: number;
  droppedCount: number;
  names: string[];
};

function readRegressionRows(
  collection: CatalogFeatureCollection,
  values: Values,
  minimumPredictors = 1
): RegressionRows {
  const responseName = (values['responseProperty'] as string).trim();
  const names = (values['predictorProperties'] as string)
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  // The model needs a response and at least one predictor, at most ten.
  if (!responseName || names.length < minimumPredictors || names.length > MAXIMUM_PREDICTORS)
    throw new RangeError('Name a response and 1 to 10 comma separated predictor properties');
  const frame = makeLocalFrame(
    values['coordinateSystem'] as CatalogCoordinateSystem,
    collectAllPositions(collection),
    {shift: 'minimum'}
  );
  const response: number[] = [];
  const predictors: number[] = [];
  const positions: number[] = [];
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
    const [x, localY] = frame.forward(position);
    positions.push(Math.fround(x), Math.fround(localY));
    featureIndex.push(index);
    coordinates.push([position[0], position[1]]);
  });
  return {
    frame,
    response: Float64Array.from(response),
    predictors: Float64Array.from(predictors),
    predictorCount: names.length,
    positions: Float64Array.from(positions),
    featureIndex: Int32Array.from(featureIndex),
    coordinates,
    count: response.length,
    droppedCount,
    names
  };
}

function addDroppedRows(diagnostics: CatalogDiagnostic[], count: number): void {
  if (count)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position or with a missing or non-numeric response or predictor are skipped (complete cases only).',
      count
    );
}

function nullIfNaN(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

// ---------------------------------------------------------------------------------------------
// Geographically weighted regression
// ---------------------------------------------------------------------------------------------

type GwrFit = {
  /** Row-major rows x (predictors + 1); NaN for singular rows. */
  coefficients: Float64Array;
  standardErrors: Float64Array;
  fitted: Float64Array;
  residuals: Float64Array;
  hat: Float64Array;
};

function getGwrWeight(kernel: string, distance: number, bandwidth: number): number {
  const ratio = distance / bandwidth;
  return kernel === 'gaussian'
    ? Math.exp(-0.5 * ratio * ratio)
    : ratio < 1
      ? (1 - ratio * ratio) ** 2
      : 0;
}

function fitGwrCpu(rows: RegressionRows, kernel: string, bandwidth: number): GwrFit {
  const {count, predictorCount} = rows;
  const size = predictorCount + 1;
  const fit: GwrFit = {
    coefficients: new Float64Array(count * size).fill(Number.NaN),
    standardErrors: new Float64Array(count * size).fill(Number.NaN),
    fitted: new Float64Array(count).fill(Number.NaN),
    residuals: new Float64Array(count).fill(Number.NaN),
    hat: new Float64Array(count).fill(Number.NaN)
  };
  const weights = new Float64Array(count);
  for (let location = 0; location < count; location++) {
    // Center about the focal observation before accumulating means, then equilibrate each
    // predictor to weighted unit variance. A coordinate translation must not imply singularity.
    const origins = rows.predictors.slice(
      location * predictorCount,
      (location + 1) * predictorCount
    );
    const responseOrigin = rows.response[location];
    const means = new Float64Array(predictorCount);
    const momentMeans = new Float64Array(predictorCount);
    const centeredMoments = new Float64Array(predictorCount);
    let responseMean = 0;
    let weightSum = 0;
    for (let row = 0; row < count; row++) {
      const weight = getGwrWeight(
        kernel,
        Math.hypot(
          rows.positions[2 * row] - rows.positions[2 * location],
          rows.positions[2 * row + 1] - rows.positions[2 * location + 1]
        ),
        bandwidth
      );
      weights[row] = weight;
      if (!weight) continue;
      const previousWeightSum = weightSum;
      weightSum += weight;
      const fraction = weight / weightSum;
      const momentWeight = weight * (previousWeightSum / weightSum);
      const differences = Array.from(
        momentMeans,
        (mean, column) => rows.predictors[row * predictorCount + column] - origins[column] - mean
      );
      for (let column = 0; column < predictorCount; column++) {
        centeredMoments[column] += momentWeight * differences[column] * differences[column];
        momentMeans[column] += fraction * differences[column];
        means[column] +=
          weight * (rows.predictors[row * predictorCount + column] - origins[column]);
      }
      responseMean += weight * (rows.response[row] - responseOrigin);
    }
    responseMean /= weightSum;
    for (let column = 0; column < predictorCount; column++) means[column] /= weightSum;
    const scales = Float64Array.from(centeredMoments, moment => Math.sqrt(moment / weightSum));
    if (scales.some(scale => !(scale > 0))) continue;
    const design = (row: number) => [
      1,
      ...Array.from(
        scales,
        (scale, column) =>
          (rows.predictors[row * predictorCount + column] - origins[column] - means[column]) / scale
      )
    ];
    const gram = Array.from({length: size}, () => new Float64Array(size));
    const cross = Array.from({length: size}, () => new Float64Array(1));
    // Re-evaluate centered cross moments after the means are fixed: online response covariance
    // can lose a distant group's contribution when its weight is tiny relative to the focal group.
    for (let row = 0; row < count; row++) {
      const weight = weights[row];
      if (!weight) continue;
      const x = design(row);
      for (let a = 0; a < size; a++) {
        cross[a][0] += weight * x[a] * (rows.response[row] - responseOrigin - responseMean);
        for (let b = 0; b < size; b++) gram[a][b] += weight * x[a] * x[b];
      }
    }
    // The inverse is needed for the hat value and the covariance; solve against the identity too.
    const inverse = Array.from({length: size}, (_, a) => {
      const unit = new Float64Array(size);
      unit[a] = 1;
      return unit;
    });
    if (
      !solveDenseLinearSystem(
        gram.map(row => row.slice()),
        inverse
      )
    )
      continue;
    const solution = Array.from({length: size}, (_, a) => {
      let value = 0;
      for (let b = 0; b < size; b++) value += inverse[a][b] * cross[b][0];
      return value;
    });
    const x = design(location);
    let fitted = 0;
    let hat = 0;
    for (let a = 0; a < size; a++) {
      fitted += x[a] * solution[a];
      for (let b = 0; b < size; b++) hat += x[a] * inverse[a][b] * x[b];
    }
    // Transform the normalized coefficients and covariance to the caller's coordinates.
    // Keep the large origin cancellation separate from the small centered correction.
    let interceptOrigin = responseOrigin;
    let interceptCorrection = responseMean + solution[0];
    for (let column = 0; column < predictorCount; column++) {
      const slope = solution[column + 1] / scales[column];
      fit.coefficients[location * size + column + 1] = slope;
      interceptOrigin -= origins[column] * slope;
      interceptCorrection -= means[column] * slope;
    }
    fit.coefficients[location * size] = interceptOrigin + interceptCorrection;
    fit.fitted[location] = responseOrigin + (responseMean + fitted);
    fit.residuals[location] = rows.response[location] - responseOrigin - responseMean - fitted;
    fit.hat[location] = hat * weights[location];
    // Transform each weighted observation influence before squaring. W^2 can underflow even
    // when the original-coordinate influence is substantial (e.g. a distant, unique group).
    const inverseRows = Array.from({length: size}, () => new Float64Array(size));
    for (let a = 0; a < size; a++) {
      const inverseRow = inverseRows[a];
      for (let b = 0; b < size; b++) {
        if (a > 0) inverseRow[b] = inverse[a][b] / scales[a - 1];
        else {
          inverseRow[b] = inverse[0][b];
          for (let column = 0; column < predictorCount; column++)
            inverseRow[b] -=
              ((origins[column] + means[column]) / scales[column]) * inverse[column + 1][b];
        }
      }
      fit.standardErrors[location * size + a] = 0;
    }
    for (let row = 0; row < count; row++) {
      if (!weights[row]) continue;
      const weightedDesign = design(row).map(value => weights[row] * value);
      for (let coefficient = 0; coefficient < size; coefficient++) {
        let influence = 0;
        for (let column = 0; column < size; column++)
          influence += inverseRows[coefficient][column] * weightedDesign[column];
        fit.standardErrors[location * size + coefficient] += influence * influence;
      }
    }
  }
  // Scale the unit covariance by sigma squared = RSS / (n - tr(S)) over non-singular rows.
  const summary = summarizeGwr(fit);
  for (let index = 0; index < fit.standardErrors.length; index++)
    fit.standardErrors[index] = Math.sqrt(fit.standardErrors[index] * summary.sigmaSquared);
  return fit;
}

function summarizeGwr(fit: GwrFit) {
  let residualSumOfSquares = 0;
  let trace = 0;
  let included = 0;
  for (let row = 0; row < fit.fitted.length; row++) {
    if (!Number.isFinite(fit.residuals[row])) continue;
    residualSumOfSquares += fit.residuals[row] ** 2;
    trace += fit.hat[row];
    included++;
  }
  return {
    residualSumOfSquares,
    trace,
    included,
    sigmaSquared: residualSumOfSquares / (included - trace)
  };
}

async function fitGwrGpu(
  device: Device,
  rows: RegressionRows,
  kernel: string,
  bandwidth: number
): Promise<GwrFit> {
  const {count, predictorCount} = rows;
  const size = predictorCount + 1;
  return withCatalogGpuSession(device, 'catalog-regression-gwr', async session => {
    const coefficients = session.output('coefficients', 'float32', count * size);
    const standardErrors = session.output('standard-errors', 'float32', count * size);
    const fitted = session.output('fitted', 'float32', count);
    const residuals = session.output('residuals', 'float32', count);
    const hat = session.output('hat', 'float32', count);
    const status = session.output('status', 'uint32', count);
    session.graph.add(
      new GPUGeographicallyWeightedRegression({
        id: 'catalog-regression-gwr',
        positions: session.input('positions', Float32Array.from(rows.positions), 'float32x2'),
        predictors: session.input('predictors', Float32Array.from(rows.predictors), 'float32'),
        predictorCount,
        response: session.input('response', Float32Array.from(rows.response), 'float32'),
        parameters: session.input(
          'parameters',
          getGPUGeographicallyWeightedRegressionParameterValues(
            {
              kernel: kernel as 'gaussian' | 'bisquare',
              bandwidthMode: 'fixed',
              bandwidths: [bandwidth]
            },
            1
          ),
          'float32'
        ),
        maximumBandwidthCount: 1,
        refinement: false,
        indexGridSize: false,
        output: {
          coefficients: coefficients.view,
          standardErrors: standardErrors.view,
          fitted: fitted.view,
          residuals: residuals.view,
          hatDiagonal: hat.view,
          localStatus: status.view
        }
      })
    );
    session.run();
    const statusValues = await session.readUint32(status);
    // Excluded rows cannot occur for complete cases; only singular systems stay NaN.
    if (
      statusValues.some(value => value === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED)
    )
      throw new CatalogGpuUnsupportedError('The GPU excluded rows with non finite values');
    return {
      coefficients: Float64Array.from(await session.readFloat32(coefficients)),
      standardErrors: Float64Array.from(await session.readFloat32(standardErrors)),
      fitted: Float64Array.from(await session.readFloat32(fitted)),
      residuals: Float64Array.from(await session.readFloat32(residuals)),
      hat: Float64Array.from(await session.readFloat32(hat))
    };
  });
}

function buildGwrOutput(
  rows: RegressionRows,
  fit: GwrFit,
  bandwidth: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedRows(diagnostics, rows.droppedCount);
  const terms = ['intercept', ...rows.names];
  const size = terms.length;
  const table = makeCatalogTable(
    'gwr-local',
    'GWR local coefficients',
    {
      feature_index: 'int32',
      ...Object.fromEntries(terms.map(term => [`coef_${term}`, 'float64' as const])),
      ...Object.fromEntries(terms.map(term => [`se_${term}`, 'float64' as const])),
      fitted: 'float64',
      residual: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let row = 0; row < rows.count; row++)
    appendCatalogRow(table, {
      feature_index: rows.featureIndex[row],
      ...Object.fromEntries(
        terms.map((term, a) => [`coef_${term}`, nullIfNaN(fit.coefficients[row * size + a])])
      ),
      ...Object.fromEntries(
        terms.map((term, a) => [`se_${term}`, nullIfNaN(fit.standardErrors[row * size + a])])
      ),
      fitted: nullIfNaN(fit.fitted[row]),
      residual: nullIfNaN(fit.residuals[row]),
      geometry: makeCatalogGeometryText('Point', rows.coordinates[row])
    });
  const summary = summarizeGwr(fit);
  const singularCount = rows.count - summary.included;
  validateRequirement(
    'regression-gwr',
    singularCount <= rows.count * 0.1,
    `predictorProperties/bandwidth produced ${singularCount}/${rows.count} singular local fits, exceeding the 10 percent limit; remove collinear predictors or increase bandwidth`
  );
  if (singularCount)
    addCatalogDiagnostic(
      diagnostics,
      'singular-local-system',
      'warning',
      'Locations whose weighted local system is singular have null coefficients; they are excluded from the fit statistics.',
      singularCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {
      rowCount: rows.count,
      bandwidth,
      residualSumOfSquares: summary.residualSumOfSquares,
      effectiveParameters: summary.trace,
      sigmaSquared: summary.sigmaSquared,
      singularCount
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readGwrArguments(inputs: Inputs, values: Values) {
  const rows = readRegressionRows(inputs['points'], values);
  // Local fits need more locations than coefficients.
  if (rows.count < rows.predictorCount + 3) throw new RangeError('Too few complete rows');
  let bandwidth = values['bandwidth'] as number | undefined;
  if (bandwidth === undefined) {
    // One fixed bandwidth includes at least half the observations at every location.
    const neighborCount = Math.min(
      rows.count - 1,
      Math.max(rows.predictorCount + 3, Math.ceil(rows.count / 2))
    );
    bandwidth = 0;
    for (let location = 0; location < rows.count; location++) {
      const distances = Array.from({length: rows.count}, (_, row) =>
        Math.hypot(
          rows.positions[2 * row] - rows.positions[2 * location],
          rows.positions[2 * row + 1] - rows.positions[2 * location + 1]
        )
      ).sort((first, second) => first - second);
      bandwidth = Math.max(bandwidth, distances[neighborCount] * 1.05);
    }
    bandwidth = Math.max(bandwidth, 1e-6);
  }
  validateRequirement(
    'regression-gwr',
    bandwidth > 0,
    'bandwidth must exceed zero; omit bandwidth for estimation'
  );
  return {rows, kernel: values['kernel'] as string, bandwidth};
}

const GWR_BINDING: CatalogOperationBinding = {
  id: 'regression-gwr',
  title: 'Geographically weighted regression',
  domain: 'analysis',
  summary:
    'Fits a local weighted least squares model at every point with a fixed kernel bandwidth and publishes local coefficients, standard errors, fitted values and residuals.',
  capabilityIds: ['spatial-regression'],
  contributorNames: ['GPUGeographicallyWeightedRegression'],
  gpuContributorNames: ['GPUGeographicallyWeightedRegression'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Observations', ['point'], 3000, 3000)],
  parameters: [
    makeStringParameter('responseProperty', 'Response property'),
    makeStringParameter('predictorProperties', 'Predictor properties (comma separated)'),
    makeRecipeEnumParameter('kernel', 'Kernel', ['gaussian', 'bisquare'], 'bisquare'),
    makeNumberParameter(
      'bandwidth',
      'Fixed bandwidth (default: derive from neighbors)',
      undefined,
      1e-9,
      1e9,
      'crs'
    ),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec('gwr-local', 'GWR local coefficients', 'point', ['feature_index'], 3000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Complete cases only; positions are mapped to a local meter frame (planar units for planar data) and rounded to float32, so the bandwidth is in meters for EPSG:4326.',
    'Omitted bandwidth defaults to 1.05 times the maximum distance to the k-th nearest point, where k=max(predictors+3, ceil(n/2)), capped at n-1; explicit positive bandwidths override this estimate. Fixed bandwidth with a Gaussian exp(-0.5 (d/h)^2) or bisquare (1 - (d/h)^2)^2 kernel; every location includes itself with weight 1.',
    "Standard errors follow mgwr: sqrt(diag(A^-1 X'W^2X A^-1) sigma^2) with sigma^2 = RSS / (n - tr(S)) over the non-singular locations.",
    'A location with a singular local system gets null coefficients and standard errors and is excluded from the fit statistics; more than 10 percent singular local fits throws an error naming predictorProperties/bandwidth and remedies.'
  ],
  limitation:
    'Fixed bandwidth (derived from the data by default): cross-validated bandwidth selection, adaptive bandwidths, local t values, collinearity diagnostics and the nonstationarity test of the contributor are not published.',
  run: (inputs, values) => {
    const {rows, kernel, bandwidth} = readGwrArguments(inputs, values);
    return buildGwrOutput(rows, fitGwrCpu(rows, kernel, bandwidth), bandwidth);
  },
  runGpu: async (context, inputs, values) => {
    const {rows, kernel, bandwidth} = readGwrArguments(inputs, values);
    return buildGwrOutput(
      rows,
      await fitGwrGpu(context.device, rows, kernel, bandwidth),
      bandwidth,
      ['GPUGeographicallyWeightedRegression']
    );
  }
};

// ---------------------------------------------------------------------------------------------
// Probit regression
// ---------------------------------------------------------------------------------------------

/** Standard normal density. */
function getNormalDensity(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Standard normal CDF from the all-positive Taylor series of erf (no cancellation). */
function getNormalCdf(x: number): number {
  const z = Math.abs(x);
  if (z > 8.5) return x > 0 ? 1 : 0;
  const argument = z / Math.SQRT2;
  let term = argument;
  let sum = argument;
  for (let n = 1; n < 400; n++) {
    term *= (2 * argument * argument) / (2 * n + 1);
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  const erf = (2 / Math.sqrt(Math.PI)) * Math.exp(-argument * argument) * sum;
  return x > 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

const PROBABILITY_FLOOR = 1e-7;

type ProbitFit = {
  coefficients: number[];
  standardErrors: number[];
  fitted: Float64Array;
  marginalEffects: number[];
  logLikelihood: number;
  deviance: number;
};

function fitProbitCpu(rows: RegressionRows): ProbitFit {
  const {count, predictorCount} = rows;
  const size = predictorCount + 1;
  // The outcome must be binary with both classes present.
  if (rows.response.some(value => value !== 0 && value !== 1))
    throw new RangeError('The response must be 0 or 1');
  const successes = rows.response.reduce((sum, value) => sum + value, 0);
  if (successes === 0 || successes === count) throw new RangeError('The response has one class');
  const design = (row: number) => [
    1,
    ...Array.from(rows.predictors.subarray(row * predictorCount, (row + 1) * predictorCount))
  ];
  const beta = new Array<number>(size).fill(0);
  const getProbability = (linear: number) =>
    Math.min(1 - PROBABILITY_FLOOR, Math.max(PROBABILITY_FLOOR, getNormalCdf(linear)));
  const getLogLikelihood = (candidate: readonly number[]) => {
    let total = 0;
    for (let row = 0; row < count; row++) {
      const x = design(row);
      const probability = getProbability(
        x.reduce((sum, value, a) => sum + value * candidate[a], 0)
      );
      total +=
        rows.response[row] * Math.log(probability) +
        (1 - rows.response[row]) * Math.log(1 - probability);
    }
    return total;
  };
  let current = getLogLikelihood(beta);
  let information: Float64Array[] = [];
  for (let iteration = 0; iteration < 200; iteration++) {
    information = Array.from({length: size}, () => new Float64Array(size));
    const gradient = [new Float64Array(size)];
    for (let row = 0; row < count; row++) {
      const x = design(row);
      const linear = x.reduce((sum, value, a) => sum + value * beta[a], 0);
      const probability = getProbability(linear);
      const density = getNormalDensity(linear);
      const score =
        (density * (rows.response[row] - probability)) / (probability * (1 - probability));
      const weight = (density * density) / (probability * (1 - probability));
      for (let a = 0; a < size; a++) {
        gradient[0][a] += score * x[a];
        for (let b = 0; b < size; b++) information[a][b] += weight * x[a] * x[b];
      }
    }
    const system = information.map(row => row.slice());
    const rightHandSide = Array.from(gradient[0], value => Float64Array.of(value));
    // A singular information matrix means collinear predictors or separation.
    if (!solveDenseLinearSystem(system, rightHandSide))
      throw new RangeError('Singular information matrix (collinear predictors)');
    let scale = 1;
    let candidate = beta;
    let candidateLikelihood = current;
    for (let attempt = 0; attempt < 30; attempt++) {
      candidate = beta.map((value, a) => value + scale * rightHandSide[a][0]);
      candidateLikelihood = getLogLikelihood(candidate);
      if (candidateLikelihood >= current - 1e-12) break;
      scale /= 2;
    }
    const largest = Math.max(...rightHandSide.map((value, a) => Math.abs(scale * value[0])));
    for (let a = 0; a < size; a++) beta[a] = candidate[a];
    current = candidateLikelihood;
    if (largest < 1e-10) break;
  }
  const inverse = Array.from({length: size}, (_, a) => {
    const unit = new Float64Array(size);
    unit[a] = 1;
    return unit;
  });
  if (
    !solveDenseLinearSystem(
      information.map(row => row.slice()),
      inverse
    )
  )
    throw new RangeError('Singular information matrix (collinear predictors)');
  const fitted = new Float64Array(count);
  const averageDensity = {value: 0};
  for (let row = 0; row < count; row++) {
    const linear = design(row).reduce((sum, value, a) => sum + value * beta[a], 0);
    fitted[row] = getProbability(linear);
    averageDensity.value += getNormalDensity(linear) / count;
  }
  return {
    coefficients: beta,
    standardErrors: inverse.map((row, a) => Math.sqrt(row[a])),
    fitted,
    marginalEffects: beta.slice(1).map(value => value * averageDensity.value),
    logLikelihood: current,
    deviance: -2 * current
  };
}

async function fitProbitGpu(device: Device, rows: RegressionRows): Promise<ProbitFit> {
  const {count, predictorCount} = rows;
  const size = predictorCount + 1;
  return withCatalogGpuSession(device, 'catalog-regression-probit', async session => {
    const coefficients = session.output('coefficients', 'float32', size);
    const standardErrors = session.output('standard-errors', 'float32', size);
    const fitted = session.output('fitted', 'float32', count);
    const marginalEffects = session.output('marginal-effects', 'float32', predictorCount);
    const summary = session.output('summary', 'float32', 4);
    const status = session.output('status', 'uint32', 1);
    session.graph.add(
      new GPUProbitRegression({
        id: 'catalog-regression-probit',
        predictors: session.input('predictors', Float32Array.from(rows.predictors), 'float32'),
        predictorCount,
        response: session.input('response', Float32Array.from(rows.response), 'float32'),
        output: {
          coefficients: coefficients.view,
          standardErrors: standardErrors.view,
          fitted: fitted.view,
          marginalEffects: marginalEffects.view,
          summary: summary.view,
          status: status.view
        }
      })
    );
    session.run();
    // Status 0 is a converged fit; the CPU engine reports separation and collinearity itself.
    if ((await session.readScalar(status)) !== 0)
      throw new CatalogGpuUnsupportedError('The GPU probit fit did not converge');
    const summaryValues = await session.readFloat32(summary);
    return {
      coefficients: Array.from(await session.readFloat32(coefficients)),
      standardErrors: Array.from(await session.readFloat32(standardErrors)),
      fitted: Float64Array.from(await session.readFloat32(fitted)),
      marginalEffects: Array.from(await session.readFloat32(marginalEffects)),
      logLikelihood: summaryValues[1],
      deviance: summaryValues[2]
    };
  });
}

function buildProbitOutput(
  rows: RegressionRows,
  fit: ProbitFit,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedRows(diagnostics, rows.droppedCount);
  const coefficients = makeCatalogTable('probit-coefficients', 'Probit coefficients', {
    term: 'string',
    estimate: 'float64',
    standard_error: 'float64',
    z_statistic: 'float64',
    average_marginal_effect: 'float64'
  });
  ['(intercept)', ...rows.names].forEach((term, index) =>
    appendCatalogRow(coefficients, {
      term,
      estimate: fit.coefficients[index],
      standard_error: fit.standardErrors[index],
      z_statistic: fit.coefficients[index] / fit.standardErrors[index],
      average_marginal_effect: index ? fit.marginalEffects[index - 1] : null
    })
  );
  const probabilities = makeCatalogTable(
    'probit-fitted',
    'Probit fitted probabilities',
    {feature_index: 'int32', probability: 'float64', geometry: 'string'},
    'geometry'
  );
  for (let row = 0; row < rows.count; row++)
    appendCatalogRow(probabilities, {
      feature_index: rows.featureIndex[row],
      probability: fit.fitted[row],
      geometry: makeCatalogGeometryText('Point', rows.coordinates[row])
    });
  return {
    tables: [coefficients, probabilities],
    diagnostics,
    summary: {
      rowCount: rows.count,
      predictorCount: rows.predictorCount,
      logLikelihood: fit.logLikelihood,
      deviance: fit.deviance
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function readProbitArguments(inputs: Inputs, values: Values) {
  const rows = readRegressionRows(inputs['points'], values);
  // The fit needs more complete rows than coefficients.
  if (rows.count <= rows.predictorCount + 1) throw new RangeError('Too few complete rows');
  return rows;
}

const PROBIT_BINDING: CatalogOperationBinding = {
  id: 'regression-probit',
  title: 'Probit regression',
  domain: 'analysis',
  summary:
    'Fits a binary (0 or 1) point property on numeric predictor properties by maximum likelihood and publishes coefficients, expected-information standard errors, z statistics, average marginal effects and fitted probabilities.',
  capabilityIds: ['spatial-regression'],
  contributorNames: ['GPUProbitRegression'],
  gpuContributorNames: ['GPUProbitRegression'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Observations', ['point'], 20000, 20000)],
  parameters: [
    makeStringParameter('responseProperty', 'Binary response property (0 or 1)'),
    makeStringParameter('predictorProperties', 'Predictor properties (comma separated)'),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec('probit-coefficients', 'Probit coefficients', 'none', ['term'], 16),
    makeOutputSpec(
      'probit-fitted',
      'Probit fitted probabilities',
      'point',
      ['feature_index'],
      20000
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Complete cases only; the response must be exactly 0 or 1 with both classes present, otherwise the run is refused.',
    'The CPU engine runs Fisher scoring with step halving in float64; the GPU engine runs the same scoring in float32 on standardized predictors, so estimates agree to float32 tolerance.',
    'Collinear predictors and separated samples are refused on the CPU engine; the GPU engine hands a non-converged fit to the CPU engine.',
    'Probabilities are floored at 1e-7. Average marginal effects are averages of the density times the coefficient over the sample, not effects at the means.'
  ],
  limitation:
    'Aspatial binary probit: the Pinkse, Kelejian-Prucha and Pinkse-Slade residual diagnostics of the contributor need spatial weights and are not published.',
  run: (inputs, values) => {
    const rows = readProbitArguments(inputs, values);
    return buildProbitOutput(rows, fitProbitCpu(rows));
  },
  runGpu: async (context, inputs, values) => {
    const rows = readProbitArguments(inputs, values);
    return buildProbitOutput(rows, await fitProbitGpu(context.device, rows), [
      'GPUProbitRegression'
    ]);
  }
};

// ---------------------------------------------------------------------------------------------
// Seemingly unrelated regression
// ---------------------------------------------------------------------------------------------

type SurRows = {
  equationCount: number;
  predictorCount: number;
  count: number;
  responseNames: string[];
  predictorNames: string[][];
  /** Equation-major responses, then row-major predictors per equation. */
  responses: Float64Array;
  predictors: Float64Array;
  featureIndex: Int32Array;
  coordinates: [number, number][];
  droppedCount: number;
};

function readSurRows(collection: CatalogFeatureCollection, values: Values): SurRows {
  const responseNames = (values['responseProperties'] as string)
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  const predictorNames = (values['predictorProperties'] as string)
    .split(';')
    .map(group =>
      group
        .split(',')
        .map(name => name.trim())
        .filter(Boolean)
    )
    .filter(group => group.length);
  const equationCount = responseNames.length;
  const predictorCount = predictorNames[0]?.length ?? 0;
  // Two to four equations with one to seven predictors each, one predictor list per equation.
  if (
    equationCount < 2 ||
    equationCount > 4 ||
    predictorNames.length !== equationCount ||
    predictorCount < 1 ||
    predictorCount > 7 ||
    predictorNames.some(group => group.length !== predictorCount) ||
    equationCount * (predictorCount + 1) > 16
  )
    throw new RangeError(
      'Name 2 to 4 responses and one list of equally many predictors (at most 7) per equation, separated by semicolons'
    );
  const responses: number[][] = responseNames.map(() => []);
  const predictors: number[][] = responseNames.map(() => []);
  const featureIndex: number[] = [];
  const coordinates: [number, number][] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const position = getPointPositions(feature.geometry)[0];
    const rowResponses = responseNames.map(name => readNumericProperty(feature, name));
    const rowPredictors = predictorNames.map(group =>
      group.map(name => readNumericProperty(feature, name))
    );
    const all = [...rowResponses, ...rowPredictors.flat()];
    // Complete cases across every equation, finite after float32 rounding.
    if (
      !position ||
      all.some(value => value === null || !Number.isFinite(Math.fround(value as number)))
    ) {
      droppedCount++;
      return;
    }
    rowResponses.forEach((value, equation) => responses[equation].push(Math.fround(value!)));
    rowPredictors.forEach((group, equation) =>
      group.forEach(value => predictors[equation].push(Math.fround(value!)))
    );
    featureIndex.push(index);
    coordinates.push([position[0], position[1]]);
  });
  const count = featureIndex.length;
  // The system fit needs more rows than coefficients per equation.
  if (count <= predictorCount + 1) throw new RangeError('Too few complete rows');
  return {
    equationCount,
    predictorCount,
    count,
    responseNames,
    predictorNames,
    responses: Float64Array.from(responses.flat()),
    predictors: Float64Array.from(predictors.flat()),
    featureIndex: Int32Array.from(featureIndex),
    coordinates,
    droppedCount
  };
}

type SurFit = {
  coefficients: Float64Array;
  covariance: Float64Array;
  residuals: Float64Array;
};

function fitSurCpu(rows: SurRows): SurFit {
  const {equationCount, predictorCount, count} = rows;
  const columns = predictorCount + 1;
  const size = equationCount * columns;
  const getDesign = (equation: number, row: number) => [
    1,
    ...Array.from(
      rows.predictors.subarray(
        (equation * count + row) * predictorCount,
        (equation * count + row + 1) * predictorCount
      )
    )
  ];
  const getResponse = (equation: number, row: number) => rows.responses[equation * count + row];
  // Cross products X_k'X_l and X_k'y_l.
  const crossDesign = (k: number, l: number) => {
    const block = Array.from({length: columns}, () => new Float64Array(columns));
    for (let row = 0; row < count; row++) {
      const xk = getDesign(k, row);
      const xl = getDesign(l, row);
      for (let a = 0; a < columns; a++)
        for (let b = 0; b < columns; b++) block[a][b] += xk[a] * xl[b];
    }
    return block;
  };
  const crossResponse = (k: number, l: number) => {
    const vector = new Float64Array(columns);
    for (let row = 0; row < count; row++) {
      const xk = getDesign(k, row);
      for (let a = 0; a < columns; a++) vector[a] += xk[a] * getResponse(l, row);
    }
    return vector;
  };
  // First stage: equation OLS residuals and their covariance with divisor n.
  const residuals = new Float64Array(equationCount * count);
  for (let equation = 0; equation < equationCount; equation++) {
    const gram = crossDesign(equation, equation);
    const right = [crossResponse(equation, equation)].flatMap(vector =>
      Array.from(vector, value => Float64Array.of(value))
    );
    if (!solveDenseLinearSystem(gram, right))
      throw new RangeError('Singular design in an equation');
    for (let row = 0; row < count; row++)
      residuals[equation * count + row] =
        getResponse(equation, row) -
        getDesign(equation, row).reduce((sum, value, a) => sum + value * right[a][0], 0);
  }
  const errorCovariance = Array.from(
    {length: equationCount},
    () => new Float64Array(equationCount)
  );
  for (let k = 0; k < equationCount; k++)
    for (let l = 0; l < equationCount; l++) {
      for (let row = 0; row < count; row++)
        errorCovariance[k][l] += residuals[k * count + row] * residuals[l * count + row];
      errorCovariance[k][l] /= count;
    }
  const inverseCovariance = Array.from({length: equationCount}, (_, k) => {
    const unit = new Float64Array(equationCount);
    unit[k] = 1;
    return unit;
  });
  // A singular residual covariance means linearly dependent equations.
  if (
    !solveDenseLinearSystem(
      errorCovariance.map(row => row.slice()),
      inverseCovariance
    )
  )
    throw new RangeError('Singular error covariance between equations');
  const information = Array.from({length: size}, () => new Float64Array(size));
  const right = Array.from({length: size}, () => new Float64Array(1));
  for (let k = 0; k < equationCount; k++) {
    for (let l = 0; l < equationCount; l++) {
      const block = crossDesign(k, l);
      for (let a = 0; a < columns; a++)
        for (let b = 0; b < columns; b++)
          information[k * columns + a][l * columns + b] = inverseCovariance[k][l] * block[a][b];
    }
    for (let m = 0; m < equationCount; m++) {
      const vector = crossResponse(k, m);
      for (let a = 0; a < columns; a++)
        right[k * columns + a][0] += inverseCovariance[k][m] * vector[a];
    }
  }
  const inverse = Array.from({length: size}, (_, a) => {
    const unit = new Float64Array(size);
    unit[a] = 1;
    return unit;
  });
  if (
    !solveDenseLinearSystem(
      information.map(row => row.slice()),
      inverse
    )
  )
    throw new RangeError('Singular system information matrix');
  const coefficients = new Float64Array(size);
  for (let a = 0; a < size; a++)
    for (let b = 0; b < size; b++) coefficients[a] += inverse[a][b] * right[b][0];
  const finalResiduals = new Float64Array(equationCount * count);
  for (let equation = 0; equation < equationCount; equation++)
    for (let row = 0; row < count; row++)
      finalResiduals[equation * count + row] =
        getResponse(equation, row) -
        getDesign(equation, row).reduce(
          (sum, value, a) => sum + value * coefficients[equation * columns + a],
          0
        );
  return {
    coefficients,
    covariance: Float64Array.from(inverse.flatMap(row => Array.from(row))),
    residuals: finalResiduals
  };
}

async function fitSurGpu(device: Device, rows: SurRows): Promise<SurFit> {
  const {equationCount, predictorCount, count} = rows;
  const size = equationCount * (predictorCount + 1);
  return withCatalogGpuSession(device, 'catalog-regression-sur', async session => {
    const coefficients = session.output('coefficients', 'float32', size);
    const covariance = session.output('covariance', 'float32', size * size);
    const residualCovariance = session.output(
      'residual-covariance',
      'float32',
      equationCount * equationCount
    );
    const residuals = session.output('residuals', 'float32', equationCount * count);
    const status = session.output('status', 'uint32', 1);
    session.graph.add(
      new GPUSeeminglyUnrelatedRegression({
        id: 'catalog-regression-sur',
        equationCount,
        predictorCount,
        rowCount: count,
        predictors: session.input('predictors', Float32Array.from(rows.predictors), 'float32'),
        responses: session.input('responses', Float32Array.from(rows.responses), 'float32'),
        output: {
          coefficients: coefficients.view,
          covariance: covariance.view,
          residualCovariance: residualCovariance.view,
          residuals: residuals.view,
          status: status.view
        }
      })
    );
    session.run();
    // Status 0 is a successful FGLS; the CPU engine reports singular designs itself.
    if ((await session.readScalar(status)) !== 0)
      throw new CatalogGpuUnsupportedError('The GPU system fit was singular or underdetermined');
    return {
      coefficients: Float64Array.from(await session.readFloat32(coefficients)),
      covariance: Float64Array.from(await session.readFloat32(covariance)),
      residuals: Float64Array.from(await session.readFloat32(residuals))
    };
  });
}

function buildSurOutput(
  rows: SurRows,
  fit: SurFit,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  addDroppedRows(diagnostics, rows.droppedCount);
  const columns = rows.predictorCount + 1;
  const size = rows.equationCount * columns;
  const coefficients = makeCatalogTable('sur-coefficients', 'SUR coefficients', {
    equation: 'string',
    term: 'string',
    estimate: 'float64',
    standard_error: 'float64',
    z_statistic: 'float64'
  });
  for (let equation = 0; equation < rows.equationCount; equation++)
    ['(intercept)', ...rows.predictorNames[equation]].forEach((term, a) => {
      const index = equation * columns + a;
      const standardError = Math.sqrt(fit.covariance[index * size + index]);
      appendCatalogRow(coefficients, {
        equation: rows.responseNames[equation],
        term,
        estimate: fit.coefficients[index],
        standard_error: standardError,
        z_statistic: fit.coefficients[index] / standardError
      });
    });
  const residuals = makeCatalogTable(
    'sur-residuals',
    'SUR residuals',
    {
      feature_index: 'int32',
      ...Object.fromEntries(
        rows.responseNames.map((_, k) => [`residual_${k}`, 'float64' as const])
      ),
      geometry: 'string'
    },
    'geometry'
  );
  for (let row = 0; row < rows.count; row++)
    appendCatalogRow(residuals, {
      feature_index: rows.featureIndex[row],
      ...Object.fromEntries(
        rows.responseNames.map((_, k) => [`residual_${k}`, fit.residuals[k * rows.count + row]])
      ),
      geometry: makeCatalogGeometryText('Point', rows.coordinates[row])
    });
  return {
    tables: [coefficients, residuals],
    diagnostics,
    summary: {rowCount: rows.count, equationCount: rows.equationCount},
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const SUR_BINDING: CatalogOperationBinding = {
  id: 'regression-sur',
  title: 'Seemingly unrelated regression',
  domain: 'analysis',
  summary:
    'Fits two to four related equations jointly by one-step feasible GLS (equation OLS, joint residual covariance, system GLS) and publishes coefficients, standard errors and residuals per equation.',
  capabilityIds: ['spatial-regression'],
  contributorNames: ['GPUSeeminglyUnrelatedRegression'],
  gpuContributorNames: ['GPUSeeminglyUnrelatedRegression'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Observations', ['point'], 20000, 20000)],
  parameters: [
    makeStringParameter(
      'responseProperties',
      'Response properties, one per equation (comma separated)'
    ),
    makeStringParameter(
      'predictorProperties',
      'Predictors per equation (comma separated, equations separated by semicolons)'
    ),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec('sur-coefficients', 'SUR coefficients', 'none', ['equation', 'term'], 64),
    makeOutputSpec('sur-residuals', 'SUR residuals', 'point', ['feature_index'], 20000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Complete cases across all equations; every equation has the same number of predictors (at most 7) and an intercept, and the system has at most 16 coefficients.',
    'The residual covariance of the first-stage equation fits uses divisor n; standard errors are the conditional GLS information inverse using that covariance (no small-sample correction).',
    'The CPU engine solves the system in float64; the GPU engine uses float32 reductions, so estimates agree to float32 tolerance.'
  ],
  limitation:
    'Nonspatial one-step SUR: the spatial-error SUR mode, iterated SUR and a user supplied first-stage residual of the contributor are not exposed.',
  run: (inputs, values) => {
    const rows = readSurRows(inputs['points'], values);
    return buildSurOutput(rows, fitSurCpu(rows));
  },
  runGpu: async (context, inputs, values) => {
    const rows = readSurRows(inputs['points'], values);
    return buildSurOutput(rows, await fitSurGpu(context.device, rows), [
      'GPUSeeminglyUnrelatedRegression'
    ]);
  }
};

export const PLATFORM5_REGRESSION_BINDINGS: readonly CatalogOperationBinding[] = [
  GWR_BINDING,
  PROBIT_BINDING,
  SUR_BINDING
];
