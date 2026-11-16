// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getChiSquareSurvival,
  getStudentTTwoSidedPValue
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature
} from '../../analysis/catalog-geojson';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  SEED_PARAMETER_SPEC,
  getFeatureGeometryText,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  readNumericProperty,
  type Inputs,
  type Values
} from '../../analysis/catalog-platform3-b-common';
import {buildAnalysisSet} from '../../analysis/catalog-platform3-b-statistics';
import {KeplerAllocationError} from '../../runtime/dataset-registry';
import {
  MAXIMUM_CURATED_AREA_FEATURES,
  MAXIMUM_CURATED_AREA_VERTICES,
  MAXIMUM_PERMUTATION_WORK,
  isConstantList,
  makeColumnParameter,
  makeCuratedError,
  getFeatureAnchors,
  getCanonicalOrder,
  makeDerivedCollection,
  makePermutationParameter,
  readColumnParameter,
  runCanonicalMoranPermutationTest
} from './curated-binding-utils';

import {runCuratedRegressionOnGpu} from './curated-regression-with-diagnostics-gpu';

const BINDING_ID = 'curated-regression-with-diagnostics';

/** Same cap as the catalog `regression-ols` binding: a design of at most 16 columns including the intercept. */
const MAXIMUM_PREDICTORS = 15;

/** The Moran statistic of the residuals needs at least four rows that have a neighbour. */
const MINIMUM_ROWS_FOR_MORAN = 4;

/**
 * The comma separated predictor list may name 15 columns; 512 characters allow 15 names of 32
 * characters and the separators.
 */
const MAXIMUM_PREDICTOR_LIST_LENGTH = 512;

/**
 * A residual sum of squares below this fraction of the total sum of squares is a perfect fit: the
 * log-likelihood and the residual tests are not defined for residuals that are rounding error.
 */
export const PERFECT_FIT_FRACTION = 1e-24;

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeColumnParameter('responseProperty', 'Response property'),
  makeColumnParameter(
    'predictorProperties',
    'Predictor properties (comma separated)',
    MAXIMUM_PREDICTOR_LIST_LENGTH
  ),
  {
    name: 'neighborhood',
    label: 'Neighbours for the residual test',
    kind: 'enum',
    options: ['auto', 'queen', 'rook', 'knn'],
    defaultValue: 'auto'
  },
  makeNumberParameter('neighborCount', 'Nearest neighbors (knn)', 6, 1, 64, undefined, 'integer'),
  makePermutationParameter(),
  SEED_PARAMETER_SPEC,
  COORDINATE_SYSTEM_PARAMETER
];

/**
 * Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations (Golub and Van Loan,
 * Matrix Computations, section 8.4). The design matrix of a regression has at most 16 columns, so
 * the cost is negligible; the method keeps high relative accuracy for the smallest eigenvalue of a
 * positive definite matrix, which the condition number needs.
 */
export function computeSymmetricEigenvalues(matrix: readonly (readonly number[])[]): number[] {
  const size = matrix.length;
  const working = matrix.map(row => [...row]);
  for (let sweep = 0; sweep < 100; sweep++) {
    let offDiagonal = 0;
    for (let row = 0; row < size; row++)
      for (let column = row + 1; column < size; column++) offDiagonal += working[row][column] ** 2;
    if (offDiagonal < 1e-300) break;
    for (let row = 0; row < size; row++)
      for (let column = row + 1; column < size; column++) {
        const entry = working[row][column];
        if (entry === 0) continue;
        const theta = (working[column][column] - working[row][row]) / (2 * entry);
        const tangent = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cosine = 1 / Math.sqrt(tangent * tangent + 1);
        const sine = tangent * cosine;
        for (let index = 0; index < size; index++) {
          const first = working[index][row];
          const second = working[index][column];
          working[index][row] = cosine * first - sine * second;
          working[index][column] = sine * first + cosine * second;
        }
        for (let index = 0; index < size; index++) {
          const first = working[row][index];
          const second = working[column][index];
          working[row][index] = cosine * first - sine * second;
          working[column][index] = sine * first + cosine * second;
        }
      }
  }
  return working.map((row, index) => row[index]);
}

/**
 * Belsley, Kuh and Welsch (1980) condition number of a design matrix with its intercept column:
 * the columns are scaled to unit length and the condition number is the ratio of the largest to
 * the smallest singular value of the scaled matrix, taken here as the square root of the ratio of
 * the extreme eigenvalues of the scaled `X'X`. Values above about 30 flag harmful collinearity.
 */
export function computeBelsleyConditionNumber(
  predictors: ArrayLike<number>,
  predictorCount: number,
  rowCount: number
): number {
  const width = predictorCount + 1;
  const norms = new Array<number>(width).fill(0);
  const scales = new Float64Array(width);
  scales[0] = 1;
  // Unit-length scaling must not square predictors in their original units either.
  for (let row = 0; row < rowCount; row++)
    for (let column = 0; column < predictorCount; column++)
      scales[column + 1] = Math.max(
        scales[column + 1],
        Math.abs(predictors[row * predictorCount + column])
      );
  for (let row = 0; row < rowCount; row++) {
    norms[0] += 1;
    for (let column = 0; column < predictorCount; column++)
      norms[column + 1] +=
        (predictors[row * predictorCount + column] / (scales[column + 1] || 1)) ** 2;
  }
  for (let column = 0; column < width; column++) norms[column] = Math.sqrt(norms[column]);
  const gram = Array.from({length: width}, () => new Array<number>(width).fill(0));
  for (let row = 0; row < rowCount; row++) {
    const scaled = [1 / norms[0]];
    for (let column = 0; column < predictorCount; column++)
      scaled.push(
        predictors[row * predictorCount + column] / (scales[column + 1] || 1) / norms[column + 1]
      );
    for (let first = 0; first < width; first++)
      for (let second = 0; second < width; second++)
        gram[first][second] += scaled[first] * scaled[second];
  }
  const eigenvalues = computeSymmetricEigenvalues(gram);
  return Math.sqrt(Math.max(...eigenvalues) / Math.min(...eigenvalues));
}

/**
 * A predictor column that cannot be used: constant, or an almost exact linear combination of the
 * others. The decision is made on standardised columns, so it does not depend on the units.
 */
export class PredictorIdentificationError extends Error {
  constructor(
    readonly kind: 'constant' | 'collinear',
    readonly column: number
  ) {
    super(`predictor column ${column} is ${kind}`);
  }
}

/**
 * Smallest share of a predictor's variance that its fit on the other predictors may leave over
 * (1 - R squared of that fit). Below 1e-10 the variance inflation factor exceeds 1e10, more than
 * float64 normal equations can resolve, and the column is treated as collinear.
 */
const MINIMUM_UNEXPLAINED_PREDICTOR_VARIANCE = 1e-10;

export type StandardizedOlsFit = {
  /** Intercept first, in the units of the data. */
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

/**
 * Ordinary least squares with an intercept on standardised predictors. Each predictor is centred
 * and divided by its standard deviation, the normal equations are those of the correlation matrix
 * (Cholesky, so a rank deficiency shows as a vanishing pivot, whatever the units), and the
 * coefficients, standard errors and intercept are transformed back exactly:
 * `slope_j = b_j / sd_j`, `se(slope_j) = se(b_j) / sd_j` and
 * `var(intercept) = sigma^2 (1/n + sum_jk (m_j/sd_j)(m_k/sd_k) (R^-1)_jk / n)` with `m` the means.
 * Fitted values use the centred form, so a predictor with a large mean loses no precision.
 * The catalog OLS now standardises too, but this fit also scales offsets before squaring to
 * preserve extreme predictor units, and rejects unexplained variance below 1e-10 rather than
 * the catalog's 1e-12 pivot threshold. AX-87 differential probes retain these distinct safeguards.
 * @throws PredictorIdentificationError for a constant or collinear predictor.
 */
export function fitStandardizedLeastSquares(
  response: ArrayLike<number>,
  predictors: ArrayLike<number>,
  predictorCount: number,
  rowCount: number
): StandardizedOlsFit {
  const means = new Float64Array(predictorCount);
  const origins = new Float64Array(predictorCount);
  const meanOffsets = new Float64Array(predictorCount);
  const deviations = new Float64Array(predictorCount);
  const offsetScales = new Float64Array(predictorCount);
  for (let column = 0; column < predictorCount; column++) {
    const origin = predictors[column];
    let offsetScale = 0;
    for (let row = 0; row < rowCount; row++)
      offsetScale = Math.max(
        offsetScale,
        Math.abs(predictors[row * predictorCount + column] - origin)
      );
    if (!(offsetScale > 0)) throw new PredictorIdentificationError('constant', column);
    // Scale offsets before summing or squaring: units as small as 1e-250 or as large as
    // 1e250 must give the same correlation matrix, without underflow or overflow.
    let meanOffset = 0;
    for (let row = 0; row < rowCount; row++)
      meanOffset += (predictors[row * predictorCount + column] - origin) / offsetScale / rowCount;
    let squares = 0;
    for (let row = 0; row < rowCount; row++)
      squares +=
        ((predictors[row * predictorCount + column] - origin) / offsetScale - meanOffset) ** 2;
    const deviation = Math.sqrt(squares / rowCount);
    // Constancy has a zero bound on centred spread, independent of the mean. Nearby binary64
    // values subtract exactly (Sterbenz's lemma), so even one representable step is real data.
    // Relative identification of varying columns is checked on the correlation pivots below.
    if (!(deviation > 0)) throw new PredictorIdentificationError('constant', column);
    origins[column] = origin;
    meanOffsets[column] = meanOffset;
    means[column] = origin + meanOffset * offsetScale;
    offsetScales[column] = offsetScale;
    deviations[column] = deviation;
  }
  const standardized = new Float64Array(rowCount * predictorCount);
  for (let row = 0; row < rowCount; row++)
    for (let column = 0; column < predictorCount; column++)
      standardized[row * predictorCount + column] =
        ((predictors[row * predictorCount + column] - origins[column]) / offsetScales[column] -
          meanOffsets[column]) /
        deviations[column];
  let responseMean = 0;
  for (let row = 0; row < rowCount; row++) responseMean += response[row];
  responseMean /= rowCount;
  // Correlation matrix R = Z'Z / n and the correlations with the centred response.
  const correlation = Array.from({length: predictorCount}, () => new Float64Array(predictorCount));
  const cross = new Float64Array(predictorCount);
  let totalSumOfSquares = 0;
  for (let row = 0; row < rowCount; row++) {
    const centered = response[row] - responseMean;
    totalSumOfSquares += centered * centered;
    for (let first = 0; first < predictorCount; first++) {
      const firstValue = standardized[row * predictorCount + first];
      cross[first] += firstValue * centered;
      for (let second = 0; second <= first; second++)
        correlation[first][second] += firstValue * standardized[row * predictorCount + second];
    }
  }
  for (let first = 0; first < predictorCount; first++) {
    cross[first] /= rowCount;
    for (let second = 0; second <= first; second++) {
      correlation[first][second] /= rowCount;
      correlation[second][first] = correlation[first][second];
    }
  }
  // Cholesky factor L of R. The squared pivot is the share of the column left unexplained by the
  // columns before it.
  const factor = Array.from({length: predictorCount}, () => new Float64Array(predictorCount));
  for (let first = 0; first < predictorCount; first++)
    for (let second = 0; second <= first; second++) {
      let sum = correlation[first][second];
      for (let inner = 0; inner < second; inner++)
        sum -= factor[first][inner] * factor[second][inner];
      if (first === second) {
        if (!(sum > MINIMUM_UNEXPLAINED_PREDICTOR_VARIANCE))
          throw new PredictorIdentificationError('collinear', first);
        factor[first][first] = Math.sqrt(sum);
      } else factor[first][second] = sum / factor[second][second];
    }
  const solve = (right: Float64Array): Float64Array => {
    const forward = new Float64Array(predictorCount);
    for (let first = 0; first < predictorCount; first++) {
      let sum = right[first];
      for (let inner = 0; inner < first; inner++) sum -= factor[first][inner] * forward[inner];
      forward[first] = sum / factor[first][first];
    }
    const result = new Float64Array(predictorCount);
    for (let first = predictorCount - 1; first >= 0; first--) {
      let sum = forward[first];
      for (let inner = first + 1; inner < predictorCount; inner++)
        sum -= factor[inner][first] * result[inner];
      result[first] = sum / factor[first][first];
    }
    return result;
  };
  const standardizedSlopes = solve(cross);
  const inverse = Array.from({length: predictorCount}, (_, column) => {
    const unit = new Float64Array(predictorCount);
    unit[column] = 1;
    return solve(unit);
  });
  const fitted = new Float64Array(rowCount);
  const residuals = new Float64Array(rowCount);
  let residualSumOfSquares = 0;
  for (let row = 0; row < rowCount; row++) {
    let value = responseMean;
    for (let column = 0; column < predictorCount; column++)
      value += standardizedSlopes[column] * standardized[row * predictorCount + column];
    fitted[row] = value;
    residuals[row] = response[row] - value;
    residualSumOfSquares += residuals[row] ** 2;
  }
  const freedom = rowCount - predictorCount - 1;
  const sigmaSquared = residualSumOfSquares / freedom;
  const slopes = Array.from(
    standardizedSlopes,
    (value, column) => value / deviations[column] / offsetScales[column]
  );
  const slopeErrors = slopes.map(
    (_, column) =>
      Math.sqrt((sigmaSquared * inverse[column][column]) / rowCount) /
      deviations[column] /
      offsetScales[column]
  );
  let intercept = responseMean;
  for (let column = 0; column < predictorCount; column++)
    intercept -= slopes[column] * means[column];
  let interceptQuadratic = 0;
  for (let first = 0; first < predictorCount; first++)
    for (let second = 0; second < predictorCount; second++)
      interceptQuadratic +=
        (means[first] / offsetScales[first] / deviations[first]) *
        (means[second] / offsetScales[second] / deviations[second]) *
        inverse[first][second];
  const interceptError = Math.sqrt((sigmaSquared * (1 + interceptQuadratic)) / rowCount);
  const coefficients = [intercept, ...slopes];
  const standardErrors = [interceptError, ...slopeErrors];
  const rSquared =
    totalSumOfSquares > 0 ? 1 - residualSumOfSquares / totalSumOfSquares : Number.NaN;
  return {
    coefficients,
    standardErrors,
    tStatistics: coefficients.map((value, index) => value / standardErrors[index]),
    fitted,
    residuals,
    rSquared,
    adjustedRSquared: 1 - ((1 - rSquared) * (rowCount - 1)) / freedom,
    residualSumOfSquares,
    sigmaSquared
  };
}

type RegressionRow = {
  sourceIndex: number;
  feature: CatalogFeature;
};

function readPredictorNames(values: Values): {response: string; names: string[]} {
  const response = readColumnParameter(BINDING_ID, values, 'responseProperty');
  const names = (values['predictorProperties'] as string)
    .split(',')
    .map(name => name.trim())
    .filter(Boolean);
  if (names.length < 1 || names.length > MAXIMUM_PREDICTORS)
    throw makeCuratedError(
      BINDING_ID,
      `parameter predictorProperties must name 1 to ${MAXIMUM_PREDICTORS} columns, got ${names.length}`
    );
  if (new Set(names).size !== names.length)
    throw makeCuratedError(BINDING_ID, 'parameter predictorProperties names a column twice');
  if (names.includes(response))
    throw makeCuratedError(
      BINDING_ID,
      `parameter predictorProperties includes the response ${response}`
    );
  return {response, names};
}

type DiagnosticRow = {
  diagnostic: string;
  value: number | null;
  expected?: number | null;
  degreesOfFreedom?: number | null;
  pValue?: number | null;
  /** What the row means where the name alone could mislead. */
  note?: string;
};

/**
 * Convention of AIC and AICc, stated in the output so that a reader comparing with another
 * package can reconcile the number.
 */
export const INFORMATION_CRITERION_NOTE =
  'k counts the coefficients and the error variance (as R AIC(lm) and MuMIn AICc); statsmodels and spreg count the coefficients only and report an AIC exactly 2 lower';

/** Shared complete-case preparation and float64 identification/covariance reference. */
export function prepareCuratedRegression(inputs: Inputs, values: Values) {
  const areas = inputs['areas'];
  const diagnostics: CatalogDiagnostic[] = [];
  const {response: responseName, names} = readPredictorNames(values);
  const predictorCount = names.length;
  const permutations = values['permutations'] as number;
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;

  // Complete cases: a geometry (the residual test needs neighbours) and finite numbers in every column.
  const rows: RegressionRow[] = [];
  const responseValues: number[] = [];
  const predictorValues: number[] = [];
  const statuses: string[] = [];
  const drop = (sourceIndex: number, code: string, message: string) => {
    statuses[sourceIndex] = code;
    addCatalogDiagnostic(diagnostics, code, 'warning', message);
  };
  areas.features.forEach((feature, sourceIndex) => {
    statuses[sourceIndex] = 'analysed';
    if (!feature.geometry) {
      drop(
        sourceIndex,
        'missing-geometry',
        'Features without geometry are skipped: they have no neighbours.'
      );
      return;
    }
    const response = readNumericProperty(feature, responseName);
    const row = names.map(name => readNumericProperty(feature, name));
    if (response === null || row.some(value => value === null)) {
      drop(
        sourceIndex,
        'missing-value',
        'Features whose response or a predictor is missing, not a number or not finite are skipped (complete cases only).'
      );
      return;
    }
    rows.push({sourceIndex, feature});
    responseValues.push(response);
    for (const value of row as number[]) predictorValues.push(value);
  });
  const rowCount = rows.length;
  const coefficientCount = predictorCount + 1;
  // AICc divides by n - k - 1 with k = coefficients + 1 (the error variance counts), so n must
  // exceed predictors + 3, and Moran's I needs four rows.
  const minimumRows = Math.max(coefficientCount + 3, MINIMUM_ROWS_FOR_MORAN);
  if (rowCount < minimumRows)
    throw makeCuratedError(
      BINDING_ID,
      `${rowCount} complete rows for ${predictorCount} predictors; at least ${minimumRows} are needed`
    );
  if (isConstantList(responseValues))
    throw makeCuratedError(
      BINDING_ID,
      `the response ${responseName} is constant, so there is no variation to explain`
    );

  // Coefficients, standard errors, t statistics, fitted values and residuals, on standardised
  // predictors so that the units of the columns do not matter.
  // The catalog now breaks neighbour ties canonically. Ordering here also fixes the fit's
  // floating-point accumulation order before residual inference; sourceIndex restores output order.
  const {order} = getCanonicalOrder(
    getFeatureAnchors(
      makeDerivedCollection(
        rows.map(row => row.feature),
        () => ({})
      ),
      coordinateSystem
    ) as [number, number][]
  );
  const originalRows = [...rows];
  order.forEach((row, position) => {
    rows[position] = originalRows[row];
  });
  const responseArray = Float64Array.from(order, row => responseValues[row]);
  const predictorArray = Float64Array.from(
    order.flatMap(row => predictorValues.slice(row * predictorCount, (row + 1) * predictorCount))
  );
  let fit: StandardizedOlsFit;
  try {
    fit = fitStandardizedLeastSquares(responseArray, predictorArray, predictorCount, rowCount);
  } catch (failure) {
    if (failure instanceof PredictorIdentificationError)
      throw makeCuratedError(
        BINDING_ID,
        failure.kind === 'constant'
          ? `the predictor ${names[failure.column]} is constant, so it duplicates the intercept; drop it`
          : `the predictors are collinear: ${names[failure.column]} is an almost exact linear combination of the ones before it (less than 1e-10 of its variance is left), so the coefficients are not identified; drop or combine the predictors`
      );
    throw failure;
  }

  // Spatial autocorrelation of the residuals: neighbours among the complete rows, row-standardised;
  // rows with no neighbour are left out of the statistic (they would add a zero lag and distort I).
  const geometryKinds = new Set(
    rows.map(row => (/Polygon/.test(row.feature.geometry!.type) ? 'polygon' : 'point'))
  );
  const requested = values['neighborhood'] as 'auto' | 'queen' | 'rook' | 'knn';
  if (requested === 'auto' && geometryKinds.size > 1)
    throw makeCuratedError(
      BINDING_ID,
      'the input mixes points and polygons; choose neighborhood knn to link them by nearest centroid or point'
    );
  if ((requested === 'queen' || requested === 'rook') && geometryKinds.has('point'))
    throw makeCuratedError(
      BINDING_ID,
      `neighborhood ${requested} needs polygons but the input has points; use auto or knn`
    );
  const neighborhood =
    requested === 'auto' ? (geometryKinds.has('polygon') ? 'queen' : 'knn') : requested;
  const derivedCollection = makeDerivedCollection(
    rows.map(row => row.feature),
    () => ({})
  );
  const set = buildAnalysisSet(derivedCollection, '', {
    neighborhood,
    neighborCount: values['neighborCount'] as number,
    distanceBand: 1,
    rowStandardize: true,
    coordinateSystem
  });
  const includedRowCount = set.weights.neighbors.filter(neighbors => neighbors.length > 0).length;
  let responseMean = 0;
  for (const value of responseValues) responseMean += value;
  responseMean /= rowCount;
  let totalSumOfSquares = 0;
  for (const value of responseValues) totalSumOfSquares += (value - responseMean) ** 2;
  if (
    includedRowCount >= MINIMUM_ROWS_FOR_MORAN &&
    fit.residualSumOfSquares > PERFECT_FIT_FRACTION * totalSumOfSquares &&
    includedRowCount * permutations > MAXIMUM_PERMUTATION_WORK
  )
    throw new KeplerAllocationError('capacity', {
      message: `${BINDING_ID}: ${includedRowCount} rows times ${permutations} permutations exceeds the ${MAXIMUM_PERMUTATION_WORK} the CPU engine runs; lower permutations`
    });
  return {
    totalSumOfSquares,
    areas,
    values,
    diagnostics,
    names,
    predictorCount,
    permutations,
    coordinateSystem,
    rows,
    statuses,
    rowCount,
    coefficientCount,
    responseValues: responseArray,
    predictorArray,
    referenceFit: fit,
    neighborhood,
    derivedCollection,
    set
  };
}

export type PreparedCuratedRegression = ReturnType<typeof prepareCuratedRegression>;

/** Shared table publication: every fit-dependent value comes from the supplied fit. */
export function publishCuratedRegression(
  prepared: PreparedCuratedRegression,
  fit: StandardizedOlsFit
): CatalogRunOutput {
  const {
    areas,
    values,
    names,
    predictorCount,
    permutations,
    coordinateSystem,
    rows,
    statuses,
    rowCount,
    coefficientCount,
    predictorArray,
    neighborhood,
    derivedCollection,
    set,
    totalSumOfSquares
  } = prepared;
  const diagnostics = prepared.diagnostics.map(diagnostic => ({...diagnostic}));
  const freedom = rowCount - coefficientCount;
  const residuals = fit.residuals;
  const residualSumOfSquares = fit.residualSumOfSquares;
  const isPerfectFit = residualSumOfSquares <= PERFECT_FIT_FRACTION * totalSumOfSquares;

  let informationCriteria: {aic: number; aicc: number} | undefined;
  const diagnosticRows: DiagnosticRow[] = [
    {diagnostic: 'r-squared', value: fit.rSquared},
    {diagnostic: 'adjusted-r-squared', value: fit.adjustedRSquared},
    {diagnostic: 'sigma-squared', value: fit.sigmaSquared, degreesOfFreedom: freedom}
  ];
  if (isPerfectFit) {
    addCatalogDiagnostic(
      diagnostics,
      'perfect-fit',
      'warning',
      'The model reproduces the response exactly, so the log-likelihood, information criteria and residual tests are not defined and are left empty.'
    );
  } else {
    // Gaussian log-likelihood at the maximum likelihood variance RSS / n. One convention for both
    // criteria: k = coefficients + 1, the error variance counted as a parameter, as R `AIC(lm)` and
    // MuMIn `AICc` do. AIC = 2k - 2 logL; AICc = AIC + 2k(k+1)/(n-k-1) (Sugiura 1978; Hurvich and
    // Tsai 1989; Burnham and Anderson 2002). statsmodels and spreg count the coefficients only, so
    // their AIC is exactly 2 lower.
    const parameterCount = coefficientCount + 1;
    const logLikelihood =
      -0.5 * rowCount * (Math.log(2 * Math.PI) + Math.log(residualSumOfSquares / rowCount) + 1);
    const aic = 2 * parameterCount - 2 * logLikelihood;
    const aicc =
      aic + (2 * parameterCount * (parameterCount + 1)) / (rowCount - parameterCount - 1);
    informationCriteria = {aic, aicc};
    diagnosticRows.push(
      {diagnostic: 'log-likelihood', value: logLikelihood},
      {diagnostic: 'aic', value: aic, note: INFORMATION_CRITERION_NOTE},
      {diagnostic: 'aicc', value: aicc, note: INFORMATION_CRITERION_NOTE}
    );
    // Jarque and Bera (1980): n/6 (S^2 + (K - 3)^2 / 4) with moments of the residuals about their
    // mean (zero with an intercept) divided by n; the 2 degrees of freedom chi-square tail is exp(-JB/2).
    const secondMoment = residualSumOfSquares / rowCount;
    let thirdSum = 0;
    let fourthSum = 0;
    for (const value of residuals) {
      thirdSum += value ** 3;
      fourthSum += value ** 4;
    }
    const skewness = thirdSum / rowCount / secondMoment ** 1.5;
    const kurtosis = fourthSum / rowCount / secondMoment ** 2;
    const jarqueBera = (rowCount / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4);
    diagnosticRows.push({
      diagnostic: 'jarque-bera',
      value: jarqueBera,
      degreesOfFreedom: 2,
      pValue: Math.exp(-jarqueBera / 2)
    });
    // Heteroskedasticity: regress the squared residuals on the predictors. Koenker and Bassett
    // (1982): n R^2 of that auxiliary regression. Breusch and Pagan (1979): half the explained sum
    // of squares of e^2 / (RSS/n) - 1, which is n R^2 (K - 1) / 2 with K the residual kurtosis.
    // Both are chi-square with one degree of freedom per predictor.
    let koenkerBassett: number | null = null;
    let breuschPagan: number | null = null;
    try {
      const auxiliary = fitStandardizedLeastSquares(
        Float64Array.from(residuals, value => value * value),
        predictorArray,
        predictorCount,
        rowCount
      );
      if (Number.isFinite(auxiliary.rSquared)) {
        koenkerBassett = rowCount * auxiliary.rSquared;
        breuschPagan = 0.5 * auxiliary.rSquared * rowCount * (kurtosis - 1);
      }
    } catch {
      // The squared residuals are constant or the auxiliary design is singular: the tests are undefined.
    }
    diagnosticRows.push(
      {
        diagnostic: 'breusch-pagan',
        value: breuschPagan,
        degreesOfFreedom: predictorCount,
        pValue: breuschPagan === null ? null : getChiSquareSurvival(breuschPagan, predictorCount)
      },
      {
        diagnostic: 'koenker-bassett',
        value: koenkerBassett,
        degreesOfFreedom: predictorCount,
        pValue:
          koenkerBassett === null ? null : getChiSquareSurvival(koenkerBassett, predictorCount)
      }
    );
  }
  diagnosticRows.push({
    diagnostic: 'condition-number',
    value: computeBelsleyConditionNumber(predictorArray, predictorCount, rowCount)
  });

  let tiedAnchorCount = 0;
  const neighborCounts = new Array<number | null>(rowCount).fill(null);
  const included: number[] = [];
  set.sourceIndex.forEach((rowIndex, position) => {
    neighborCounts[rowIndex] = set.weights.neighbors[position].length;
    if (set.weights.neighbors[position].length) included.push(position);
  });
  const islandCount = set.sourceIndex.length - included.length;
  if (islandCount)
    addCatalogDiagnostic(
      diagnostics,
      'island-features',
      'warning',
      'Rows with no neighbour are left out of the residual Moran statistic.',
      islandCount
    );
  let moranValue: number | null = null;
  let moranExpected: number | null = null;
  let moranPValue: number | null = null;
  if (included.length >= MINIMUM_ROWS_FOR_MORAN && !isPerfectFit) {
    const compactIndex = new Map(included.map((position, compact) => [position, compact]));
    const anchors = getFeatureAnchors(derivedCollection, coordinateSystem);
    const result = runCanonicalMoranPermutationTest(
      included.map(position => residuals[set.sourceIndex[position]]),
      included.map(position =>
        set.weights.neighbors[position].map(other => compactIndex.get(other) as number)
      ),
      included.map(position => set.weights.weights[position]),
      included.map(position => anchors[set.sourceIndex[position]] as [number, number]),
      permutations,
      values['seed'] as number
    );
    moranValue = result.statistic;
    // The reference of the permutation test, the mean of the statistic over all orderings of the
    // residuals when they sum to zero. It is not the Cliff-Ord expectation tr(MW)/(n-k) of OLS residuals.
    moranExpected = -1 / (included.length - 1);
    moranPValue = result.pseudoPValue;
    tiedAnchorCount = result.tiedCount;
    if (tiedAnchorCount)
      addCatalogDiagnostic(
        diagnostics,
        'row-order-dependent-inference',
        'warning',
        'Coincident anchors retain input order for neighbour ties and permutation streams, so reordering them can change residual Moran inference throughout the dataset.',
        tiedAnchorCount
      );
  } else {
    addCatalogDiagnostic(
      diagnostics,
      'residual-moran-not-computed',
      'warning',
      isPerfectFit
        ? 'The residuals are rounding error, so their spatial autocorrelation is not computed.'
        : `Fewer than ${MINIMUM_ROWS_FOR_MORAN} rows have a neighbour, so the residual Moran statistic is not computed.`
    );
  }
  diagnosticRows.push({
    diagnostic: 'residual-morans-i',
    value: moranValue,
    expected: moranExpected,
    pValue: moranPValue,
    note: 'expected is the mean of the statistic over all orderings of the residuals (-1/(m-1)), the reference of the permutation test, not the Cliff-Ord expectation tr(MW)/(n-k) of OLS residuals; p_value is the two-sided permutation pseudo p-value'
  });

  const coefficientTable = makeCatalogTable('coefficients', 'Regression coefficients', {
    term: 'string',
    estimate: 'float64',
    standard_error: 'float64',
    t_statistic: 'float64',
    p_value: 'float64'
  });
  ['(intercept)', ...names].forEach((term, index) =>
    appendCatalogRow(coefficientTable, {
      term,
      estimate: fit.coefficients[index],
      standard_error: fit.standardErrors[index],
      t_statistic: fit.tStatistics[index],
      p_value: getStudentTTwoSidedPValue(fit.tStatistics[index], freedom)
    })
  );
  const diagnosticTable = makeCatalogTable('diagnostics', 'Regression diagnostics', {
    diagnostic: 'string',
    value: 'float64',
    expected: 'float64',
    degrees_of_freedom: 'int32',
    p_value: 'float64',
    note: 'string'
  });
  for (const row of diagnosticRows)
    appendCatalogRow(diagnosticTable, {
      diagnostic: row.diagnostic,
      value: row.value === null || !Number.isFinite(row.value) ? null : row.value,
      expected: row.expected ?? null,
      degrees_of_freedom: row.degreesOfFreedom ?? null,
      p_value: row.pValue ?? null,
      note: row.note ?? null
    });
  const residualTable = makeCatalogTable(
    'residuals',
    'Fitted values and residuals',
    {
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      fitted: 'float64',
      residual: 'float64',
      neighbor_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const rowByInputIndex = new Map(rows.map((row, index) => [row.sourceIndex, index]));
  areas.features.forEach((feature, sourceIndex) => {
    const rowIndex = rowByInputIndex.get(sourceIndex);
    appendCatalogRow(residualTable, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: statuses[sourceIndex],
      fitted: rowIndex === undefined ? null : fit.fitted[rowIndex],
      residual: rowIndex === undefined ? null : residuals[rowIndex],
      neighbor_count: rowIndex === undefined ? null : neighborCounts[rowIndex],
      geometry: getFeatureGeometryText(feature)
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'residual-permutation',
    'info',
    `The residual Moran p-value is a two-sided pseudo p-value from ${permutations} seeded permutations of the residuals over the rows; residuals are not exactly exchangeable, so it is an approximation to the Cliff-Ord test.`
  );
  return {
    tables: [coefficientTable, diagnosticTable, residualTable],
    diagnostics,
    summary: {
      inputCount: areas.features.length,
      rowCount,
      droppedCount: areas.features.length - rowCount,
      predictorCount,
      rSquared: fit.rSquared,
      adjustedRSquared: fit.adjustedRSquared,
      residualMoransI: moranValue ?? 'not computed',
      neighborhood,
      aic: informationCriteria?.aic ?? 'not computed',
      aicc: informationCriteria?.aicc ?? 'not computed',
      informationCriterionConvention: INFORMATION_CRITERION_NOTE,
      // Run-time caution `row-order-dependent-inference`: rows whose anchor ties with another's.
      tiedAnchorCount
    }
  };
}

/**
 * Regression with diagnostics: ordinary least squares with standard errors and p-values, fit,
 * information, normality, heteroskedasticity and conditioning diagnostics, and the spatial
 * autocorrelation of the residuals.
 */
export const CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Regression with spatial diagnostics',
  domain: 'analysis',
  summary:
    'Fits a response on one to fifteen predictors by ordinary least squares and reports coefficients with standard errors, t statistics and p-values, R squared and adjusted R squared, AIC and AICc, the condition number, the Jarque-Bera, Breusch-Pagan and Koenker-Bassett tests, the residual of every feature and the Moran statistic of the residuals with a permutation pseudo p-value. The GPU recipe adds LM tests, a specification recommendation and local residual Moran values.',
  capabilityIds: ['spatial-regression'],
  contributorNames: [
    'addSpatialRegressionRecipe',
    'GPULocalMoran',
    'GPUOrdinaryLeastSquares',
    'GPUSpatialRegressionDiagnostics',
    'GPUContiguityWeights',
    'GPUSpatialWeightsTransform'
  ],
  gpuContributorNames: [
    'GPUOrdinaryLeastSquares',
    'GPUSpatialRegressionDiagnostics',
    'GPULocalMoran'
  ],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec(
      'areas',
      'Areas or points with a response and predictors',
      ['polygon', 'point'],
      MAXIMUM_CURATED_AREA_FEATURES,
      MAXIMUM_CURATED_AREA_VERTICES
    )
  ],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec('coefficients', 'Regression coefficients', 'none', ['term'], 16),
    makeOutputSpec('diagnostics', 'Regression diagnostics', 'none', ['diagnostic'], 32),
    makeOutputSpec(
      'residuals',
      'Fitted values and residuals',
      'polygon',
      ['source_index', 'source_id'],
      MAXIMUM_CURATED_AREA_FEATURES
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Complete cases only: a feature with no geometry, or with a missing, non-numeric or non-finite response or predictor, is skipped and counted under its own diagnostic; the residuals table still lists it, in input order, with a status naming why and no fitted value.',
    'Fewer complete rows than the larger of predictors + 4 and 4 throws (AICc needs n above predictors + 3, so one predictor needs 5 rows and each further predictor one more); a constant response throws, as there is nothing to explain; a constant predictor, or a predictor that is an almost exact linear combination of the others (less than 1e-10 of its variance left, decided on standardised columns so the units do not matter), throws with a message naming the column. Predictors of very different scales, such as dollars beside a share, are fitted normally: each predictor is standardised before the fit and the coefficients, standard errors and intercept are transformed back exactly.',
    'A model that reproduces the response exactly leaves the log-likelihood, information criteria, residual tests and residual Moran statistic empty and says so in a diagnostic.',
    'Standard errors use RSS / (n - c) with c the coefficients including the intercept, and p-values are two-sided Student t on n - c degrees of freedom. Information criteria use one convention, k = c + 1 (the error variance counts as a parameter), as R AIC(lm) and MuMIn AICc: AIC = 2k - 2 logL and AICc = AIC + 2k(k+1)/(n-k-1), with logL the Gaussian log-likelihood at the maximum likelihood variance. statsmodels and spreg count the coefficients only and report an AIC exactly 2 lower; the convention is written in the note column of the aic and aicc rows and in the summary key informationCriterionConvention.',
    'Jarque-Bera uses the residual skewness and kurtosis (2 degrees of freedom); Breusch-Pagan and Koenker-Bassett regress the squared residuals on the predictors (one degree of freedom per predictor); the condition number is the Belsley, Kuh and Welsch value of the unit-length scaled design with its intercept.',
    'The residual Moran statistic uses queen contiguity for polygons and k nearest neighbours for points (auto), row-standardised over the complete rows; rows with no neighbour are left out and counted; its p-value is a two-sided permutation pseudo p-value, (m + 1) / (permutations + 1) with m the permutations at least as far from the mean of the permuted statistics as the observed one. Rows are ordered by anchor (point, or centroid in the local frame) x then y before fitting, selecting neighbours and shuffling, so distinct-anchor results do not depend on input order, including equal-distance neighbour ties. Coincident anchors keep input order and can affect inference throughout the dataset (summary key tiedAnchorCount). The expected value reported is the permutation mean, -1/(m-1), not the Cliff-Ord expectation of OLS residuals.',
    'The GPU path dispatches addSpatialRegressionRecipe with CPU-prepared row-standardised CSR weights in canonical complete-case order. GPU coefficients, fitted values and residuals are transformed back to source units; standard errors use the float64 reference covariance scaled by the GPU residual variance. Reference-perfect fits and uploads whose rounding error exceeds 1 percent of the reference residual norm use the CPU fallback before GPU allocation. Failed solver or diagnostic status, or nonfinite decoded inference, refuses GPU publication and permits the catalog CPU fallback. Compiled graphs, uploads, outputs and the recipe parameter buffer are released on success and failure.',
    'The input may hold polygons or points but not a mix under auto; queen and rook with points throw; at most 5000 features; rows times permutations above 20 million throws a capacity error, which can only bind above 2000 rows because permutations are at most 9999.'
  ],
  limitation:
    'The GPU path runs addSpatialRegressionRecipe on centred, scaled float32 columns. Input preparation, identification and covariance checks, supplementary fit diagnostics and the seeded global residual permutation test remain on the CPU. GPU-only rows report LM tests, Cliff-Ord analytic residual Moran and the specification decision; local Moran columns use uncorrected analytic inference, not the global permutation p-value. Spatial lag/error models, geographically weighted fits, robust standard errors and variance inflation factors are not published.',
  run: (inputs, values) => {
    const prepared = prepareCuratedRegression(inputs, values);
    return publishCuratedRegression(prepared, prepared.referenceFit);
  },
  runGpu: (context, inputs, values) => runCuratedRegressionOnGpu(context.device, inputs, values)
};
