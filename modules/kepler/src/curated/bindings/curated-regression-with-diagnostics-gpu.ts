// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUParameterBuffer, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  addSpatialRegressionRecipe,
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_EXPECTATION,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_Z,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_P_VALUE,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {appendCatalogRow, addCatalogDiagnostic} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import type {Inputs, Values} from '../../analysis/catalog-platform3-b-common';
import {
  PERFECT_FIT_FRACTION,
  prepareCuratedRegression,
  publishCuratedRegression,
  type PreparedCuratedRegression,
  type StandardizedOlsFit
} from './curated-regression-with-diagnostics-binding';

const BINDING_ID = 'curated-regression-with-diagnostics';

/** Below this residual share, float32 rounding can dominate the fit's inferential diagnostics. */
const MINIMUM_FLOAT32_RESIDUAL_VARIANCE_FRACTION = 1e-12;

/**
 * At the reference coefficients, upload perturbations may have at most 1 percent of the
 * reference residual L2 norm (1e-4 of its squared norm). Larger errors can drive inference.
 * This conservative bound needs no second solver and includes response and predictor rounding.
 */
const MAXIMUM_UPLOAD_ROUNDING_RESIDUAL_FRACTION = 1e-4;

/** Recipe default; shared by its specification decision and residual local Moran. */
const SIGNIFICANCE_LEVEL = 0.05;

/** Affine scaling before float32 upload retains representable variation near a large offset. */
function scaleColumn(values: ArrayLike<number>, center = true) {
  const origin = center ? values[0] : 0;
  let scale = 0;
  for (let row = 0; row < values.length; row++)
    scale = Math.max(scale, Math.abs(values[row] - origin));
  validateRequirement(
    BINDING_ID,
    Number.isFinite(scale) && scale > 0,
    'column spread must be finite and greater than 0 for float32 scaling'
  );
  return {origin, scale, values: Float32Array.from(values, value => (value - origin) / scale)};
}

/** Use the exact CPU publication cutoff before dividing by its residual variance anywhere. */
function validateReferenceInference(prepared: PreparedCuratedRegression): void {
  const {referenceFit, totalSumOfSquares} = prepared;
  validateRequirement(
    BINDING_ID,
    Number.isFinite(totalSumOfSquares) &&
      totalSumOfSquares > 0 &&
      Number.isFinite(referenceFit.residualSumOfSquares) &&
      referenceFit.residualSumOfSquares > PERFECT_FIT_FRACTION * totalSumOfSquares,
    `reference residual variance must exceed the CPU perfect-fit cutoff (${PERFECT_FIT_FRACTION} of response variance); use the CPU reference`
  );
}

/** CPU preparation only; no copied solver or weights implementation. */
export function prepareCuratedRegressionGpu(prepared: PreparedCuratedRegression) {
  validateReferenceInference(prepared);
  const {rowCount, predictorCount, predictorArray, set} = prepared;
  const columns = Array.from({length: predictorCount}, (_, column) =>
    scaleColumn(
      Float64Array.from(
        {length: rowCount},
        (_, row) => predictorArray[row * predictorCount + column]
      )
    )
  );
  const predictors = Float32Array.from(
    {length: rowCount * predictorCount},
    (_, index) => columns[index % predictorCount].values[Math.floor(index / predictorCount)]
  );
  // Scaling y without translating it preserves lag diagnostics even when W has island rows
  // (W * 1 is then not constant). Predictor translations preserve the design span with intercept.
  const response = scaleColumn(prepared.responseValues, false);
  let roundingSquares = 0;
  let referenceResidualSquares = 0;
  const normalizedSlopes = columns.map(
    (column, index) =>
      (prepared.referenceFit.coefficients[index + 1] * column.scale) / response.scale
  );
  for (let row = 0; row < rowCount; row++) {
    let roundingError = response.values[row] - prepared.responseValues[row] / response.scale;
    for (let column = 0; column < predictorCount; column++) {
      const index = row * predictorCount + column;
      const exactPredictor =
        (predictorArray[index] - columns[column].origin) / columns[column].scale;
      roundingError -= normalizedSlopes[column] * (predictors[index] - exactPredictor);
    }
    roundingSquares += roundingError * roundingError;
    referenceResidualSquares += (prepared.referenceFit.residuals[row] / response.scale) ** 2;
  }
  validateRequirement(
    BINDING_ID,
    Number.isFinite(roundingSquares) &&
      Number.isFinite(referenceResidualSquares) &&
      referenceResidualSquares > 0 &&
      roundingSquares <= MAXIMUM_UPLOAD_ROUNDING_RESIDUAL_FRACTION * referenceResidualSquares,
    `upload rounding error must be at most ${MAXIMUM_UPLOAD_ROUNDING_RESIDUAL_FRACTION} of reference residual squared norm; use the CPU reference`
  );
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  const positionByRow = new Map(set.sourceIndex.map((row, position) => [row, position]));
  for (let row = 0; row < rowCount; row++) {
    const position = positionByRow.get(row);
    if (position !== undefined) {
      set.weights.neighbors[position].forEach((neighbor, slot) => {
        neighbors.push(set.sourceIndex[neighbor]);
        weights.push(set.weights.weights[position][slot]);
      });
    }
    offsets.push(neighbors.length);
  }
  return {
    predictors,
    response,
    columns,
    weights: {
      offsets: Uint32Array.from(offsets),
      neighbors: Uint32Array.from(neighbors),
      weights: Float32Array.from(weights)
    }
  };
}

export type CuratedRegressionGpuReadback = {
  coefficients: Float32Array;
  fitted: Float32Array;
  residuals: Float32Array;
  olsStatus: number;
  diagnosticsStatus: number;
  tests: Float32Array;
  diagnosticsSummary: Float32Array;
  decision: number;
  localI: Float32Array;
  localPValues: Float32Array;
};

/** Decode actual GPU fit values; the existing float64 covariance is scaled by GPU residual variance. */
export function publishCuratedRegressionGpu(
  prepared: PreparedCuratedRegression,
  scaled: ReturnType<typeof prepareCuratedRegressionGpu>,
  readback: CuratedRegressionGpuReadback
) {
  validateReferenceInference(prepared);
  const {rowCount, predictorCount, referenceFit} = prepared;
  validateRequirement(
    BINDING_ID,
    readback.coefficients.length === predictorCount + 1 &&
      [readback.fitted, readback.residuals, readback.localI, readback.localPValues].every(
        values => values.length === rowCount
      ) &&
      readback.tests.length === GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH &&
      readback.diagnosticsSummary.length === GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
    'readback lengths must match the complete rows, coefficients and recipe diagnostic layouts'
  );
  validateRequirement(
    BINDING_ID,
    readback.olsStatus === 0,
    `ols status code must be 0, got ${readback.olsStatus}; use the CPU reference for an ill-conditioned design`
  );
  const normalizedResidualSquares = Array.from(readback.residuals).reduce(
    (sum, value) => sum + value * value,
    0
  );
  const residualSumOfSquares =
    normalizedResidualSquares * scaled.response.scale * scaled.response.scale;
  const normalizedMean = Array.from(scaled.response.values).reduce(
    (sum, value) => sum + value / rowCount,
    0
  );
  const totalSquares = Array.from(scaled.response.values).reduce(
    (sum, value) => sum + (value - normalizedMean) ** 2,
    0
  );
  // A float32 fit cannot support inference on residuals at its rounding floor. Refuse GPU
  // publication so the catalog can use the established float64 perfect-fit behavior.
  validateRequirement(
    BINDING_ID,
    normalizedResidualSquares > totalSquares * MINIMUM_FLOAT32_RESIDUAL_VARIANCE_FRACTION,
    `residual variance must exceed the float32 rounding floor (${MINIMUM_FLOAT32_RESIDUAL_VARIANCE_FRACTION} of response variance); use the CPU reference`
  );
  validateRequirement(
    BINDING_ID,
    readback.diagnosticsStatus === 0,
    `spatial diagnostics status code must be 0, got ${readback.diagnosticsStatus}; no GPU diagnostics were published`
  );
  validateRequirement(
    BINDING_ID,
    [
      readback.coefficients,
      readback.fitted,
      readback.residuals,
      readback.tests,
      readback.diagnosticsSummary
    ].every(values => Array.from(values).every(Number.isFinite)),
    'GPU fit and diagnostic values must be finite'
  );
  validateRequirement(
    BINDING_ID,
    Number.isInteger(readback.decision) &&
      readback.decision >= 0 &&
      readback.decision < GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED,
    `specification decision must be in [0, ${GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED - 1}], got ${readback.decision}`
  );
  const slopes = scaled.columns.map(
    (column, index) => (readback.coefficients[index + 1] * scaled.response.scale) / column.scale
  );
  let intercept = scaled.response.origin + readback.coefficients[0] * scaled.response.scale;
  slopes.forEach((slope, index) => {
    intercept -= slope * scaled.columns[index].origin;
  });
  const coefficients = [intercept, ...slopes];
  const varianceRatio = Math.sqrt(residualSumOfSquares / referenceFit.residualSumOfSquares);
  const standardErrors = referenceFit.standardErrors.map(value => value * varianceRatio);
  const rSquared = 1 - normalizedResidualSquares / totalSquares;
  const freedom = rowCount - predictorCount - 1;
  const fit: StandardizedOlsFit = {
    coefficients,
    standardErrors,
    tStatistics: coefficients.map((value, index) => value / standardErrors[index]),
    fitted: Float64Array.from(
      readback.fitted,
      value => scaled.response.origin + value * scaled.response.scale
    ),
    residuals: Float64Array.from(readback.residuals, value => value * scaled.response.scale),
    residualSumOfSquares,
    sigmaSquared: residualSumOfSquares / freedom,
    rSquared,
    adjustedRSquared: 1 - ((1 - rSquared) * (rowCount - 1)) / freedom
  };
  validateRequirement(
    BINDING_ID,
    Number.isFinite(varianceRatio) &&
      varianceRatio > 0 &&
      [fit.coefficients, fit.standardErrors, fit.tStatistics, fit.fitted, fit.residuals].every(
        values => Array.from(values).every(Number.isFinite)
      ) &&
      fit.standardErrors.every(value => value > 0) &&
      [fit.residualSumOfSquares, fit.sigmaSquared, fit.rSquared, fit.adjustedRSquared].every(
        Number.isFinite
      ) &&
      fit.sigmaSquared > 0,
    'decoded fit inference must have finite coefficients, positive finite standard errors and variance, and finite t statistics'
  );
  validateRequirement(
    BINDING_ID,
    Array.from(readback.tests).every(
      (value, index) =>
        index % GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE !== 2 || (value >= 0 && value <= 1)
    ) &&
      readback.localPValues.every(value => !Number.isFinite(value) || (value >= 0 && value <= 1)) &&
      readback.diagnosticsSummary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_P_VALUE] >= 0 &&
      readback.diagnosticsSummary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_P_VALUE] <= 1,
    'decoded inference p-values must be in [0, 1] when defined'
  );
  const result = publishCuratedRegression(prepared, fit);
  validateRequirement(
    BINDING_ID,
    result.tables.every(table =>
      Object.values(table.columns).every(values =>
        values.every(value => typeof value !== 'number' || Number.isFinite(value))
      )
    ) &&
      Object.values(result.summary).every(
        value => typeof value !== 'number' || Number.isFinite(value)
      ),
    'decoded fit diagnostics must be finite when defined'
  );
  const diagnostics = result.tables.find(table => table.id === 'diagnostics')!;
  ['lm-lag', 'lm-error', 'robust-lm-lag', 'robust-lm-error', 'lm-sarma'].forEach((name, row) => {
    const offset = row * GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
    appendCatalogRow(diagnostics, {
      diagnostic: name,
      value: readback.tests[offset],
      degrees_of_freedom: readback.tests[offset + 1],
      p_value: readback.tests[offset + 2],
      note: 'GPU spatial-regression recipe; asymptotic chi-square inference'
    });
  });
  const summary = readback.diagnosticsSummary;
  const finiteOrNull = (value: number) => (Number.isFinite(value) ? value : null);
  appendCatalogRow(diagnostics, {
    diagnostic: 'residual-morans-i-analytic',
    value: finiteOrNull(summary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I]),
    expected: finiteOrNull(summary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_EXPECTATION]),
    p_value: finiteOrNull(summary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_P_VALUE]),
    note: `Cliff-Ord regression-residual normal inference; z = ${summary[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_Z]}; includes all complete rows, including islands`
  });
  appendCatalogRow(diagnostics, {
    diagnostic: 'specification-decision',
    value: readback.decision,
    note: 'Anselin rule at alpha 0.05: 0 OLS, 1 spatial lag, 2 spatial error, 3 lag and error; recommendation only, no spatial model is fitted'
  });
  const residuals = result.tables.find(table => table.id === 'residuals')!;
  const positionByRow = new Map(prepared.set.sourceIndex.map((row, position) => [row, position]));
  const rowBySourceIndex = new Map(prepared.rows.map((row, index) => [row.sourceIndex, index]));
  for (const [name, values] of [
    ['local_morans_i', readback.localI],
    ['local_moran_p_value', readback.localPValues]
  ] as const) {
    residuals.columnTypes[name] = 'float64';
    residuals.columns[name] = prepared.areas.features.map((_, sourceIndex) => {
      const row = rowBySourceIndex.get(sourceIndex);
      const position = row === undefined ? undefined : positionByRow.get(row);
      return row === undefined ||
        position === undefined ||
        !prepared.set.weights.neighbors[position].length ||
        !Number.isFinite(values[row])
        ? null
        : values[row];
    });
  }
  addCatalogDiagnostic(
    result.diagnostics,
    'gpu-inference',
    'info',
    'LM tests and the specification decision use the GPU recipe. Local Moran p-values are uncorrected analytic values on all complete rows; the global permutation test excludes islands and remains a CPU supplement.'
  );
  result.summary['specificationDecision'] = readback.decision;
  result.summary['localMoranInference'] = 'analytic; uncorrected; alpha 0.05';
  result.summary['gpuOlsStatus'] = readback.olsStatus;
  result.summary['gpuDiagnosticsStatus'] = readback.diagnosticsStatus;
  return result;
}

/** Recipe dispatch with one session owning all graph resources; the parameter buffer is caller-owned. */
export async function runCuratedRegressionOnGpu(device: Device, inputs: Inputs, values: Values) {
  const prepared = prepareCuratedRegression(inputs, values);
  const scaled = prepareCuratedRegressionGpu(prepared);
  const parameters = new GPUParameterBuffer(device, {
    id: `${BINDING_ID}-parameters`,
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  try {
    return await withCatalogGpuSession(device, BINDING_ID, async session => {
      const coefficients = session.output('coefficients', 'float32', prepared.coefficientCount);
      const fitted = session.output('fitted', 'float32', prepared.rowCount);
      const residuals = session.output('residuals', 'float32', prepared.rowCount);
      const olsSummary = session.output(
        'ols-summary',
        'float32',
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
      );
      const olsStatus = session.output('ols-status', 'uint32', 1);
      const tests = session.output(
        'tests',
        'float32',
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
      );
      const diagnosticsSummary = session.output(
        'diagnostics-summary',
        'float32',
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
      );
      const diagnosticsStatus = session.output('diagnostics-status', 'uint32', 1);
      const decision = session.output('decision', 'uint32', 1);
      const localI = session.output('local-i', 'float32', prepared.rowCount);
      const localPValues = session.output('local-p-values', 'float32', prepared.rowCount);
      const recipe = addSpatialRegressionRecipe(session.graph, {
        predictors: session.input('predictors', scaled.predictors, 'float32'),
        response: session.input('response', scaled.response.values, 'float32'),
        predictorCount: prepared.predictorCount,
        weights: {
          offsets: session.input('offsets', scaled.weights.offsets, 'uint32'),
          neighbors: session.input('neighbors', scaled.weights.neighbors, 'uint32'),
          weights: session.input('weights', scaled.weights.weights, 'float32')
        },
        parameters,
        settings: {significanceLevel: SIGNIFICANCE_LEVEL},
        output: {
          ols: {
            coefficients: coefficients.view,
            fitted: fitted.view,
            residuals: residuals.view,
            summary: olsSummary.view,
            status: {code: olsStatus.view}
          },
          diagnostics: {
            tests: tests.view,
            summary: diagnosticsSummary.view,
            status: {code: diagnosticsStatus.view}
          },
          specification: {decision: decision.view},
          residualMoran: {localI: localI.view, pValues: localPValues.view}
        }
      });
      session.run();
      const readback: CuratedRegressionGpuReadback = {
        coefficients: await session.readFloat32(coefficients),
        fitted: await session.readFloat32(fitted),
        residuals: await session.readFloat32(residuals),
        olsStatus: await session.readScalar(olsStatus),
        diagnosticsStatus: await session.readScalar(diagnosticsStatus),
        tests: await session.readFloat32(tests),
        diagnosticsSummary: await session.readFloat32(diagnosticsSummary),
        decision: await session.readScalar(decision),
        localI: await session.readFloat32(localI),
        localPValues: await session.readFloat32(localPValues)
      };
      const dispatchedContributors = recipe.contributors.flatMap(contributor =>
        contributor instanceof GPUOrdinaryLeastSquares
          ? ['GPUOrdinaryLeastSquares']
          : contributor instanceof GPUSpatialRegressionDiagnostics
            ? ['GPUSpatialRegressionDiagnostics']
            : contributor instanceof GPULocalMoran
              ? ['GPULocalMoran']
              : []
      );
      return {...publishCuratedRegressionGpu(prepared, scaled, readback), dispatchedContributors};
    });
  } finally {
    parameters.destroy();
  }
}
