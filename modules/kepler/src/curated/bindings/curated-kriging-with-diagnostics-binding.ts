// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogGpuContext,
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem
} from '../../analysis/catalog-geojson';
import {
  fitKeplerVariogram,
  summarizeKrigingCrossValidation
} from '../../analysis/kriging-variogram-fit';
import {
  computeKrigingField,
  makeGridTable,
  preparePointGrid,
  type KrigingSettings
} from '../../analysis/catalog-platform3-b-grids';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  type GridSpec,
  type Inputs,
  type Sample,
  type Values
} from '../../analysis/catalog-platform3-b-common';
import {KeplerAllocationError} from '../../runtime/dataset-registry';
import {
  getCanonicalOrder,
  isConstantList,
  makeColumnParameter,
  makeCuratedError,
  readColumnParameter
} from './curated-binding-utils';

import {computeCuratedKrigingGpu} from './curated-kriging-with-diagnostics-gpu';

const BINDING_ID = 'curated-kriging-with-diagnostics';

/** Same sample cap as the catalog kriging-grid binding: every cell solves a system over its neighbours. */
const MAXIMUM_SAMPLES = 2000;

/** Same cell cap as the catalog kriging-grid binding. */
const MAXIMUM_CELLS = 40000;

/**
 * Fewest distinct locations: a variogram needs two populated lag bins and a leave-one-out
 * prediction needs at least as many neighbours as the system has constraints (one) plus data.
 */
const MINIMUM_SAMPLES = 8;

/**
 * Sample-times-cell work the catalog kriging field accepts on the CPU
 * (`MAXIMUM_CPU_WORK` in catalog-platform3-b-grids.ts, not exported; the value is repeated here
 * only to name it in the error).
 */
const CPU_WORK_BOUND = 6e7;

/**
 * Bounds of the model checks that raise the `poor-model-fit` caution. They are screening bounds,
 * not tests.
 *
 * Nugget share of the total sill above which the data show essentially no spatial structure:
 * Cambardella et al. (1994) call a nugget-to-sill ratio above 75 percent weak spatial dependence.
 */
export const MAXIMUM_NUGGET_TO_SILL_RATIO = 0.75;

/**
 * Band of the root mean square standardised leave-one-out error. When the kriging variance is
 * right the standardised errors have mean square 1 (Webster and Oliver, Geostatistics for
 * Environmental Scientists), so a value near 1 means the variance is neither too small nor too
 * large. The band 0.8 to 1.25 (a ratio of 1.25 either way, that is a variance 56 percent off at the
 * ends) lets sampling variation of a few dozen correlated errors through and catches a variance
 * wrong by more than a quarter in deviation; it is wider for fewer than about 30 samples.
 */
export const STANDARDIZED_ERROR_BAND = [0.8, 1.25] as const;

/**
 * Smallest share of leave-one-out errors inside 1.96 standard errors. Nominal is 0.95; with 40
 * samples the binomial deviation is 3.4 points, so 0.90 is about 1.5 deviations below nominal.
 */
export const MINIMUM_STANDARDIZED_COVERAGE = 0.9;

type VariogramModel = KrigingSettings['model'];

const PARAMETERS: readonly CatalogParameterSpec[] = [
  makeColumnParameter('valueProperty', 'Value property'),
  makeNumberParameter('cellSize', 'Cell size', 250, 1e-6, 1e7, 'crs'),
  {
    name: 'variogram',
    label: 'Variogram',
    kind: 'enum',
    options: ['fit', 'stated'],
    defaultValue: 'fit'
  },
  {
    name: 'model',
    label: 'Variogram model (auto picks the best fit)',
    kind: 'enum',
    options: ['auto', 'spherical', 'exponential', 'gaussian'],
    defaultValue: 'auto'
  },
  makeNumberParameter('nugget', 'Nugget (stated variogram)', 0, 0, 1e12),
  makeNumberParameter('sill', 'Partial sill (stated variogram)', 1, 1e-12, 1e12),
  makeNumberParameter('range', 'Range (stated variogram)', 1000, 1e-6, 1e9, 'crs'),
  makeNumberParameter(
    'binCount',
    'Lag bins of the empirical variogram',
    15,
    4,
    60,
    undefined,
    'integer'
  ),
  makeNumberParameter(
    'maximumLag',
    'Largest lag (0 = a third of the data diagonal)',
    0,
    0,
    1e9,
    'crs'
  ),
  makeNumberParameter(
    'neighborCount',
    'Nearest samples per prediction',
    16,
    4,
    64,
    undefined,
    'integer'
  ),
  makeNumberParameter('searchRadius', 'Search radius (0 = unlimited)', 0, 0, 1e9, 'crs'),
  COORDINATE_SYSTEM_PARAMETER
];

/** Semivariance of a model at a distance; zero at distance zero, nugget plus partial sill times the shape beyond. */
function getModelSemivariance(
  model: VariogramModel,
  nugget: number,
  sill: number,
  range: number,
  distance: number
): number {
  // Same definition as the catalog kriging field (`getSemivariance` in catalog-platform3-b-grids.ts).
  if (distance <= 0) return 0;
  const ratio = distance / range;
  const shape =
    model === 'spherical'
      ? ratio >= 1
        ? 1
        : 1.5 * ratio - 0.5 * ratio ** 3
      : model === 'exponential'
        ? 1 - Math.exp(-3 * ratio)
        : 1 - Math.exp(-3 * ratio * ratio);
  return nugget + sill * shape;
}

export type FrameEmpiricalVariogram = {
  distances: Float64Array;
  semivariances: Float64Array;
  pairCounts: Float64Array;
  maximumLag: number;
};

/**
 * Binned Matheron semivariogram of point values in local frame coordinates: for the pairs with
 * `0 < h <= maximumLag`, equal-width bins of `h`, `gamma = sum (z_i - z_j)^2 / (2 N)` and the mean
 * pair distance per bin; an empty bin is NaN. `estimateKeplerEmpiricalVariogram` does the same in
 * great-circle metres for geographic input only, and cannot take planar coordinates; this one works
 * in the frame the kriging itself uses, so the variogram and the prediction share one metric.
 */
export function computeFrameEmpiricalVariogram(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  value: ArrayLike<number>,
  binCount: number,
  maximumLag: number
): FrameEmpiricalVariogram {
  const distanceSums = new Float64Array(binCount);
  const gammaSums = new Float64Array(binCount);
  const pairCounts = new Float64Array(binCount);
  for (let first = 0; first < x.length; first++)
    for (let second = first + 1; second < x.length; second++) {
      const distance = Math.hypot(x[first] - x[second], y[first] - y[second]);
      if (!(distance > 0) || distance > maximumLag) continue;
      const bin = Math.min(binCount - 1, Math.floor((distance / maximumLag) * binCount));
      distanceSums[bin] += distance;
      gammaSums[bin] += 0.5 * (value[first] - value[second]) ** 2;
      pairCounts[bin]++;
    }
  const distances = new Float64Array(binCount);
  const semivariances = new Float64Array(binCount);
  for (let bin = 0; bin < binCount; bin++) {
    distances[bin] = pairCounts[bin] ? distanceSums[bin] / pairCounts[bin] : Number.NaN;
    semivariances[bin] = pairCounts[bin] ? gammaSums[bin] / pairCounts[bin] : Number.NaN;
  }
  return {distances, semivariances, pairCounts, maximumLag};
}

/**
 * Samples with coincident locations merged to one location holding the mean value: two samples at
 * one place make the kriging system singular. `sourceIndex` keeps the first member.
 */
function mergeCoincidentSamples(sample: Sample): {
  merged: Sample;
  mergedCount: number;
  /** Merged location of every original sample, and the original samples of every location. */
  groupOfSample: number[];
  membersOfGroup: number[][];
} {
  const groups = new Map<string, number[]>();
  // The catalog now breaks neighbour ties canonically. Keep a stable location order here too:
  // empirical variogram pair sums and fitting must accumulate in the same order after a shuffle.
  const {order} = getCanonicalOrder(
    Array.from({length: sample.count}, (_, index) => [sample.x[index], sample.y[index]])
  );
  for (const index of order) {
    const key = `${sample.x[index]},${sample.y[index]}`;
    const list = groups.get(key) ?? [];
    list.push(index);
    groups.set(key, list);
  }
  const x: number[] = [];
  const y: number[] = [];
  const value: number[] = [];
  const sourceIndex: number[] = [];
  const groupOfSample = new Array<number>(sample.count);
  const membersOfGroup: number[][] = [];
  for (const members of groups.values()) {
    for (const member of members) groupOfSample[member] = membersOfGroup.length;
    membersOfGroup.push(members);
    x.push(sample.x[members[0]]);
    y.push(sample.y[members[0]]);
    value.push(members.reduce((sum, member) => sum + sample.value[member], 0) / members.length);
    sourceIndex.push(sample.sourceIndex[members[0]]);
  }
  return {
    merged: {
      x: Float64Array.from(x),
      y: Float64Array.from(y),
      value: Float64Array.from(value),
      sourceIndex: Int32Array.from(sourceIndex),
      partIndex: new Int32Array(x.length),
      count: x.length,
      droppedCount: sample.droppedCount
    },
    mergedCount: sample.count - x.length,
    groupOfSample,
    membersOfGroup
  };
}

export type LeaveOneOutResult = {
  predicted: Float64Array;
  variance: Float64Array;
  /** Samples whose leave-one-out system was singular or had no neighbour in reach. */
  failedCount: number;
};

/**
 * Leave-one-out prediction: each sample is predicted by ordinary kriging from all the others with
 * the same variogram, neighbour count and search radius as the grid (no refit of the variogram).
 * The prediction at a sample location reuses `computeKrigingField` over a one-cell grid centred on
 * that location, with the sample itself removed.
 */
export function computeLeaveOneOut(sample: Sample, settings: KrigingSettings): LeaveOneOutResult {
  const predicted = new Float64Array(sample.count).fill(Number.NaN);
  const variance = new Float64Array(sample.count).fill(Number.NaN);
  let failedCount = 0;
  const others: Sample = {
    x: new Float64Array(sample.count - 1),
    y: new Float64Array(sample.count - 1),
    value: new Float64Array(sample.count - 1),
    sourceIndex: new Int32Array(sample.count - 1),
    partIndex: new Int32Array(sample.count - 1),
    count: sample.count - 1,
    droppedCount: 0
  };
  for (let held = 0; held < sample.count; held++) {
    for (const name of ['x', 'y', 'value'] as const) {
      others[name].set(sample[name].subarray(0, held), 0);
      others[name].set(sample[name].subarray(held + 1), held);
    }
    // A one-cell grid of side 2 whose centre is exactly the held-out location.
    const grid: GridSpec = {
      minX: sample.x[held] - 1,
      minY: sample.y[held] - 1,
      cellSize: 2,
      columns: 1,
      rows: 1
    };
    const field = computeKrigingField(others, grid, settings);
    if (Number.isFinite(field.values[0])) {
      predicted[held] = field.values[0];
      variance[held] = (field.variances as ArrayLike<number>)[0];
    } else failedCount++;
  }
  return {predicted, variance, failedCount};
}

/** Shared input preparation and variogram fitting; does not execute either kriging engine. */
export function prepareCuratedKriging(inputs: Inputs, values: Values) {
  const valueProperty = readColumnParameter(BINDING_ID, values, 'valueProperty');
  const coordinateSystem = values['coordinateSystem'] as CatalogCoordinateSystem;
  const cellSize = values['cellSize'] as number;
  let context: ReturnType<typeof preparePointGrid>;
  try {
    context = preparePointGrid(inputs['points'], valueProperty, coordinateSystem, cellSize, 0);
  } catch (failure) {
    if (failure instanceof KeplerAllocationError) throw failure;
    throw makeCuratedError(
      BINDING_ID,
      `no feature has a point position and a numeric ${valueProperty}`
    );
  }
  const {grid, diagnostics} = context;
  const {
    merged: sample,
    mergedCount,
    groupOfSample,
    membersOfGroup
  } = mergeCoincidentSamples(context.sample);
  if (mergedCount)
    addCatalogDiagnostic(
      diagnostics,
      'coincident-samples',
      'warning',
      'Samples at exactly the same location are merged into one sample holding their mean value, because two samples at one place make the kriging system singular.',
      mergedCount
    );
  if (sample.count < MINIMUM_SAMPLES)
    throw makeCuratedError(
      BINDING_ID,
      `${sample.count} distinct sample locations; at least ${MINIMUM_SAMPLES} are needed`
    );
  if (grid.columns * grid.rows > MAXIMUM_CELLS)
    throw new KeplerAllocationError('capacity', {
      message: `${BINDING_ID}: ${grid.columns * grid.rows} cells exceed the ${MAXIMUM_CELLS} allowed; raise cellSize`
    });

  // Empirical semivariogram in the frame metric.
  let extentX = 0;
  let extentY = 0;
  let minimumX = Infinity;
  let minimumY = Infinity;
  for (let index = 0; index < sample.count; index++) {
    minimumX = Math.min(minimumX, sample.x[index]);
    minimumY = Math.min(minimumY, sample.y[index]);
    extentX = Math.max(extentX, sample.x[index]);
    extentY = Math.max(extentY, sample.y[index]);
  }
  const diagonal = Math.hypot(extentX - minimumX, extentY - minimumY);
  const requestedLag = values['maximumLag'] as number;
  const maximumLag = requestedLag > 0 ? requestedLag : diagonal / 3;
  const empirical = computeFrameEmpiricalVariogram(
    sample.x,
    sample.y,
    sample.value,
    values['binCount'] as number,
    maximumLag
  );

  // The variogram: fitted to the empirical one, or stated.
  const requestedModel = values['model'] as 'auto' | VariogramModel;
  let model: VariogramModel;
  let nugget: number;
  let sill: number;
  let range: number;
  let fitCriterion: number | string = 'stated';
  if (values['variogram'] === 'stated') {
    if (requestedModel === 'auto')
      throw makeCuratedError(
        BINDING_ID,
        'a stated variogram needs a model: choose spherical, exponential or gaussian'
      );
    model = requestedModel;
    nugget = values['nugget'] as number;
    sill = values['sill'] as number;
    range = values['range'] as number;
  } else {
    if (isConstantList(sample.value))
      throw makeCuratedError(
        BINDING_ID,
        `the values of ${valueProperty} are all equal, so no variogram can be fitted; state one or choose another column`
      );
    try {
      const fit = fitKeplerVariogram(
        empirical,
        requestedModel === 'auto' ? ['spherical', 'exponential', 'gaussian'] : [requestedModel]
      );
      // Only the three families of KrigingSettings are offered to the fit above.
      model = fit.model as VariogramModel;
      nugget = fit.nugget;
      sill = fit.sill;
      range = fit.rangeMeters;
      fitCriterion = fit.criterion;
    } catch {
      throw makeCuratedError(
        BINDING_ID,
        'no variogram model could be fitted to the empirical semivariogram (fewer than two populated lag bins, or no model with a positive range); raise maximumLag, lower binCount or state a variogram'
      );
    }
  }
  const settings: KrigingSettings = {
    model,
    nugget,
    sill,
    range,
    neighborCount: values['neighborCount'] as number,
    searchRadius: values['searchRadius'] as number
  };

  return {
    context,
    sample,
    groupOfSample,
    membersOfGroup,
    empirical,
    settings,
    maximumLag,
    fitCriterion
  };
}

/** Shared publication keeps source identities, units and diagnostic interpretation identical. */
function finishCuratedKriging(
  inputs: Inputs,
  prepared: ReturnType<typeof prepareCuratedKriging>,
  field: ReturnType<typeof computeKrigingField>,
  leaveOneOut: LeaveOneOutResult,
  gpuSummary: Record<string, number> = {}
): CatalogRunOutput {
  const {
    context,
    sample,
    groupOfSample,
    membersOfGroup,
    empirical,
    settings,
    maximumLag,
    fitCriterion
  } = prepared;
  const {frame, grid, diagnostics} = context;
  const {model, nugget, sill, range} = settings;
  // Both engines publish the same grid and validation schema.
  if (gpuSummary['gpuInvalidCellCount'])
    addCatalogDiagnostic(
      diagnostics,
      'insufficient-kriging-neighbors',
      'warning',
      'Cells with no usable sample within the search radius are nodata.',
      gpuSummary['gpuInvalidCellCount']
    );
  if (field.singularCount)
    addCatalogDiagnostic(
      diagnostics,
      'singular-kriging-system',
      'warning',
      'Cells whose kriging system is singular or numerically unreliable are nodata.',
      field.singularCount
    );
  const cellTable = makeGridTable(frame, grid, field, {
    tableId: 'kriging-cells',
    label: 'Kriged cells',
    valueColumn: 'value',
    includeEmptyCells: false,
    isEmpty: index => !Number.isFinite(field.values[index])
  });

  // Leave-one-out cross-validation.
  if (leaveOneOut.failedCount)
    addCatalogDiagnostic(
      diagnostics,
      'leave-one-out-failed',
      'warning',
      'Samples whose leave-one-out prediction failed (a singular system, or no other sample within the search radius) have empty errors and are left out of the summary.',
      leaveOneOut.failedCount
    );
  const validationTable = makeCatalogTable(
    'cross-validation',
    'Leave-one-out cross-validation',
    {
      sample_index: 'int32',
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      location_sample_count: 'int32',
      observed: 'float64',
      predicted: 'float64',
      error: 'float64',
      kriging_variance: 'float64',
      standardized_error: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  // One row per input feature (per point of a multipoint feature), in input order. The first
  // sample at a location carries the error of the merged location; later samples at the same
  // location are listed as merged, with their own value and the location's prediction.
  const samplesOfFeature = new Map<number, number[]>();
  context.sample.sourceIndex.forEach((featureIndex, originalIndex) => {
    samplesOfFeature.set(featureIndex, [
      ...(samplesOfFeature.get(featureIndex) ?? []),
      originalIndex
    ]);
  });
  inputs['points'].features.forEach((feature, featureIndex) => {
    const originals = samplesOfFeature.get(featureIndex);
    if (!originals) {
      appendCatalogRow(validationTable, {
        source_index: featureIndex,
        source_id: getFeatureIdentity(feature),
        status: getPointPositions(feature.geometry).length ? 'missing-value' : 'missing-geometry',
        geometry: feature.geometry ? JSON.stringify(feature.geometry) : null
      });
      return;
    }
    for (const original of originals) {
      const group = groupOfSample[original];
      const isFirst = membersOfGroup[group][0] === original;
      const predicted = leaveOneOut.predicted[group];
      const variance = leaveOneOut.variance[group];
      const observed = isFirst ? sample.value[group] : context.sample.value[original];
      const error = isFirst ? predicted - observed : Number.NaN;
      appendCatalogRow(validationTable, {
        sample_index: group,
        source_index: featureIndex,
        source_id: getFeatureIdentity(feature),
        status: isFirst ? 'analysed' : 'merged-into-earlier-sample',
        location_sample_count: membersOfGroup[group].length,
        observed,
        predicted: Number.isFinite(predicted) ? predicted : null,
        error: Number.isFinite(error) ? error : null,
        kriging_variance: Number.isFinite(variance) ? variance : null,
        standardized_error:
          Number.isFinite(error) && variance > 0 ? error / Math.sqrt(variance) : null,
        geometry: makeCatalogGeometryText('Point', frame.inverse(sample.x[group], sample.y[group]))
      });
    }
  });
  const summary = summarizeKrigingCrossValidation(
    sample.value,
    leaveOneOut.predicted,
    leaveOneOut.variance
  );
  const variogramTable = makeCatalogTable('variogram', 'Empirical and fitted variogram', {
    bin: 'int32',
    distance: 'float64',
    semivariance: 'float64',
    pair_count: 'int32',
    model_semivariance: 'float64'
  });
  for (let bin = 0; bin < empirical.pairCounts.length; bin++) {
    if (!empirical.pairCounts[bin]) continue;
    appendCatalogRow(variogramTable, {
      bin,
      distance: empirical.distances[bin],
      semivariance: empirical.semivariances[bin],
      pair_count: empirical.pairCounts[bin],
      model_semivariance: getModelSemivariance(model, nugget, sill, range, empirical.distances[bin])
    });
  }
  // Model checks behind the `poor-model-fit` caution.
  const nuggetToSillRatio = nugget + sill > 0 ? nugget / (nugget + sill) : Number.NaN;
  const poorFitReasons: string[] = [];
  if (range > maximumLag)
    poorFitReasons.push(
      `the range (${range.toPrecision(4)}) exceeds the largest lag (${maximumLag.toPrecision(4)}), so the empirical variogram does not reach it and the range is not identified`
    );
  if (nuggetToSillRatio > MAXIMUM_NUGGET_TO_SILL_RATIO)
    poorFitReasons.push(
      `the nugget is ${(100 * nuggetToSillRatio).toFixed(0)} percent of the total sill (above ${100 * MAXIMUM_NUGGET_TO_SILL_RATIO} percent): the isotropic screen shows weak omnidirectional dependence, which can mask anisotropy`
    );
  const standardizedRoot = summary.rootMeanSquareStandardizedError;
  if (
    Number.isFinite(standardizedRoot) &&
    (standardizedRoot < STANDARDIZED_ERROR_BAND[0] || standardizedRoot > STANDARDIZED_ERROR_BAND[1])
  )
    poorFitReasons.push(
      `the root mean square standardised error is ${standardizedRoot.toFixed(2)}, outside ${STANDARDIZED_ERROR_BAND[0]} to ${STANDARDIZED_ERROR_BAND[1]}: the kriging variance is ${standardizedRoot > 1 ? 'too small' : 'too large'}`
    );
  if (Number.isFinite(summary.coverage) && summary.coverage < MINIMUM_STANDARDIZED_COVERAGE)
    poorFitReasons.push(
      `only ${(100 * summary.coverage).toFixed(0)} percent of leave-one-out errors are within 1.96 standard errors (below ${100 * MINIMUM_STANDARDIZED_COVERAGE} percent)`
    );
  if (poorFitReasons.length)
    addCatalogDiagnostic(
      diagnostics,
      'poor-model-fit',
      'warning',
      `The variogram model is not well identified or does not describe the data: ${poorFitReasons.join('; ')}. Read the predictions and the kriging variance with care.`,
      poorFitReasons.length
    );
  let mean = 0;
  for (const value of sample.value) mean += value;
  mean /= sample.count;
  let sumOfSquares = 0;
  for (const value of sample.value) sumOfSquares += (value - mean) ** 2;
  return {
    tables: [cellTable, validationTable, variogramTable],
    diagnostics,
    summary: {
      ...gpuSummary,
      sampleCount: sample.count,
      cellCount: grid.columns * grid.rows,
      model,
      nugget,
      sill,
      range,
      fitCriterion,
      maximumLag,
      crossValidationCount: gpuSummary['crossValidationCount'] ?? summary.count,
      rootMeanSquareError: gpuSummary['rootMeanSquareError'] ?? summary.rootMeanSquareError,
      meanError: gpuSummary['meanError'] ?? summary.meanError,
      meanAbsoluteError: gpuSummary['meanAbsoluteError'] ?? summary.meanAbsoluteError,
      meanStandardizedError: summary.meanStandardizedError,
      rootMeanSquareStandardizedError: summary.rootMeanSquareStandardizedError,
      coverage95: summary.coverage,
      valueStandardDeviation: Math.sqrt(sumOfSquares / (sample.count - 1)),
      nuggetToSillRatio,
      // Run-time caution `poor-model-fit`: poorModelFit is 1 when any check fails and
      // poorModelFitReasons lists them (empty otherwise).
      poorModelFit: poorFitReasons.length ? 1 : 0,
      poorModelFitReasons: poorFitReasons.join('; ')
    }
  };
}

function runKrigingWithDiagnostics(inputs: Inputs, values: Values): CatalogRunOutput {
  const prepared = prepareCuratedKriging(inputs, values);
  const {
    sample,
    settings,
    context: {grid}
  } = prepared;
  const cellSize = grid.cellSize;
  let field: ReturnType<typeof computeKrigingField>;
  try {
    field = computeKrigingField(sample, grid, settings);
  } catch (failure) {
    if (failure instanceof KeplerAllocationError) {
      const cells = grid.columns * grid.rows;
      const work = cells * sample.count;
      throw new KeplerAllocationError('capacity', {
        cause: failure,
        message: `${BINDING_ID}: ${cells} cells times ${sample.count} samples is ${work} units of work, over the CPU reference limit of ${CPU_WORK_BOUND}; raise cellSize to at least ${Math.ceil(cellSize * Math.sqrt(work / CPU_WORK_BOUND))} or use fewer samples`
      });
    }
    throw failure;
  }
  return finishCuratedKriging(inputs, prepared, field, computeLeaveOneOut(sample, settings));
}

async function runGpuKrigingWithDiagnostics(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const prepared = prepareCuratedKriging(inputs, values);
  const result = await computeCuratedKrigingGpu(
    context,
    prepared.sample,
    prepared.context.grid,
    prepared.settings
  );
  return {
    ...finishCuratedKriging(inputs, prepared, result.field, result.leaveOneOut, result.summary),
    dispatchedContributors: ['GPUKriging']
  };
}

/**
 * Kriging with diagnostics: an empirical variogram, a fitted model, ordinary kriging prediction and
 * variance on a grid, and leave-one-out cross-validation errors.
 */
export const CURATED_KRIGING_WITH_DIAGNOSTICS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Kriging prediction, variance and diagnostics',
  domain: 'interpolation',
  summary:
    'Fits a variogram to the measured values, predicts the value and the kriging variance on a grid by ordinary kriging, and predicts every measurement from the others to report the root mean square error, the mean error and the mean standardised error.',
  capabilityIds: ['density-and-interpolation', 'coverage-local-prediction'],
  contributorNames: ['GPUKriging'],
  gpuContributorNames: ['GPUKriging'],
  engine: 'gpu-or-cpu',
  inputs: [makeInputSpec('points', 'Measurements', ['point'], MAXIMUM_SAMPLES, MAXIMUM_SAMPLES)],
  parameters: PARAMETERS,
  outputs: [
    makeOutputSpec(
      'kriging-cells',
      'Kriged cells',
      'polygon',
      ['cell_id', 'column', 'row'],
      MAXIMUM_CELLS
    ),
    makeOutputSpec(
      'cross-validation',
      'Leave-one-out cross-validation',
      'point',
      ['sample_index', 'source_index', 'source_id'],
      MAXIMUM_SAMPLES
    ),
    makeOutputSpec('variogram', 'Empirical and fitted variogram', 'none', ['bin'], 60)
  ],
  cleanup: `${PLATFORM3_GPU_OR_CPU_CLEANUP} Cleanup also runs on dispatch or readback failure; the caller retains its device.`,
  declaredBehavior: [
    'The cross-validation table lists every input feature in input order with a status: analysed, merged-into-earlier-sample (a sample at exactly the same location as an earlier one; it carries its own value and the location prediction, and no error), missing-value or missing-geometry. Samples at exactly the same location are merged to their mean value and counted, so each location is one sample in the fit and the summary; fewer than 8 distinct locations throw.',
    'The empirical semivariogram is the Matheron estimator in equal-width lag bins up to the largest lag (a third of the data diagonal unless set), in metres in a local frame for EPSG:4326 and coordinate units for planar; a model (spherical, exponential or gaussian) is fitted by Cressie-weighted least squares and, with model auto, the one with the smallest common criterion is kept. Parameters are rounded to float32 by the fit.',
    'A fit needs two populated bins and a positive range, otherwise it throws with that message; constant values cannot be fitted and throw; variogram stated skips the fit and uses the given model, nugget, partial sill and range.',
    'Model checks (screening bounds, not tests) raise the run-time caution poor-model-fit: summary key poorModelFit is 1 and poorModelFitReasons lists the reasons, with a diagnostic, when the range exceeds the largest lag (not identified: white noise can give a range several times the lag), the nugget is above 75 percent of the total sill (Cambardella et al. 1994: weak spatial dependence), the root mean square standardised leave-one-out error is outside 0.8 to 1.25 (it is 1 when the kriging variance is right; the band allows a variance 56 percent off at its ends), or fewer than 90 percent of leave-one-out errors are within 1.96 standard errors (nominal 95; about 1.5 binomial deviations below it for 40 samples). These incomplete isotropic screens may miss white noise, trends or periodic data; no warning does not establish a good or identified fit. A high omnidirectional nugget can mask anisotropy. The bounds are fixed, so they are noisier for fewer than about 30 samples.',
    'Ordinary kriging uses the nearest samples within the search radius (at most the neighbour count) per cell; locations are ordered by local-frame x then y before neighbour selection, so equal-distance ties select the same locations under input reordering. A cell whose system is singular, or with no sample in reach, is nodata and counted.',
    'Leave-one-out predicts each sample from the nearest others within the search radius (at most the neighbour count, so it is local, as the grid is) with the same variogram, without refitting; the error is predicted minus observed and the standardised error is the error over the square root of the kriging variance; a failed prediction is counted and left out of the summary.',
    'The GPU path dispatches addInterpolationDiagnosticsRecipe (GPUKriging plus residual and reduction nodes) on the same local-frame samples and fitted or stated model; the empirical variogram and fitting remain on the CPU. It supports 4 to 16 neighbours; larger requests fall back to the CPU with a visible reason, never clamp. Before GPU allocation, measurement origins are compared for float32 round-trip error (at most 1e-6 * max(1, abs(value)) in measurement units); unrepresentable measurements or distinct locations that collide in float32 cause a visible CPU fallback. GPU solves use float32 and reject ill-conditioned systems as nodata; recipe stage invalid/uncertain counts are retained in the summary. GPU sample_count is null because GPUKriging does not expose retained neighbourhood counts.',
    'At most 2000 samples and 40000 cells; CPU sample times cell work above 60 million throws a capacity error that names the binding, the limit and the cell size that would fit.'
  ],
  limitation:
    'GPU float32 solves can mark ill-conditioned cells and leave-one-out predictions nodata where the float64 CPU reference succeeds; the GPU neighbourhood limit is 16, versus 64 on the CPU. Isotropic model with no trend, anisotropy or block kriging; the variogram is fitted once and not refitted inside each leave-one-out step, so the errors are slightly optimistic. The kriging neighbourhood is local, so the system differs from a global solve when the neighbour count is below the sample count.',
  run: (inputs, values) => runKrigingWithDiagnostics(inputs, values),
  runGpu: runGpuKrigingWithDiagnostics
};
