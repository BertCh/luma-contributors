// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  buildVecchiaConditioningSets,
  type GPUVecchiaCovariance
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  evaluateValidationVecchia,
  fitValidationVecchia,
  type ValidationVecchiaFit
} from './catalog-validation-vecchia';
import {makeLocalFrame} from './catalog-local-frame';
import type {KeplerVariogramFit} from './kriging-variogram-fit';
import {MAXIMUM_VALIDATION_ROWS} from './catalog-validation-common';

/** Fits the spatial covariance only; a space-time model must be fitted in the time-aware catalog. */
export async function fitKeplerVecchiaVariogram(
  device: Device,
  longitudes: ArrayLike<number>,
  latitudes: ArrayLike<number>,
  values: ArrayLike<number>,
  options: {
    method: 'likelihood' | 'fisher';
    initial: GPUVecchiaCovariance;
    neighborCount?: number;
    maximumIterations?: number;
  }
): Promise<
  KeplerVariogramFit &
    Pick<
      ValidationVecchiaFit,
      'stopReason' | 'stallReason' | 'iterationLimitReached' | 'history' | 'logLikelihood'
    > & {nonconvergenceDiagnostic?: string}
> {
  const rows = Array.from({length: values.length}, (_, index) => index).filter(
    index =>
      Number.isFinite(longitudes[index]) &&
      Number.isFinite(latitudes[index]) &&
      Math.abs(longitudes[index]) <= 180 &&
      Math.abs(latitudes[index]) <= 90 &&
      Number.isFinite(values[index])
  );
  if (rows.length < 3)
    throw new Error(
      `Vecchia variogram fitting has fewer than 3 valid samples (found ${rows.length})`
    );
  if (rows.length > MAXIMUM_VALIDATION_ROWS)
    throw new Error(
      `Vecchia variogram fitting supports at most ${MAXIMUM_VALIDATION_ROWS} valid samples`
    );
  if (rows.every(index => values[index] === values[rows[0]]))
    throw new Error('Vecchia variogram fitting response has no variance among valid samples');
  const coordinates = rows.map(index => [longitudes[index], latitudes[index]] as const);
  const frame = makeLocalFrame('EPSG:4326', coordinates, {shift: 'minimum'});
  const positions = Float32Array.from(coordinates.flatMap(position => frame.forward(position)));
  if (!positions.every(Number.isFinite))
    throw new Error('Vecchia variogram positions must project to finite Float32 coordinates');
  const mean = rows.reduce((sum, index) => sum + values[index], 0) / rows.length;
  const observations = Float32Array.from(rows.flatMap(index => [values[index] - mean, 0, 0, 1]));
  if (rows.every((_, index) => observations[index * 4] === observations[0]))
    throw new Error('Vecchia variogram fitting response has no variance at Float32 precision');
  const neighborCount = options.neighborCount ?? 8;
  if (!Number.isInteger(neighborCount) || neighborCount < 1 || neighborCount > 32)
    throw new Error('Vecchia variogram neighborCount must be 1 to 32');
  const neighborhoods = buildVecchiaConditioningSets({positions, neighborCount});
  const data = {positions, observations, neighborhoods, neighborCount};
  const result = await fitValidationVecchia(
    device,
    data,
    options.initial,
    options.method,
    options.maximumIterations ?? (options.method === 'fisher' ? 12 : 48)
  );
  if (result.failed)
    throw new Error(
      `Vecchia variogram fitting failed (${result.stopReason ?? 'unusable-likelihood'}); change the starting covariance or remove duplicate samples`
    );
  const evaluation = await evaluateValidationVecchia(
    device,
    data,
    result.covariance,
    options.method === 'fisher'
  );
  if (!Number.isFinite(evaluation.logLikelihood))
    throw new Error('Vecchia variogram final likelihood is invalid');
  const candidate = {
    model: result.covariance.model,
    nugget: Math.fround(result.covariance.nugget ?? 0),
    sill: Math.fround(result.covariance.sill),
    rangeMeters: Math.fround(result.covariance.range),
    criterion: -evaluation.logLikelihood
  };
  return {
    ...candidate,
    method: options.method === 'fisher' ? 'vecchia-fisher' : 'vecchia-likelihood',
    weighting: 'conditional-likelihood',
    candidates: [candidate],
    binCount: 0,
    pairCount: neighborhoods.neighbors.length,
    ...(result.stopReason === undefined
      ? {}
      : {
          stopReason: result.stopReason,
          ...(result.stallReason === undefined ? {} : {stallReason: result.stallReason}),
          iterationLimitReached: result.iterationLimitReached,
          history: result.history,
          logLikelihood: result.logLikelihood,
          ...(!result.converged
            ? {
                nonconvergenceDiagnostic: `Vecchia fit stopped: ${result.stopReason}${result.stallReason ? ` (${result.stallReason})` : ''}; retained last usable covariance after ${result.budgetUsed} accepted steps`
              }
            : {})
        }),
    diagnostics: {
      valid_rows: rows.length,
      excluded_rows: values.length - rows.length,
      known_mean: mean,
      neighbor_count: neighborCount,
      converged: Number(result.converged),
      ...(result.iterationLimitReached === undefined
        ? {}
        : {iteration_limit: Number(result.iterationLimitReached)}),
      budget_used: result.budgetUsed,
      log_likelihood: evaluation.logLikelihood,
      failed_sites: evaluation.failedSiteCount
    }
  };
}
