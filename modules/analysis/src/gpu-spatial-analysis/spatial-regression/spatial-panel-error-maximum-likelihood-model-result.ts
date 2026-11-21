// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegressionStatusReadback} from './regression-status-readback';

import type {GPURegressionModelResult} from '../../analysis-core/contracts/model-result';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialErrorMaximumLikelihoodModelResult,
  type GPUSpatialMaximumLikelihoodReadbackInput,
  type GPUSpatialMaximumLikelihoodModelResultOptions
} from './spatial-maximum-likelihood-model-result';

/** Balanced-panel shape required to account for the profiled unit intercepts. */
export type GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions =
  GPUSpatialMaximumLikelihoodModelResultOptions & {entityCount: number};

/** The intercept table row is absorbed; parameterCount includes all profiled entity effects. */
export type GPUSpatialPanelErrorMaximumLikelihoodModelResult = Omit<
  GPURegressionModelResult,
  'kind'
> & {
  kind: 'spatial-panel-error-maximum-likelihood';
  absorbedEntityCount: number;
};

/**
 * Reads unit fixed-effects panel-error ML, preserving the published Gaussian likelihood
 * `-NT/2 * (log(2*pi*sigma2) + 1) + T*log|I-lambda*W|` and NaN absorbed-intercept row.
 * This is the NT variance convention, not spreg's reported panel-error `logll` or the panel-lag
 * convention. All information criteria remain unavailable (`null`). With p predictors and N
 * entities, k = N+p+1 includes lambda and all N absorbed intercepts, so residual df = NT-N-p-1.
 * `sigmaSquaredMaximumLikelihood` (RSS/NT) scales the published standard errors; `sigmaSquared`
 * is the descriptive RSS/df form. RSS here is the spatially filtered residual sum of squares.
 */
export function getGPUSpatialPanelErrorMaximumLikelihoodModelResult(
  readback: GPUSpatialMaximumLikelihoodReadbackInput,
  options: GPUSpatialPanelErrorMaximumLikelihoodModelResultOptions
): GPUSpatialPanelErrorMaximumLikelihoodModelResult {
  const statusCode = validateRegressionStatusReadback(
    'getGPUSpatialPanelErrorMaximumLikelihoodModelResult',
    readback.status
  );
  const {table, summary} = readback;
  const status = new Uint32Array([statusCode]);
  const result = getGPUSpatialErrorMaximumLikelihoodModelResult({table, summary, status}, options);
  const id = 'spatial-panel-error-maximum-likelihood-reader';
  validateCapacity(id, 'entityCount', options.entityCount);
  validateCapacity(
    id,
    'rowCount / entityCount (balanced panel periods)',
    result.rowCount / options.entityCount,
    2
  );
  // Replace the single absorbed placeholder intercept by N profiled intercepts.
  const parameterCount = result.parameterCount - 1 + options.entityCount;
  const degreesOfFreedom = result.rowCount - parameterCount;
  return {
    ...result,
    kind: 'spatial-panel-error-maximum-likelihood',
    absorbedEntityCount: options.entityCount,
    parameterCount,
    degreesOfFreedom,
    sigmaSquared: degreesOfFreedom > 0 ? result.residualSumOfSquares / degreesOfFreedom : NaN,
    akaikeInformationCriterion: null,
    schwarzCriterion: null,
    correctedAkaikeInformationCriterion: null
  };
}
