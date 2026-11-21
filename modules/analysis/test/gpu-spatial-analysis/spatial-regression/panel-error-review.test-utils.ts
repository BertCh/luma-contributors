// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUSpatialPanelErrorMaximumLikelihoodInput} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import type {SpatialPanelErrorMaximumLikelihoodOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';

export function getSmallSlopeInput(delta: number): CPUSpatialPanelErrorMaximumLikelihoodInput {
  return {
    predictors: Float32Array.from([-1, 1, -1, 1].flatMap(value => [value, 0, 0])),
    response: Float32Array.from([-1, 1, -1, 1].flatMap(value => [delta * value, -value, value])),
    weights: {
      offsets: Uint32Array.of(0, 1, 2, 3),
      neighbors: Uint32Array.of(1, 2, 0),
      weights: Float32Array.of(1, 1, 1)
    }
  };
}

export function getReviewProps(
  entityCount: number,
  periodCount: number,
  predictorCount = 1
): SpatialPanelErrorMaximumLikelihoodOperationProps {
  return {
    entityCount,
    periodCount,
    predictorCount,
    predictors: 'predictors',
    response: 'response',
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    maximumPower: 8,
    output: {
      table: 'table',
      summary: 'summary',
      residuals: 'residuals',
      powerTraceStandardErrors: 'traceErrors',
      status: {code: 'code', converged: 'converged'}
    },
    diagnostics: {tests: 'tests', status: 'diagnosticStatus'}
  };
}

export const reviewOutputNames = [
  'table',
  'summary',
  'residuals',
  'traceErrors',
  'code',
  'converged',
  'tests',
  'diagnosticStatus'
];

export function getReviewInputs(input: CPUSpatialPanelErrorMaximumLikelihoodInput) {
  return {
    predictors: {format: 'float32' as const, values: input.predictors},
    response: {format: 'float32' as const, values: input.response},
    offsets: {format: 'uint32' as const, values: input.weights.offsets},
    neighbors: {format: 'uint32' as const, values: input.weights.neighbors},
    weights: {format: 'float32' as const, values: input.weights.weights}
  };
}
