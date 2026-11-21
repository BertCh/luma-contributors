// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialErrorMaximumLikelihoodOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import fixtures from './likelihood-flat-profile-fixtures.json';

export {fixtures};

export function getFlatProfileInput(fixture: (typeof fixtures)[number]) {
  return {
    offsets: {format: 'uint32' as const, values: Uint32Array.from({length: 9}, (_, row) => row)},
    neighbors: {format: 'uint32' as const, values: Uint32Array.from({length: 8}, (_, row) => row)},
    weights: {format: 'float32' as const, values: new Float32Array(8).fill(fixture.scalarWeight)},
    eigenvalues: {
      format: 'float32' as const,
      values: new Float32Array(8).fill(fixture.scalarWeight)
    },
    predictors: {format: 'float32' as const, values: Float32Array.from(fixture.predictors)},
    response: {format: 'float32' as const, values: Float32Array.from(fixture.response)}
  };
}

export function getFlatProfileProps(
  fixture: (typeof fixtures)[number]
): SpatialErrorMaximumLikelihoodOperationProps {
  return {
    rowCount: 8,
    predictorCount: 1,
    maximumPower: fixture.maximumPower,
    logDeterminant: 'exact',
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    eigenvalues: 'eigenvalues',
    output: {
      status: {code: 'status', converged: 'converged'},
      table: 'table',
      summary: 'summary',
      residuals: 'residuals',
      powerTraceStandardErrors: 'powerTraceStandardErrors'
    }
  };
}
