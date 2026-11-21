// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ProbitRegressionOperationProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import fixture from './probit-spatial-lag-fixture.json';

export function getProbitSpatialLagInput(data = fixture) {
  return {
    predictors: Float32Array.from(data.predictors),
    response: Float32Array.from(data.response),
    mask: Uint32Array.from(data.mask),
    spatialLag: {
      weights: {
        offsets: Uint32Array.from(data.offsets),
        neighbors: Uint32Array.from(data.neighbors),
        weights: Float32Array.from(data.weights)
      }
    }
  };
}

export const PROBIT_SPATIAL_LAG_PROPS: ProbitRegressionOperationProps = {
  rowCount: 60,
  predictorCount: 2,
  predictors: 'predictors',
  response: 'response',
  mask: 'mask',
  spatialDiagnostics: {
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {tests: 'tests', status: 'diagnosticStatus'}
  },
  spatialLag: {
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {
      coefficients: 'lagCoefficients',
      standardErrors: 'lagStandardErrors',
      status: 'lagStatus'
    }
  },
  output: {
    coefficients: 'coefficients',
    standardErrors: 'standardErrors',
    fitted: 'fitted',
    marginalEffects: 'marginalEffects',
    summary: 'summary',
    status: 'status'
  }
};
