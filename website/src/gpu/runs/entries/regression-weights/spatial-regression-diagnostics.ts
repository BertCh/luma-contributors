// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialRegressionDiagnostics tests the residuals of an OLS fit for spatial dependence. The OLS
// fit runs in the same graph. Left: the residual map. Right: the six test statistics the operation
// wrote (LM lag, LM error, robust lag, robust error, LM SARMA and Moran's I z); the lag and error
// tests rise with the matching spatial process in the data.

import {
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-regression-diagnostics.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  MAP_SHARE,
  uploadRegressionMap,
  uploadStandardizedWeights
} from './shared/regression';
import {addTableColumn} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Residual map (left) and the spatial dependence tests on it (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const weights = uploadStandardizedWeights(builder, inputs.weights);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    // A third of the tracts is left out of the fit; their links are dropped, which leaves rows of
    // the row-standardized weights short of 1 unless `restandardizeRows` rescales them.
    const mask = builder.upload(
      'mask',
      'uint32',
      Uint32Array.from({length: rowCount}, (_, row) =>
        ((row * 2654435761) >>> 0) % 100 < 35 ? 0 : 1
      )
    );
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        mask: mask.view,
        output: {
          status: {code: builder.allocate('ols-status', 'uint32', 1).view},
          coefficients: builder.allocate('ols-coefficients', 'float32', 3).view,
          standardErrors: builder.allocate('ols-standard-errors', 'float32', 3).view,
          tStatistics: builder.allocate('ols-t-statistics', 'float32', 3).view,
          summary: builder.allocate(
            'ols-summary',
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ).view,
          residuals: residuals.view
        }
      })
    );
    const tests = builder.allocate('tests', 'float32', 18);
    builder.graph.add(
      new GPUSpatialRegressionDiagnostics({
        weights: weights.input,
        predictors: predictors.view,
        response: response.view,
        residuals: residuals.view,
        mask: mask.view,
        predictorCount: 2,
        restandardizeRows: values.restandardizeRows === 1,
        output: {
          tests: tests.view,
          summary: builder.allocate('summary', 'float32', 16).view,
          status: {code: builder.allocate('status', 'uint32', 1).view}
        }
      })
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals,
        statistics: addTableColumn(builder, 'tests', tests, 3, 0, 6)
      },
      update: current =>
        response.write(
          createResponse(inputs, {
            noise: 0.4,
            lagRho: current.lagRho,
            errorLambda: current.errorLambda > 0 ? current.errorLambda : undefined
          })
        )
    };
  },
  rebuildOn: ['restandardizeRows'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.residuals),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.statistics,
      ramp: 'sequential',
      range: [0, 40],
      rectangle: [MAP_SHARE + 0.08, 0.14, 0.97, 0.86]
    }
  ]
});
