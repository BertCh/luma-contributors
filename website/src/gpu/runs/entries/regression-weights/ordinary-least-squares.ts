// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOrdinaryLeastSquares on tract data with two predictors. Left: the residuals the operation
// wrote per tract (clustered when the response is spatially dependent, which is what OLS cannot
// model). Right: the fitted coefficients as bars with the true values as dots; `ridgeLambda`
// shrinks the slopes.

import {
  GPUOrdinaryLeastSquares,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  getGPUOrdinaryLeastSquaresParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './ordinary-least-squares.parameters';
import {
  createCoefficientPanel,
  createRegressionInputs,
  createResponse,
  drawCoefficientPanel,
  drawTractMap,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';

export default defineOperationRun({
  parameters,
  caption: 'Residual map and fitted coefficients (bars) against the true ones (dots).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs) {
    const count = inputs.tracts.count;
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', count);
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const coefficients = builder.allocate('coefficients', 'float32', 3);
    const standardErrors = builder.allocate('standard-errors', 'float32', 3);
    const tStatistics = builder.allocate('t-statistics', 'float32', 3);
    const summary = builder.allocate(
      'summary',
      'float32',
      GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
    );
    const status = builder.allocate('status', 'uint32', 1);
    const residuals = builder.allocate('residuals', 'float32', count);
    const fitted = builder.allocate('fitted', 'float32', count);
    builder.graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        parameters: parameterBuffer.view,
        output: {
          status: {code: status.view},
          coefficients: coefficients.view,
          standardErrors: standardErrors.view,
          tStatistics: tStatistics.view,
          summary: summary.view,
          residuals: residuals.view,
          fitted: fitted.view
        }
      })
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals,
        panel: createCoefficientPanel(builder, coefficients, 1, TRUE_COEFFICIENTS)
      },
      update: current => {
        response.write(createResponse(inputs, {noise: current.noise, lagRho: current.rho}));
        parameterBuffer.write(
          getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: current.ridgeLambda})
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.residuals),
    ...drawCoefficientPanel(outputs.panel)
  ]
});
